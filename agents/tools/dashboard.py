"""Constructeurs de visuels du Dashboard Generator : déterministes, sans LLM.

Chaque ``build_*`` retourne ``(visuals, excluded)`` ; rien n'est jamais inventé :
toute valeur affichée vient du KPIReport, de l'AnalystReport ou d'une agrégation
tracée sur la table nettoyée (``origin="dashboard_computed"``).
Gouvernance : pas de coordonnées individuelles, cellules < ``min_cell_rows`` masquées,
colonnes de fuite de cible jamais exposées.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from agents.bi_analyst.models import AnalystReport
from agents.dashboard.catalog import LEAKAGE_COLUMNS, LEAKAGE_KPIS
from agents.dashboard.models import (
    DashboardOptions,
    Encoding,
    ExcludedVisual,
    KPICard,
    VisualReference,
    VisualSpec,
)
from agents.kpi.models import KPIReport

# Clés interdites dans ``VisualSpec.data`` (PII / fuite de cible).
# ``lat_mean`` / ``lon_mean`` (moyennes par zone) restent autorisées.
_FORBIDDEN_KEYS = {
    "customerid", "latitude", "longitude", "lat", "lon", "lng", "latlong",
    "zipcode", "zip", "phone", "email", "name",
} | {re.sub(r"[^a-z0-9]", "", c) for c in LEAKAGE_COLUMNS}

_LOWER_IS_BETTER = ("churn", "lost", "cancel", "complaint", "attrition")
_TENURE_BINS = [-1, 6, 12, 24, 36, 48, 60, np.inf]
_TENURE_LABELS = ["0-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61+"]


# ── Utilitaires ──────────────────────────────────────────────────────────────

def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    index = {norm(c): c for c in df.columns}
    for alias in aliases:
        found = index.get(norm(alias))
        if found is not None:
            return found
    return None


def churn_series(df: pd.DataFrame) -> pd.Series | None:
    """Flag churn 0/1 (Churn Value / Churn Label / Churn), ou None."""
    col = find_col(df, ["churn_value", "churn", "churn_label"])
    if col is None:
        return None
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return (s.fillna(0) > 0).astype(int)
    return s.astype(str).str.strip().str.lower().isin({"yes", "true", "1", "oui", "y"}).astype(int)


def clean_value(v: Any, ndigits: int = 2) -> Any:
    """Types Python natifs, NaN/inf -> None, flottants arrondis (JSON sûr)."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, ndigits)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if v is pd.NA or v is pd.NaT:
        return None
    return v


def clean_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: clean_value(v) for k, v in r.items()} for r in rows]


def _slug(*parts: str) -> str:
    return "_".join(re.sub(r"[^a-z0-9]+", "_", str(p).lower()).strip("_") for p in parts if p)


def validate_visual(v: VisualSpec) -> list[str]:
    """Contrôles de cohérence d'un visuel ; liste vide = valide."""
    problems: list[str] = []
    if v.type == "kpi_card":
        if v.kpi is None:
            problems.append("kpi_card sans contenu 'kpi'")
        return problems
    if v.type != "text" and not v.data:
        problems.append("données vides")
    keys = {k for row in v.data for k in row}
    bad = sorted(k for k in keys if norm(k) in _FORBIDDEN_KEYS)
    if bad:
        problems.append(f"colonnes interdites (PII / fuite de cible) : {bad}")
    if v.data:
        for field in (v.encoding.x, v.encoding.y, v.encoding.color, v.encoding.size, v.encoding.geo_key):
            if field and field not in keys:
                problems.append(f"encodage '{field}' absent des données")
    for row in v.data:
        for val in row.values():
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                problems.append("valeur non finie dans les données")
                break
        else:
            continue
        break
    return problems


# ── Cartes KPI ───────────────────────────────────────────────────────────────

def build_kpi_cards(kpi: KPIReport) -> tuple[list[VisualSpec], list[ExcludedVisual]]:
    visuals: list[VisualSpec] = []
    excluded: list[ExcludedVisual] = []
    for k in kpi.kpis:
        if k.id in LEAKAGE_KPIS:
            excluded.append(ExcludedVisual(
                candidate=k.id, reason="fuite de cible : dérivé du churn, exclu des cartes de pilotage"))
            continue
        if k.status == "not_computable" or k.value is None:
            missing = ", ".join(f"{r}: {a}" for r, a in k.missing_inputs.items()) or "entrées manquantes"
            excluded.append(ExcludedVisual(candidate=k.id, reason=f"non calculable ({missing})"))
            continue
        lower = any(tok in k.id.lower() for tok in _LOWER_IS_BETTER)
        visuals.append(VisualSpec(
            id=_slug("kpi", k.id), type="kpi_card", title=k.name, section="overview", span=3,
            kpi=KPICard(
                kpi_id=k.id, name=k.name, value=clean_value(k.value), unit=k.unit,
                status="suspect" if k.status == "suspect" else "ok",
                higher_is_better=False if lower else None),
            note=k.definition, reliable=k.status != "suspect", warnings=list(k.warnings),
            references=VisualReference(
                kpi_id=k.id, source_columns=sorted(set(k.columns_used.values())),
                formula=k.formula, origin="kpi_report")))
    return visuals, excluded


# ── Ventilations (barres) ────────────────────────────────────────────────────

def build_breakdown_bars(
    kpi: KPIReport, analyst: AnalystReport, opts: DashboardOptions,
) -> tuple[list[VisualSpec], list[ExcludedVisual]]:
    visuals: list[VisualSpec] = []
    excluded: list[ExcludedVisual] = []
    insight_dims: dict[str, set[str]] = {}
    for ins in analyst.insights:
        for st in ins.segment_stats:
            insight_dims.setdefault(st.dimension, set()).add(ins.id)

    for k in kpi.kpis:
        if k.id in LEAKAGE_KPIS or k.status == "not_computable" or not k.breakdowns:
            continue
        dims = sorted(k.breakdowns, key=lambda d: (d not in insight_dims, d))
        for dim in dims:
            segs = k.breakdowns[dim]
            kept = [s for s in segs if s.value is not None and s.rows >= opts.min_cell_rows]
            masked = len(segs) - len(kept)
            if len(kept) < 2:
                excluded.append(ExcludedVisual(
                    candidate=f"{k.id} par {dim}",
                    reason=f"moins de 2 segments d'effectif ≥ {opts.min_cell_rows}"))
                continue
            kept = sorted(kept, key=lambda s: s.value, reverse=True)[: opts.top_n_segments]
            warnings = [f"{masked} segment(s) masqué(s) (effectif < {opts.min_cell_rows})"] if masked else []
            visuals.append(VisualSpec(
                id=_slug("bar", k.id, dim), type="bar", title=f"{k.name} par {dim}",
                subtitle=f"Référence globale : {clean_value(k.value)} {k.unit}".strip(),
                section="segments", span=6,
                data=clean_rows([{"segment": s.segment, "value": s.value, "rows": s.rows,
                                  "global": k.value} for s in kept]),
                encoding=Encoding(x="segment", y="value", sort="desc"),
                warnings=warnings,
                references=VisualReference(
                    kpi_id=k.id, insight_ids=sorted(insight_dims.get(dim, set())),
                    source_columns=[dim], formula=k.formula,
                    aggregation=f"{k.name} par {dim}", origin="kpi_report")))
    return _prioritize_bars(visuals, kpi), excluded


_DERIVED_TOKENS = ("churn", "lost")  # KPIs dérivés du churn : barres très proches de celles du taux


def _prioritize_bars(visuals: list[VisualSpec], kpi: KPIReport) -> list[VisualSpec]:
    """Ordre d'affichage des barres (la limite max_visuals retire la fin de liste).

    0 : KPI principal (churn_rate) sur toutes ses dimensions ;
    1 : autres KPIs indépendants du churn, sur leur dimension la plus pertinente seulement ;
    2 : le reste (KPIs dérivés du churn, dimensions secondaires).
    """
    primary = "churn_rate" if any(k.id == "churn_rate" for k in kpi.kpis) else None
    seen_first: set[str] = set()

    def score(v: VisualSpec) -> int:
        kid = v.references.kpi_id or ""
        if primary and kid == primary:
            return 0
        if any(tok in kid for tok in _DERIVED_TOKENS):
            return 2
        if kid not in seen_first:
            seen_first.add(kid)
            return 1
        return 2

    scored = [(score(v), i, v) for i, v in enumerate(visuals)]
    return [v for _, _, v in sorted(scored, key=lambda t: (t[0], t[1]))]


# ── Temporel : ancienneté et dates ───────────────────────────────────────────

def build_temporal(
    frame: pd.DataFrame | None, opts: DashboardOptions,
) -> tuple[list[VisualSpec], list[ExcludedVisual]]:
    if frame is None:
        return [], [ExcludedVisual(candidate="temporal", reason="table nettoyée indisponible")]
    flag = churn_series(frame)
    if flag is None:
        return [], [ExcludedVisual(candidate="temporal", reason="colonne de churn introuvable")]
    visuals: list[VisualSpec] = []
    excluded: list[ExcludedVisual] = []

    tenure = find_col(frame, ["tenure_months", "tenure", "tenure_month"])
    if tenure is None:
        excluded.append(ExcludedVisual(candidate="churn par ancienneté", reason="colonne d'ancienneté introuvable"))
    else:
        t = pd.to_numeric(frame[tenure], errors="coerce")
        bucket = pd.cut(t, bins=_TENURE_BINS, labels=_TENURE_LABELS)
        g = pd.DataFrame({"b": bucket, "f": flag}).dropna().groupby("b", observed=True)["f"].agg(["size", "sum", "mean"])
        g = g[g["size"] >= opts.min_cell_rows]
        if len(g) < 3:
            excluded.append(ExcludedVisual(candidate="churn par ancienneté", reason="moins de 3 tranches fiables"))
        else:
            rows = [{"tenure_band": str(i), "churn_rate": r["mean"] * 100, "n": int(r["size"]),
                     "n_churned": int(r["sum"])} for i, r in g.iterrows()]
            visuals.append(VisualSpec(
                id="line_churn_by_tenure", type="line", title="Taux de churn par tranche d'ancienneté",
                subtitle="Mois d'abonnement", section="temporal", span=6, data=clean_rows(rows),
                encoding=Encoding(x="tenure_band", y="churn_rate"),
                note="Taux de churn observé par tranche d'ancienneté (association, pas causalité).",
                references=VisualReference(
                    kpi_id="churn_rate", source_columns=[tenure, "churn flag"],
                    aggregation="mean(churn) par tranche de tenure", origin="dashboard_computed")))

    date_col = find_col(frame, ["signup_date", "start_date", "subscription_date", "activation_date",
                                "contract_start", "date"])
    if date_col is not None:
        d = pd.to_datetime(frame[date_col], errors="coerce")
        m = pd.DataFrame({"m": d.dt.to_period("M").astype(str), "f": flag})[d.notna()]
        g = m.groupby("m")["f"].agg(["size", "sum", "mean"])
        g = g[g["size"] >= opts.min_cell_rows]
        if len(g) >= 3:
            rows = [{"month": i, "churn_rate": r["mean"] * 100, "n": int(r["size"])} for i, r in g.iterrows()]
            visuals.append(VisualSpec(
                id="line_churn_by_month", type="line", title="Taux de churn par mois d'entrée",
                section="temporal", span=6, data=clean_rows(rows),
                encoding=Encoding(x="month", y="churn_rate"),
                references=VisualReference(
                    kpi_id="churn_rate", source_columns=[date_col, "churn flag"],
                    aggregation="mean(churn) par mois", origin="dashboard_computed")))
    else:
        excluded.append(ExcludedVisual(
            candidate="série calendaire", reason="aucune colonne de date : la dimension temporelle repose sur l'ancienneté"))
    return visuals, excluded


# ── Géospatial agrégé ────────────────────────────────────────────────────────

def _geo_visual(
    frame: pd.DataFrame, flag: pd.Series, state: str | None, city: str | None,
    use_city: bool, opts: DashboardOptions, global_rate: float | None,
) -> tuple[VisualSpec | None, str]:
    """Agrège par zone ; retourne (visuel, raison d'échec). Aucun point individuel."""
    keys = [c for c in ([state] if state else []) + ([city] if use_city else [])] or [state or city]
    work = frame[keys].copy()
    work["_f"] = flag
    lat = find_col(frame, ["latitude", "lat"])
    lon = find_col(frame, ["longitude", "lon", "lng"])
    if use_city and lat and lon:
        work["_lat"] = pd.to_numeric(frame[lat], errors="coerce")
        work["_lon"] = pd.to_numeric(frame[lon], errors="coerce")
    agg = {"_f": ["size", "sum", "mean"]}
    if "_lat" in work:
        agg.update({"_lat": "mean", "_lon": "mean"})
    g = work.groupby(keys, observed=True).agg(agg)
    g.columns = ["n", "n_churned", "rate"] + (["lat_mean", "lon_mean"] if "_lat" in work else [])
    g = g.reset_index()
    masked = int((g["n"] < opts.min_cell_rows).sum())
    g = g[g["n"] >= opts.min_cell_rows]
    if len(g) < 2:
        return None, f"moins de 2 zones d'effectif ≥ {opts.min_cell_rows}"

    zone_col = keys[-1]
    rows = []
    for _, r in g.sort_values("rate", ascending=False).iterrows():
        row: dict[str, Any] = {"zone": str(r[zone_col]), "churn_rate": r["rate"] * 100,
                               "n": int(r["n"]), "n_churned": int(r["n_churned"])}
        if state:
            row["state"] = str(r[state])
        if use_city:
            row["city"] = str(r[city])
        if global_rate is not None:
            row["excess_rate"] = r["rate"] * 100 - global_rate
        if "lat_mean" in g:
            row["lat_mean"], row["lon_mean"] = r["lat_mean"], r["lon_mean"]
        rows.append(row)
    level = "city" if use_city else "state"
    return VisualSpec(
        id=f"geo_churn_by_{level}", type="geo_map",
        title=f"Taux de churn par {'ville' if use_city else 'état'}",
        subtitle="Agrégation par zone, aucun point individuel", section="geo", span=12, data=clean_rows(rows),
        encoding=Encoding(x="zone", y="churn_rate", color="churn_rate", size="n", geo_key=level),
        warnings=[f"{masked} zone(s) masquée(s) (effectif < {opts.min_cell_rows})"] if masked else [],
        note="Écart de churn par zone, à lire comme une association.",
        references=VisualReference(
            kpi_id="churn_rate", source_columns=list(keys), aggregation=f"mean(churn) par {level}",
            origin="dashboard_computed")), ""


def build_geo(
    frame: pd.DataFrame | None, opts: DashboardOptions, global_rate: float | None,
) -> tuple[list[VisualSpec], list[ExcludedVisual]]:
    """Carte agrégée. Repli automatique état -> ville si l'état n'a pas 2 zones fiables."""
    if frame is None:
        return [], [ExcludedVisual(candidate="geo_map", reason="table nettoyée indisponible")]
    flag = churn_series(frame)
    state = find_col(frame, ["state", "region", "province"])
    city = find_col(frame, ["city", "town"])
    if flag is None or (state is None and city is None):
        return [], [ExcludedVisual(candidate="geo_map", reason="colonne géographique (state/city) ou churn introuvable")]

    use_city = opts.geo_level == "city" and city is not None
    visual, reason = _geo_visual(frame, flag, state, city, use_city, opts, global_rate)
    if visual is None and not use_city and city is not None:
        visual, reason2 = _geo_visual(frame, flag, state, city, True, opts, global_rate)
        if visual is not None:
            visual.warnings.insert(0, "niveau état insuffisant (moins de 2 zones fiables) : repli automatique sur la ville")
        else:
            reason = f"{reason} (état) ; {reason2} (ville)"
    if visual is None:
        return [], [ExcludedVisual(candidate="geo_map", reason=reason)]
    return [visual], []


# ── Heatmap croisée ──────────────────────────────────────────────────────────

def build_heatmap(
    frame: pd.DataFrame | None, analyst: AnalystReport, opts: DashboardOptions,
) -> tuple[list[VisualSpec], list[ExcludedVisual]]:
    if frame is None:
        return [], []
    flag = churn_series(frame)
    if flag is None:
        return [], []
    cols: list[str] = []
    for ins in analyst.insights:
        for st in ins.segment_stats:
            col = find_col(frame, [st.dimension])
            if (col and col not in cols and col not in {c for c in frame.columns if norm(c) in _FORBIDDEN_KEYS}
                    and not pd.api.types.is_numeric_dtype(frame[col]) and 2 <= frame[col].nunique() <= 6):
                cols.append(col)
    if len(cols) < 2:
        return [], [ExcludedVisual(candidate="heatmap", reason="moins de 2 dimensions catégorielles issues des insights")]
    a, b = cols[:2]
    g = pd.DataFrame({"a": frame[a].astype(str), "b": frame[b].astype(str), "f": flag}).groupby(["a", "b"])["f"].agg(["size", "mean"])
    g = g[g["size"] >= opts.min_cell_rows]
    if len(g) < 4:
        return [], [ExcludedVisual(candidate=f"heatmap {a} × {b}", reason="moins de 4 cellules fiables")]
    rows = [{a: i[0], b: i[1], "churn_rate": r["mean"] * 100, "n": int(r["size"])} for i, r in g.iterrows()]
    ids = sorted({ins.id for ins in analyst.insights for st in ins.segment_stats
                  if find_col(frame, [st.dimension]) in (a, b)})
    return [VisualSpec(
        id=_slug("heatmap", a, b), type="heatmap", title=f"Churn : {a} × {b}", section="segments", span=6,
        data=clean_rows(rows), encoding=Encoding(x=a, y=b, color="churn_rate"),
        note="Croisement de deux dimensions ; cellules à faible effectif masquées.",
        references=VisualReference(
            kpi_id="churn_rate", insight_ids=ids, source_columns=[a, b],
            aggregation=f"mean(churn) par {a} × {b}", origin="dashboard_computed"))], []


# ── Profils churnés vs retenus ───────────────────────────────────────────────

def build_profiles(analyst: AnalystReport) -> tuple[list[VisualSpec], list[ExcludedVisual]]:
    visuals = []
    for pc in analyst.profile_comparisons:
        warn = [] if pc.p_value_adjusted < 0.05 else ["différence non significative après correction BH"]
        visuals.append(VisualSpec(
            id=_slug("profile", pc.metric), type="bar", title=f"{pc.metric} : churnés vs retenus (médiane)",
            section="profiles", span=4,
            data=clean_rows([
                {"group": "churnés", "median": pc.median_churned, "mean": pc.mean_churned, "n": pc.n_churned},
                {"group": "retenus", "median": pc.median_retained, "mean": pc.mean_retained, "n": pc.n_retained}]),
            encoding=Encoding(x="group", y="median", color="group"),
            note=(f"Cliff's delta {clean_value(pc.cliffs_delta)} (effet {pc.effect_size_label}) ; "
                  "association observée, pas causalité."),
            warnings=warn,
            references=VisualReference(
                kpi_id=pc.kpi_id, source_columns=[pc.metric], aggregation="médiane par groupe (Mann-Whitney)",
                origin="analyst_report")))
    return visuals, []


# ── Tables : insights et recommandations ─────────────────────────────────────

def build_insights_table(analyst: AnalystReport) -> list[VisualSpec]:
    if not analyst.insights:
        return []
    rows = [{"rank": i.rank, "title": i.title, "finding": i.finding, "direction": i.direction,
             "impact_churners": i.impact_churners, "revenue_monthly_lost_excess": i.revenue_monthly_lost_excess,
             "confidence": i.confidence} for i in analyst.insights]
    return [VisualSpec(
        id="table_insights", type="table", title="Insights classés par impact", section="segments", span=12,
        data=clean_rows(rows), note="Impacts non additifs : les segments peuvent se chevaucher.",
        warnings=["chevauchement possible entre segments"] if analyst.overlap_warning else [],
        references=VisualReference(insight_ids=[i.id for i in analyst.insights], origin="analyst_report"))]


def build_recommendations_table(analyst: AnalystReport) -> list[VisualSpec]:
    if not analyst.recommendations:
        return []
    rows = [{"id": r.id, "action": r.action, "target": r.target, "priority": r.priority,
             "test_type": r.test_type, "impact_churners": r.impact_churners,
             "confidence": r.confidence, "insight_ids": ", ".join(r.insight_ids)}
            for r in analyst.recommendations]
    return [VisualSpec(
        id="table_recommendations", type="table", title="Hypothèses d'action à tester", section="actions", span=12,
        data=clean_rows(rows), note="Chaque recommandation est une hypothèse à valider (A/B test, pilote), pas une certitude.",
        references=VisualReference(
            insight_ids=sorted({i for r in analyst.recommendations for i in r.insight_ids}),
            recommendation_ids=[r.id for r in analyst.recommendations], origin="analyst_report"))]


def build_caveats_block(caveats: list[str]) -> list[VisualSpec]:
    if not caveats:
        return []
    return [VisualSpec(
        id="text_caveats", type="text", title="Réserves et limites", section="quality", span=12,
        data=[{"text": c} for c in caveats], references=VisualReference(origin="analyst_report"))]