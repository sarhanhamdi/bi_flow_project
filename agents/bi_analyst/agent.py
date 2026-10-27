"""Graphe LangGraph du BI Analyst Agent.

    load -> segment -> test -> impact -> rank -> recommend -> narrate -> persist

Principe :
  - Les calculs statistiques (proportions, Mann-Whitney, correction BH) sont
    entièrement déterministes et tracés.
  - Le LLM (optionnel, ChatOllama) ne fait que rédiger la narration à partir
    des chiffres déjà calculés. Le garde-fou ``_numbers_grounded`` rejette toute
    synthèse contenant un nombre inventé.
  - ``run()`` ne lève jamais d'exception : les erreurs vont dans ``report.errors``.

Garde-fous :
  - Fuite de cible : exclure churn_label, churn_value, churn_score, churn_reason.
  - avg_churn_score est listé dans ``excluded_kpis`` avec la raison.
  - PII : colonnes ``is_pii=True`` dans le rapport ETL (+ heuristique géo).
  - Causalité : formulation « associé à », jamais « cause de ».
  - Segments < ``min_segment_rows`` → ``reliable=False``.
  - Recommandation = hypothèse à tester, liée à au moins un ``insight_id``.
  - Caveat fixe sur l'analyse univariée non ajustée.

Correction BH (Benjamini-Hochberg) :
  Une seule famille : z-tests de proportion (un par segment) + tests Mann-Whitney
  (un par variable continue). ``n_tests`` = taille totale de cette famille.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph

from agents.bi_analyst.catalog import ACTION_RULES, find_rules
from agents.bi_analyst.models import (
    AnalystOptions,
    AnalystReport,
    AnalystRequest,
    Insight,
    ProfileComparison,
    Recommendation,
    SegmentStat,
)
from agents.bi_analyst.prompts import NARRATIVE_PROMPT, SYSTEM_PROMPT
from agents.kpi.models import KPIReport, KPIResult
from agents.profiler.models import TraceEntry
from agents.tools.analyst import (
    _numbers_grounded,
    benjamini_hochberg,
    chi2_independence,
    churn_columns,
    constant_columns,
    effect_label,
    geo_columns,
    mann_whitney_test,
    pii_columns,
    proportion_ztest,
    reconstruct_counts,
    tenure_bins,
)
from agents.tools.kpi import as_flag

# KPIs exclus comme fuite de cible
_LEAKAGE_KPIS: frozenset[str] = frozenset({"avg_churn_score"})

# Caveat fixe affiché dans chaque rapport.
_UNIVARIATE_CAVEAT = (
    "Analyse univariée non ajustée : chaque dimension est analysée indépendamment. "
    "Les associations observées peuvent être confondues par d'autres variables. "
    "Toutes les formulations sont en termes d'association (« associé à »), "
    "jamais de causalité."
)

# Caveat chevauchement (segments non mutuellement exclusifs)
_OVERLAP_CAVEAT = (
    "Chevauchement des segments : les insights portent sur des dimensions différentes "
    "et leurs populations peuvent se recouper. Les impacts (départs en excès) "
    "ne sont pas additifs."
)

# Caveat troncature (top N affiché)
_TRUNCATION_CAVEAT_TPL = (
    "Troncature : {n_passing} segments passent les filtres ; "
    "seuls les {top_n} à plus fort impact sont affichés."
)


# ── État du graphe ────────────────────────────────────────────────────────────

class AnalystState(TypedDict, total=False):
    request: AnalystRequest
    options: AnalystOptions
    report: AnalystReport
    kpi_obj: KPIReport                 # KPIReport parsé
    frames: dict[str, pd.DataFrame]   # tables nettoyées en mémoire
    etl_columns: list[dict]            # ColumnSummary de l'ETL
    excluded_cols: set[str]            # colonnes à ne pas segmenter
    churn_col: str | None              # colonne de flag churn dans la table
    monthly_col: str | None            # colonne monthly_charges
    tenure_col: str | None             # colonne tenure_months
    rate_global: float                 # taux de churn global (%)
    n_global: int                      # nombre total de clients
    n_churned_global: int              # nombre total de churnés
    arpu_global: float                 # ARPU global
    rev_lookup: dict[tuple, float]     # (dim, seg) -> monthly_revenue_lost
    raw_segs: list[dict]               # segments bruts (avant tests)
    mw_results: list[dict]             # résultats Mann-Whitney


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _trace(tool: str, started: float, **params: Any) -> TraceEntry:
    return TraceEntry(
        timestamp=datetime.now(), tool=tool, parameters=params,
        duration_ms=(time.perf_counter() - started) * 1000,
    )


def _confidence(
    p_adj: float, excess_rate: float, n: int,
) -> str:
    """Règle déterministe de confiance documentée.

    - ``"élevée"``  : p_adj < 0.001 ET |excess_rate| >= 10 pts ET n >= 100
    - ``"moyenne"`` : p_adj < 0.05  ET |excess_rate| >= 5 pts  ET n >= 30
    - ``"faible"``  : tous les autres cas qui passent les filtres
    """
    ae = abs(excess_rate)
    if p_adj < 0.001 and ae >= 10.0 and n >= 100:
        return "élevée"
    if p_adj < 0.05 and ae >= 5.0 and n >= 30:
        return "moyenne"
    return "faible"


def _find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    """Cherche la première colonne du DataFrame correspondant à un alias (casse/_ ignorés)."""
    index = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in df.columns}
    for a in aliases:
        found = index.get(re.sub(r"[^a-z0-9]", "", a.lower()))
        if found is not None:
            return found
    return None


def _get_frame(state: AnalystState) -> pd.DataFrame | None:
    """Retourne le premier DataFrame disponible en mémoire ou None."""
    frames = state.get("frames") or {}
    if frames:
        return next(iter(frames.values()))
    return None


# ── Nœuds du graphe ───────────────────────────────────────────────────────────

def _load_node(state: AnalystState) -> AnalystState:
    """Charge et valide le KPIReport, le rapport ETL et la table nettoyée.

    Correction F.3 : le statut final est déterminé dans _persist_node,
    pas ici (sauf erreur de parsing bloquante).
    Correction D : KPIReport vide → caveat explicite (pas d'erreur bloquante).
    """
    request = state["request"]
    options = request.options
    started = time.perf_counter()

    # Guard: KPIReport vide
    empty_kpi = not request.kpi_report
    if empty_kpi:
        report = AnalystReport(
            status="warning",
            errors=["KPIReport vide : aucun KPI churn_rate exploitable."],
            caveats=[_UNIVARIATE_CAVEAT],
        )
        return {
            "options": options, "report": report, "kpi_obj": KPIReport(),
            "frames": {}, "etl_columns": [], "excluded_cols": set(),
            "churn_col": None, "monthly_col": None, "tenure_col": None,
            "rate_global": 0.0, "n_global": 0, "n_churned_global": 0,
            "arpu_global": 0.0, "rev_lookup": {}, "raw_segs": [], "mw_results": [],
        }

    # Parse KPIReport
    try:
        kpi_obj = KPIReport.model_validate(request.kpi_report)
    except Exception as exc:
        report = AnalystReport(status="error", errors=[f"KPIReport invalide : {exc}"])
        return {"options": options, "report": report, "kpi_obj": KPIReport(),
                "frames": {}, "etl_columns": [], "excluded_cols": set(),
                "churn_col": None, "monthly_col": None, "tenure_col": None,
                "rate_global": 0.0, "n_global": 0, "n_churned_global": 0,
                "arpu_global": 0.0, "rev_lookup": {}, "raw_segs": [], "mw_results": []}

    etl_raw = request.etl_report or {}

    # Colonnes ETL
    etl_columns: list[dict] = []
    output_path: str | None = None
    for table in etl_raw.get("tables", []):
        etl_columns.extend(table.get("columns", []))
        if not output_path:
            output_path = table.get("output_path")

    # Exclusions
    pii = pii_columns(etl_columns)
    churn = churn_columns(etl_columns)
    all_col_names = [c["name"] for c in etl_columns] if etl_columns else []
    geo = geo_columns(all_col_names)
    excluded_cols = pii | churn | geo | {"count", "country", "state"}

    # Charge la table nettoyée
    frames: dict[str, pd.DataFrame] = dict(state.get("frames") or {})
    if not frames and output_path:
        try:
            path = Path(output_path)
            frames["main"] = (pd.read_parquet(path) if path.suffix == ".parquet"
                              else pd.read_csv(path))
        except Exception:
            pass  # Pas bloquant : l'analyse se limitera aux breakdowns du KPIReport.

    df = _get_frame({"frames": frames})

    # Affiner l'exclusion avec les colonnes constantes de la table réelle.
    if df is not None:
        excluded_cols |= constant_columns(df)
        geo |= geo_columns(list(df.columns))
        excluded_cols |= geo

    # Extraire les valeurs globales du KPIReport
    kpi_by_id = {k.id: k for k in kpi_obj.kpis}

    rate_kpi = (
        kpi_by_id.get("churn_rate")
        or kpi_by_id.get("cdr_incident_rate")
        or kpi_by_id.get("ticket_resolution_rate")
        or kpi_by_id.get("dispute_rate")
    )
    rate_global = rate_kpi.value if (rate_kpi and rate_kpi.value is not None) else 0.0
    rows_global = rate_kpi.rows_used if rate_kpi else 0
    n_churned_global = reconstruct_counts(rate_global, rows_global)

    arpu_kpi = kpi_by_id.get("arpu")
    arpu_global = arpu_kpi.value if (arpu_kpi and arpu_kpi.value is not None) else 0.0

    # Revenue lookup depuis les breakdowns du KPI monthly_revenue_lost
    rev_lookup: dict[tuple[str, str], float] = {}
    rev_kpi = kpi_by_id.get("monthly_revenue_lost")
    if rev_kpi:
        for dim, segs in rev_kpi.breakdowns.items():
            for sv in segs:
                if sv.value is not None:
                    rev_lookup[(dim, sv.segment)] = sv.value

    # Initialisation du rapport
    report = AnalystReport(
        business_goal=kpi_obj.business_goal,
        kpi_status=kpi_obj.status,
        caveats=list(kpi_obj.caveats) + [_UNIVARIATE_CAVEAT],
        excluded_kpis=[
            {"kpi_id": kid, "reason": "fuite de cible (corrélé directement à churn_label)"}
            for kid in _LEAKAGE_KPIS
            if kid in kpi_by_id
        ],
    )

    # Colonnes utiles dans la table
    churn_col = monthly_col = tenure_col = None
    if df is not None:
        churn_col = _find_col(df, [
            "churn_value", "churn_flag", "churn_label", "churn", "is_churn",
            "cdr_flag", "dropped_connection", "incident_flag",
            "consumer_disputed", "consumer_disputed?", "disputed",
            "ticket_status", "status"
        ])
        monthly_col = _find_col(df, [
            "monthly_charges", "monthly_charge", "monthly_fee", "monthly_revenue", "total_day_charge", "day_charge",
            "throughput_mbps", "download_speed_mbps", "resolution_time_hours"
        ])
        tenure_col = _find_col(df, [
            "tenure_months", "tenure", "months_active", "account_length",
            "latency_ms", "signal_strength_dbm", "customer_age"
        ])

    report.trace.append(_trace(
        "load", started,
        kpis_computed=sum(1 for k in kpi_obj.kpis if k.status != "not_computable"),
        table_loaded=df is not None,
        excluded_cols=sorted(excluded_cols),
    ))

    return {
        "options": options, "report": report, "kpi_obj": kpi_obj,
        "frames": frames, "etl_columns": etl_columns, "excluded_cols": excluded_cols,
        "churn_col": churn_col, "monthly_col": monthly_col, "tenure_col": tenure_col,
        "rate_global": rate_global, "n_global": rows_global,
        "n_churned_global": n_churned_global, "arpu_global": arpu_global,
        "rev_lookup": rev_lookup, "raw_segs": [], "mw_results": [],
    }


def _segment_node(state: AnalystState) -> AnalystState:
    """Collecte tous les segments bruts : breakdowns KPI + dimensions de la table."""
    opts = state["options"]
    started = time.perf_counter()
    report = state["report"]
    kpi_obj = state["kpi_obj"]
    rate_global = state["rate_global"]
    n_global = state["n_global"]
    n_churned_global = state["n_churned_global"]
    rev_lookup = state["rev_lookup"]
    excluded_cols = state["excluded_cols"]
    df = _get_frame(state)
    churn_col = state["churn_col"]
    monthly_col = state["monthly_col"]
    tenure_col = state["tenure_col"]

    raw_segs: list[dict] = []

    # KPI source_file (pour la traçabilité XAI)
    kpi_source_file = (state["kpi_obj"].output_files or {}).get("report", "kpi_report.json")
    etl_source_file = "telco_clean.csv"
    if df is not None:
        for table in (state["request"].etl_report or {}).get("tables", []):
            if table.get("output_path"):
                etl_source_file = table["output_path"]
                break

    kpi_by_id = {k.id: k for k in kpi_obj.kpis}

    # 1. Segments depuis les breakdowns du KPIReport
    churn_kpi = kpi_by_id.get("churn_rate")
    if churn_kpi and churn_kpi.status != "not_computable":
        dims_done: set[str] = set()
        for dim, seg_list in churn_kpi.breakdowns.items():
            if dim in excluded_cols:
                continue
            dims_done.add(dim)
            for sv in seg_list:
                if sv.value is None or sv.rows < 1:
                    continue
                n_seg = sv.rows
                rate_seg = sv.value
                n_churned_seg = reconstruct_counts(rate_seg, n_seg)
                rev = rev_lookup.get((dim, sv.segment))
                raw_segs.append({
                    "kpi_id": "churn_rate",
                    "dimension": dim,
                    "segment": sv.segment,
                    "n_segment": n_seg,
                    "n_churned_segment": n_churned_seg,
                    "rate_segment": rate_seg,
                    "monthly_revenue_lost_segment": rev,
                    "source": "kpi_report",
                    "source_file": kpi_source_file,
                    "reliable": n_seg >= opts.min_segment_rows,
                })

    # 2. Segments supplémentaires depuis la table nettoyée
    if df is not None and churn_col is not None:
        flag_series = as_flag(df[churn_col])
        if flag_series is None:
            report.errors.append(
                f"Colonne de churn '{churn_col}' non convertible en flag binaire.")
        else:
            df_work = df.copy()
            if tenure_col and tenure_col in df_work.columns:
                df_work["tenure_bin"] = tenure_bins(df_work[tenure_col])

            kpi_dims: set[str] = set()
            if churn_kpi:
                kpi_dims = set(churn_kpi.breakdowns.keys())

            candidate_dims: list[str] = []
            for col in df_work.columns:
                if col in excluded_cols:
                    continue
                if col == churn_col or col == monthly_col or col == tenure_col:
                    continue
                if col in kpi_dims:
                    continue
                card = df_work[col].nunique(dropna=True)
                if card < 2 or card > opts.max_segments:
                    continue
                candidate_dims.append(col)

            for dim in candidate_dims:
                for seg_val, grp in df_work.groupby(dim, dropna=True, observed=True):
                    n_seg = len(grp)
                    if n_seg < 1:
                        continue
                    seg_flag = flag_series[grp.index]
                    n_churned_seg = int(seg_flag.sum())
                    rate_seg = round(float(seg_flag.mean()) * 100.0, 2)
                    rev = None
                    if monthly_col and monthly_col in grp.columns:
                        rev = round(float(grp.loc[seg_flag == 1, monthly_col].sum()), 2)
                    raw_segs.append({
                        "kpi_id": "churn_rate",
                        "dimension": dim,
                        "segment": str(seg_val),
                        "n_segment": n_seg,
                        "n_churned_segment": n_churned_seg,
                        "rate_segment": rate_seg,
                        "monthly_revenue_lost_segment": rev,
                        "source": "analyst_computed",
                        "source_file": etl_source_file,
                        "reliable": n_seg >= opts.min_segment_rows,
                    })

    # 3. Mann-Whitney : partis vs restés sur tenure et monthly_charges
    mw_results: list[dict] = []
    if df is not None and churn_col is not None:
        flag_series = as_flag(df[churn_col])
        if flag_series is not None:
            churned_mask = flag_series == 1
            retained_mask = flag_series == 0
            for col, kpi_id in [(tenure_col, "avg_tenure"), (monthly_col, "arpu")]:
                if col is None or col not in df.columns:
                    continue
                a = pd.to_numeric(df.loc[churned_mask, col], errors="coerce").dropna().values
                b = pd.to_numeric(df.loc[retained_mask, col], errors="coerce").dropna().values
                if len(a) < 2 or len(b) < 2:
                    continue
                U, p, delta = mann_whitney_test(a, b)
                mw_results.append({
                    "kpi_id": kpi_id,
                    "metric_col": col,
                    "n_churned": int(churned_mask.sum()),
                    "n_retained": int(retained_mask.sum()),
                    "median_churned": float(np.median(a)),
                    "median_retained": float(np.median(b)),
                    "mean_churned": float(np.mean(a)),
                    "mean_retained": float(np.mean(b)),
                    "U": U, "p_value_raw": p, "cliffs_delta": delta,
                    "effect_size_label": effect_label(delta),
                    "source": "analyst_computed",
                    "source_file": etl_source_file,
                })

    report.trace.append(_trace(
        "segment", started,
        n_segments_kpi=sum(1 for s in raw_segs if s["source"] == "kpi_report"),
        n_segments_analyst=sum(1 for s in raw_segs if s["source"] == "analyst_computed"),
        n_mw=len(mw_results),
    ))
    return {"report": report, "raw_segs": raw_segs, "mw_results": mw_results}


def _test_node(state: AnalystState) -> AnalystState:
    """Applique les z-tests de proportion et la correction Benjamini-Hochberg globale.

    - Un z-test par segment vs reste de la population (pour churn_rate).
    - Chi2 par dimension (indicateur d'ensemble, hors correction BH).
    - Les tests Mann-Whitney rejoignent la même famille BH (correction commune).
    - BH appliqué une seule fois sur l'ensemble : z-tests + Mann-Whitney.
    - ``n_tests`` exposé dans le rapport = taille totale de la famille.
    """
    started = time.perf_counter()
    report = state["report"]
    raw_segs = list(state.get("raw_segs") or [])
    mw_results = list(state.get("mw_results") or [])
    n_global = state["n_global"]
    n_churned_global = state["n_churned_global"]

    if not raw_segs and not mw_results:
        report.trace.append(_trace("test", started, n_tests=0))
        report.n_tests = 0
        return {"report": report}

    # Calcul des z-tests (segment vs reste)
    p_values_raw: list[float] = []
    effect_sizes: list[float] = []

    for seg in raw_segs:
        n1 = seg["n_segment"]
        k1 = seg["n_churned_segment"]
        n2 = max(n_global - n1, 0)
        k2 = max(n_churned_global - k1, 0)
        _, p = proportion_ztest(n1, k1, n2, k2)
        obs = np.array([[k1, n1 - k1], [k2, n2 - k2]], dtype=float)
        v = 0.0
        if obs.min() >= 0 and obs.sum() > 0:
            try:
                _, _, v = chi2_independence(obs)
            except Exception:
                v = 0.0
        seg["p_value_raw"] = p
        seg["effect_size"] = v
        seg["test_used"] = "proportion_z"
        p_values_raw.append(p)
        effect_sizes.append(v)

    n_ztests = len(p_values_raw)

    # Ajouter les p-values Mann-Whitney à la même famille BH
    for mw in mw_results:
        p_values_raw.append(mw["p_value_raw"])

    # Correction BH globale (z-tests + Mann-Whitney ensemble)
    if p_values_raw:
        p_adj_arr = benjamini_hochberg(p_values_raw)
    else:
        p_adj_arr = []

    # Distribuer les p ajustées aux z-tests
    for i, seg in enumerate(raw_segs):
        seg["p_value_adjusted"] = float(p_adj_arr[i]) if i < len(p_adj_arr) else 1.0
        seg["effect_size_label"] = effect_label(seg["effect_size"])

    # Distribuer les p ajustées aux Mann-Whitney
    for j, mw in enumerate(mw_results):
        idx = n_ztests + j
        mw["p_value_adjusted"] = float(p_adj_arr[idx]) if idx < len(p_adj_arr) else 1.0

    report.n_tests = len(p_values_raw)
    report.trace.append(_trace("test", started, n_tests=report.n_tests,
                                n_ztests=n_ztests, n_mw=len(mw_results),
                                min_p_raw=float(min(p_values_raw)) if p_values_raw else 1.0,
                                max_p_adj=float(max(p_adj_arr)) if len(p_adj_arr) > 0 else 1.0))
    return {"report": report, "raw_segs": raw_segs, "mw_results": mw_results}


def _impact_node(state: AnalystState) -> AnalystState:
    """Calcule l'impact : départs en excès et revenu mensuel perdu en excès."""
    started = time.perf_counter()
    report = state["report"]
    rate_global = state["rate_global"]
    n_churned_global = state["n_churned_global"]

    for seg in state.get("raw_segs") or []:
        n_seg = seg["n_segment"]
        rate_seg = seg["rate_segment"]
        n_churned_seg = seg["n_churned_segment"]

        excess_rate = rate_seg - rate_global
        excess_churners = excess_rate / 100.0 * n_seg

        # Revenu mensuel perdu en excès
        rev_seg = seg.get("monthly_revenue_lost_segment")
        if rev_seg is not None and n_churned_seg > 0:
            rev_excess = rev_seg * excess_churners / n_churned_seg
        else:
            rev_excess = 0.0

        seg["excess_rate"] = round(excess_rate, 2)
        seg["excess_churners"] = round(excess_churners, 1)
        seg["revenue_monthly_lost_excess"] = round(rev_excess, 2)

        # Références pour la traçabilité XAI
        seg["references"] = {
            "kpi_id": seg["kpi_id"],
            "dimension": seg["dimension"],
            "segment": seg["segment"],
            "source": seg["source"],
            "source_file": seg["source_file"],
        }

    report.trace.append(_trace("impact", started, n_segs=len(state.get("raw_segs") or [])))
    return {"report": report}


def _rank_node(state: AnalystState) -> AnalystState:
    """Filtre, trie et transforme les segments en Insight.

    Corrections A + E :
    - Les résultats Mann-Whitney vont dans ``report.profile_comparisons`` (section dédiée),
      pas dans ``report.insights``.
    - ``direction`` est ajouté à chaque Insight ("risque" si excess_rate > 0,
      "protecteur" si <= 0).
    - ``n_significant_segments`` comptabilise tous les segments passant les filtres.
    - Le caveat de chevauchement est ajouté si top > 1.
    - Le caveat de troncature est ajouté si passing > top_n_insights.

    Filtre : p_adjusted <= significance_level ET |excess_rate| >= min_excess_rate
             ET reliable → puis top_n par excess_churners décroissant.

    null_result = True si aucun segment ne passe tous les filtres.
    """
    started = time.perf_counter()
    opts = state["options"]
    report = state["report"]
    rate_global = state["rate_global"]
    n_churned_global = state["n_churned_global"]
    mw_results = list(state.get("mw_results") or [])

    raw_segs = list(state.get("raw_segs") or [])

    # Filtrage
    passing = [
        s for s in raw_segs
        if (s.get("p_value_adjusted", 1.0) <= opts.significance_level
            and abs(s.get("excess_rate", 0.0)) >= opts.min_excess_rate
            and s.get("reliable", False))
    ]

    # Comptabiliser les segments significatifs (correction E)
    report.n_significant_segments = len(passing)

    if not passing:
        report.null_result = True
        report.caveats.append(
            "Aucun segment ne passe les filtres combinés "
            "(p ajustée <= {:.2f}, écart >= {:.1f} pts, n >= {}).".format(
                opts.significance_level, opts.min_excess_rate, opts.min_segment_rows))
        report.trace.append(_trace("rank", started, n_passing=0, n_mw=len(mw_results)))
        # Construire quand même les ProfileComparisons MW
        _build_profile_comparisons(report, mw_results, opts)
        return {"report": report}

    # Tri par excess_churners décroissant (impact absolu)
    passing.sort(key=lambda s: s.get("excess_churners", 0.0), reverse=True)
    top = passing[: opts.top_n_insights]

    # Caveat de troncature (correction E)
    if len(passing) > opts.top_n_insights:
        report.caveats.append(
            _TRUNCATION_CAVEAT_TPL.format(n_passing=len(passing), top_n=opts.top_n_insights)
        )

    insights: list[Insight] = []
    for rank, seg in enumerate(top, start=1):
        n = seg["n_segment"]
        rate = seg["rate_segment"]
        excess_rate = seg["excess_rate"]
        excess_ch = seg["excess_churners"]
        p_adj = seg["p_value_adjusted"]
        v = seg["effect_size"]
        rev_excess = seg["revenue_monthly_lost_excess"]
        conf = _confidence(p_adj, excess_rate, n)

        # Direction (correction A)
        direction = "risque" if excess_rate > 0 else "protecteur"

        p_str = f"< 0,001" if p_adj < 0.001 else f"= {p_adj:.3f}".replace(".", ",")

        title = (
            f"{seg['dimension']} = {seg['segment']} : "
            f"churn {rate:.2f} % ({excess_rate:+.1f} pts)"
        )
        finding = (
            f"Le segment « {seg['segment']} » (dimension : {seg['dimension']}) "
            f"est associé à un taux de churn de {rate:.2f} %, "
            f"contre {rate_global:.2f} % sur l'ensemble du portefeuille "
            f"({excess_rate:+.2f} points d'écart)."
        )
        evidence = (
            f"n={n:,} clients dans le segment ({seg['n_churned_segment']:,} churnés). "
            f"Écart : {excess_rate:+.2f} pts. "
            f"Départs en excès : {excess_ch:,.0f}. "
            f"Revenu mensuel perdu en excès : {rev_excess:,.0f} (devise/mois). "
            f"p ajustée (BH, z-test proportion) {p_str}. "
            f"Taille d'effet : {v:.3f} ({seg['effect_size_label']}, Cramér's V). "
            f"Source : {seg['source']}."
        )

        stat = SegmentStat(
            dimension=seg["dimension"],
            segment=seg["segment"],
            n=n,
            n_churned=seg["n_churned_segment"],
            rate_or_mean=rate,
            global_rate=rate_global,
            excess_rate=excess_rate,
            excess_churners=excess_ch,
            revenue_monthly_lost_excess=rev_excess,
            p_value_raw=seg.get("p_value_raw", 1.0),
            p_value_adjusted=p_adj,
            effect_size=v,
            effect_size_label=seg["effect_size_label"],
            test_used=seg.get("test_used", "proportion_z"),
            reliable=seg.get("reliable", True),
            source=seg["source"],
            references=seg.get("references", {}),
        )

        warnings: list[str] = []
        if not seg.get("reliable", True):
            warnings.append(f"Segment potentiellement peu fiable (n={n} < seuil).")

        insights.append(Insight(
            id=f"insight_{seg['dimension']}_{re.sub(r'[^a-z0-9]', '_', seg['segment'].lower())}",
            kpi_id=seg["kpi_id"],
            title=title,
            finding=finding,
            evidence=evidence,
            segment_stats=[stat],
            impact_churners=excess_ch,
            revenue_monthly_lost_excess=rev_excess,
            confidence=conf,
            direction=direction,
            rank=rank,
            warnings=warnings,
            references=seg.get("references", {}),
        ))

    # Caveat de chevauchement (correction E)
    if len(top) > 1:
        report.overlap_warning = True
        report.caveats.append(_OVERLAP_CAVEAT)

    report.insights = insights

    # Construire les ProfileComparisons MW (correction A) — section séparée
    _build_profile_comparisons(report, mw_results, opts)

    report.trace.append(_trace("rank", started, n_insights=len(insights),
                                n_proportion=len(top), n_mw=len(mw_results)))
    return {"report": report}


def _build_profile_comparisons(
    report: AnalystReport,
    mw_results: list[dict],
    opts: AnalystOptions,
) -> None:
    """Construit les ProfileComparison depuis les résultats Mann-Whitney.

    Ces comparaisons vont dans ``report.profile_comparisons``, jamais dans
    ``report.insights``. Elles ne génèrent pas de recommandations directes.
    """
    comparisons: list[ProfileComparison] = []
    for mw in mw_results:
        comparisons.append(ProfileComparison(
            metric=mw["metric_col"],
            kpi_id=mw["kpi_id"],
            n_churned=mw["n_churned"],
            n_retained=mw["n_retained"],
            median_churned=mw["median_churned"],
            median_retained=mw["median_retained"],
            mean_churned=mw.get("mean_churned", 0.0),
            mean_retained=mw.get("mean_retained", 0.0),
            cliffs_delta=mw["cliffs_delta"],
            effect_size_label=mw["effect_size_label"],
            p_value_raw=mw["p_value_raw"],
            p_value_adjusted=mw.get("p_value_adjusted", mw["p_value_raw"]),
            test_used="mann-whitney",
            source="analyst_computed",
            references={
                "kpi_id": mw["kpi_id"],
                "metric_col": mw["metric_col"],
                "source": mw["source"],
                "source_file": mw["source_file"],
            },
        ))
    report.profile_comparisons = comparisons


def _recommend_node(state: AnalystState) -> AnalystState:
    """Génère les recommandations depuis le catalogue de règles d'actions.

    Corrections B.1 + B.2 :
    - Le bug ``pass`` est corrigé : si une règle déjà vue est déclenchée par un
      nouvel insight (même règle, segment différent), on cumule impact_churners
      et revenue_monthly_lost_excess sur la recommandation existante.
    - Les insights de direction "protecteur" (excess_rate <= 0) sont exclus.
    - Les ProfileComparisons MW ne génèrent jamais de recommandations.
    """
    started = time.perf_counter()
    report = state["report"]
    if report.null_result or not report.insights:
        report.trace.append(_trace("recommend", started, n_rec=0))
        return {"report": report}

    seen_rule_ids: set[str] = set()
    recommendations: list[Recommendation] = []

    for insight in report.insights:
        stat = insight.segment_stats[0] if insight.segment_stats else None
        if stat is None:
            continue
        # Exclure les insights protecteurs (correction B.1)
        if insight.direction == "protecteur":
            continue
        # Exclure les insights sans impact (excess_rate <= 0 cas résiduel)
        if stat.excess_rate <= 0:
            continue

        rules = find_rules(
            kpi_id=insight.kpi_id,
            dimension=stat.dimension,
            segment=stat.segment,
            excess_rate=stat.excess_rate,
        )
        for rule in rules:
            rec_id = f"rec_{rule.id}"
            if rule.id in seen_rule_ids:
                # Correction B.1 : enrichir la recommandation existante
                for i, existing_rec in enumerate(recommendations):
                    if existing_rec.id == rec_id:
                        if insight.id not in existing_rec.insight_ids:
                            new_ids = list(existing_rec.insight_ids) + [insight.id]
                            new_impact = existing_rec.impact_churners + insight.impact_churners
                            new_rev = (existing_rec.revenue_monthly_lost_excess
                                       + insight.revenue_monthly_lost_excess)
                            recommendations[i] = existing_rec.model_copy(update={
                                "insight_ids": new_ids,
                                "impact_churners": new_impact,
                                "revenue_monthly_lost_excess": new_rev,
                            })
                continue
            seen_rule_ids.add(rule.id)
            recommendations.append(Recommendation(
                id=rec_id,
                action=rule.action,
                target=rule.target,
                rationale=(
                    f"Insight #{insight.rank} : {insight.finding} "
                    f"(p ajustée <= {stat.p_value_adjusted:.3f}, "
                    f"excédent {stat.excess_churners:.0f} clients). "
                    f"Association statistique, pas de causalité établie."
                ),
                insight_ids=[insight.id],
                impact_churners=insight.impact_churners,
                revenue_monthly_lost_excess=insight.revenue_monthly_lost_excess,
                priority=rule.priority,
                confidence=insight.confidence,
                test_type=rule.test_type,
                references={
                    **insight.references,
                    "rule_id": rule.id,
                    "rule_description": rule.description,
                },
            ))

    report.recommendations = recommendations
    report.trace.append(_trace("recommend", started, n_rec=len(recommendations)))
    return {"report": report}


# ── Narration ─────────────────────────────────────────────────────────────────

def _fallback_narrative(report: AnalystReport) -> str:
    """Synthèse déterministe (sans LLM)."""
    if report.null_result:
        return (
            "Analyse terminée. Aucun segment ne présente à la fois une significativité "
            "statistique (p ajustée <= 0,05), un écart suffisant et un effectif fiable. "
            + ("; ".join(report.errors) if report.errors else "")
        )
    lines: list[str] = [
        f"Analyse BI — {len(report.insights)} insight(s), "
        f"{len(report.recommendations)} recommandation(s). "
        f"Tests : {report.n_tests} ({report.n_tests} tests dans la famille BH).",
    ]
    for ins in report.insights[:5]:
        lines.append(
            f"[#{ins.rank}] {ins.title} — {ins.finding} "
            f"Impact : {ins.impact_churners:.0f} départs en excès, "
            f"{ins.revenue_monthly_lost_excess:,.0f} (revenu/mois en excès). "
            f"Confiance : {ins.confidence}."
        )
    if report.profile_comparisons:
        lines.append("\nComparaisons de profils (Mann-Whitney) :")
        for pc in report.profile_comparisons:
            direction_str = "inférieure" if pc.cliffs_delta < 0 else "supérieure"
            col_label = "ancienneté" if "tenure" in pc.metric else "charges mensuelles"
            lines.append(
                f"  {col_label} : médiane churnés={pc.median_churned:.1f}, "
                f"retenus={pc.median_retained:.1f} "
                f"(Cliff's Δ={pc.cliffs_delta:.3f}, {pc.effect_size_label}, "
                f"p ajustée={pc.p_value_adjusted:.3f})."
            )
    if report.recommendations:
        lines.append("\nRecommandations (hypothèses à tester) :")
        for rec in report.recommendations:
            lines.append(
                f"  [{rec.priority}] {rec.action} — Cible : {rec.target}. "
                f"Test suggéré : {rec.test_type}."
            )
    if report.caveats:
        lines.append("\nRéserves :")
        lines += [f"  • {c}" for c in report.caveats]
    return "\n".join(lines)


def _load_llm():
    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.1", temperature=0)
    except Exception:
        return None


def _narrate_node(state: AnalystState) -> AnalystState:
    """Rédige la narration (LLM optionnel avec garde-fou, sinon déterministe)."""
    started = time.perf_counter()
    opts = state["options"]
    report = state["report"]

    fallback = _fallback_narrative(report)

    if not opts.use_llm:
        report.llm_narrative = fallback
        report.trace.append(_trace("narrate", started, llm=False))
        return {"report": report}

    llm = _load_llm()
    if llm is None:
        report.llm_narrative = fallback
        report.trace.append(_trace("narrate", started, llm=False, reason="LLM non disponible"))
        return {"report": report}

    data = {
        "insights": [i.model_dump(mode="json",
                                   include={"id", "rank", "title", "finding", "evidence",
                                            "impact_churners", "revenue_monthly_lost_excess",
                                            "confidence"})
                     for i in report.insights[:5]],
        "recommendations": [r.model_dump(mode="json",
                                          include={"id", "action", "target", "rationale",
                                                   "priority", "test_type"})
                             for r in report.recommendations[:5]],
    }
    source_json = json.dumps(data, ensure_ascii=False)
    try:
        resp = llm.invoke([
            ("system", SYSTEM_PROMPT),
            ("human", NARRATIVE_PROMPT.format(
                business_goal=report.business_goal,
                insights_json=source_json,
                recommendations_json=source_json,
                caveats="\n".join(f"• {c}" for c in report.caveats),
            )),
        ])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        report.llm_narrative = text if _numbers_grounded(text, source_json) else fallback
    except Exception:
        report.llm_narrative = fallback

    report.trace.append(_trace("narrate", started, llm=True))
    return {"report": report}


def _persist_node(state: AnalystState) -> AnalystState:
    """Écrit analyst_report.json, insights.csv et recommendations.csv.

    Correction F.2 : le statut final est déterminé AVANT le test write_outputs,
    pour qu'il soit correct même quand write_outputs=False.

    Correction F.3 : null_result=True ne peut pas être écrasé par "success".
    """
    opts = state["options"]
    report = state["report"]

    # Statut final — AVANT le guard write_outputs
    if report.null_result:
        report.status = "warning"
    elif report.errors:
        report.status = "warning"
    elif report.status not in {"error", "warning"}:
        report.status = "success"

    if not opts.write_outputs:
        return {"report": report}

    try:
        out = Path(opts.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        json_path = out / "analyst_report.json"
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

        # insights.csv
        ins_rows = []
        for ins in report.insights:
            for st in ins.segment_stats:
                p_adj = st.p_value_adjusted
                p_adj_display = (
                    "< 0,001" if p_adj < 0.001
                    else f"= {p_adj:.3f}".replace(".", ",")
                )
                ins_rows.append({
                    "insight_id": ins.id, "kpi_id": ins.kpi_id,
                    "dimension": st.dimension, "segment": st.segment,
                    "rate": st.rate_or_mean, "global_rate": st.global_rate,
                    "excess_rate": st.excess_rate,
                    "excess_churners": st.excess_churners,
                    "revenue_monthly_lost_excess": st.revenue_monthly_lost_excess,
                    "p_adj": p_adj,
                    "p_adj_display": p_adj_display,
                    "effect_size": st.effect_size,
                    "effect_label": st.effect_size_label, "test_used": st.test_used,
                    "confidence": ins.confidence, "direction": ins.direction,
                    "rank": ins.rank,
                    "source": st.source,
                })
        pd.DataFrame(ins_rows).to_csv(out / "insights.csv", index=False)

        # profile_comparisons.csv
        pc_rows = []
        for pc in report.profile_comparisons:
            pc_rows.append({
                "metric": pc.metric, "kpi_id": pc.kpi_id,
                "n_churned": pc.n_churned, "n_retained": pc.n_retained,
                "median_churned": pc.median_churned, "median_retained": pc.median_retained,
                "mean_churned": pc.mean_churned, "mean_retained": pc.mean_retained,
                "cliffs_delta": pc.cliffs_delta, "effect_size_label": pc.effect_size_label,
                "p_value_raw": pc.p_value_raw, "p_value_adjusted": pc.p_value_adjusted,
                "test_used": pc.test_used,
            })
        pd.DataFrame(pc_rows).to_csv(out / "profile_comparisons.csv", index=False)

        # recommendations.csv
        rec_rows = []
        for rec in report.recommendations:
            rec_rows.append({
                "rec_id": rec.id, "action": rec.action, "target": rec.target,
                "priority": rec.priority, "test_type": rec.test_type,
                "impact_churners": rec.impact_churners,
                "revenue_monthly_lost_excess": rec.revenue_monthly_lost_excess,
                "confidence": rec.confidence,
                "insight_ids": ";".join(rec.insight_ids),
            })
        pd.DataFrame(rec_rows).to_csv(out / "recommendations.csv", index=False)

        report.output_files = {
            "report": str(json_path),
            "insights": str(out / "insights.csv"),
            "profile_comparisons": str(out / "profile_comparisons.csv"),
            "recommendations": str(out / "recommendations.csv"),
        }
    except OSError as exc:
        report.errors.append(f"Écriture des résultats impossible : {exc}")
        report.status = "warning"

    return {"report": report}


# ── Assemblage du graphe ──────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AnalystState)
    g.add_node("load",      _load_node)
    g.add_node("segment",   _segment_node)
    g.add_node("test",      _test_node)
    g.add_node("impact",    _impact_node)
    g.add_node("rank",      _rank_node)
    g.add_node("recommend", _recommend_node)
    g.add_node("narrate",   _narrate_node)
    g.add_node("persist",   _persist_node)
    g.add_edge(START,       "load")
    g.add_edge("load",      "segment")
    g.add_edge("segment",   "test")
    g.add_edge("test",      "impact")
    g.add_edge("impact",    "rank")
    g.add_edge("rank",      "recommend")
    g.add_edge("recommend", "narrate")
    g.add_edge("narrate",   "persist")
    g.add_edge("persist",   END)
    return g.compile()


_analyst_graph = build_graph()


# ── API publique ──────────────────────────────────────────────────────────────

def run(
    request: AnalystRequest,
    frames: dict[str, pd.DataFrame] | None = None,
) -> AnalystReport:
    """Point d'entrée public pour l'Orchestrator.

    Parameters
    ----------
    request : AnalystRequest
        KPIReport obligatoire + ETLReport optionnel (pour les métadonnées PII
        et le chemin de la table nettoyée).
    frames : dict[str, pd.DataFrame] | None
        DataFrames nettoyés déjà en mémoire (depuis ``run_with_data`` de l'ETL).
        Si absent, la table est lue depuis ``etl_report.tables[0].output_path``.

    Returns
    -------
    AnalystReport
        Ne lève jamais d'exception : les erreurs vont dans ``report.errors``.
    """
    try:
        result = _analyst_graph.invoke(
            {"request": request, "frames": frames or {}},
        )
        return result["report"]
    except Exception as exc:
        return AnalystReport(
            business_goal=request.kpi_report.get("business_goal", "") if request.kpi_report else "",
            status="error",
            errors=[f"Erreur inattendue du BI Analyst Agent : {exc}"],
        )
