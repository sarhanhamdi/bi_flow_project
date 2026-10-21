"""Graphe LangGraph du Dashboard Generator Agent.

    load -> build -> assemble -> report -> narrate -> persist

Principe (identique au BI Analyst) :
  - La construction des visuels est entièrement déterministe et tracée (``tools/dashboard.py``).
  - Le LLM (optionnel, même client que le BI Analyst) ne rédige que la synthèse du rapport ;
    toute synthèse contenant un nombre absent des données est rejetée (repli déterministe).
  - ``run()`` ne lève jamais d'exception : les erreurs vont dans ``report.errors``.

Garde-fous :
  - Fuite de cible : avg_churn_score et colonnes churn_* jamais affichés comme pilotage.
  - PII : aucune coordonnée individuelle ; géo agrégé par zone avec effectif minimal.
  - Causalité : formulation « associé à » ; recommandations = hypothèses à tester.
  - Traçabilité : chaque visuel porte ses ``references`` (kpi_id, insight_ids, colonnes, formule).
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agents.bi_analyst.models import AnalystReport
from agents.dashboard.catalog import SECTIONS
from agents.dashboard.models import (
    DashboardOptions,
    DashboardReport,
    DashboardRequest,
    DashboardSection,
    ExcludedVisual,
    ReportBlock,
    ReportSpec,
    VisualSpec,
)
from agents.dashboard.prompts import NARRATIVE_PROMPT, SYSTEM_PROMPT
from agents.kpi.models import KPIReport
from agents.profiler.models import TraceEntry
from agents.tools.dashboard import (
    build_breakdown_bars,
    build_caveats_block,
    build_geo,
    build_heatmap,
    build_insights_table,
    build_kpi_cards,
    build_profiles,
    build_recommendations_table,
    build_temporal,
    churn_series,
    validate_visual,
)

_SECTION_ORDER = {sid: i for i, (sid, _) in enumerate(SECTIONS)}
_SECTION_TITLE = dict(SECTIONS)
_FIXED_TYPES = {"table", "text"}  # jamais supprimés par la limite max_visuals


class DashboardState(TypedDict, total=False):
    request: DashboardRequest
    options: DashboardOptions
    report: DashboardReport
    analyst: AnalystReport
    kpi: KPIReport | None
    frames: dict[str, pd.DataFrame]
    frame: pd.DataFrame | None
    global_rate: float | None
    n_global: int
    visuals: list[VisualSpec]
    excluded: list[ExcludedVisual]
    caveats: list[str]
    abort: bool


def _trace(tool: str, started: float, **params: Any) -> TraceEntry:
    return TraceEntry(timestamp=datetime.now(), tool=tool, parameters=params,
                      duration_ms=(time.perf_counter() - started) * 1000)


def _read_frame_from_etl(etl_report: dict[str, Any] | None) -> pd.DataFrame | None:
    """Relit la table nettoyée depuis ``etl_report.tables[0].output_path`` (csv / parquet)."""
    try:
        tables = (etl_report or {}).get("tables") or []
        path = tables[0].get("output_path") if tables else None
        if not path or not Path(path).exists():
            return None
        return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    except Exception:
        return None


def _load_llm():
    """Même client LLM que le BI Analyst (ChatOllama) ; None si indisponible."""
    try:
        from agents.bi_analyst.agent import _load_llm as analyst_load_llm
        return analyst_load_llm()
    except Exception:
        return None


def _grounded(text: str, source: str) -> bool:
    try:
        from agents.tools.analyst import _numbers_grounded
        return bool(_numbers_grounded(text, source))
    except Exception:
        pat = r"\d+(?:[.,]\d+)?"
        known = {n.replace(",", ".") for n in re.findall(pat, source)}
        return all(n.replace(",", ".") in known for n in re.findall(pat, text))


# ── Nœuds ────────────────────────────────────────────────────────────────────

def _load_node(state: DashboardState) -> DashboardState:
    req = state["request"]
    opts = req.options
    started = time.perf_counter()
    caveats: list[str] = []

    base: DashboardState = {"options": opts, "visuals": [], "excluded": [], "kpi": None,
                            "frame": None, "global_rate": None, "n_global": 0}

    if req.analyst_report:
        try:
            analyst = AnalystReport.model_validate(req.analyst_report)
        except Exception as exc:
            report = DashboardReport(status="error", errors=[f"AnalystReport invalide : {exc}"])
            return {**base, "report": report, "analyst": AnalystReport(), "caveats": [], "abort": True}
    else:
        analyst = AnalystReport()
        caveats.append("AnalystReport vide : le dashboard est limité aux KPIs et aux agrégations de la table.")

    kpi: KPIReport | None = None
    if req.kpi_report:
        try:
            kpi = KPIReport.model_validate(req.kpi_report)
        except Exception as exc:
            caveats.append(f"KPIReport invalide, cartes KPI non produites : {exc}")
    else:
        caveats.append("KPIReport absent : cartes KPI et ventilations non produites.")

    frames = state.get("frames") or {}
    frame = next(iter(frames.values())) if frames else _read_frame_from_etl(req.etl_report)
    if frame is None:
        caveats.append("Table nettoyée indisponible : visuels temporels, géographiques et croisés non produits.")

    flag = churn_series(frame) if frame is not None else None
    global_rate: float | None = None
    n_global = 0
    if flag is not None:
        global_rate, n_global = float(flag.mean() * 100), int(len(flag))
    elif kpi is not None:
        churn = next((k for k in kpi.kpis if k.id == "churn_rate" and k.value is not None), None)
        if churn is not None:
            global_rate, n_global = float(churn.value), churn.rows_used
    if global_rate is None:
        for ins in analyst.insights:
            if ins.segment_stats:
                global_rate = ins.segment_stats[0].global_rate
                break

    caveats.extend(c for c in analyst.caveats if c not in caveats)
    if kpi is not None:
        caveats.extend(c for c in kpi.caveats if c not in caveats)

    report = DashboardReport(
        business_goal=analyst.business_goal or (kpi.business_goal if kpi else ""),
        analyst_status=analyst.status, null_result=analyst.null_result)
    report.trace.append(_trace("load", started, has_frame=frame is not None,
                               has_kpi=kpi is not None, n_insights=len(analyst.insights)))
    return {**base, "report": report, "analyst": analyst, "kpi": kpi, "frame": frame,
            "global_rate": global_rate, "n_global": n_global, "caveats": caveats}


def _build_node(state: DashboardState) -> DashboardState:
    if state.get("abort"):
        return {}
    started = time.perf_counter()
    opts, analyst, kpi, frame = state["options"], state["analyst"], state.get("kpi"), state.get("frame")
    visuals: list[VisualSpec] = []
    excluded: list[ExcludedVisual] = []

    def add(result: tuple[list[VisualSpec], list[ExcludedVisual]]) -> None:
        visuals.extend(result[0])
        excluded.extend(result[1])

    # Ordre = priorité d'affichage (la limite max_visuals retire d'abord la fin de liste).
    if kpi is not None:
        add(build_kpi_cards(kpi))
    visuals.extend(build_insights_table(analyst))
    add(build_temporal(frame, opts))
    add(build_geo(frame, opts, state.get("global_rate")))
    add(build_heatmap(frame, analyst, opts))
    add(build_profiles(analyst))
    if kpi is not None:
        add(build_breakdown_bars(kpi, analyst, opts))
    visuals.extend(build_recommendations_table(analyst))
    visuals.extend(build_caveats_block(state.get("caveats", [])))

    state["report"].trace.append(_trace("build", started, n_candidates=len(visuals)))
    return {"visuals": visuals, "excluded": excluded}


def _assemble_node(state: DashboardState) -> DashboardState:
    if state.get("abort"):
        return {}
    started = time.perf_counter()
    opts, report = state["options"], state["report"]
    excluded = list(state["excluded"])
    caveats = list(state["caveats"])

    valid: list[VisualSpec] = []
    seen: set[str] = set()
    for v in state["visuals"]:
        problems = validate_visual(v)
        if v.id in seen:
            problems.append("identifiant dupliqué")
        if problems:
            excluded.append(ExcludedVisual(candidate=v.id, reason="; ".join(problems)))
            continue
        seen.add(v.id)
        valid.append(v)

    kept: list[VisualSpec] = []
    counted = 0
    dropped = 0
    for v in valid:
        if v.type in _FIXED_TYPES:
            kept.append(v)
        elif counted < opts.max_visuals:
            kept.append(v)
            counted += 1
        else:
            dropped += 1
            excluded.append(ExcludedVisual(candidate=v.id, reason=f"limite max_visuals={opts.max_visuals}"))
    if dropped:
        caveats.append(f"Troncature : {dropped} visuel(s) non affiché(s) (max_visuals={opts.max_visuals}).")

    kept.sort(key=lambda v: _SECTION_ORDER.get(v.section, 99))
    sections = []
    for sid, title in SECTIONS:
        ids = [v.id for v in kept if v.section == sid]
        if ids:
            sections.append(DashboardSection(id=sid, title=title, order=_SECTION_ORDER[sid], visual_ids=ids))

    report.title = f"Dashboard BI : {report.business_goal}" if report.business_goal else "Dashboard BI"
    report.visuals, report.sections = kept, sections
    report.excluded_visuals, report.caveats = excluded, caveats
    report.trace.append(_trace("assemble", started, n_visuals=len(kept), n_excluded=len(excluded)))
    return {"report": report}


def _fallback_summary(state: DashboardState) -> str:
    analyst, rate, n = state["analyst"], state.get("global_rate"), state.get("n_global", 0)
    parts = []
    if rate is not None:
        parts.append(f"Taux de churn global : {rate:.1f} %" + (f" sur {n} clients." if n else "."))
    if analyst.null_result:
        parts.append("Aucun segment ne passe les filtres statistiques de l'analyse.")
    elif analyst.insights:
        top = analyst.insights[0]
        parts.append(f"Le segment associé au plus fort excès de départs est « {top.title} » "
                     f"({top.impact_churners:.0f} départs en excès, confiance {top.confidence}).")
    return " ".join(parts)


def _report_node(state: DashboardState) -> DashboardState:
    if state.get("abort"):
        return {}
    started = time.perf_counter()
    report, analyst = state["report"], state["analyst"]
    by_section: dict[str, list[str]] = {}
    for v in report.visuals:
        by_section.setdefault(v.section, []).append(v.id)

    blocks: list[ReportBlock] = []
    if by_section.get("overview"):
        blocks.append(ReportBlock(heading="Indicateurs clés", visual_ids=by_section["overview"]))
    if analyst.insights:
        text = " ".join(f"{i.rank}. {i.finding}" for i in analyst.insights[:3])
        blocks.append(ReportBlock(heading="Segments à risque", text=text, visual_ids=by_section.get("segments", [])))
    for sid, heading in (("temporal", "Évolution et ancienneté"), ("geo", "Répartition géographique"),
                         ("profiles", "Profils churnés vs retenus")):
        if by_section.get(sid):
            blocks.append(ReportBlock(heading=heading, visual_ids=by_section[sid]))
    if analyst.recommendations:
        text = " ".join(f"{r.action} (cible : {r.target}, test : {r.test_type})." for r in analyst.recommendations[:3])
        blocks.append(ReportBlock(heading="Hypothèses d'action à tester", text=text,
                                  visual_ids=by_section.get("actions", [])))

    report.report = ReportSpec(
        title=report.title, executive_summary=_fallback_summary(state), blocks=blocks,
        recommendations_table=[r.model_dump(mode="json", include={
            "id", "action", "target", "priority", "test_type", "impact_churners", "confidence", "insight_ids"})
            for r in analyst.recommendations],
        caveats=list(report.caveats))
    report.trace.append(_trace("report", started, n_blocks=len(blocks)))
    return {"report": report}


def _narrate_node(state: DashboardState) -> DashboardState:
    if state.get("abort"):
        return {}
    started = time.perf_counter()
    opts, report, analyst = state["options"], state["report"], state["analyst"]
    fallback = report.report.executive_summary
    report.llm_narrative = fallback

    llm = _load_llm() if opts.use_llm else None
    if llm is None:
        report.trace.append(_trace("narrate", started, llm=False))
        return {"report": report}

    data = {
        "resume": fallback,
        "insights": [i.model_dump(mode="json", include={"title", "finding", "evidence", "confidence"})
                     for i in analyst.insights[:5]],
        "recommendations": [r.model_dump(mode="json", include={"action", "target", "priority", "test_type"})
                            for r in analyst.recommendations[:5]],
    }
    source = json.dumps(data, ensure_ascii=False)
    try:
        resp = llm.invoke([("system", SYSTEM_PROMPT), ("human", NARRATIVE_PROMPT.format(
            business_goal=report.business_goal, data_json=source,
            caveats="\n".join(f"• {c}" for c in report.caveats[:5])))])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        if _grounded(text, source):
            report.llm_narrative = text
            report.report.executive_summary = text
    except Exception:
        pass
    report.trace.append(_trace("narrate", started, llm=True))
    return {"report": report}


def _persist_node(state: DashboardState) -> DashboardState:
    opts, report = state["options"], state["report"]
    if report.status != "error":
        report.status = "warning" if (report.errors or report.null_result or not report.visuals) else "success"
    if state.get("abort") or not opts.write_outputs:
        return {"report": report}
    try:
        out = Path(opts.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "dashboard_report.json"
        report.output_files = {"report": str(path)}
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    except OSError as exc:
        report.output_files = {}
        report.errors.append(f"Écriture des résultats impossible : {exc}")
        report.status = "warning"
    return {"report": report}


# ── Assemblage ───────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(DashboardState)
    for name, fn in (("load", _load_node), ("build", _build_node), ("assemble", _assemble_node),
                     ("report", _report_node), ("narrate", _narrate_node), ("persist", _persist_node)):
        g.add_node(name, fn)
    g.add_edge(START, "load")
    for a, b in (("load", "build"), ("build", "assemble"), ("assemble", "report"),
                 ("report", "narrate"), ("narrate", "persist")):
        g.add_edge(a, b)
    g.add_edge("persist", END)
    return g.compile()


_dashboard_graph = build_graph()


def run(request: DashboardRequest, frames: dict[str, pd.DataFrame] | None = None) -> DashboardReport:
    """Point d'entrée public pour l'Orchestrator. Ne lève jamais d'exception."""
    try:
        return _dashboard_graph.invoke({"request": request, "frames": frames or {}})["report"]
    except Exception as exc:
        return DashboardReport(status="error", errors=[f"Erreur inattendue du Dashboard Agent : {exc}"])