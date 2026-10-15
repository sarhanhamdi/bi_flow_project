"""Graphe LangGraph du KPI & Semantic Agent."""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agents.kpi.models import (
    DimensionDefinition,
    KPICatalogReport,
    KPIRequest,
    KPIResult,
    MeasureDefinition,
)
from agents.kpi.prompts import KPI_SUMMARY_PROMPT, SYSTEM_PROMPT
from agents.profiler.models import TraceEntry
from agents.tools.kpi_calculator import calculate_telecom_kpis
from agents.tools.semantic_catalog import extract_semantic_layer


class KPIState(TypedDict, total=False):
    request: KPIRequest
    report: KPICatalogReport
    df: pd.DataFrame
    trace: list[TraceEntry]


def _make_fallback_summary(report: KPICatalogReport) -> str:
    """Synthèse déterministe claire sans dépendance LLM."""
    if report.status == "error":
        err_msg = "; ".join(report.errors) or "erreur inconnue"
        return f"Échec du calcul des KPI : {err_msg}"

    kpi_lines = [
        f"- {k.name} ({k.category.upper()}) : {k.formatted_value} [Statut: {k.status_alert.upper()}] — Formule: {k.formula}"
        for k in report.kpis
    ]
    dim_names = [d.name for d in report.dimensions]
    measure_names = [m.name for m in report.measures]

    return (
        f"=== CATALOGUE DES KPI & COUCHE SÉMANTIQUE BI ===\n"
        f"Objectif : {report.business_goal}\n\n"
        f"Indicateurs Clés de Performance ({len(report.kpis)} métriques) :\n" + "\n".join(kpi_lines) + "\n\n"
        f"Couche Sémantique :\n"
        f"- Dimensions ({len(report.dimensions)}) : {', '.join(dim_names)}\n"
        f"- Mesures ({len(report.measures)}) : {', '.join(measure_names)}\n"
    )


def _load_llm():
    """Charge le LLM local si disponible."""
    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.1", temperature=0)
    except Exception:
        return None


def _summarize_with_llm(report: KPICatalogReport) -> str:
    """Interprète le catalogue de KPI via LLM ou bascule sur la synthèse déterministe."""
    llm = _load_llm()
    if llm is None:
        return _make_fallback_summary(report)

    data = {
        "business_goal": report.business_goal,
        "kpis": [k.model_dump(mode="json") for k in report.kpis],
        "dimensions": [d.model_dump(mode="json") for d in report.dimensions],
        "measures": [m.model_dump(mode="json") for m in report.measures],
    }
    prompt = KPI_SUMMARY_PROMPT.format(
        business_goal=report.business_goal,
        results=json.dumps(data, ensure_ascii=False, indent=2),
    )

    try:
        response = llm.invoke([
            ("system", SYSTEM_PROMPT),
            ("human", prompt),
        ])
        content = response.content
        return content if isinstance(content, str) else str(content)
    except Exception:
        return _make_fallback_summary(report)


def _load_data_node(state: KPIState) -> KPIState:
    """Étape 1 : Chargement des données propres issues de l'ETL."""
    request = state["request"]
    report = KPICatalogReport(business_goal=request.business_goal)
    trace: list[TraceEntry] = []

    started = time.perf_counter()
    try:
        # Priorité au fichier unifié telco_clean.csv ou parquet
        path = request.clean_data_path
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        trace.append(TraceEntry(
            timestamp=datetime.now(),
            tool="load_clean_data",
            parameters={"path": path, "rows": len(df)},
            duration_ms=(time.perf_counter() - started) * 1000,
        ))

        return {"request": request, "report": report, "df": df, "trace": trace}

    except Exception as exc:
        report.status = "error"
        report.errors.append(f"Impossible de charger les données propres pour le calcul des KPI : {exc}")
        return {"request": request, "report": report, "trace": trace}


def _semantic_layer_node(state: KPIState) -> KPIState:
    """Étape 2 : Extraction et structuration de la couche sémantique."""
    report = state["report"]
    df = state.get("df")
    trace = state.get("trace", [])

    if df is None or df.empty:
        return state

    started = time.perf_counter()
    dimensions, measures = extract_semantic_layer(df)
    report.dimensions = dimensions
    report.measures = measures

    trace.append(TraceEntry(
        timestamp=datetime.now(),
        tool="extract_semantic_layer",
        parameters={"dimensions_count": len(dimensions), "measures_count": len(measures)},
        duration_ms=(time.perf_counter() - started) * 1000,
    ))

    return {"report": report, "trace": trace}


def _calculate_kpis_node(state: KPIState) -> KPIState:
    """Étape 3 : Calcul déterministe de tous les KPIs télécoms."""
    report = state["report"]
    df = state.get("df")
    trace = state.get("trace", [])

    if df is None or df.empty:
        return state

    started = time.perf_counter()
    kpis = calculate_telecom_kpis(df)
    report.kpis = kpis

    trace.append(TraceEntry(
        timestamp=datetime.now(),
        tool="calculate_telecom_kpis",
        parameters={"kpis_calculated": len(kpis)},
        duration_ms=(time.perf_counter() - started) * 1000,
    ))

    report.trace = trace
    return {"report": report, "trace": trace}


def _summary_node(state: KPIState) -> KPIState:
    """Étape 4 : Synthèse du rapport officiel de KPIs."""
    report = state["report"]
    report.llm_summary = _summarize_with_llm(report)
    return {"report": report}


def build_kpi_graph():
    """Assemble le graphe LangGraph de l'agent KPI."""
    graph = StateGraph(KPIState)
    graph.add_node("load_data", _load_data_node)
    graph.add_node("semantic_layer", _semantic_layer_node)
    graph.add_node("calculate_kpis", _calculate_kpis_node)
    graph.add_node("summarize", _summary_node)

    graph.add_edge(START, "load_data")
    graph.add_edge("load_data", "semantic_layer")
    graph.add_edge("semantic_layer", "calculate_kpis")
    graph.add_edge("calculate_kpis", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()


_kpi_graph = build_kpi_graph()


def run(request: KPIRequest) -> KPICatalogReport:
    """Point d'entrée public du KPI & Semantic Agent."""
    try:
        result = _kpi_graph.invoke({"request": request})
        return result["report"]
    except Exception as exc:
        return KPICatalogReport(
            business_goal=request.business_goal,
            status="error",
            errors=[f"Erreur inattendue du KPI Agent : {exc}"],
        )
