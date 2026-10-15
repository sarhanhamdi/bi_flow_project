"""Graphe LangGraph du Data Quality / ETL Agent."""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agents.etl.models import (
    DataQualityReport,
    ETLRequest,
    QualityCheckResult,
    StarSchemaTable,
    TransformationStep,
)
from agents.etl.prompts import ETL_SUMMARY_PROMPT, SYSTEM_PROMPT
from agents.profiler.models import TraceEntry
from agents.tools.cleaner import clean_dataframe
from agents.tools.loader import LoaderError, load_source
from agents.tools.modeler import build_star_schema
from agents.tools.validator import validate_dataframe


class ETLState(TypedDict, total=False):
    request: ETLRequest
    report: DataQualityReport
    raw_df: pd.DataFrame
    clean_df: pd.DataFrame
    trace: list[TraceEntry]


def _make_fallback_summary(report: DataQualityReport) -> str:
    """Synthèse déterministe lisible sans dépendance LLM."""
    if report.status == "error":
        err_msg = "; ".join(report.errors) or "erreur inconnue"
        return f"L'exécution ETL a échoué : {err_msg}"

    transform_lines = [
        f"- {t.step_name} : {t.description} ({t.rows_affected} ligne(s) affectée(s))"
        for t in report.transformations
    ]
    check_lines = [
        f"- [{c.status.upper()}] {c.rule_name} ({c.column or 'table'}) : {c.details}"
        for c in report.quality_checks
    ]
    table_lines = [
        f"- {t.table_name} ({t.table_type.upper()}) : {t.row_count} lignes, {t.column_count} colonnes (PK: {', '.join(t.primary_key)})"
        for t in report.star_schema_tables
    ]

    return (
        f"=== RAPPORT DATA QUALITY & ETL ===\n"
        f"Objectif : {report.business_goal}\n"
        f"Volumétrie : {report.initial_rows} lignes x {report.initial_cols} colonnes -> {report.cleaned_rows} lignes x {report.cleaned_cols} colonnes.\n"
        f"Évolution de la qualité : {report.quality_score_initial:.1f}% -> {report.quality_score_final:.1f}%\n\n"
        f"Transformations appliquées :\n" + "\n".join(transform_lines) + "\n\n"
        f"Règles de validation (Gouvernance) :\n" + "\n".join(check_lines) + "\n\n"
        f"Modèle dimensionnel généré (Star Schema) :\n" + "\n".join(table_lines)
    )


def _load_llm():
    """Charge le LLM local Ollama si configuré et disponible."""
    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.1", temperature=0)
    except Exception:
        return None


def _summarize_with_llm(report: DataQualityReport) -> str:
    """Interprète le rapport ETL via LLM, sinon bascule sur la synthèse déterministe."""
    llm = _load_llm()
    if llm is None:
        return _make_fallback_summary(report)

    data = {
        "business_goal": report.business_goal,
        "initial_rows": report.initial_rows,
        "cleaned_rows": report.cleaned_rows,
        "quality_score_initial": report.quality_score_initial,
        "quality_score_final": report.quality_score_final,
        "transformations": [t.model_dump(mode="json") for t in report.transformations],
        "quality_checks": [c.model_dump(mode="json") for c in report.quality_checks],
        "star_schema_tables": [s.model_dump(mode="json") for s in report.star_schema_tables],
    }
    prompt = ETL_SUMMARY_PROMPT.format(
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


def _load_and_audit_node(state: ETLState) -> ETLState:
    """Étape 1 : Chargement des sources brutes et audit initial."""
    request = state["request"]
    report = DataQualityReport(business_goal=request.business_goal)
    trace: list[TraceEntry] = []

    started = time.perf_counter()
    try:
        primary_source = request.sources[0]
        raw_df, meta = load_source(primary_source)

        report.initial_rows = len(raw_df)
        report.initial_cols = len(raw_df.columns)

        # Audit initial de qualité avant nettoyage
        _, initial_score = validate_dataframe(raw_df)
        report.quality_score_initial = initial_score

        trace.append(TraceEntry(
            timestamp=datetime.now(),
            tool="load_source",
            parameters={"source": primary_source.name, "rows": len(raw_df)},
            duration_ms=(time.perf_counter() - started) * 1000,
        ))

        return {"request": request, "report": report, "raw_df": raw_df, "trace": trace}

    except LoaderError as exc:
        report.status = "error"
        report.errors.append(f"Erreur de chargement ETL : {exc}")
        return {"request": request, "report": report, "trace": trace}


def _clean_node(state: ETLState) -> ETLState:
    """Étape 2 : Nettoyage, imputation et harmonisation."""
    report = state["report"]
    raw_df = state.get("raw_df")
    trace = state.get("trace", [])

    if raw_df is None or raw_df.empty:
        return state

    started = time.perf_counter()
    issues = report.errors  # Ou request.profile_report.issues
    clean_df, steps = clean_dataframe(raw_df, issues=issues)

    report.cleaned_rows = len(clean_df)
    report.cleaned_cols = len(clean_df.columns)
    report.transformations.extend(steps)

    trace.append(TraceEntry(
        timestamp=datetime.now(),
        tool="clean_dataframe",
        parameters={"steps_count": len(steps)},
        duration_ms=(time.perf_counter() - started) * 1000,
    ))

    return {"clean_df": clean_df, "report": report, "trace": trace}


def _validate_node(state: ETLState) -> ETLState:
    """Étape 3 : Validation de la qualité selon les règles de gouvernance."""
    report = state["report"]
    clean_df = state.get("clean_df")
    trace = state.get("trace", [])

    if clean_df is None or clean_df.empty:
        return state

    started = time.perf_counter()
    checks, final_score = validate_dataframe(clean_df)
    report.quality_checks = checks
    report.quality_score_final = final_score

    trace.append(TraceEntry(
        timestamp=datetime.now(),
        tool="validate_dataframe",
        parameters={"checks_count": len(checks), "score": final_score},
        duration_ms=(time.perf_counter() - started) * 1000,
    ))

    return {"report": report, "trace": trace}


def _model_node(state: ETLState) -> ETLState:
    """Étape 4 : Découpage en modèle dimensionnel en étoile (Star Schema)."""
    request = state["request"]
    report = state["report"]
    clean_df = state.get("clean_df")
    trace = state.get("trace", [])

    if clean_df is None or clean_df.empty or not request.enable_star_schema:
        return state

    started = time.perf_counter()
    tables = build_star_schema(clean_df, output_dir=request.output_dir)
    report.star_schema_tables = tables

    trace.append(TraceEntry(
        timestamp=datetime.now(),
        tool="build_star_schema",
        parameters={"tables_generated": len(tables)},
        duration_ms=(time.perf_counter() - started) * 1000,
    ))

    report.trace = trace
    return {"report": report, "trace": trace}


def _summary_node(state: ETLState) -> ETLState:
    """Étape 5 : Synthèse globale du rapport Data Quality."""
    report = state["report"]
    report.llm_summary = _summarize_with_llm(report)
    return {"report": report}


def build_etl_graph():
    """Assemble le graphe LangGraph de l'agent ETL."""
    graph = StateGraph(ETLState)
    graph.add_node("load_and_audit", _load_and_audit_node)
    graph.add_node("clean", _clean_node)
    graph.add_node("validate", _validate_node)
    graph.add_node("model", _model_node)
    graph.add_node("summarize", _summary_node)

    graph.add_edge(START, "load_and_audit")
    graph.add_edge("load_and_audit", "clean")
    graph.add_edge("clean", "validate")
    graph.add_edge("validate", "model")
    graph.add_edge("model", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()


_etl_graph = build_etl_graph()


def run(request: ETLRequest) -> DataQualityReport:
    """Point d'entrée public de l'ETL Agent."""
    try:
        result = _etl_graph.invoke({"request": request})
        return result["report"]
    except Exception as exc:
        return DataQualityReport(
            business_goal=request.business_goal,
            status="error",
            errors=[f"Erreur inattendue de l'ETL Agent : {exc}"],
        )
