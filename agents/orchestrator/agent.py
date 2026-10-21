"""Graphe LangGraph de l'Orchestrateur Global BIFlow."""

from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agents.bi_analyst import AnalystOptions, AnalystRequest
from agents.bi_analyst import run as run_analyst
from agents.dashboard import DashboardOptions, DashboardRequest
from agents.dashboard import run as run_dashboard
from agents.etl import ETLOptions, ETLRequest, run_with_data
from agents.kpi import KPIOptions, KPIRequest
from agents.kpi import run as run_kpi
from agents.orchestrator.models import (
    AgentExecutionStep,
    OrchestratorRequest,
    PipelineRunResult,
)
from agents.profiler.agent import run as run_profiler
from agents.profiler.models import ProfileOptions, ProfileRequest, SourceConfig
from agents.xai import XAIOptions, XAIRequest
from agents.xai import run as run_xai


class GlobalWorkflowState(TypedDict, total=False):
    request: OrchestratorRequest
    dataset_name: str
    target_dir: Path
    source_config: SourceConfig
    frames: dict[str, pd.DataFrame]
    result: PipelineRunResult
    start_time: float


def node_init(state: GlobalWorkflowState) -> dict[str, Any]:
    req = state["request"]
    path = Path(req.source_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier source introuvable : {path}")

    dataset_name = req.dataset_name or re.sub(r"\W+", "_", path.stem).strip("_").lower()
    target_dir = Path(req.options.output_dir) / dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)

    source_cfg = SourceConfig(
        name=dataset_name,
        type="csv",
        uri=str(path),
    )

    result = PipelineRunResult(
        status="success",
        dataset_name=dataset_name,
        business_goal=req.business_goal,
        execution_start=datetime.utcnow().isoformat(),
    )

    return {
        "dataset_name": dataset_name,
        "target_dir": target_dir,
        "source_config": source_cfg,
        "result": result,
        "start_time": time.time(),
        "frames": {},
    }


def node_profiler(state: GlobalWorkflowState) -> dict[str, Any]:
    req = state["request"]
    t0 = time.time()
    result = state["result"]

    try:
        profiling = run_profiler(
            ProfileRequest(
                business_goal=req.business_goal,
                sources=[state["source_config"]],
                options=ProfileOptions(),
            )
        )
        duration = round(time.time() - t0, 2)
        result.profiler_report = profiling
        result.steps.append(
            AgentExecutionStep(
                agent_name="profiler",
                status="success" if profiling.status != "error" else "error",
                duration_seconds=duration,
                message=f"{len(profiling.issues)} anomalies détectées",
            )
        )
        # Écriture de l'artefact
        (state["target_dir"] / "profiler_output.json").write_text(
            profiling.model_dump_json(indent=2), encoding="utf-8"
        )
    except Exception as exc:
        duration = round(time.time() - t0, 2)
        result.steps.append(
            AgentExecutionStep(
                agent_name="profiler",
                status="error",
                duration_seconds=duration,
                message=str(exc),
            )
        )
        result.errors.append(f"Profiler: {exc}")

    return {"result": result}


def node_etl(state: GlobalWorkflowState) -> dict[str, Any]:
    req = state["request"]
    result = state["result"]
    t0 = time.time()

    if not result.profiler_report or result.profiler_report.status == "error":
        result.steps.append(
            AgentExecutionStep(
                agent_name="etl",
                status="skipped",
                duration_seconds=0.0,
                message="Skipped: Profiler échoué",
            )
        )
        return {"result": result}

    try:
        report, frames = run_with_data(
            ETLRequest(
                profiling_report=result.profiler_report,
                sources=[state["source_config"]],
                options=ETLOptions(
                    preset=req.options.preset,
                    use_llm=req.options.use_llm,
                    output_dir=str(state["target_dir"]),
                ),
            )
        )
        duration = round(time.time() - t0, 2)
        result.etl_report = report
        result.steps.append(
            AgentExecutionStep(
                agent_name="etl",
                status="success" if report.status != "error" else "error",
                duration_seconds=duration,
                message=f"{len(report.actions)} actions appliquées, étoile validée",
            )
        )
        return {"result": result, "frames": frames}
    except Exception as exc:
        duration = round(time.time() - t0, 2)
        result.steps.append(
            AgentExecutionStep(
                agent_name="etl",
                status="error",
                duration_seconds=duration,
                message=str(exc),
            )
        )
        result.errors.append(f"ETL: {exc}")
        return {"result": result}


def node_kpi(state: GlobalWorkflowState) -> dict[str, Any]:
    req = state["request"]
    result = state["result"]
    t0 = time.time()

    if req.options.skip_kpi or not result.etl_report or result.etl_report.status == "error":
        result.steps.append(
            AgentExecutionStep(
                agent_name="kpi",
                status="skipped",
                duration_seconds=0.0,
                message="Skipped par configuration ou prérequis manquant",
            )
        )
        return {"result": result}

    try:
        kpi_out = run_kpi(
            KPIRequest(
                etl_report=result.etl_report.model_dump(mode="json"),
                options=KPIOptions(
                    output_dir=str(state["target_dir"] / "kpi"),
                    use_llm=req.options.use_llm,
                ),
            ),
            frames=state.get("frames", {}),
        )
        duration = round(time.time() - t0, 2)
        result.kpi_report = kpi_out
        result.steps.append(
            AgentExecutionStep(
                agent_name="kpi",
                status="success" if kpi_out.status != "error" else "error",
                duration_seconds=duration,
                message=f"{len([k for k in kpi_out.kpis if k.status != 'not_computable'])} KPIs calculés",
            )
        )
    except Exception as exc:
        duration = round(time.time() - t0, 2)
        result.steps.append(
            AgentExecutionStep(
                agent_name="kpi",
                status="error",
                duration_seconds=duration,
                message=str(exc),
            )
        )
        result.errors.append(f"KPI: {exc}")

    return {"result": result}


def node_analyst(state: GlobalWorkflowState) -> dict[str, Any]:
    req = state["request"]
    result = state["result"]
    t0 = time.time()

    if (
        req.options.skip_analyst
        or not result.kpi_report
        or result.kpi_report.status == "error"
        or not result.etl_report
    ):
        result.steps.append(
            AgentExecutionStep(
                agent_name="bi_analyst",
                status="skipped",
                duration_seconds=0.0,
                message="Skipped par configuration ou prérequis manquant",
            )
        )
        return {"result": result}

    try:
        analyst_out = run_analyst(
            AnalystRequest(
                kpi_report=result.kpi_report.model_dump(mode="json"),
                etl_report=result.etl_report.model_dump(mode="json"),
                options=AnalystOptions(
                    output_dir=str(state["target_dir"] / "analyst"),
                    use_llm=req.options.use_llm,
                ),
            ),
            frames=state.get("frames", {}),
        )
        duration = round(time.time() - t0, 2)
        result.analyst_report = analyst_out
        result.steps.append(
            AgentExecutionStep(
                agent_name="bi_analyst",
                status="success" if analyst_out.status != "error" else "error",
                duration_seconds=duration,
                message=f"{len(analyst_out.insights)} insights, {len(analyst_out.recommendations)} recommandations",
            )
        )
    except Exception as exc:
        duration = round(time.time() - t0, 2)
        result.steps.append(
            AgentExecutionStep(
                agent_name="bi_analyst",
                status="error",
                duration_seconds=duration,
                message=str(exc),
            )
        )
        result.errors.append(f"BI Analyst: {exc}")

    return {"result": result}


def node_dashboard(state: GlobalWorkflowState) -> dict[str, Any]:
    req = state["request"]
    result = state["result"]
    t0 = time.time()

    if (
        req.options.skip_dashboard
        or not result.analyst_report
        or not result.kpi_report
        or not result.etl_report
    ):
        result.steps.append(
            AgentExecutionStep(
                agent_name="dashboard",
                status="skipped",
                duration_seconds=0.0,
                message="Skipped par configuration ou prérequis manquant",
            )
        )
        return {"result": result}

    try:
        dash_out = run_dashboard(
            DashboardRequest(
                analyst_report=result.analyst_report.model_dump(mode="json"),
                kpi_report=result.kpi_report.model_dump(mode="json"),
                etl_report=result.etl_report.model_dump(mode="json"),
                options=DashboardOptions(
                    output_dir=str(state["target_dir"] / "dashboard"),
                    use_llm=req.options.use_llm,
                ),
            ),
            frames=state.get("frames", {}),
        )
        duration = round(time.time() - t0, 2)
        result.dashboard_report = dash_out
        result.steps.append(
            AgentExecutionStep(
                agent_name="dashboard",
                status="success" if dash_out.status != "error" else "error",
                duration_seconds=duration,
                message=f"{len(dash_out.visuals)} visuels générés dans {len(dash_out.sections)} sections",
            )
        )
    except Exception as exc:
        duration = round(time.time() - t0, 2)
        result.steps.append(
            AgentExecutionStep(
                agent_name="dashboard",
                status="error",
                duration_seconds=duration,
                message=str(exc),
            )
        )
        result.errors.append(f"Dashboard: {exc}")

    return {"result": result}


def node_xai(state: GlobalWorkflowState) -> dict[str, Any]:
    req = state["request"]
    result = state["result"]
    t0 = time.time()

    clean_file = state["target_dir"] / f"{state['dataset_name']}_clean.csv"

    if req.options.skip_xai or not clean_file.exists():
        result.steps.append(
            AgentExecutionStep(
                agent_name="xai",
                status="skipped",
                duration_seconds=0.0,
                message="Skipped ou données clean absentes",
            )
        )
        return {"result": result}

    try:
        xai_out = run_xai(
            XAIRequest(
                clean_data_path=str(clean_file),
                business_goal=req.business_goal,
                options=XAIOptions(
                    output_dir=str(state["target_dir"] / "xai"),
                    use_llm=req.options.use_llm,
                ),
            )
        )
        duration = round(time.time() - t0, 2)
        result.xai_report = xai_out
        result.steps.append(
            AgentExecutionStep(
                agent_name="xai",
                status="success" if xai_out.status != "error" else "error",
                duration_seconds=duration,
                message=(
                    f"AUC: {xai_out.model_metrics.auc_roc:.3f}, Top: "
                    f"{', '.join([f.feature for f in xai_out.shap_summary.top_features[:3]])}"
                    if xai_out.model_metrics and xai_out.shap_summary
                    else "Audit terminé"
                ),
            )
        )
    except Exception as exc:
        duration = round(time.time() - t0, 2)
        result.steps.append(
            AgentExecutionStep(
                agent_name="xai",
                status="error",
                duration_seconds=duration,
                message=str(exc),
            )
        )
        result.errors.append(f"XAI: {exc}")

    return {"result": result}


def node_finalize(state: GlobalWorkflowState) -> dict[str, Any]:
    result = state["result"]
    result.execution_end = datetime.utcnow().isoformat()
    result.total_duration_seconds = round(time.time() - state["start_time"], 2)

    has_error = any(step.status == "error" for step in result.steps)
    has_skip = any(step.status == "skipped" for step in result.steps)

    if has_error:
        result.status = "error"
    elif has_skip:
        result.status = "partial"
    else:
        result.status = "success"

    # Enregistrement du compte rendu consolidé
    (state["target_dir"] / "orchestrator_summary.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )

    return {"result": result}


def build_orchestrator_graph():
    """Assemble et compile le StateGraph de l'Orchestrateur global."""
    g = StateGraph(GlobalWorkflowState)

    g.add_node("init", node_init)
    g.add_node("profiler", node_profiler)
    g.add_node("etl", node_etl)
    g.add_node("kpi", node_kpi)
    g.add_node("analyst", node_analyst)
    g.add_node("dashboard", node_dashboard)
    g.add_node("xai", node_xai)
    g.add_node("finalize", node_finalize)

    g.add_edge(START, "init")
    g.add_edge("init", "profiler")
    g.add_edge("profiler", "etl")
    g.add_edge("etl", "kpi")
    g.add_edge("kpi", "analyst")
    g.add_edge("analyst", "dashboard")
    g.add_edge("dashboard", "xai")
    g.add_edge("xai", "finalize")
    g.add_edge("finalize", END)

    return g.compile()


_orchestrator = build_orchestrator_graph()


def run(request: OrchestratorRequest) -> PipelineRunResult:
    """Exécute de bout en bout l'orchestration multi-agents BIFlow."""
    try:
        final_state = _orchestrator.invoke({"request": request})
        return final_state["result"]
    except Exception as exc:
        return PipelineRunResult(
            status="error",
            dataset_name=request.dataset_name or "unknown",
            business_goal=request.business_goal,
            errors=[str(exc)],
        )
