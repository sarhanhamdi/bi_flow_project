"""Schémas d'entrée et de sortie de l'Orchestrateur Global BIFlow."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agents.bi_analyst.models import AnalystReport
from agents.dashboard.models import DashboardReport
from agents.etl.models import ETLReport
from agents.kpi.models import KPIReport
from agents.profiler.models import ProfilingReport
from agents.xai.models import XAIReport


class OrchestratorOptions(BaseModel):
    """Options d'exécution globale du pipeline."""

    model_config = ConfigDict(extra="forbid")

    preset: str | None = Field(default="telco", description="Preset ETL (ex: telco)")
    output_dir: str = Field(default="output", description="Dossier racine des artefacts")
    skip_kpi: bool = Field(default=False)
    skip_analyst: bool = Field(default=False)
    skip_dashboard: bool = Field(default=False)
    skip_xai: bool = Field(default=False)
    use_llm: bool = Field(default=False)


class OrchestratorRequest(BaseModel):
    """Requête de lancement globale du pipeline BIFlow."""

    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(description="Chemin du fichier source CSV")
    dataset_name: str | None = Field(default=None, description="Nom logique de la table")
    business_goal: str = Field(default="Analyse BI et prédiction de churn telecom")
    options: OrchestratorOptions = Field(default_factory=OrchestratorOptions)


class AgentExecutionStep(BaseModel):
    """Trace de l'exécution d'un agent."""

    model_config = ConfigDict(extra="forbid")

    agent_name: str
    status: Literal["success", "warning", "skipped", "error"]
    duration_seconds: float
    message: str = ""


class PipelineRunResult(BaseModel):
    """Résultat consolidé retourné par l'Orchestrateur."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "partial", "error"]
    dataset_name: str
    business_goal: str
    execution_start: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    execution_end: str | None = None
    total_duration_seconds: float = 0.0
    steps: list[AgentExecutionStep] = Field(default_factory=list)
    profiler_report: ProfilingReport | None = None
    etl_report: ETLReport | None = None
    kpi_report: KPIReport | None = None
    analyst_report: AnalystReport | None = None
    dashboard_report: DashboardReport | None = None
    xai_report: XAIReport | None = None
    errors: list[str] = Field(default_factory=list)
