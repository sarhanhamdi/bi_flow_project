"""Schémas d'entrée et de sortie du Data Quality / ETL Agent."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agents.profiler.models import ProfilingReport, SourceConfig, TraceEntry


class TransformationStep(BaseModel):
    """Description d'une étape de transformation appliquée."""

    model_config = ConfigDict(extra="forbid")

    step_name: str = Field(description="Nom de la règle appliquée")
    description: str = Field(description="Explication lisible par un humain")
    columns_affected: list[str] = Field(default_factory=list)
    rows_affected: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0, ge=0.0)


class QualityCheckResult(BaseModel):
    """Résultat d'une règle de qualité (type Great Expectations)."""

    model_config = ConfigDict(extra="forbid")

    check_id: str
    rule_name: str
    column: str | None = None
    status: Literal["passed", "failed", "warning"]
    details: str
    passed_rows: int = Field(default=0, ge=0)
    failed_rows: int = Field(default=0, ge=0)


class StarSchemaTable(BaseModel):
    """Description d'une table du modèle dimensionnel (en étoile)."""

    model_config = ConfigDict(extra="forbid")

    table_name: str
    table_type: Literal["fact", "dimension"]
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    primary_key: list[str]
    foreign_keys: dict[str, str] = Field(default_factory=dict)
    columns: list[str]
    file_path_csv: str
    file_path_parquet: str


class ETLRequest(BaseModel):
    """Requête envoyée au Data Quality / ETL Agent."""

    model_config = ConfigDict(extra="forbid")

    business_goal: str = Field(default="Nettoyage et modélisation des données")
    sources: list[SourceConfig] = Field(min_length=1)
    profile_report: ProfilingReport | None = Field(
        default=None,
        description="Rapport d'audit transmis par le Data Profiler Agent",
    )
    output_dir: str = Field(
        default="data/clean",
        description="Répertoire où stocker les données nettoyées et le modèle en étoile",
    )
    enable_star_schema: bool = Field(
        default=True,
        description="Générer automatiquement les tables dimensionnelles et de faits",
    )


class DataQualityReport(BaseModel):
    """Livrable officiel : Rapport de qualité et d'ingénierie des données."""

    model_config = ConfigDict(extra="forbid")

    business_goal: str
    timestamp: datetime = Field(default_factory=datetime.now)
    status: Literal["success", "warning", "error"] = "success"

    initial_rows: int = Field(default=0, ge=0)
    initial_cols: int = Field(default=0, ge=0)
    cleaned_rows: int = Field(default=0, ge=0)
    cleaned_cols: int = Field(default=0, ge=0)

    quality_score_initial: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Score de qualité des données brutes en %",
    )
    quality_score_final: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Score de qualité après nettoyage ETL en %",
    )

    transformations: list[TransformationStep] = Field(default_factory=list)
    quality_checks: list[QualityCheckResult] = Field(default_factory=list)
    star_schema_tables: list[StarSchemaTable] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    trace: list[TraceEntry] = Field(default_factory=list)
    llm_summary: str | None = None
