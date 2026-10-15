"""Schémas d'entrée et de sortie du KPI & Semantic Agent."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agents.profiler.models import TraceEntry


class KPIDefinition(BaseModel):
    """Définition sémantique et métier d'un KPI."""

    model_config = ConfigDict(extra="forbid")

    kpi_id: str
    name: str
    category: Literal["financial", "operational", "customer_retention", "risk"]
    business_definition: str
    formula: str
    calculation_type: Literal["ratio", "sum", "average", "count", "percentage"]
    unit: str
    target_direction: Literal["minimize", "maximize", "monitor"]


class KPIResult(BaseModel):
    """Valeur calculée et ventilation d'un indicateur clé de performance."""

    model_config = ConfigDict(extra="forbid")

    kpi_id: str
    name: str
    value: float
    formatted_value: str
    unit: str
    category: str
    formula: str
    target_direction: str
    status_alert: Literal["healthy", "warning", "critical"] = "healthy"
    commentary: str
    breakdowns: dict[str, dict[str, float]] = Field(default_factory=dict)


class DimensionDefinition(BaseModel):
    """Définition d'un axe d'analyse (dimension)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    table: str
    business_name: str
    cardinality: int
    sample_values: list[str]


class MeasureDefinition(BaseModel):
    """Définition d'une valeur chiffrée agrégeable (mesure)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    table: str
    business_name: str
    unit: str
    allowed_aggregations: list[str]


class KPIRequest(BaseModel):
    """Requête de calcul et de construction de la couche sémantique."""

    model_config = ConfigDict(extra="forbid")

    business_goal: str = Field(default="Calcul des KPI Churn et couche sémantique télécom")
    clean_data_path: str = Field(default="data/clean/telco_clean.csv")
    fact_table_path: str = Field(default="data/clean/fact_churn.csv")
    dim_customers_path: str = Field(default="data/clean/dim_customers.csv")
    dim_subscriptions_path: str = Field(default="data/clean/dim_subscriptions.csv")


class KPICatalogReport(BaseModel):
    """Livrable officiel : Catalogue des KPI et Couche Sémantique BI."""

    model_config = ConfigDict(extra="forbid")

    business_goal: str
    timestamp: datetime = Field(default_factory=datetime.now)
    status: Literal["success", "warning", "error"] = "success"

    dimensions: list[DimensionDefinition] = Field(default_factory=list)
    measures: list[MeasureDefinition] = Field(default_factory=list)
    kpis: list[KPIResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    trace: list[TraceEntry] = Field(default_factory=list)
    llm_summary: str | None = None
