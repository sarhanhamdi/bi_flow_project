"""Schémas d'entrée et de sortie du Dashboard Generator Agent.

Contrat avec les autres agents :
  - entrée  : ``DashboardRequest`` = AnalystReport (obligatoire) + KPIReport / ETLReport (optionnels) ;
  - sortie  : ``DashboardReport`` = spec de dashboard + spec de rapport, 100 % sérialisables en JSON,
              prêtes pour le renderer (HTML autonome, Dash...) et pour l'Auditor / XAI Agent.

Principes :
  - La spec est AUTOPORTANTE : chaque visuel embarque ses données agrégées (``data``),
    le renderer n'a besoin ni du CSV ni des DataFrames.
  - Traçabilité (XAI) : chaque visuel porte des ``references`` (kpi_id, insight_ids,
    colonnes sources, formule, origine) pour que l'Auditor remonte jusqu'à la donnée.
  - Gouvernance : pas de géo au niveau individuel (PII), cellules < ``min_cell_rows`` masquées,
    KPIs en fuite de cible exclus, formulation « associé à », jamais de causalité.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agents.profiler.models import TraceEntry

VisualType = Literal[
    "kpi_card", "bar", "line", "area", "heatmap",
    "geo_map", "histogram", "box", "table", "text",
]
Origin = Literal["kpi_report", "analyst_report", "analyst_computed", "dashboard_computed"]


# ── Options ──────────────────────────────────────────────────────────────────

class DashboardOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str = "output/telco/dashboard"
    write_outputs: bool = True
    use_llm: bool = True

    max_visuals: int = Field(default=20, ge=1, description="Nombre maximum de visuels dans le dashboard.")
    min_cell_rows: int = Field(
        default=30, ge=1,
        description="Effectif minimal d'une cellule (segment, ville...) pour être affichée (confidentialité + fiabilité).",
    )
    geo_level: Literal["state", "city"] = Field(
        default="state",
        description="Granularité géographique maximale. Jamais de coordonnées individuelles.",
    )
    top_n_segments: int = Field(default=10, ge=2)
    language: Literal["fr", "en"] = "fr"
    theme: Literal["light", "dark"] = "light"


# ── Requête ──────────────────────────────────────────────────────────────────

class DashboardRequest(BaseModel):
    """Demande transmise par l'Orchestrator.

    ``analyst_report`` = ``AnalystReport.model_dump(mode="json")`` ou contenu de ``analyst_report.json``.
    ``kpi_report``     = KPIReport (valeurs, unités, breakdowns) ; recommandé pour les cartes KPI.
    ``etl_report``     = ETLReport ; sert à repérer colonnes géo / date / PII et le schéma en étoile.
    Les DataFrames nettoyés sont passés à ``run()``, pas stockés ici.
    """

    model_config = ConfigDict(extra="forbid")

    analyst_report: dict[str, Any]
    kpi_report: dict[str, Any] | None = None
    etl_report: dict[str, Any] | None = None
    options: DashboardOptions = Field(default_factory=DashboardOptions)


# ── Traçabilité ──────────────────────────────────────────────────────────────

class VisualReference(BaseModel):
    """Lien d'un visuel vers sa source (consommé par l'Auditor / XAI)."""

    model_config = ConfigDict(extra="forbid")

    kpi_id: str | None = None
    insight_ids: list[str] = Field(default_factory=list)
    recommendation_ids: list[str] = Field(default_factory=list)
    source_columns: list[str] = Field(default_factory=list)
    formula: str | None = None
    aggregation: str | None = Field(default=None, description="Ex. 'mean(churn_value) par state'.")
    origin: Origin = "kpi_report"


# ── Visuels ──────────────────────────────────────────────────────────────────

class KPICard(BaseModel):
    """Contenu d'une carte KPI."""

    model_config = ConfigDict(extra="forbid")

    kpi_id: str
    name: str
    value: float | None
    unit: str = ""
    status: Literal["ok", "warning", "critical", "suspect", "not_computable"] = "ok"
    target: float | None = Field(default=None, description="Seuil ou objectif, si défini.")
    delta: float | None = Field(default=None, description="Écart vs référence (période ou global).")
    higher_is_better: bool | None = None


class Encoding(BaseModel):
    """Mapping colonnes -> axes, indépendant du moteur de rendu."""

    model_config = ConfigDict(extra="forbid")

    x: str | None = None
    y: str | None = None
    color: str | None = None
    size: str | None = None
    geo_key: str | None = Field(default=None, description="Colonne de jointure géographique (state, city).")
    sort: Literal["asc", "desc", "none"] = "none"


class VisualSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: VisualType
    title: str
    subtitle: str = ""
    section: str = Field(description="Id de la section du dashboard qui contient ce visuel.")
    span: int = Field(default=6, ge=1, le=12, description="Largeur en colonnes sur une grille de 12.")
    kpi: KPICard | None = Field(default=None, description="Renseigné uniquement si type == 'kpi_card'.")
    data: list[dict[str, Any]] = Field(default_factory=list, description="Données agrégées embarquées.")
    encoding: Encoding = Field(default_factory=Encoding)
    note: str = Field(default="", description="Lecture du graphique, formulation « associé à ».")
    reliable: bool = True
    warnings: list[str] = Field(default_factory=list)
    references: VisualReference = Field(default_factory=VisualReference)


class FilterSpec(BaseModel):
    """Filtre global du dashboard (le renderer l'applique aux visuels concernés)."""

    model_config = ConfigDict(extra="forbid")

    dimension: str
    label: str
    kind: Literal["select", "multiselect", "range"] = "multiselect"
    values: list[str] = Field(default_factory=list)
    applies_to: list[str] = Field(default_factory=list, description="Ids des visuels filtrables.")


class DashboardSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    order: int = Field(ge=0)
    visual_ids: list[str] = Field(default_factory=list)


class ExcludedVisual(BaseModel):
    """Visuel non produit, avec la raison (ex. fuite de cible, PII, effectif insuffisant)."""

    model_config = ConfigDict(extra="forbid")

    candidate: str
    reason: str


# ── Rapport automatisé ───────────────────────────────────────────────────────

class ReportBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heading: str
    text: str = ""
    visual_ids: list[str] = Field(default_factory=list)


class ReportSpec(BaseModel):
    """Rapport exécutif généré automatiquement (rendu HTML / PDF par ``tools/report.py``)."""

    model_config = ConfigDict(extra="forbid")

    title: str = ""
    executive_summary: str = ""
    blocks: list[ReportBlock] = Field(default_factory=list)
    recommendations_table: list[dict[str, Any]] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


# ── Rapport de sortie ────────────────────────────────────────────────────────

class DashboardReport(BaseModel):
    """Rapport remis à l'Orchestrator, au renderer et à l'Auditor / XAI Agent."""

    model_config = ConfigDict(extra="forbid")

    business_goal: str = ""
    status: Literal["success", "warning", "error"] = "success"
    analyst_status: str = "unknown"

    title: str = ""
    sections: list[DashboardSection] = Field(default_factory=list)
    visuals: list[VisualSpec] = Field(default_factory=list)
    filters: list[FilterSpec] = Field(default_factory=list)
    report: ReportSpec = Field(default_factory=ReportSpec)

    excluded_visuals: list[ExcludedVisual] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    null_result: bool = Field(
        default=False, description="True si l'Analyst n'a retenu aucun insight (dashboard KPI seul).",
    )

    llm_narrative: str = ""
    output_files: dict[str, str] = Field(default_factory=dict)
    trace: list[TraceEntry] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)