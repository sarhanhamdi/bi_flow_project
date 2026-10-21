"""Catalogue du Dashboard Generator : quel visuel pour quelle forme de donnée.

Le planner (déterministe d'abord, LLM ensuite) choisit dans ce catalogue ;
il n'invente jamais un type de visuel hors catalogue.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisualRule:
    type: str
    when: str                 # forme de donnée qui déclenche ce visuel
    requires: tuple[str, ...]  # rôles de colonnes nécessaires
    min_rows: int = 1


VISUAL_RULES: tuple[VisualRule, ...] = (
    VisualRule("kpi_card", "valeur unique d'un KPI (+ statut, seuil, delta)", ("value",)),
    VisualRule("bar", "KPI ventilé par dimension catégorielle (segment, contrat, offre)", ("category", "value"), 2),
    VisualRule("line", "KPI ordonné : ancienneté (tenure) ou date si disponible", ("ordered", "value"), 3),
    VisualRule("heatmap", "KPI croisé sur deux dimensions catégorielles", ("category", "category", "value"), 4),
    VisualRule("geo_map", "KPI agrégé par zone géographique (state / city), jamais par client", ("geo", "value"), 2),
    VisualRule("histogram", "distribution d'une variable continue", ("numeric",), 10),
    VisualRule("box", "variable continue churnés vs retenus (ProfileComparison)", ("numeric", "group"), 10),
    VisualRule("table", "top insights / recommandations avec preuves chiffrées", ("rows",), 1),
    VisualRule("text", "narration ou réserves", ("text",)),
)

# Statut d'une carte KPI selon le statut du KPIReport
KPI_STATUS_MAP: dict[str, str] = {
    "ok": "ok", "computed": "ok", "suspect": "suspect", "not_computable": "not_computable",
}

# Fuite de cible : jamais affichés comme KPI de pilotage (cohérent avec le BI Analyst)
LEAKAGE_KPIS: frozenset[str] = frozenset({"avg_churn_score"})
LEAKAGE_COLUMNS: frozenset[str] = frozenset({"churn_label", "churn_value", "churn_score", "churn_reason"})

# Sections du dashboard, dans l'ordre d'affichage
SECTIONS: tuple[tuple[str, str], ...] = (
    ("overview", "Vue d'ensemble"),
    ("segments", "Segments à risque"),
    ("temporal", "Évolution et ancienneté"),
    ("geo", "Répartition géographique"),
    ("profiles", "Profils churnés vs retenus"),
    ("actions", "Recommandations"),
    ("quality", "Qualité et réserves"),
)

# Colonnes géo autorisées pour l'agrégation (jamais lat/long individuels)
GEO_AGGREGATION_COLUMNS: tuple[str, ...] = ("state", "city")