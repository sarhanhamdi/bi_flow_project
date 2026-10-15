"""Construction du catalogue sémantique (dimensions et mesures)."""

from __future__ import annotations

import pandas as pd

from agents.kpi.models import DimensionDefinition, MeasureDefinition


def extract_semantic_layer(
    df: pd.DataFrame,
    dim_cust: pd.DataFrame | None = None,
    dim_sub: pd.DataFrame | None = None,
    fact: pd.DataFrame | None = None,
) -> tuple[list[DimensionDefinition], list[MeasureDefinition]]:
    """Construit les définitions sémantiques des axes d'analyse (dimensions) et des faits (mesures)."""
    dimensions: list[DimensionDefinition] = []
    measures: list[MeasureDefinition] = []

    # 1. Dimensions stratégiques
    dim_specs = [
        {"name": "Contract", "table": "dim_subscriptions", "desc": "Type d'engagement contractuel"},
        {"name": "Payment Method", "table": "dim_subscriptions", "desc": "Canal de règlement des factures"},
        {"name": "Internet Service", "table": "dim_subscriptions", "desc": "Technologie d'accès internet"},
        {"name": "Gender", "table": "dim_customers", "desc": "Genre du client"},
        {"name": "Senior Citizen", "table": "dim_customers", "desc": "Statut senior (>=65 ans)"},
        {"name": "City", "table": "dim_customers", "desc": "Ville de résidence"},
        {"name": "Tech Support", "table": "dim_subscriptions", "desc": "Souscription à l'assistance technique"},
        {"name": "Online Security", "table": "dim_subscriptions", "desc": "Option de cybersécurité active"},
    ]

    for spec in dim_specs:
        col = spec["name"]
        if col in df.columns:
            series = df[col].dropna()
            samples = [str(x) for x in series.unique()[:5]]
            dimensions.append(DimensionDefinition(
                name=col,
                table=spec["table"],
                business_name=spec["desc"],
                cardinality=int(series.nunique()),
                sample_values=samples,
            ))

    # 2. Mesures quantitatives
    measure_specs = [
        {"name": "Monthly Charges", "desc": "Montant facturé mensuellement", "unit": "$", "aggs": ["SUM", "AVG", "MIN", "MAX"]},
        {"name": "Total Charges", "desc": "Montant total cumulé dépensé", "unit": "$", "aggs": ["SUM", "AVG"]},
        {"name": "Tenure Months", "desc": "Ancienneté d'abonnement en mois", "unit": "mois", "aggs": ["AVG", "MEDIAN", "MAX"]},
        {"name": "CLTV", "desc": "Valeur vie estimée du client (Customer Lifetime Value)", "unit": "$", "aggs": ["AVG", "SUM"]},
        {"name": "Churn Score", "desc": "Score de risque de résiliation (0-100)", "unit": "pts", "aggs": ["AVG", "MAX"]},
    ]

    for spec in measure_specs:
        col = spec["name"]
        if col in df.columns:
            measures.append(MeasureDefinition(
                name=col,
                table="fact_churn",
                business_name=spec["desc"],
                unit=spec["unit"],
                allowed_aggregations=spec["aggs"],
            ))

    return dimensions, measures
