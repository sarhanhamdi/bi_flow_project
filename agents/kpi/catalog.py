"""Catalogue de KPI télécom enrichi et sectorisé par domaine métier."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class KPIDefinition:
    id: str
    name: str
    category: str
    definition: str
    formula: str
    agg: str
    inputs: dict[str, list[str]]
    unit: str = ""
    scale: float = 1.0
    valid_range: tuple[float | None, float | None] = (0.0, None)
    tags: list[str] = field(default_factory=list)


# Dimensions à privilégier pour les ventilations
PREFERRED_DIMENSIONS = [
    "contract", "internet_service", "payment_method", "paperless_billing",
    "senior_citizen", "gender", "partner", "dependents", "phone_service",
    "segment", "network_type", "carrier", "device_model", "band",
    "product", "submitted_via", "company_response_to_consumer", "ticket_type", "ticket_channel"
]

_CHURN_FLAG = ["churn_value", "churn_flag", "churn", "is_churn", "churned", "churn_label"]
_MONTHLY = ["monthly_charges", "monthly_charge", "monthly_fee", "monthly_revenue", "total_day_charge", "day_charge"]

CATALOG: list[KPIDefinition] = [
    # =================================================================== 1. DOMAINE CHURN
    KPIDefinition(
        id="churn_rate", name="Taux de churn", category="Churn", unit="%",
        definition="Part des clients ayant quitté l'opérateur.",
        formula="clients_churnés / clients_total × 100",
        agg="rate", inputs={"flag": _CHURN_FLAG}, valid_range=(0.0, 100.0), tags=["churn"]),
    KPIDefinition(
        id="churned_customers", name="Clients churnés", category="Churn", unit="clients",
        definition="Nombre de clients ayant quitté l'opérateur.",
        formula="somme(indicateur_churn)",
        agg="count_flag", inputs={"flag": _CHURN_FLAG}, tags=["churn"]),
    KPIDefinition(
        id="avg_churn_score", name="Score de churn moyen", category="Churn", unit="points",
        definition="Score moyen de risque de départ (0 à 100) du portefeuille.",
        formula="moyenne(churn_score)",
        agg="mean", inputs={"value": ["churn_score"]}, valid_range=(0.0, 100.0), tags=["churn"]),
    KPIDefinition(
        id="monthly_revenue_lost", name="Revenu mensuel perdu par churn", category="Churn",
        unit="devise/mois",
        definition="Revenu mensuel récurrent des clients qui sont partis.",
        formula="somme(frais_mensuels) pour les clients churnés",
        agg="sum_where", inputs={"value": _MONTHLY, "flag": _CHURN_FLAG}, tags=["churn"]),
    KPIDefinition(
        id="customer_count", name="Nombre de clients", category="Customer Base", unit="clients",
        definition="Nombre de clients distincts dans la table.",
        formula="nombre_distinct(identifiant_client)",
        agg="count_distinct", inputs={"id": ["customer_id", "customerid", "subscriber_id", "phone_number"]}, tags=["churn", "general"]),
    KPIDefinition(
        id="avg_tenure", name="Ancienneté moyenne", category="Customer Base", unit="mois",
        definition="Durée moyenne depuis la souscription.",
        formula="moyenne(ancienneté_en_mois)",
        agg="mean", inputs={"value": ["tenure_months", "tenure", "months_active", "account_length"]}, tags=["churn"]),
    KPIDefinition(
        id="arpu", name="ARPU mensuel", category="Billing & Revenue", unit="devise/client/mois",
        definition="Revenu mensuel moyen par client.",
        formula="moyenne(frais_mensuels)",
        agg="mean", inputs={"value": _MONTHLY}, tags=["churn"]),
    KPIDefinition(
        id="total_revenue", name="Revenu cumulé", category="Billing & Revenue", unit="devise",
        definition="Somme des montants facturés depuis la souscription.",
        formula="somme(total_facturé)",
        agg="sum", inputs={"value": ["total_charges", "total_revenue", "total_billed"]}, tags=["churn"]),
    KPIDefinition(
        id="avg_cltv", name="CLTV moyen", category="Billing & Revenue", unit="devise",
        definition="Valeur vie client moyenne estimée.",
        formula="moyenne(cltv)",
        agg="mean", inputs={"value": ["cltv", "clv", "customer_lifetime_value"]}, tags=["churn"]),
    KPIDefinition(
        id="unpaid_rate", name="Taux d'impayés", category="Billing & Revenue", unit="%",
        definition="Part du montant facturé restée impayée.",
        formula="somme(montant_impayé) / somme(montant_facturé) × 100",
        agg="ratio", scale=100.0, valid_range=(0.0, 100.0),
        inputs={"num": ["unpaid_amount", "overdue_amount"],
                "den": ["billed_amount", "invoice_amount"]}, tags=["churn"]),

    # =================================================================== 2. DOMAINE QUALITÉ & PERFORMANCE RÉSEAU
    KPIDefinition(
        id="avg_mos_voice", name="Score MOS Voix moyen", category="Network Performance", unit="score",
        definition="Score moyen d'opinion (MOS) de la qualité vocale (1 à 5).",
        formula="moyenne(mos_voice)",
        agg="mean", inputs={"value": ["mos_voice", "mos", "streaming_mos"]}, tags=["network"]),
    KPIDefinition(
        id="avg_throughput", name="Débit moyen", category="Network Performance", unit="Mbps",
        definition="Débit réseau moyen observé.",
        formula="moyenne(throughput)",
        agg="mean", inputs={"value": ["throughput_mbps", "effective_bitrate_mbps", "throughput", "download_speed_mbps", "download_speed"]}, tags=["network"]),
    KPIDefinition(
        id="avg_upload_speed", name="Débit montant moyen", category="Network Performance", unit="Mbps",
        definition="Débit montant (upload) moyen observé.",
        formula="moyenne(upload_speed)",
        agg="mean", inputs={"value": ["upload_speed_mbps", "upload_speed"]}, tags=["network"]),
    KPIDefinition(
        id="avg_latency", name="Latence moyenne", category="Network Performance", unit="ms",
        definition="Latence réseau moyenne observée.",
        formula="moyenne(latence)",
        agg="mean", inputs={"value": ["latency_ms", "latency", "dns_latency_ms", "ping_to_google_ms"]}, tags=["network"]),
    KPIDefinition(
        id="avg_packet_loss", name="Taux de perte de paquets", category="Network Performance", unit="%",
        definition="Pourcentage moyen de paquets perdus.",
        formula="moyenne(plr)",
        agg="mean", inputs={"value": ["plr", "packet_loss"]}, tags=["network"]),
    KPIDefinition(
        id="network_availability", name="Disponibilité réseau", category="Network Performance", unit="%",
        definition="Taux moyen de disponibilité des équipements.",
        formula="moyenne(availability)",
        agg="mean", inputs={"value": ["availability"]}, tags=["network"]),
    KPIDefinition(
        id="dropped_call_rate", name="Taux d'appels coupés", category="Network Performance", unit="%",
        definition="Part des appels interrompus avant la fin normale.",
        formula="somme(appels_coupés) / somme(appels_total) × 100",
        agg="ratio", scale=100.0, valid_range=(0.0, 100.0),
        inputs={"num": ["dropped_calls"], "den": ["total_calls", "calls"]}, tags=["network"]),
    KPIDefinition(
        id="cdr_incident_rate", name="Taux d'incidents réseau / CDR", category="Network Performance", unit="%",
        definition="Part des sessions réseau ayant subi un incident ou coupure.",
        formula="sessions_en_incident / sessions_total × 100",
        agg="rate", inputs={"flag": ["cdr_flag", "incident_flag", "call_drop", "dropped_connection"]}, tags=["network"]),
    KPIDefinition(
        id="avg_signal_strength", name="Force du signal moyen", category="Network Performance", unit="dBm",
        definition="Puissance moyenne du signal radio.",
        formula="moyenne(signal_strength_dbm)",
        agg="mean", inputs={"value": ["signal_strength_dbm", "signal_strength", "rssi", "rsrp"]}, tags=["network"]),
    KPIDefinition(
        id="avg_data_usage", name="Consommation data moyenne", category="Usage & Consumption", unit="Mo",
        definition="Volume de données moyen consommé par session/ligne.",
        formula="moyenne(volume_data)",
        agg="mean", inputs={"value": ["data_usage_mb", "data_mb", "data_usage", "internet_mb"]}, tags=["network", "churn"]),

    # =================================================================== 3. DOMAINE SUPPORT & EXPÉRIENCE CLIENT
    KPIDefinition(
        id="ticket_count", name="Nombre de tickets / réclamations", category="Operations & Support", unit="tickets",
        definition="Nombre total de tickets ou réclamations distinctes.",
        formula="nombre_distinct(ticket_id)",
        agg="count_distinct", inputs={"id": ["ticket_id", "ticket", "case_id", "complaint_id", "issue_id"]}, tags=["support"]),
    KPIDefinition(
        id="ticket_resolution_rate", name="Taux de résolution des tickets", category="Operations & Support", unit="%",
        definition="Part des tickets ou réclamations résolus / clos.",
        formula="tickets_clos / total_tickets × 100",
        agg="rate", inputs={"flag": ["ticket_status", "status", "case_status"]}, tags=["support"]),
    KPIDefinition(
        id="dispute_rate", name="Taux de litiges clients", category="Operations & Support", unit="%",
        definition="Part des réclamations ayant entraîné un litige client.",
        formula="réclamations_contestées / total_réclamations × 100",
        agg="rate", inputs={"flag": ["consumer_disputed", "consumer_disputed?", "is_disputed", "disputed"]}, tags=["support"]),
    KPIDefinition(
        id="timely_response_rate", name="Taux de réponse dans les délais", category="Operations & Support", unit="%",
        definition="Part des réclamations traitées dans le délai imparti.",
        formula="réponses_dans_les_délais / total × 100",
        agg="rate", inputs={"flag": ["timely_response", "timely_response?", "in_time"]}, tags=["support"]),
    KPIDefinition(
        id="avg_csat", name="Score de satisfaction moyen (CSAT)", category="Operations & Support", unit="score",
        definition="Score moyen de satisfaction client.",
        formula="moyenne(satisfaction_client)",
        agg="mean", inputs={"value": ["customer_satisfaction_rating", "csat", "satisfaction_rating", "rating"]}, tags=["support"]),
    KPIDefinition(
        id="avg_resolution_time", name="Temps de résolution moyen", category="Operations & Support", unit="h",
        definition="Délai moyen de résolution d'un ticket.",
        formula="moyenne(temps_de_résolution)",
        agg="mean", inputs={"value": ["resolution_time_hours", "resolution_time", "resolution_hours"]}, tags=["support"]),
]
