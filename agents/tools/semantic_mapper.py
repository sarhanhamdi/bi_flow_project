"""Moteur de détection de domaine et d'usage télécom (Telecom Domain & Use-Case Classifier).

Identifie l'usage métier (customer_churn, network_performance, customer_support)
et mappe les concepts sémantiques clés à partir des colonnes et types d'un DataFrame.
"""

from __future__ import annotations

import re
from typing import Any
import pandas as pd


# Dictionnaire de règles pondérées par domaine
DOMAIN_SIGNALS: dict[str, dict[str, Any]] = {
    "network_performance": {
        "keywords": [
            r"mos", r"latency", r"throughput", r"packet_loss", r"plr", r"jitter",
            r"signal_strength", r"download_speed", r"upload_speed", r"ping",
            r"cdr", r"switch", r"port_no", r"vonr", r"band", r"congestion",
            r"drop.*connection", r"call_drop", r"qoe", r"qos", r"cell.*id",
            # Radio & technique réseau
            r"rsrp", r"rsrq", r"sinr", r"rssi", r"bandwidth", r"frequency",
            r"handover", r"handoff", r"antenna", r"cell.*site", r"tower",
            r"base.*station", r"uplink", r"downlink", r"bit.*rate", r"delay",
            r"retransmission", r"network.*type", r"technology", r"data.*usage",
            r"volume.*data", r"payload",
            # Termes français
            r"debit", r"qualite.*signal", r"perte.*paquet", r"disponibilite",
            r"incident.*reseau", r"cellule", r"couverture", r"interference", r"noeud"
        ],
        "weight": 3.0,
    },
    "customer_support": {
        "keywords": [
            r"ticket", r"complaint", r"issue", r"dispute", r"resolution",
            r"csat", r"satisfaction", r"inquiry", r"solved", r"pending",
            r"consumer_disputed", r"timely_response", r"channel", r"narrative",
            # CRM & Support
            r"case", r"incident", r"request", r"claim", r"feedback",
            r"nps", r"escalat", r"priority", r"severity", r"agent.*id",
            r"reopened", r"first.*contact", r"response.*time", r"handle.*time",
            # Termes français
            r"reclamation", r"plainte", r"delai.*traitement", r"satisfaction.*client",
            r"motif", r"niveau.*priorite", r"resolu", r"en_attente", r"ferme", r"categorie"
        ],
        "weight": 2.5,
    },
    "customer_churn": {
        "keywords": [
            r"churn", r"tenure", r"contract", r"account_length", r"monthly_charge",
            r"total_charge", r"cltv", r"internetservice", r"paperless", r"partner",
            r"dependents", r"senior_citizen", r"payment_method", r"international_plan",
            # Gestion client & Télécom
            r"attrition", r"retention", r"lifetime", r"subscriber", r"subscription",
            r"cancel", r"renewal", r"upgrade", r"downgrade", r"voicemail",
            r"roaming", r"overage", r"arpu", r"revenue", r"mrr",
            r"days.*since", r"last.*activity", r"num.*calls",
            # Termes français
            r"resiliation", r"desabonnement", r"fidelisation", r"portabilite",
            r"abonne", r"anciennete", r"facture", r"offre", r"engagement"
        ],
        "weight": 2.0,
    },
}

# Règles d'extraction des concepts spécifiques
CONCEPT_PATTERNS: dict[str, list[str]] = {
    # Churn / Rétention
    "target_churn": [
        r"^churn.*", r".*churn_value.*", r".*churn_label.*", r".*is_churn.*",
        r".*resiliation.*", r".*attrition.*", r".*desabonnement.*"
    ],
    "tenure": [
        r".*tenure.*", r".*account_length.*", r".*months_customer.*",
        r".*anciennete.*", r".*duree_engagement.*"
    ],
    "monthly_charge": [
        r".*monthly_charge.*", r".*monthlyrevenue.*", r".*monthly_fee.*", r".*day_charge.*",
        r".*facture.*", r".*montant_mensuel.*"
    ],
    "total_revenue": [
        r".*total_charge.*", r".*total_revenue.*", r".*totalrecurringcharge.*",
        r".*chiffre_affaires.*", r".*revenu_total.*"
    ],
    "contract": [r".*contract.*", r".*plan.*", r".*internetservice.*", r".*contrat.*", r".*forfait.*"],

    # Réseau / Performance
    "mos_score": [r".*mos.*", r".*mean_opinion.*"],
    "latency": [r".*latency.*", r".*ping.*", r".*rtt.*", r".*delai.*"],
    "throughput": [r".*throughput.*", r".*download_speed.*", r".*bitrate.*", r".*debit.*"],
    "upload_speed": [r".*upload_speed.*", r".*debit_montant.*", r".*uplink.*"],
    "packet_loss": [r".*packet_loss.*", r".*plr.*", r".*perte.*paquet.*"],
    "signal_strength": [r".*signal_strength.*", r".*rssi.*", r".*rsrp.*", r".*qualite.*signal.*"],
    "target_network_incident": [
        r".*cdr_flag.*", r".*drop.*connection.*", r".*call_drop.*", r".*incident_flag.*",
        r".*incident.*reseau.*", r".*panne.*"
    ],
    "availability": [r".*availability.*", r".*uptime.*", r".*disponibilite.*"],

    # Support / Réclamations
    "ticket_id": [r".*ticket.*", r".*complaint_id.*", r".*case_id.*", r".*issue_id.*", r".*id_reclamation.*"],
    "ticket_status": [r".*ticket_status.*", r"^status$", r".*case_status.*", r".*statut.*"],
    "dispute_flag": [r".*consumer_disputed.*", r".*is_disputed.*", r".*dispute.*", r".*litige.*", r".*contestation.*"],
    "timely_response": [r".*timely_response.*", r".*in_time.*", r".*delai_respecte.*"],
    "csat_score": [r".*satisfaction.*", r".*csat.*", r".*rating.*", r".*note.*"],
    "resolution_time": [r".*resolution_time.*", r".*time_to_resolution.*", r".*delai_traitement.*"],
}


def _normalize_name(name: str) -> str:
    """Normalise un nom de colonne (minuscule, sans ponctuation/espace)."""
    return re.sub(r"[^a-z0-9]", "_", str(name).strip().lower()).strip("_")


def detect_telecom_usecase(df: pd.DataFrame) -> dict[str, Any]:
    """Détecte l'usage métier (domain), la variable cible et les concepts mappés."""
    columns = list(df.columns)
    normalized_cols = {col: _normalize_name(col) for col in columns}

    scores = {
        "customer_churn": 0.0,
        "network_performance": 0.0,
        "customer_support": 0.0,
    }

    # Calcul des scores par domaine
    for original_col, norm_col in normalized_cols.items():
        for domain, conf in DOMAIN_SIGNALS.items():
            for kw in conf["keywords"]:
                if re.search(kw, norm_col):
                    scores[domain] += conf["weight"]

    # Identification du domaine dominant
    sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_domain, best_score = sorted_domains[0]

    # Par défaut si aucun signal fort
    if best_score < 2.0:
        best_domain = "customer_churn"

    # Extraction des concepts
    concepts: dict[str, str | None] = {}
    for concept, patterns in CONCEPT_PATTERNS.items():
        matched_col = None
        for pattern in patterns:
            for original_col, norm_col in normalized_cols.items():
                if re.match(pattern, norm_col):
                    matched_col = original_col
                    break
            if matched_col:
                break
        concepts[concept] = matched_col

    # Détection de la cible principale selon le domaine
    primary_target = None
    target_type = "binary"

    if best_domain == "customer_churn":
        primary_target = concepts.get("target_churn")
    elif best_domain == "network_performance":
        primary_target = concepts.get("target_network_incident") or concepts.get("mos_score") or concepts.get("latency")
    elif best_domain == "customer_support":
        primary_target = concepts.get("dispute_flag") or concepts.get("ticket_status")

    # Si aucune cible trouvée par motif direct, recherche d'une colonne binaire
    if not primary_target:
        for c in columns:
            if c.lower() in ["customer_id", "customerid", "id", "count", "country", "state", "city", "zip_code"]:
                continue
            uniq = df[c].dropna().unique()
            if len(uniq) == 2:
                primary_target = c
                break

    return {
        "domain": best_domain,
        "confidence_scores": scores,
        "primary_target": primary_target,
        "concepts": concepts,
    }
