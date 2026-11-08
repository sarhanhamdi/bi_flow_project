"""Gestionnaire de persistance et d'application des règles de gouvernance des données.

Permet à l'administrateur de modifier les règles de nettoyage, validation et conformité
directement depuis l'interface web (Data Governance Admin Console).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.etl.models import (
    ConditionalFillRule,
    ETLOptions,
    MappingRule,
    RangeRule,
)

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
GOVERNANCE_CONFIG_FILE = CONFIG_DIR / "governance_rules.json"


def get_default_governance_rules() -> dict[str, Any]:
    """Retourne la configuration par défaut de la gouvernance des données."""
    return {
        "global_settings": {
            "drop_column_missing_pct": 80.0,
            "iqr_multiplier": 1.5,
            "numeric_missing_strategy": "median",
            "categorical_missing_strategy": "unknown",
            "unknown_label": "Unknown",
            "pii_policy": "hash",
            "strip_whitespace": True,
            "standardize_column_names": True,
            "drop_exact_duplicates": True,
        },
        "range_rules": [
            {"column": "tenure", "min": 0, "max": None, "fix": "report", "description": "L'ancienneté doit être positive"},
            {"column": "monthly_charges", "min": 0, "max": None, "fix": "report", "description": "Le montant mensuel doit être positif"},
            {"column": "total_charges", "min": 0, "max": None, "fix": "report", "description": "Le montant total doit être positif"},
            {"column": "total_day_minutes", "min": 0, "max": 1440, "fix": "report", "description": "Minutes/jour <= 1440 (24h)"},
            {"column": "total_eve_minutes", "min": 0, "max": 1440, "fix": "report", "description": "Minutes/soirée <= 1440"},
            {"column": "total_night_minutes", "min": 0, "max": 1440, "fix": "report", "description": "Minutes/nuit <= 1440"},
            {"column": "customer_age", "min": 18, "max": 120, "fix": "report", "description": "Âge client légal (18-120 ans)"},
            {"column": "customer_satisfaction_rating", "min": 1, "max": 5, "fix": "report", "description": "Score CSAT borné de 1 à 5"},
        ],
        "mapping_rules": [
            {
                "column_a": "churn_label",
                "column_b": "churn_value",
                "mapping": {"Yes": 1, "No": 0},
                "description": "Standardisation binaire du churn (Yes/No -> 1/0)"
            },
            {
                "column_a": "phone_service",
                "column_b": "multiple_lines",
                "mapping": {"No": "No phone service"},
                "description": "Cohérence conditionnelle de la téléphonie"
            }
        ],
        "conditional_fills": [
            {
                "column": "total_charges",
                "value": 0.0,
                "when_column": "tenure",
                "when_equals": 0,
                "rationale": "Client à 0 mois d'ancienneté n'a pas encore de facture cumulée (total = 0)"
            }
        ]
    }


def load_governance_rules() -> dict[str, Any]:
    """Charge les règles de gouvernance personnalisées ou renvoie les règles par défaut."""
    if GOVERNANCE_CONFIG_FILE.exists():
        try:
            content = GOVERNANCE_CONFIG_FILE.read_text(encoding="utf-8")
            data = json.loads(content)
            # S'assurer que les clés essentielles sont présentes
            defaults = get_default_governance_rules()
            for k, v in defaults.items():
                if k not in data:
                    data[k] = v
            return data
        except Exception:
            pass
    return get_default_governance_rules()


def save_governance_rules(rules: dict[str, Any]) -> dict[str, Any]:
    """Sauvegarde les règles de gouvernance modifiées par l'administrateur."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    GOVERNANCE_CONFIG_FILE.write_text(json.dumps(rules, indent=2, ensure_ascii=False), encoding="utf-8")
    return rules


def reset_governance_rules() -> dict[str, Any]:
    """Réinitialise la politique de gouvernance aux valeurs par défaut."""
    defaults = get_default_governance_rules()
    save_governance_rules(defaults)
    return defaults


def apply_governance_to_options(options: ETLOptions) -> ETLOptions:
    """Injecte les règles de gouvernance configurées dans les options de l'agent ETL."""
    rules = load_governance_rules()

    # 1. Paramètres globaux de qualité
    glob = rules.get("global_settings", {})
    update: dict[str, Any] = {}

    if "drop_column_missing_pct" in glob:
        update["drop_column_missing_pct"] = float(glob["drop_column_missing_pct"])
    if "iqr_multiplier" in glob:
        update["iqr_multiplier"] = float(glob["iqr_multiplier"])
    if "numeric_missing_strategy" in glob:
        update["numeric_missing_strategy"] = glob["numeric_missing_strategy"]
    if "categorical_missing_strategy" in glob:
        update["categorical_missing_strategy"] = glob["categorical_missing_strategy"]
    if "pii_policy" in glob:
        update["pii_policy"] = glob["pii_policy"]
    if "strip_whitespace" in glob:
        update["strip_whitespace"] = bool(glob["strip_whitespace"])
    if "drop_exact_duplicates" in glob:
        update["drop_exact_duplicates"] = bool(glob["drop_exact_duplicates"])

    # 2. Règles d'intervalle (Range Rules)
    custom_ranges: list[RangeRule] = []
    for r in rules.get("range_rules", []):
        try:
            custom_ranges.append(RangeRule(
                column=r["column"],
                min=r.get("min"),
                max=r.get("max"),
                fix=r.get("fix", "report"),
            ))
        except Exception:
            pass
    if custom_ranges:
        update["range_rules"] = options.range_rules + custom_ranges

    # 3. Règles de mapping
    custom_mappings: list[MappingRule] = []
    for m in rules.get("mapping_rules", []):
        try:
            custom_mappings.append(MappingRule(
                column_a=m["column_a"],
                column_b=m["column_b"],
                mapping=m["mapping"],
                fix=m.get("fix", "report")
            ))
        except Exception:
            pass
    if custom_mappings:
        update["mapping_rules"] = options.mapping_rules + custom_mappings

    # 4. Remplissages conditionnels
    custom_fills: list[ConditionalFillRule] = []
    for cf in rules.get("conditional_fills", []):
        try:
            custom_fills.append(ConditionalFillRule(
                column=cf["column"],
                value=cf["value"],
                when_column=cf["when_column"],
                when_equals=cf["when_equals"],
                rationale=cf.get("rationale", "Règle de gouvernance admin")
            ))
        except Exception:
            pass
    if custom_fills:
        update["conditional_fills"] = options.conditional_fills + custom_fills

    return options.model_copy(update=update)
