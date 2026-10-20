"""Catalogue de règles d'actions du BI Analyst Agent.

Chaque règle lie un signal statistique (insight) à une action proposée au métier.

Principe d'extensibilité :
  - Ajouter une règle = ajouter une entrée dans ``ACTION_RULES``.
  - Aucune autre modification du code n'est nécessaire.

Structure d'une règle :
  ``trigger_kpi``       : identifiant du KPI déclencheur (ex. ``"churn_rate"``).
  ``trigger_dimension`` : dimension concernée (``None`` = toute dimension).
  ``trigger_segment``   : valeur exacte du segment (``None`` = tout segment au-dessus du seuil).
  ``min_excess_rate``   : écart minimal en points de % par rapport au taux global.
  ``action``            : action proposée (formulation au conditionnel ou à l'infinitif).
  ``target``            : population cible.
  ``test_type``         : type de test suggéré pour valider l'action.
  ``priority``          : ``"haute"`` / ``"moyenne"`` / ``"basse"``.

Règle de repli :
  Si aucune règle spécifique ne correspond, ``find_rules()`` retourne la règle
  ``"investigate_segment"`` : « investiguer ce segment ».

Priorité de ``find_rules()`` :
  1. Règle spécifique (trigger_kpi + trigger_dimension + trigger_segment) → retour immédiat.
  2. Règle générique (trigger_kpi seul, dimension=None) → retour immédiat.
  3. Règle de repli (investigate_segment) → toujours disponible.

  Pas de cumul règle spécifique + règle générique : dès qu'une règle spécifique
  correspond, les génériques sont ignorées.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ActionRule:
    """Règle d'action reliant un insight à une proposition métier."""

    id: str
    trigger_kpi: str
    trigger_dimension: str | None
    trigger_segment: str | None
    min_excess_rate: float
    action: str
    target: str
    test_type: str
    priority: Literal["haute", "moyenne", "basse"]
    description: str = field(default="")


# ── Règle de repli (générique) ────────────────────────────────────────────────

_FALLBACK_RULE = ActionRule(
    id="investigate_segment",
    trigger_kpi="churn_rate",
    trigger_dimension=None,
    trigger_segment=None,
    min_excess_rate=0.0,
    action=(
        "Investiguer ce segment : recueillir les verbatims clients, croiser avec "
        "les données de satisfaction et identifier les leviers d'action"
    ),
    target="Segment présentant un churn anormalement élevé",
    test_type="enquête qualitative",
    priority="basse",
    description="Règle de repli générique appliquée lorsqu'aucune règle spécifique ne correspond.",
)


# ── Règles spécifiques Telco ─────────────────────────────────────────────────

_TELCO_RULES: list[ActionRule] = [
    ActionRule(
        id="monthly_to_annual",
        trigger_kpi="churn_rate",
        trigger_dimension="contract",
        trigger_segment="Month-to-month",
        min_excess_rate=5.0,
        action=(
            "Tester une offre d'engagement annuel avec remise tarifaire "
            "(ex. 10 % de réduction sur 12 mois prépayés)"
        ),
        target="Clients en contrat mensuel depuis moins de 12 mois d'ancienneté",
        test_type="A/B test",
        priority="haute",
        description=(
            "Le contrat mensuel est fortement associé au churn. "
            "Un engagement annuel réduit le risque sans nécessiter d'action réseau."
        ),
    ),
    ActionRule(
        id="echeck_to_autopay",
        trigger_kpi="churn_rate",
        trigger_dimension="payment_method",
        trigger_segment="Electronic check",
        min_excess_rate=5.0,
        action=(
            "Proposer une incitation au prélèvement automatique "
            "(ex. remise sur la prochaine facture ou mois de service offert)"
        ),
        target="Clients payant actuellement par chèque électronique",
        test_type="A/B test",
        priority="haute",
        description=(
            "Le chèque électronique est associé au plus fort taux de churn "
            "parmi tous les modes de paiement testés."
        ),
    ),
    ActionRule(
        id="fiber_quality",
        trigger_kpi="churn_rate",
        trigger_dimension="internet_service",
        trigger_segment="Fiber optic",
        min_excess_rate=5.0,
        action=(
            "Lancer une enquête qualité sur le service fibre et mettre en place "
            "un SLA de remboursement automatique en cas de coupure"
        ),
        target="Clients abonnés à la fibre optique",
        test_type="enquête + test pilote",
        priority="moyenne",
        description=(
            "Le churn fibre est nettement supérieur au churn DSL, "
            "malgré un positionnement premium. Cause probable : déception qualité."
        ),
    ),
    ActionRule(
        id="early_onboarding",
        trigger_kpi="churn_rate",
        trigger_dimension="tenure_bin",
        trigger_segment="0-12 mois",
        min_excess_rate=3.0,
        action=(
            "Déployer un parcours d'accueil 0-12 mois : appel de bienvenue J+7, "
            "coaching usage J+30, offre de fidélisation J+90"
        ),
        target="Clients en première année d'ancienneté (tenure < 12 mois)",
        test_type="test pilote",
        priority="moyenne",
        description=(
            "Les nouveaux clients présentent un risque de churn disproportionné "
            "pendant leur première année."
        ),
    ),
    ActionRule(
        id="no_techsupport",
        trigger_kpi="churn_rate",
        trigger_dimension="tech_support",
        trigger_segment="No",
        min_excess_rate=3.0,
        action="Proposer un essai gratuit de 3 mois d'assistance technique",
        target="Clients sans support technique actif",
        test_type="A/B test",
        priority="basse",
        description="L'absence de support technique est associée à un churn significativement plus élevé.",
    ),
    ActionRule(
        id="no_online_security",
        trigger_kpi="churn_rate",
        trigger_dimension="online_security",
        trigger_segment="No",
        min_excess_rate=3.0,
        action=(
            "Communiquer sur les bénéfices de la sécurité en ligne "
            "et proposer un essai offert de 1 mois"
        ),
        target="Clients sans option de sécurité en ligne",
        test_type="A/B test",
        priority="basse",
        description="L'absence de sécurité en ligne est associée à un churn plus élevé.",
    ),
    ActionRule(
        id="paperless_billing",
        trigger_kpi="churn_rate",
        trigger_dimension="paperless_billing",
        trigger_segment="Yes",
        min_excess_rate=3.0,
        action=(
            "Améliorer l'expérience de la facturation dématérialisée : "
            "alertes claires, accès facile aux historiques"
        ),
        target="Clients en facturation dématérialisée",
        test_type="test pilote",
        priority="basse",
        description=(
            "Les clients en facturation dématérialisée (paperless=Yes) présentent "
            "un taux de churn de 33,57 % contre 16,33 % pour ceux en facturation papier."
        ),
    ),
    ActionRule(
        id="no_online_backup",
        trigger_kpi="churn_rate",
        trigger_dimension="online_backup",
        trigger_segment="No",
        min_excess_rate=3.0,
        action="Proposer un essai gratuit de sauvegarde en ligne",
        target="Clients sans sauvegarde en ligne active",
        test_type="A/B test",
        priority="moyenne",
        description="L'absence de sauvegarde en ligne est associée à un churn élevé (+13,4 pts).",
    ),
    ActionRule(
        id="no_device_protection",
        trigger_kpi="churn_rate",
        trigger_dimension="device_protection",
        trigger_segment="No",
        min_excess_rate=3.0,
        action="Proposer une offre de protection d'équipement",
        target="Clients sans protection d'équipement",
        test_type="A/B test",
        priority="moyenne",
        description="L'absence de protection d'équipement est associée à un churn élevé (+12,6 pts).",
    ),
    ActionRule(
        id="no_dependents",
        trigger_kpi="churn_rate",
        trigger_dimension="dependents",
        trigger_segment="No",
        min_excess_rate=3.0,
        action="Cibler les offres famille pour les clients sans dépendants",
        target="Clients sans dépendants à charge",
        test_type="test pilote",
        priority="basse",
        description="Les clients sans dépendants présentent un churn significativement plus élevé (+6,0 pts).",
    ),
]


# ── Règles génériques (non spécifiques Telco) ────────────────────────────────

_GENERIC_RULES: list[ActionRule] = [
    ActionRule(
        id="high_churn_any_segment",
        trigger_kpi="churn_rate",
        trigger_dimension=None,
        trigger_segment=None,
        min_excess_rate=5.0,
        action=(
            "Analyser les verbatims et données de satisfaction pour ce segment "
            "afin d'identifier les leviers d'action spécifiques"
        ),
        target="Segment présentant un excès de churn significatif",
        test_type="enquête qualitative",
        priority="moyenne",
        description=(
            "Règle générique pour tout segment présentant un écart de churn important "
            "sans règle spécifique disponible."
        ),
    ),
]


# ── Catalogue complet ─────────────────────────────────────────────────────────

ACTION_RULES: list[ActionRule] = _TELCO_RULES + _GENERIC_RULES


# ── Recherche de règles ───────────────────────────────────────────────────────

def find_rules(
    kpi_id: str,
    dimension: str,
    segment: str,
    excess_rate: float,
) -> list[ActionRule]:
    """Retourne les règles applicables à un insight donné.

    Priorité de correspondance (sans cumul) :
    1. Règles spécifiques : ``trigger_kpi`` ET ``trigger_dimension`` ET ``trigger_segment``.
       → Si au moins une correspond, on retourne uniquement celles-là.
    2. Règles génériques  : ``trigger_kpi`` seul, ``trigger_dimension=None``.
       → Si au moins une correspond, on retourne uniquement celles-là.
    3. Règle de repli     : toujours disponible si rien d'autre ne correspond.

    Pas de cumul règle spécifique + règle générique : dès qu'une règle spécifique
    est trouvée, les génériques sont ignorées. Cela évite les doublons de recommandations.

    Parameters
    ----------
    kpi_id : identifiant du KPI de l'insight.
    dimension : dimension de l'insight (ex. ``"contract"``).
    segment : valeur du segment (ex. ``"Month-to-month"``).
    excess_rate : écart en points de % par rapport au taux global.
    """
    # 1. Règles spécifiques (dimension ET segment définis)
    specific: list[ActionRule] = []
    for rule in ACTION_RULES:
        if rule.trigger_kpi != kpi_id:
            continue
        if rule.trigger_dimension is None or rule.trigger_segment is None:
            continue  # pas spécifique
        if excess_rate < rule.min_excess_rate:
            continue
        if rule.trigger_dimension == dimension and rule.trigger_segment == segment:
            specific.append(rule)
    if specific:
        return specific

    # 2. Règles génériques (dimension=None)
    generic: list[ActionRule] = []
    for rule in ACTION_RULES:
        if rule.trigger_kpi != kpi_id:
            continue
        if rule.trigger_dimension is not None:
            continue  # pas générique
        if excess_rate < rule.min_excess_rate:
            continue
        generic.append(rule)
    if generic:
        return generic

    # 3. Repli
    return [_FALLBACK_RULE]
