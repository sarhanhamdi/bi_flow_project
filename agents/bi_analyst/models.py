"""Schémas d'entrée et de sortie du BI Analyst Agent.

Contrat avec les autres agents :
  - entrée  : ``AnalystRequest`` = KPIReport (obligatoire) + ETLReport (optionnel) ;
  - sortie  : ``AnalystReport``  = insights classés + recommandations + réserves,
              prêts pour le Dashboard Agent et l'Auditor / XAI Agent.

Règle de traçabilité (XAI) :
  Chaque ``SegmentStat`` porte un champ ``source`` qui distingue :
    - ``"kpi_report"``      : valeur issue directement des breakdowns du KPIReport ;
    - ``"analyst_computed"`` : valeur calculée par l'Analyst Agent sur la table nettoyée.
  Cela permet à l'agent XAI de remonter jusqu'à la donnée originale.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agents.profiler.models import TraceEntry


# ── Options ──────────────────────────────────────────────────────────────────

class AnalystOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str = "output/analyst"
    write_outputs: bool = True
    use_llm: bool = True

    min_segment_rows: int = Field(
        default=30, ge=1,
        description="Taille minimale d'un segment pour être considéré fiable (cohérent avec le KPI Agent).",
    )
    max_segments: int = Field(
        default=10, ge=2,
        description="Cardinalité maximale d'une dimension (cohérent avec le KPI Agent).",
    )
    significance_level: float = Field(
        default=0.05, gt=0.0, le=1.0,
        description="Seuil α pour la p-value ajustée (après correction BH).",
    )
    min_excess_rate: float = Field(
        default=3.0, ge=0.0,
        description="Écart minimal (en points de %) par rapport au taux global pour retenir un insight.",
    )
    top_n_insights: int = Field(
        default=10, ge=1,
        description="Nombre maximum d'insights retenus après filtrage et tri.",
    )


# ── Requête ──────────────────────────────────────────────────────────────────

class AnalystRequest(BaseModel):
    """Demande transmise par l'Orchestrator.

    ``kpi_report``  = ``KPIReport.model_dump(mode="json")`` ou contenu de ``kpi_report.json``.
    ``etl_report``  = ``ETLReport.model_dump(mode="json")`` ou ``None``.
                      Requis pour les métadonnées PII (``is_pii``) et le chemin de la table nettoyée.
                      Si absent, seules les ventilations du KPIReport sont analysées.
    ``frames``      est passé directement à ``run()``, pas stocké ici.
    """

    model_config = ConfigDict(extra="forbid")

    kpi_report: dict[str, Any]
    etl_report: dict[str, Any] | None = None
    options: AnalystOptions = Field(default_factory=AnalystOptions)


# ── Résultats élémentaires ────────────────────────────────────────────────────

class SegmentStat(BaseModel):
    """Statistiques d'un segment pour un KPI donné.

    ``source`` distingue l'origine de la valeur pour la traçabilité XAI :
    ``"kpi_report"`` (breakdown calculé par le KPI Agent) ou
    ``"analyst_computed"`` (calculé par l'Analyst Agent sur la table nettoyée).
    """

    model_config = ConfigDict(extra="forbid")

    dimension: str
    segment: str
    n: int = Field(ge=0, description="Effectif total du segment.")
    n_churned: int = Field(default=0, ge=0, description="Nombre de clients churnés dans le segment.")
    rate_or_mean: float = Field(description="Taux de churn (%) ou moyenne selon le KPI.")
    global_rate: float = Field(description="Taux global de référence (% sur l'ensemble du portefeuille).")
    excess_rate: float = Field(description="Écart en points : rate_or_mean − global_rate.")
    excess_churners: float = Field(
        description="Départs en excès : (taux_segment − taux_global) / 100 × n_segment.",
    )
    revenue_monthly_lost_excess: float = Field(
        description=(
            "Revenu mensuel perdu en excès (départs passés) : "
            "monthly_revenue_lost_segment × excess_churners / n_churned_segment. "
            "0.0 si non calculable."
        ),
    )
    p_value_raw: float = Field(ge=0.0, le=1.0, description="P-value brute du z-test de proportion.")
    p_value_adjusted: float = Field(ge=0.0, le=1.0, description="P-value après correction Benjamini-Hochberg.")
    effect_size: float = Field(description="Taille d'effet (Cramér's V ou |Cliff's delta|).")
    effect_size_label: str = Field(description="'faible', 'moyen' ou 'fort'.")
    test_used: str = Field(description="Test appliqué : 'proportion_z', 'mann-whitney'.")
    reliable: bool = Field(description="True si n ≥ min_segment_rows.")
    source: Literal["kpi_report", "analyst_computed"]
    references: dict[str, Any] = Field(
        default_factory=dict,
        description="Traçabilité XAI : kpi_id, dimension, segment, source_file.",
    )


# ── Insight ───────────────────────────────────────────────────────────────────

class Insight(BaseModel):
    """Un insight = un signal statistiquement significatif sur un segment.

    ``finding``    : constat en langage naturel (formulation « associé à »).
    ``evidence``   : preuve chiffrée complète (taux, effectifs, p-value, taille d'effet, impact).
    ``rank``       : 1 = impact le plus fort (trié par ``excess_churners`` décroissant).
    ``direction``  : ``"risque"`` si excess_rate > 0 (churn au-dessus du global),
                     ``"protecteur"`` si excess_rate ≤ 0 (churn en dessous du global).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    kpi_id: str
    title: str
    finding: str
    evidence: str
    segment_stats: list[SegmentStat] = Field(default_factory=list)
    impact_churners: float = Field(description="Départs en excès (même valeur que SegmentStat).")
    revenue_monthly_lost_excess: float = Field(description="Revenu mensuel perdu en excès.")
    confidence: Literal["élevée", "moyenne", "faible"]
    direction: Literal["risque", "protecteur"] = Field(
        default="risque",
        description=(
            "Direction de l'association : 'risque' si le segment churne au-dessus du global, "
            "'protecteur' si en dessous."
        ),
    )
    rank: int = Field(ge=1)
    warnings: list[str] = Field(default_factory=list)
    references: dict[str, Any] = Field(default_factory=dict)


# ── Recommandation ───────────────────────────────────────────────────────────

class Recommendation(BaseModel):
    """Une recommandation = une hypothèse à tester, jamais une certitude.

    Chaque recommandation est obligatoirement liée à au moins un insight
    statistiquement vérifié (``insight_ids``).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    action: str = Field(description="Action proposée (formulation au conditionnel ou à l'infinitif).")
    target: str = Field(description="Segment ou population cible.")
    rationale: str = Field(description="Justification : lien vers l'insight, formulation 'associé à'.")
    insight_ids: list[str] = Field(min_length=1, description="Identifiants des insights qui fondent cette recommandation.")
    impact_churners: float = Field(description="Départs en excès couverts par cette recommandation.")
    revenue_monthly_lost_excess: float = Field(description="Revenu mensuel perdu en excès couvert.")
    priority: Literal["haute", "moyenne", "basse"]
    confidence: Literal["élevée", "moyenne", "faible"]
    test_type: str = Field(description="Type de test suggéré : 'A/B test', 'test pilote', 'enquête'.")
    references: dict[str, Any] = Field(default_factory=dict)


# ── Comparaison de profils (Mann-Whitney) ─────────────────────────────────────

class ProfileComparison(BaseModel):
    """Comparaison de la distribution d'une variable continue entre churnés et retenus.

    Produit par l'Analyst Agent via Mann-Whitney U ; jamais transformé en insight
    de proportion et jamais associé à une recommandation directe.

    ``source`` = ``"analyst_computed"`` toujours (calculé sur la table nettoyée).
    """

    model_config = ConfigDict(extra="forbid")

    metric: str = Field(description="Nom de la colonne analysée (ex. 'tenure_months').")
    kpi_id: str = Field(description="Identifiant du KPI associé (ex. 'avg_tenure', 'arpu').")
    n_churned: int = Field(ge=0)
    n_retained: int = Field(ge=0)
    median_churned: float
    median_retained: float
    mean_churned: float = Field(default=0.0)
    mean_retained: float = Field(default=0.0)
    cliffs_delta: float = Field(description="Cliff's delta : mesure d'effet non paramétrique.")
    effect_size_label: str = Field(description="'faible', 'moyen' ou 'fort'.")
    p_value_raw: float = Field(ge=0.0, le=1.0)
    p_value_adjusted: float = Field(
        ge=0.0, le=1.0,
        description="P-value après correction Benjamini-Hochberg (famille commune avec les z-tests).",
    )
    test_used: str = Field(default="mann-whitney")
    source: Literal["analyst_computed"]
    references: dict[str, Any] = Field(default_factory=dict)


# ── Rapport ───────────────────────────────────────────────────────────────────

class AnalystReport(BaseModel):
    """Rapport remis au Dashboard Agent, à l'Orchestrator et à l'Auditor / XAI Agent."""

    model_config = ConfigDict(extra="forbid")

    business_goal: str = ""
    status: Literal["success", "warning", "error"] = "success"
    kpi_status: str = "unknown"

    insights: list[Insight] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    profile_comparisons: list[ProfileComparison] = Field(
        default_factory=list,
        description=(
            "Comparaisons Mann-Whitney (variable continue churné vs retenu). "
            "Section séparée des insights de proportion ; pas de recommandation directe."
        ),
    )

    caveats: list[str] = Field(
        default_factory=list,
        description=(
            "Réserves héritées du KPIReport + caveat fixe sur l'analyse univariée."
        ),
    )
    null_result: bool = Field(
        default=False,
        description="True si aucun segment ne passe tous les filtres (p ajustée, écart minimal, fiabilité).",
    )
    n_tests: int = Field(
        default=0, ge=0,
        description="Nombre total de tests soumis à la correction Benjamini-Hochberg (z-tests + Mann-Whitney).",
    )
    n_significant_segments: int = Field(
        default=0, ge=0,
        description="Nombre de segments passant les filtres (p ajustée ≤ α, écart ≥ seuil, reliable).",
    )
    overlap_warning: bool = Field(
        default=False,
        description="True si plusieurs insights portent sur des segments non mutuellement exclusifs.",
    )
    excluded_kpis: list[dict[str, str]] = Field(
        default_factory=list,
        description="KPIs exclus avec la raison (ex. avg_churn_score = fuite de cible).",
    )

    llm_narrative: str = ""
    output_files: dict[str, str] = Field(default_factory=dict)
    trace: list[TraceEntry] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)
