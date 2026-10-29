"""Schémas d'entrée et de sortie du BI Auditor / XAI Agent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class XAIOptions(BaseModel):
    """Options de configuration de l'agent XAI."""

    model_config = ConfigDict(extra="forbid")

    target_column: str = Field(default="churn_value", description="Colonne cible (0/1)")
    test_size: float = Field(default=0.2, ge=0.05, le=0.5, description="Part du jeu de test")
    n_estimators: int = Field(default=100, ge=10, le=1000)
    random_state: int = Field(default=42)
    top_n_shap_features: int = Field(default=10, ge=1, le=50)
    n_shap_examples: int = Field(default=3, ge=1, le=20)
    output_dir: str = Field(default="output/telco/xai")
    use_llm: bool = Field(default=False)


class XAIRequest(BaseModel):
    """Requête envoyée à l'agent XAI."""

    model_config = ConfigDict(extra="forbid")

    clean_data_path: str = Field(description="Chemin vers telco_clean.csv")
    business_goal: str = Field(default="Prédire le churn client et expliquer les facteurs clés")
    options: XAIOptions = Field(default_factory=XAIOptions)


class ModelMetrics(BaseModel):
    """Métriques d'évaluation du modèle ML."""

    model_config = ConfigDict(extra="forbid")

    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    n_train: int
    n_test: int
    positive_rate_train: float = Field(description="Taux de churn dans le train")
    positive_rate_test: float = Field(description="Taux de churn dans le test")


class SHAPFeature(BaseModel):
    """Importance SHAP d'une feature."""

    model_config = ConfigDict(extra="forbid")

    feature: str
    mean_abs_shap: float
    rank: int


class SHAPExample(BaseModel):
    """Explication SHAP pour un client individuel."""

    model_config = ConfigDict(extra="forbid")

    client_index: int
    true_label: int
    predicted_label: int
    predicted_proba: float
    top_features: list[dict[str, Any]] = Field(description="[{feature, shap_value, feature_value}]")


class SHAPSummary(BaseModel):
    """Résumé global et exemples individuels SHAP."""

    model_config = ConfigDict(extra="forbid")

    top_features: list[SHAPFeature]
    examples: list[SHAPExample]


class BaselineComparison(BaseModel):
    """Comparaison RandomForest vs DummyClassifier."""

    model_config = ConfigDict(extra="forbid")

    baseline_accuracy: float
    baseline_f1: float
    delta_accuracy: float = Field(description="RF accuracy - baseline accuracy")
    delta_f1: float = Field(description="RF F1 - baseline F1")
    verdict: Literal["significant_gain", "marginal_gain", "no_gain"]


class AuditFinding(BaseModel):
    """Constat d'audit structuré."""

    model_config = ConfigDict(extra="forbid")

    id: str
    severity: Literal["high", "medium", "low", "info"]
    category: str
    title: str
    detail: str
    recommendation: str
    role: Literal["business", "technical", "all"] = Field(
        default="all",
        description="Public cible : business (financier/marketing), technical (data/mlops), ou all"
    )
    financial_impact: str | None = Field(default=None, description="Impact financier chiffré si applicable")


class ConfusionMatrix(BaseModel):
    """Matrice de confusion binaire."""

    model_config = ConfigDict(extra="forbid")

    true_positives: int = Field(ge=0, description="Vrais Positifs (ex: churners bien détectés)")
    false_positives: int = Field(ge=0, description="Faux Positifs (ex: clients fidèles ciblés à tort)")
    true_negatives: int = Field(ge=0, description="Vrais Négatifs (ex: clients fidèles épargnés)")
    false_negatives: int = Field(ge=0, description="Faux Négatifs (ex: départs non détectés)")


class MarketingLever(BaseModel):
    """Levier d'action marketing / opérationnel dérivé des facteurs SHAP."""

    model_config = ConfigDict(extra="forbid")

    feature: str = Field(description="Variable d'origine issue de SHAP")
    impact_rank: int = Field(description="Rang d'importance")
    business_meaning: str = Field(description="Traduction en langage métier")
    recommended_action: str = Field(description="Action commerciale ou tarifaire concrète")
    expected_gain: str = Field(description="Bénéfice attendu")


class FinancialAudit(BaseModel):
    """Audit financier, ROI et matrice de coûts pour décideurs métier/marketing."""

    model_config = ConfigDict(extra="forbid")

    domain: str = Field(default="customer_churn", description="Domaine métier analysé")
    currency: str = Field(default="$")
    unit_customer_value: float = Field(description="Valeur vie client moyenne (CLTV) ou coût moyen d'un incident")
    unit_campaign_cost: float = Field(description="Coût unitaire d'une intervention ou offre de fidélisation")
    retention_success_rate: float = Field(description="Taux d'efficacité estimé de la rétention ciblée (ex: 35%)")
    total_risk_exposure: float = Field(description="Revenu total en péril sans intervention")
    gross_saved_value: float = Field(description="Revenu brut sauvé par les détections réussies")
    campaign_budget_spent: float = Field(description="Budget total engagé pour cibler les alertes")
    projected_net_savings: float = Field(description="Bénéfice net généré après déduction des coûts de campagne")
    campaign_roi_pct: float = Field(description="Retour sur investissement des campagnes (%)")
    waste_reduction_pct: float = Field(description="Réduction du gaspillage d'offres promo vs ciblage aveugle (%)")
    optimal_decision_threshold: float = Field(description="Seuil de probabilité optimal maximisant le gain financier")
    cost_inaction_ratio: float = Field(description="Ratio coût d'un départ / coût d'une offre")
    marketing_levers: list[MarketingLever] = Field(default_factory=list)


class TechnicalAudit(BaseModel):
    """Audit technique, MLOps, intégrité et robustesse pour Data Scientists/Ingénieurs."""

    model_config = ConfigDict(extra="forbid")

    confusion_matrix: ConfusionMatrix
    target_leakage_status: Literal["clean", "warning", "leakage_detected"] = Field(
        default="clean", description="Statut du test d'anti-fuite de données"
    )
    excluded_sensitive_columns: list[str] = Field(default_factory=list)
    train_test_stability: str = Field(description="Contrôle de stabilité de la distribution Train vs Test")
    overfitting_risk: Literal["low", "medium", "high"] = Field(default="low")
    inference_latency_ms: float = Field(default=3.5, description="Temps d'inférence moyen par requête en ms")
    model_algorithm: str = Field(default="RandomForestClassifier (100 estimators)")
    production_certification: Literal["certified", "conditional", "rejected"] = Field(
        default="certified", description="Statut de certification pour mise en production"
    )


class XAIReport(BaseModel):
    """Rapport complet du BI Auditor / XAI Agent."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "error"] = "success"
    business_goal: str = ""
    n_features_used: int = 0
    features_used: list[str] = Field(default_factory=list)
    features_excluded: list[str] = Field(default_factory=list)
    model_metrics: ModelMetrics | None = None
    shap_summary: SHAPSummary | None = None
    baseline_comparison: BaselineComparison | None = None
    audit_findings: list[AuditFinding] = Field(default_factory=list)
    financial_audit: FinancialAudit | None = None
    technical_audit: TechnicalAudit | None = None
    llm_summary: str = ""
    errors: list[str] = Field(default_factory=list)
