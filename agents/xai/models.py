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
    llm_summary: str = ""
    errors: list[str] = Field(default_factory=list)
