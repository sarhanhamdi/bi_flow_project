"""Graphe LangGraph du BI Auditor / XAI Agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agents.tools.xai_auditor import (
    compute_baseline,
    compute_shap_explanations,
    prepare_data,
    train_and_evaluate,
)
from agents.xai.models import (
    AuditFinding,
    BaselineComparison,
    ModelMetrics,
    SHAPExample,
    SHAPFeature,
    SHAPSummary,
    XAIReport,
    XAIRequest,
)
from agents.xai.prompts import SUMMARY_PROMPT, SYSTEM_PROMPT


class XAIState(TypedDict, total=False):
    request: XAIRequest
    report: XAIReport
    df: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_cols: list[str]
    excluded_cols: list[str]
    model: Any
    metrics: dict[str, Any]
    shap_top: list[dict[str, Any]]
    shap_examples: list[dict[str, Any]]
    baseline: dict[str, Any]
    findings: list[AuditFinding]


def _make_fallback_summary(report: XAIReport) -> str:
    """Synthèse déterministe en l'absence de LLM."""
    if not report.model_metrics:
        return "Audit XAI incomplet : modèle non entraîné."

    m = report.model_metrics
    b = report.baseline_comparison
    top_f = ", ".join([f.feature for f in (report.shap_summary.top_features[:3] if report.shap_summary else [])])

    text = (
        f"L'audit du modèle RandomForest indique une Accuracy de {m.accuracy:.1%}, "
        f"un Recall de {m.recall:.1%} et un F1-Score de {m.f1:.1%} (AUC-ROC: {m.auc_roc:.3f}). "
    )
    if b:
        text += (
            f"Par rapport à la baseline naïve (DummyClassifier), le gain en F1-Score est de "
            f"+{b.delta_f1:.1%}, validant l'apport prédictif du modèle ({b.verdict}). "
        )
    if top_f:
        text += f"Les principaux facteurs expliquant le risque de churn selon SHAP sont : {top_f}. "
    text += f"{len(report.audit_findings)} constats d'audit ont été relevés."
    return text


def node_load_data(state: XAIState) -> dict[str, Any]:
    req = state["request"]
    path = Path(req.clean_data_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    df = pd.read_csv(path)
    X_train, X_test, y_train, y_test, feature_cols, excluded_cols = prepare_data(
        df,
        target_column=req.options.target_column,
        test_size=req.options.test_size,
        random_state=req.options.random_state,
    )

    findings = [
        AuditFinding(
            id="AUDIT-DATA-001",
            severity="info",
            category="data_preparation",
            title="Exclusion de fuite de cible",
            detail=f"{len(excluded_cols)} colonnes exclues pour éviter fuite d'information ou biais (ex: churn_score, churn_reason, identifiants).",
            recommendation="Conserver l'exclusion stricte de churn_score en production.",
        )
    ]

    return {
        "df": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "excluded_cols": excluded_cols,
        "findings": findings,
    }


def node_train_model(state: XAIState) -> dict[str, Any]:
    req = state["request"]
    model, metrics = train_and_evaluate(
        state["X_train"],
        state["X_test"],
        state["y_train"],
        state["y_test"],
        n_estimators=req.options.n_estimators,
        random_state=req.options.random_state,
    )

    findings = list(state.get("findings", []))
    if metrics["recall"] < 0.60:
        findings.append(
            AuditFinding(
                id="AUDIT-MODEL-001",
                severity="medium",
                category="performance",
                title="Recall modéré sur le Churn",
                detail=f"Le Recall est de {metrics['recall']:.1%}. Certains clients résilients ne sont pas détectés.",
                recommendation="Ajuster le seuil de décision probabilité ou tester un modèle de boosting (XGBoost/LightGBM).",
            )
        )
    else:
        findings.append(
            AuditFinding(
                id="AUDIT-MODEL-001",
                severity="low",
                category="performance",
                title="Bon équilibre Détection/Précision",
                detail=f"Le modèle capture {metrics['recall']:.1%} des churners réels avec une précision de {metrics['precision']:.1%}.",
                recommendation="Maintenir le monitoring périodique du F1-Score.",
            )
        )

    return {"model": model, "metrics": metrics, "findings": findings}


def node_baseline_compare(state: XAIState) -> dict[str, Any]:
    metrics = state["metrics"]
    baseline = compute_baseline(
        state["X_train"],
        state["X_test"],
        state["y_train"],
        state["y_test"],
        rf_accuracy=metrics["accuracy"],
        rf_f1=metrics["f1"],
    )

    findings = list(state.get("findings", []))
    findings.append(
        AuditFinding(
            id="AUDIT-BASELINE-001",
            severity="info",
            category="baseline",
            title="Validation par rapport au modèle naïf",
            detail=f"Gain de F1-Score de +{baseline['delta_f1']:.1%} par rapport à la classe majoritaire.",
            recommendation="Le modèle ML apporte une valeur réelle par rapport à une règle naïve.",
        )
    )

    return {"baseline": baseline, "findings": findings}


def node_compute_shap(state: XAIState) -> dict[str, Any]:
    req = state["request"]
    top_features, examples = compute_shap_explanations(
        state["model"],
        state["X_test"],
        state["y_test"],
        top_n=req.options.top_n_shap_features,
        n_examples=req.options.n_shap_examples,
    )

    findings = list(state.get("findings", []))
    top_names = [f["feature"] for f in top_features[:3]]
    findings.append(
        AuditFinding(
            id="AUDIT-XAI-001",
            severity="info",
            category="explainability",
            title="Déterminants majeurs identifiés",
            detail=f"Les 3 facteurs les plus influents au niveau global sont : {', '.join(top_names)}.",
            recommendation="Cibler en priorité les offres commerciales sur ces leviers (ex: durée d'engagement, tarification).",
        )
    )

    return {"shap_top": top_features, "shap_examples": examples, "findings": findings}


def node_summarize(state: XAIState) -> dict[str, Any]:
    req = state["request"]
    metrics_model = ModelMetrics(**state["metrics"])
    baseline_model = BaselineComparison(**state["baseline"])

    shap_model = SHAPSummary(
        top_features=[SHAPFeature(**f) for f in state["shap_top"]],
        examples=[SHAPExample(**e) for e in state["shap_examples"]],
    )

    report = XAIReport(
        status="success",
        business_goal=req.business_goal,
        n_features_used=len(state["feature_cols"]),
        features_used=state["feature_cols"],
        features_excluded=state["excluded_cols"],
        model_metrics=metrics_model,
        shap_summary=shap_model,
        baseline_comparison=baseline_model,
        audit_findings=state.get("findings", []),
    )

    report.llm_summary = _make_fallback_summary(report)

    # Sauvegarde du rapport JSON
    out_dir = Path(req.options.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "xai_report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")

    return {"report": report}


def build_graph():
    """Assemble et compile le StateGraph LangGraph de l'agent XAI."""
    g = StateGraph(XAIState)

    g.add_node("load_data", node_load_data)
    g.add_node("train_model", node_train_model)
    g.add_node("baseline_compare", node_baseline_compare)
    g.add_node("compute_shap", node_compute_shap)
    g.add_node("summarize", node_summarize)

    g.add_edge(START, "load_data")
    g.add_edge("load_data", "train_model")
    g.add_edge("train_model", "baseline_compare")
    g.add_edge("baseline_compare", "compute_shap")
    g.add_edge("compute_shap", "summarize")
    g.add_edge("summarize", END)

    return g.compile()


_graph = build_graph()


def run(request: XAIRequest) -> XAIReport:
    """Point d'entrée principal de l'agent XAI."""
    try:
        final_state = _graph.invoke({"request": request})
        return final_state["report"]
    except Exception as exc:
        return XAIReport(status="error", errors=[str(exc)])
