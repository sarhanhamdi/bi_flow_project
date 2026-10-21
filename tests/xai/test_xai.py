"""Tests unitaires pour l'agent BI Auditor / XAI."""

from pathlib import Path
import pytest
from agents.xai import run, XAIRequest, XAIOptions, XAIReport


def test_xai_agent_telco_success():
    data_path = "output/telco/telco_clean.csv"
    if not Path(data_path).exists():
        pytest.skip("Données clean telco non générées.")

    req = XAIRequest(
        clean_data_path=data_path,
        business_goal="Audit Churn Test",
        options=XAIOptions(
            n_estimators=30,  # rapide pour les tests
            top_n_shap_features=5,
            n_shap_examples=2,
            output_dir="output/test_xai",
        ),
    )
    report = run(req)

    assert isinstance(report, XAIReport)
    assert report.status == "success"
    assert report.model_metrics is not None
    assert report.model_metrics.auc_roc > 0.70
    assert report.model_metrics.accuracy > 0.70
    assert report.shap_summary is not None
    assert len(report.shap_summary.top_features) == 5
    assert len(report.shap_summary.examples) == 2
    assert report.baseline_comparison is not None
    assert report.baseline_comparison.verdict in {"significant_gain", "marginal_gain"}
    assert len(report.audit_findings) >= 3
    assert (Path("output/test_xai/xai_report.json")).exists()


def test_xai_agent_invalid_path():
    req = XAIRequest(clean_data_path="fichier_inexistant.csv")
    report = run(req)
    assert report.status == "error"
    assert len(report.errors) > 0
