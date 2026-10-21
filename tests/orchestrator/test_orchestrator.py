"""Tests unitaires de l'Orchestrateur BIFlow."""

from pathlib import Path
import pytest
from agents.orchestrator import run, OrchestratorRequest, OrchestratorOptions, PipelineRunResult


def test_orchestrator_full_flow():
    source_csv = "data/telco.csv"
    if not Path(source_csv).exists():
        pytest.skip("Fichier data/telco.csv introuvable.")

    req = OrchestratorRequest(
        source_path=source_csv,
        business_goal="Test Orchestrateur BIFlow",
        options=OrchestratorOptions(
            preset="telco",
            output_dir="output",
            skip_xai=True,  # Rapide pour les tests CI
        ),
    )

    result = run(req)

    assert isinstance(result, PipelineRunResult)
    assert result.status in {"success", "partial"}
    assert result.dataset_name == "telco"
    assert result.profiler_report is not None
    assert result.etl_report is not None
    assert result.kpi_report is not None
    assert result.analyst_report is not None
    assert result.dashboard_report is not None
    assert any(step.agent_name == "profiler" for step in result.steps)
    assert any(step.agent_name == "etl" for step in result.steps)
    assert (Path("output/telco/orchestrator_summary.json")).exists()


def test_orchestrator_invalid_source():
    req = OrchestratorRequest(source_path="chemin_inexistant.csv")
    result = run(req)
    assert result.status == "error"
    assert len(result.errors) > 0
