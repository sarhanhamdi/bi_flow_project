"""Tests du Dashboard Generator Agent (données synthétiques de type Telco, sans LLM)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agents.bi_analyst.models import AnalystReport, Insight, ProfileComparison, Recommendation, SegmentStat
from agents.dashboard import DashboardOptions, DashboardRequest, run
from agents.kpi.models import KPIReport, KPIResult, SegmentValue
from agents.tools.dashboard import validate_visual


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 1200
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n)
    tenure = rng.integers(1, 73, n)
    p = np.where(contract == "Month-to-month", 0.42, 0.08)
    churn = rng.random(n) < p
    state = rng.choice(["California", "Texas", "Ohio"], n)
    return pd.DataFrame({
        "CustomerID": [f"C{i}" for i in range(n)], "State": state,
        "City": rng.choice(["Los Angeles", "Austin", "Columbus"], n),
        "Latitude": rng.uniform(30, 40, n), "Longitude": rng.uniform(-120, -80, n),
        "Contract": contract, "Internet Service": internet, "Tenure Months": tenure,
        "Churn Value": churn.astype(int), "Churn Score": rng.integers(1, 100, n)})


def _kpi() -> dict:
    return KPIReport(business_goal="Analyser le churn", kpis=[
        KPIResult(id="churn_rate", name="Taux de churn", category="churn", definition="Part de clients partis",
                  formula="churned / total", unit="%", status="computed", value=26.5, rows_used=1200,
                  columns_used={"churn": "Churn Value"},
                  breakdowns={"Contract": [SegmentValue(segment="Month-to-month", value=42.0, rows=660),
                                           SegmentValue(segment="One year", value=8.0, rows=300),
                                           SegmentValue(segment="Tiny", value=50.0, rows=5)]},
                  justification="ok"),
        KPIResult(id="avg_churn_score", name="Score moyen", category="churn", definition="d", formula="f",
                  status="computed", value=50.0, justification="ok"),
        KPIResult(id="arpu", name="ARPU", category="rev", definition="d", formula="f", status="not_computable",
                  missing_inputs={"revenue": ["monthly_charges"]}, justification="ok")]).model_dump(mode="json")


def _analyst() -> dict:
    st = SegmentStat(dimension="Contract", segment="Month-to-month", n=660, n_churned=277, rate_or_mean=42.0,
                     global_rate=26.5, excess_rate=15.5, excess_churners=102.3, revenue_monthly_lost_excess=0.0,
                     p_value_raw=0.0001, p_value_adjusted=0.0002, effect_size=0.3, effect_size_label="moyen",
                     test_used="proportion_z", reliable=True, source="kpi_report")
    st2 = st.model_copy(update={"dimension": "Internet Service", "segment": "Fiber optic"})
    ins = Insight(id="I1", kpi_id="churn_rate", title="Contrat mensuel", finding="Associé à un churn plus élevé",
                  evidence="42 % vs 26.5 %", segment_stats=[st, st2], impact_churners=102.3,
                  revenue_monthly_lost_excess=0.0, confidence="élevée", rank=1)
    rec = Recommendation(id="R1", action="Tester une offre de fidélisation", target="Contrat mensuel",
                         rationale="associé à", insight_ids=["I1"], impact_churners=102.3,
                         revenue_monthly_lost_excess=0.0, priority="haute", confidence="élevée", test_type="A/B test")
    pc = ProfileComparison(metric="tenure_months", kpi_id="avg_tenure", n_churned=300, n_retained=900,
                           median_churned=10, median_retained=40, cliffs_delta=-0.6, effect_size_label="fort",
                           p_value_raw=0.0, p_value_adjusted=0.0, source="analyst_computed")
    return AnalystReport(business_goal="Analyser le churn", insights=[ins], recommendations=[rec],
                         profile_comparisons=[pc], caveats=["Analyse univariée non ajustée."],
                         n_significant_segments=1).model_dump(mode="json")


def _run(frame, **opts):
    return run(DashboardRequest(analyst_report=_analyst(), kpi_report=_kpi(),
                                options=DashboardOptions(write_outputs=False, use_llm=False, **opts)),
               frames={"t": frame})


def test_full_run_builds_expected_visuals(frame):
    r = _run(frame)
    ids = {v.id for v in r.visuals}
    assert r.status == "success"
    assert {"kpi_churn_rate", "line_churn_by_tenure", "geo_churn_by_state", "bar_churn_rate_contract",
            "table_insights", "table_recommendations", "profile_tenure_months"} <= ids
    assert any(v.type == "heatmap" for v in r.visuals)


def test_leakage_and_not_computable_are_excluded(frame):
    r = _run(frame)
    reasons = {e.candidate: e.reason for e in r.excluded_visuals}
    assert "avg_churn_score" in reasons and "fuite de cible" in reasons["avg_churn_score"]
    assert "arpu" in reasons and "non calculable" in reasons["arpu"]
    assert not any(v.kpi and v.kpi.kpi_id == "avg_churn_score" for v in r.visuals)


def test_no_individual_pii_in_any_visual(frame):
    r = _run(frame, geo_level="city")
    dumped = json.dumps([v.model_dump(mode="json") for v in r.visuals])
    assert "C0" not in dumped and "CustomerID" not in dumped and "Churn Score" not in dumped
    for v in r.visuals:
        assert validate_visual(v) == []
    geo = next(v for v in r.visuals if v.type == "geo_map")
    assert all(row["n"] >= 30 for row in geo.data)


def test_small_segments_are_masked(frame):
    r = _run(frame)
    bar = next(v for v in r.visuals if v.id == "bar_churn_rate_contract")
    assert "Tiny" not in {row["segment"] for row in bar.data}
    assert bar.warnings


def test_every_visual_has_traceability(frame):
    for v in (v for v in _run(frame).visuals if v.type != "text"):
        ref = v.references
        assert ref.kpi_id or ref.insight_ids or ref.recommendation_ids or ref.source_columns


def test_without_frame_degrades_with_caveat():
    r = run(DashboardRequest(analyst_report=_analyst(), kpi_report=_kpi(),
                             options=DashboardOptions(write_outputs=False, use_llm=False)))
    assert any(v.type == "kpi_card" for v in r.visuals)
    assert not any(v.type in {"geo_map", "heatmap"} for v in r.visuals)
    assert any("Table nettoyée indisponible" in c for c in r.caveats)


def test_invalid_analyst_report_returns_error_without_raising():
    r = run(DashboardRequest(analyst_report={"insights": "oops"}, options=DashboardOptions(write_outputs=False)))
    assert r.status == "error" and r.errors


def test_max_visuals_truncates_but_keeps_tables(frame):
    r = _run(frame, max_visuals=2)
    assert sum(1 for v in r.visuals if v.type not in {"table", "text"}) == 2
    assert any(v.id == "table_recommendations" for v in r.visuals)
    assert any("Troncature" in c for c in r.caveats)


def test_persist_writes_json(frame, tmp_path):
    r = run(DashboardRequest(analyst_report=_analyst(), kpi_report=_kpi(),
                             options=DashboardOptions(output_dir=str(tmp_path), use_llm=False)), frames={"t": frame})
    assert json.loads((tmp_path / "dashboard_report.json").read_text(encoding="utf-8"))["visuals"]
    assert r.output_files["report"].endswith("dashboard_report.json")


def test_geo_falls_back_to_city_when_single_state(frame):
    frame = frame.assign(State="California")
    r = _run(frame)  # geo_level par défaut = "state"
    geo = next(v for v in r.visuals if v.type == "geo_map")
    assert geo.id == "geo_churn_by_city"
    assert any("repli automatique" in w for w in geo.warnings)
    assert validate_visual(geo) == []


def test_geo_stays_on_state_when_enough_states(frame):
    geo = next(v for v in _run(frame).visuals if v.type == "geo_map")
    assert geo.id == "geo_churn_by_state" and not any("repli" in w for w in geo.warnings)


def test_geo_excluded_when_no_geo_column(frame):
    r = _run(frame.drop(columns=["State", "City"]))
    assert not any(v.type == "geo_map" for v in r.visuals)
    assert any(e.candidate == "geo_map" for e in r.excluded_visuals)


def _kpi_with_derived() -> dict:
    seg = lambda v: [SegmentValue(segment="Month-to-month", value=v, rows=660),
                     SegmentValue(segment="One year", value=v / 2, rows=300)]
    mk = lambda i, n, u: KPIResult(id=i, name=n, category="c", definition="d", formula="f", unit=u,
                                   status="computed", value=10.0, rows_used=1200,
                                   breakdowns={"Contract": seg(10.0)}, justification="ok")
    return KPIReport(business_goal="x", kpis=[
        mk("churned_customers", "Clients churnés", "clients"),   # dérivé du churn -> priorité basse
        mk("arpu", "ARPU", "devise"),                             # indépendant -> priorité moyenne
        mk("churn_rate", "Taux de churn", "%")]).model_dump(mode="json")


def test_bar_priority_keeps_primary_and_independent_kpis_first(frame):
    r = run(DashboardRequest(analyst_report=_analyst(), kpi_report=_kpi_with_derived(),
                             options=DashboardOptions(write_outputs=False, use_llm=False, max_visuals=9)),
            frames={"t": frame})
    bars = [v.id for v in r.visuals if v.type == "bar" and v.id.startswith("bar_")]
    assert bars == ["bar_churn_rate_contract", "bar_arpu_contract"]
    assert any(e.candidate == "bar_churned_customers_contract" for e in r.excluded_visuals)