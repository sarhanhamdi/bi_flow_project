"""Tests unitaires du KPI & Semantic Agent."""

import unittest
from agents.kpi.models import KPIRequest
from agents.kpi.agent import run


class TestKPIAgent(unittest.TestCase):
    def test_kpi_calculation_on_clean_data(self):
        request = KPIRequest(
            business_goal="Test unitaire KPI",
            clean_data_path="data/clean/telco_clean.csv",
        )
        report = run(request)

        # Vérifications de statut et volumétrie sémantique
        self.assertEqual(report.status, "success")
        self.assertGreaterEqual(len(report.kpis), 8)
        self.assertGreaterEqual(len(report.dimensions), 8)
        self.assertGreaterEqual(len(report.measures), 5)

        # Vérification du Churn Rate calculé
        churn_kpi = next((k for k in report.kpis if k.kpi_id == "KPI_01_CHURN_RATE"), None)
        self.assertIsNotNone(churn_kpi)
        self.assertAlmostEqual(churn_kpi.value, 26.54, places=1)
        self.assertIn("Contract", churn_kpi.breakdowns)

        # Vérification du MRR
        mrr_kpi = next((k for k in report.kpis if k.kpi_id == "KPI_02_MRR"), None)
        self.assertIsNotNone(mrr_kpi)
        self.assertGreater(mrr_kpi.value, 300000.0)


if __name__ == "__main__":
    unittest.main()
