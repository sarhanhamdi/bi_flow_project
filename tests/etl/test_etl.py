"""Tests unitaires du Data Quality / ETL Agent."""

import unittest
import pandas as pd
from agents.etl.models import ETLRequest
from agents.etl.agent import run
from agents.profiler.models import SourceConfig, SourceType


class TestETLAgent(unittest.TestCase):
    def test_etl_execution_on_telco_csv(self):
        request = ETLRequest(
            business_goal="Test unitaire ETL",
            sources=[SourceConfig(name="telco", type=SourceType.CSV, uri="data/telco.csv")],
            output_dir="data/clean",
            enable_star_schema=True,
        )
        report = run(request)

        # Vérifications fondamentales du rapport de qualité
        self.assertEqual(report.status, "success")
        self.assertEqual(report.cleaned_rows, 7043)
        self.assertEqual(report.cleaned_cols, 33)
        self.assertEqual(report.quality_score_final, 100.0)
        self.assertGreater(len(report.transformations), 0)
        self.assertGreater(len(report.quality_checks), 0)
        self.assertEqual(len(report.star_schema_tables), 3)

        # Vérification du Star Schema généré
        table_names = [t.table_name for t in report.star_schema_tables]
        self.assertIn("dim_customers", table_names)
        self.assertIn("dim_subscriptions", table_names)
        self.assertIn("fact_churn", table_names)


if __name__ == "__main__":
    unittest.main()
