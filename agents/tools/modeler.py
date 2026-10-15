"""Outils de modélisation dimensionnelle (Star Schema) et persistance des données propres."""

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

from agents.etl.models import StarSchemaTable


def build_star_schema(
    df: pd.DataFrame,
    output_dir: str = "data/clean",
) -> list[StarSchemaTable]:
    """Découpe le DataFrame plat en un schéma en étoile (Dimensions + Fait) et sauvegarde en CSV/Parquet."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tables: list[StarSchemaTable] = []

    # 1. Dimension Client : Dim_Customers
    cust_cols = [
        c for c in [
            "CustomerID", "Gender", "Senior Citizen", "Partner", "Dependents",
            "Country", "State", "City", "Zip Code", "Latitude", "Longitude"
        ] if c in df.columns
    ]
    if cust_cols:
        dim_cust = df[cust_cols].drop_duplicates(subset=["CustomerID"]).copy()
        csv_file = str(out_path / "dim_customers.csv")
        parquet_file = str(out_path / "dim_customers.parquet")
        dim_cust.to_csv(csv_file, index=False)
        dim_cust.to_parquet(parquet_file, index=False)

        tables.append(StarSchemaTable(
            table_name="dim_customers",
            table_type="dimension",
            row_count=len(dim_cust),
            column_count=len(cust_cols),
            primary_key=["CustomerID"],
            columns=cust_cols,
            file_path_csv=csv_file,
            file_path_parquet=parquet_file,
        ))

    # 2. Dimension Abonnements & Services : Dim_Subscriptions
    sub_cols = [
        c for c in [
            "CustomerID", "Contract", "Paperless Billing", "Payment Method",
            "Phone Service", "Multiple Lines", "Internet Service", "Online Security",
            "Online Backup", "Device Protection", "Tech Support",
            "Streaming TV", "Streaming Movies"
        ] if c in df.columns
    ]
    if sub_cols:
        dim_sub = df[sub_cols].drop_duplicates(subset=["CustomerID"]).copy()
        csv_file = str(out_path / "dim_subscriptions.csv")
        parquet_file = str(out_path / "dim_subscriptions.parquet")
        dim_sub.to_csv(csv_file, index=False)
        dim_sub.to_parquet(parquet_file, index=False)

        tables.append(StarSchemaTable(
            table_name="dim_subscriptions",
            table_type="dimension",
            row_count=len(dim_sub),
            column_count=len(sub_cols),
            primary_key=["CustomerID"],
            foreign_keys={"CustomerID": "dim_customers.CustomerID"},
            columns=sub_cols,
            file_path_csv=csv_file,
            file_path_parquet=parquet_file,
        ))

    # 3. Table de Faits : Fact_Churn
    fact_cols = [
        c for c in [
            "CustomerID", "Tenure Months", "Monthly Charges", "Total Charges",
            "Churn Value", "Churn Score", "CLTV", "Churn Reason"
        ] if c in df.columns
    ]
    if fact_cols:
        fact_churn = df[fact_cols].copy()
        csv_file = str(out_path / "fact_churn.csv")
        parquet_file = str(out_path / "fact_churn.parquet")
        fact_churn.to_csv(csv_file, index=False)
        fact_churn.to_parquet(parquet_file, index=False)

        tables.append(StarSchemaTable(
            table_name="fact_churn",
            table_type="fact",
            row_count=len(fact_churn),
            column_count=len(fact_cols),
            primary_key=["CustomerID"],
            foreign_keys={
                "CustomerID": "dim_customers.CustomerID",
            },
            columns=fact_cols,
            file_path_csv=csv_file,
            file_path_parquet=parquet_file,
        ))

    # Sauvegarde également de la table unifiée propre complète
    unified_csv = str(out_path / "telco_clean.csv")
    unified_parquet = str(out_path / "telco_clean.parquet")
    df.to_csv(unified_csv, index=False)
    df.to_parquet(unified_parquet, index=False)

    return tables
