"""Outils de nettoyage et d'imputation des données pour l'ETL."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from agents.etl.models import TransformationStep


def clean_dataframe(
    frame: pd.DataFrame,
    issues: list[Any] | None = None,
) -> tuple[pd.DataFrame, list[TransformationStep]]:
    """Nettoie le DataFrame brut en appliquant des règles déterministes d'ingénierie des données."""
    df = frame.copy()
    steps: list[TransformationStep] = []

    # 1. Suppression des espaces blancs invisibles et conversion en NaN
    t0 = time.perf_counter()
    str_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
    whitespace_fixed = 0
    for col in str_cols:
        series_str = df[col].astype(str).str.strip()
        empty_mask = (series_str == "") | (series_str.str.lower() == "nan") | (series_str.str.lower() == "none")
        whitespace_fixed += int(empty_mask.sum())
        df[col] = df[col].astype(str).str.strip()
        df.loc[empty_mask, col] = np.nan

    steps.append(TransformationStep(
        step_name="strip_whitespace_and_nullify",
        description="Suppression des espaces superflus et conversion des chaînes vides en valeurs nulles.",
        columns_affected=str_cols,
        rows_affected=whitespace_fixed,
        duration_ms=(time.perf_counter() - t0) * 1000,
    ))

    # 2. Correction des types déclarés / inférés (ex: Total Charges)
    t0 = time.perf_counter()
    type_fixed_cols = []
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
        type_fixed_cols.append("Total Charges")

    # Colonnes d'identifiants ou codes postaux
    if "Zip Code" in df.columns:
        df["Zip Code"] = df["Zip Code"].astype(str)
        type_fixed_cols.append("Zip Code")

    steps.append(TransformationStep(
        step_name="cast_data_types",
        description="Conversion des types détectés (Total Charges en float64, Zip Code en string).",
        columns_affected=type_fixed_cols,
        rows_affected=len(df),
        duration_ms=(time.perf_counter() - t0) * 1000,
    ))

    # 3. Imputation métier intelligente des valeurs manquantes
    t0 = time.perf_counter()
    imputed_count = 0
    imputed_cols = []

    # Règle Total Charges : si Tenure Months == 0, le client vient d'arriver donc Total Charges = 0
    if "Total Charges" in df.columns:
        tenure_col = "Tenure Months" if "Tenure Months" in df.columns else "tenure"
        if tenure_col in df.columns:
            new_cust_mask = df["Total Charges"].isna() & (df[tenure_col] == 0)
            df.loc[new_cust_mask, "Total Charges"] = 0.0
            imputed_count += int(new_cust_mask.sum())

        # Imputation de sécurité par la médiane si d'autres valeurs manquent
        rem_missing = df["Total Charges"].isna()
        if rem_missing.any():
            median_val = float(df["Total Charges"].median())
            df.loc[rem_missing, "Total Charges"] = median_val
            imputed_count += int(rem_missing.sum())
        imputed_cols.append("Total Charges")

    # Règle Churn Reason : les clients fidèles n'ont pas de motif de départ (normal)
    if "Churn Reason" in df.columns:
        churn_val_col = "Churn Value" if "Churn Value" in df.columns else ("Churn" if "Churn" in df.columns else None)
        if churn_val_col:
            active_mask = (df[churn_val_col].isin([0, "0", "No", False])) & df["Churn Reason"].isna()
            df.loc[active_mask, "Churn Reason"] = "Not Applicable (Active Customer)"
            imputed_count += int(active_mask.sum())

            departed_mask = df["Churn Reason"].isna()
            if departed_mask.any():
                df.loc[departed_mask, "Churn Reason"] = "Unknown / Not Specified"
                imputed_count += int(departed_mask.sum())
        else:
            df["Churn Reason"] = df["Churn Reason"].fillna("Unknown")
            imputed_count += int(df["Churn Reason"].isna().sum())
        imputed_cols.append("Churn Reason")

    steps.append(TransformationStep(
        step_name="smart_imputation",
        description="Imputation métier (Total Charges=0 pour nouveaux clients, Churn Reason explicité pour clients actifs).",
        columns_affected=imputed_cols,
        rows_affected=imputed_count,
        duration_ms=(time.perf_counter() - t0) * 1000,
    ))

    # 4. Standardisation des modalités redondantes
    t0 = time.perf_counter()
    service_cols = [
        "Multiple Lines", "Online Security", "Online Backup",
        "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies"
    ]
    standardized_cols = []
    redundant_fixed = 0
    for col in service_cols:
        if col in df.columns:
            mask = df[col].isin(["No internet service", "No phone service"])
            redundant_fixed += int(mask.sum())
            df[col] = df[col].replace({
                "No internet service": "No",
                "No phone service": "No",
            })
            standardized_cols.append(col)

    steps.append(TransformationStep(
        step_name="standardize_categories",
        description="Harmonisation des catégories redondantes ('No internet service' -> 'No').",
        columns_affected=standardized_cols,
        rows_affected=redundant_fixed,
        duration_ms=(time.perf_counter() - t0) * 1000,
    ))

    # 5. Déduplication
    t0 = time.perf_counter()
    init_len = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dups_removed = init_len - len(df)
    if "CustomerID" in df.columns:
        init_key_len = len(df)
        df = df.drop_duplicates(subset=["CustomerID"], keep="first").reset_index(drop=True)
        dups_removed += (init_key_len - len(df))

    steps.append(TransformationStep(
        step_name="deduplication",
        description="Élimination des doublons stricts et des doublons sur la clé primaire CustomerID.",
        columns_affected=["CustomerID"] if "CustomerID" in df.columns else [],
        rows_affected=dups_removed,
        duration_ms=(time.perf_counter() - t0) * 1000,
    ))

    return df, steps
