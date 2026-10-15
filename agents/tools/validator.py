"""Outils de validation de la qualité des données (Great Expectations style)."""

from __future__ import annotations

import pandas as pd

from agents.etl.models import QualityCheckResult


def validate_dataframe(df: pd.DataFrame) -> tuple[list[QualityCheckResult], float]:
    """Exécute la suite de règles de gouvernance et calcule un score de qualité (0 à 100%)."""
    checks: list[QualityCheckResult] = []
    total_rows = len(df)

    if total_rows == 0:
        return checks, 0.0

    # 1. Règle Unicité de la clé primaire
    if "CustomerID" in df.columns:
        dups = int(df["CustomerID"].duplicated().sum())
        passed = dups == 0
        checks.append(QualityCheckResult(
            check_id="GE_01_PRIMARY_KEY_UNIQUE",
            rule_name="expect_column_values_to_be_unique",
            column="CustomerID",
            status="passed" if passed else "failed",
            details="La clé primaire CustomerID doit être unique et sans doublons.",
            passed_rows=total_rows - dups,
            failed_rows=dups,
        ))

    # 2. Règle Non-nullité sur colonnes critiques
    critical_cols = ["CustomerID", "Contract", "Monthly Charges", "Total Charges"]
    for col in critical_cols:
        if col in df.columns:
            nulls = int(df[col].isna().sum())
            passed = nulls == 0
            checks.append(QualityCheckResult(
                check_id=f"GE_02_NOT_NULL_{col.upper().replace(' ', '_')}",
                rule_name="expect_column_values_to_not_be_null",
                column=col,
                status="passed" if passed else "failed",
                details=f"La colonne {col} ne doit contenir aucune valeur nulle.",
                passed_rows=total_rows - nulls,
                failed_rows=nulls,
            ))

    # 3. Règle Domaine de valeurs : Monthly Charges >= 0
    if "Monthly Charges" in df.columns:
        num_monthly = pd.to_numeric(df["Monthly Charges"], errors="coerce")
        invalid = int((num_monthly.dropna() < 0).sum()) + int(num_monthly.isna().sum())
        passed = invalid == 0
        checks.append(QualityCheckResult(
            check_id="GE_03_MONTHLY_CHARGES_POSITIVE",
            rule_name="expect_column_values_to_be_between",
            column="Monthly Charges",
            status="passed" if passed else "failed",
            details="Les montants de facturation mensuelle doivent être strictement positifs ou nuls.",
            passed_rows=total_rows - invalid,
            failed_rows=invalid,
        ))

    # 4. Règle Domaine de valeurs : Total Charges >= 0
    if "Total Charges" in df.columns:
        num_total = pd.to_numeric(df["Total Charges"], errors="coerce")
        invalid = int((num_total.dropna() < 0).sum()) + int(num_total.isna().sum())
        passed = invalid == 0
        checks.append(QualityCheckResult(
            check_id="GE_04_TOTAL_CHARGES_POSITIVE",
            rule_name="expect_column_values_to_be_between",
            column="Total Charges",
            status="passed" if passed else "failed",
            details="Le montant total cumulé Total Charges doit être positif ou nul.",
            passed_rows=total_rows - invalid,
            failed_rows=invalid,
        ))

    # 5. Règle Domaine de valeurs : Tenure Months >= 0
    tenure_col = "Tenure Months" if "Tenure Months" in df.columns else ("tenure" if "tenure" in df.columns else None)
    if tenure_col and tenure_col in df.columns:
        invalid = int((df[tenure_col] < 0).sum())
        passed = invalid == 0
        checks.append(QualityCheckResult(
            check_id="GE_05_TENURE_POSITIVE",
            rule_name="expect_column_values_to_be_between",
            column=tenure_col,
            status="passed" if passed else "failed",
            details="L'ancienneté (tenure) doit être un entier positif ou nul.",
            passed_rows=total_rows - invalid,
            failed_rows=invalid,
        ))

    # 6. Règle Ensembles autorisés : Churn Value in [0, 1]
    churn_col = "Churn Value" if "Churn Value" in df.columns else ("Churn" if "Churn" in df.columns else None)
    if churn_col and churn_col in df.columns:
        invalid = int((~df[churn_col].isin([0, 1, "0", "1", "Yes", "No", True, False])).sum())
        passed = invalid == 0
        checks.append(QualityCheckResult(
            check_id="GE_06_CHURN_VALUE_IN_SET",
            rule_name="expect_column_values_to_be_in_set",
            column=churn_col,
            status="passed" if passed else "failed",
            details="La variable cible Churn Value doit obligatoirement valoir 0 ou 1.",
            passed_rows=total_rows - invalid,
            failed_rows=invalid,
        ))

    # Calcul du score global de qualité (pourcentage de règles validées avec succès)
    if not checks:
        return checks, 100.0

    passed_checks = sum(1 for c in checks if c.status == "passed")
    quality_score = round((passed_checks / len(checks)) * 100.0, 2)

    return checks, quality_score
