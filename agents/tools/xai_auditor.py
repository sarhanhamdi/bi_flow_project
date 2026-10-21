"""Outils déterministes pour l'audit ML et l'explicabilité SHAP."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

EXCLUDE_COLUMNS = {
    "customer_id",
    "churn_label",
    "churn_reason",
    "churn_score",  # Target leakage risk
    "count",
    "country",
    "state",
    "city",
    "zip_code",
    "lat_long",
    "latitude",
    "longitude",
}


def prepare_data(
    df: pd.DataFrame,
    target_column: str = "churn_value",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
    """Prépare et encode les features, puis sépare en train/test."""
    df_clean = df.copy()

    if target_column not in df_clean.columns:
        raise ValueError(f"Colonne cible '{target_column}' introuvable.")

    y = df_clean[target_column].astype(int)

    excluded = [col for col in df_clean.columns if col in EXCLUDE_COLUMNS or col == target_column]
    feature_cols = [col for col in df_clean.columns if col not in excluded]

    X = df_clean[feature_cols].copy()

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = encoder.fit_transform(X[cat_cols].astype(str))

    # Gestion des valeurs manquantes éventuelles
    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_cols, excluded


def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    """Entraîne un RandomForest et calcule les métriques standards."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "auc_roc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "positive_rate_train": round(float(y_train.mean()), 4),
        "positive_rate_test": round(float(y_test.mean()), 4),
    }

    return model, metrics


def compute_baseline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    rf_accuracy: float,
    rf_f1: float,
) -> dict[str, Any]:
    """Compare les performances à un DummyClassifier (baseline)."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_dummy = dummy.predict(X_test)

    base_acc = round(float(accuracy_score(y_test, y_dummy)), 4)
    base_f1 = round(float(f1_score(y_test, y_dummy, zero_division=0)), 4)

    delta_acc = round(rf_accuracy - base_acc, 4)
    delta_f1 = round(rf_f1 - base_f1, 4)

    if delta_f1 > 0.15 and delta_acc > 0.05:
        verdict = "significant_gain"
    elif delta_f1 > 0:
        verdict = "marginal_gain"
    else:
        verdict = "no_gain"

    return {
        "baseline_accuracy": base_acc,
        "baseline_f1": base_f1,
        "delta_accuracy": delta_acc,
        "delta_f1": delta_f1,
        "verdict": verdict,
    }


def compute_shap_explanations(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    top_n: int = 10,
    n_examples: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Calcule les importances globales et les explications locales via TreeExplainer."""
    explainer = shap.TreeExplainer(model)
    # SHAP TreeExplainer sur RandomForest peut renvoyer list ou array (n_samples, n_features, n_classes)
    shap_vals = explainer.shap_values(X_test)

    if isinstance(shap_vals, list):
        # Pour binary classification, classe 1 (churn)
        vals = shap_vals[1]
    elif len(shap_vals.shape) == 3:
        vals = shap_vals[:, :, 1]
    else:
        vals = shap_vals

    mean_abs = np.abs(vals).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:top_n]

    top_features = []
    for rank, idx in enumerate(top_indices, start=1):
        top_features.append({
            "feature": str(X_test.columns[idx]),
            "mean_abs_shap": round(float(mean_abs[idx]), 4),
            "rank": rank,
        })

    # Exemples locaux (ex: premiers clients du test set)
    examples = []
    y_preds = model.predict(X_test)
    y_probas = model.predict_proba(X_test)[:, 1]

    for i in range(min(n_examples, len(X_test))):
        client_vals = vals[i]
        top_client_indices = np.argsort(np.abs(client_vals))[::-1][:5]
        local_top = []
        for idx in top_client_indices:
            local_top.append({
                "feature": str(X_test.columns[idx]),
                "shap_value": round(float(client_vals[idx]), 4),
                "feature_value": float(X_test.iloc[i, idx]),
            })

        examples.append({
            "client_index": int(i),
            "true_label": int(y_test.iloc[i]),
            "predicted_label": int(y_preds[i]),
            "predicted_proba": round(float(y_probas[i]), 4),
            "top_features": local_top,
        })

    return top_features, examples
