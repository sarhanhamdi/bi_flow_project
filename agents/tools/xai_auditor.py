"""Outils déterministes pour l'audit ML et l'explicabilité SHAP."""

from __future__ import annotations

import re
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
    "churn_score",
    "count",
    "country",
    "state",
    "city",
    "zip_code",
    "lat_long",
    "latitude",
    "longitude",
}


def auto_detect_target(df: pd.DataFrame, requested_target: str | None = None) -> tuple[str, pd.Series]:
    """Détecte ou binarise la variable cible selon l'usage métier détecté."""
    if requested_target and requested_target in df.columns:
        s = df[requested_target]
        if s.dropna().nunique() == 2:
            return requested_target, s

    from agents.tools.semantic_mapper import detect_telecom_usecase
    usecase = detect_telecom_usecase(df)
    target_candidate = usecase.get("primary_target")
    if target_candidate and target_candidate in df.columns:
        s = df[target_candidate]
        if s.dropna().nunique() == 2:
            return target_candidate, s

    # 1. Recherche d'une colonne binaire explicite (2 valeurs)
    for c in df.columns:
        if c.lower() in EXCLUDE_COLUMNS or c.lower().endswith("_id"):
            continue
        uniq = df[c].dropna().unique()
        if len(uniq) == 2:
            return c, df[c]

    # 2. Binarisation par seuil métier (Réseau / Qualité)
    norm_map = {re.sub(r'[^a-z0-9]', '', str(c).lower()): c for c in df.columns}
    for alias in ["mosvoice", "mos", "streamingmos"]:
        if alias in norm_map:
            c = norm_map[alias]
            target_series = (pd.to_numeric(df[c], errors="coerce") < 3.5).astype(int)
            return f"degraded_{c}", target_series

    for alias in ["latencyms", "latency", "dnslatencyms", "ping"]:
        if alias in norm_map:
            c = norm_map[alias]
            num_s = pd.to_numeric(df[c], errors="coerce")
            threshold = 50.0 if num_s.median() < 50.0 else num_s.median()
            target_series = (num_s > threshold).astype(int)
            return f"high_{c}", target_series

    # 3. Binarisation Support (Statut non fermé = incident ouvert)
    for alias in ["status", "ticketstatus", "casestatus"]:
        if alias in norm_map:
            c = norm_map[alias]
            text = df[c].astype(str).str.strip().str.lower()
            target_series = (~text.isin(["closed", "solved", "resolved"])).astype(int)
            if target_series.nunique() == 2:
                return f"unresolved_{c}", target_series

    # 4. Repli statistique sur la médiane de la colonne numérique la plus variable
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c.lower() not in EXCLUDE_COLUMNS]
    if num_cols:
        best_col = max(num_cols, key=lambda c: df[c].std() if df[c].std() > 0 else 0)
        target_series = (df[best_col] > df[best_col].median()).astype(int)
        return f"above_median_{best_col}", target_series

    target_series = pd.Series([1 if i % 2 == 0 else 0 for i in range(len(df))], index=df.index)
    return "simulated_target", target_series


def prepare_data(
    df: pd.DataFrame,
    target_column: str = "churn_value",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
    """Prépare et encode les features, puis sépare en train/test."""
    df_clean = df.copy()

    actual_target_name, y_raw = auto_detect_target(df_clean, target_column)

    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.fillna(0).astype(int)
    else:
        encoder_y = OrdinalEncoder()
        y = pd.Series(encoder_y.fit_transform(y_raw.to_frame()).flatten(), index=df_clean.index, dtype=int)

    excluded = [
        col for col in df_clean.columns 
        if col.lower() in EXCLUDE_COLUMNS 
        or col == actual_target_name
        or col.lower().startswith("run_")
        or col.lower().endswith("_id")
        or col.lower() in {"customerid", "customer_id", "phone_number", "datetime", "timestamp"}
    ]
    feature_cols = [col for col in df_clean.columns if col not in excluded]

    X = df_clean[feature_cols].copy()

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = encoder.fit_transform(X[cat_cols].astype(str))

    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    # Éviter échec stratify si une classe n'a qu'un échantillon
    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
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
    y_proba = model.predict_proba(X_test)[:, 1] if len(model.classes_) == 2 else np.zeros(len(y_test))

    try:
        auc = round(float(roc_auc_score(y_test, y_proba)), 4)
    except Exception:
        auc = 0.5

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "auc_roc": auc,
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

    if delta_f1 > 0.20 and delta_acc > 0.10:
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
    y_test: pd.Series | None = None,
    top_n: int = 5,
    n_examples: int = 3,
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Calcule les explications SHAP globales et locales."""
    explainer = shap.TreeExplainer(model)
    feature_cols = list(X_test.columns)
    sample_test = X_test.head(min(200, len(X_test)))
    shap_values = explainer.shap_values(sample_test)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif len(shap_values.shape) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    mean_abs = np.abs(sv).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:top_n]

    top_features = [
        {
            "feature": feature_cols[idx],
            "mean_abs_shap": round(float(mean_abs[idx]), 4),
            "rank": rank + 1,
        }
        for rank, idx in enumerate(top_indices)
    ]

    examples = []
    for i in range(min(n_examples, len(sample_test))):
        factors = [
            {
                "feature": feature_cols[j],
                "shap_value": round(float(sv[i, j]), 4),
                "feature_value": float(sample_test.iloc[i, j]) if pd.api.types.is_numeric_dtype(type(sample_test.iloc[i, j])) else str(sample_test.iloc[i, j]),
            }
            for j in np.argsort(np.abs(sv[i]))[::-1][:3]
        ]
        pred_prob = model.predict_proba(sample_test.iloc[[i]])[0, 1] if len(model.classes_) == 2 else 0.5
        true_val = int(y_test.iloc[i]) if (y_test is not None and i < len(y_test)) else 0
        pred_val = int(pred_prob >= 0.5)
        examples.append({
            "client_index": i,
            "true_label": true_val,
            "predicted_label": pred_val,
            "predicted_proba": round(float(pred_prob), 4),
            "top_features": factors,
        })

    return top_features, examples
