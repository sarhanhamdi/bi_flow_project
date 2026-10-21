"""Outils statistiques déterministes pour le BI Analyst Agent.

Toutes les fonctions sont pures (pas de LLM, pas de LangGraph).
Elles peuvent être importées et testées indépendamment de l'agent.

Conventions :
- Les p-values sont toujours à deux queues.
- La correction BH est appliquée globalement sur tous les z-tests du rapport.
- Les tailles d'effet suivent les seuils de Cohen / Cramér : faible < 0.1, moyen < 0.3, fort ≥ 0.3.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

# ── Constantes ──────────────────────────────────────────────────────────────

# Noms normalisés (sans caractères spéciaux) des colonnes de fuite de cible.
_CHURN_LEAKAGE_NORM: frozenset[str] = frozenset({
    "churnlabel", "churnvalue", "churnscore", "churnreason",
    "churn", "ischurn", "churned", "churnflag",
})

# Motifs géographiques à exclure (colonnes quasi-PII ou non analytiques).
_GEO_RE = re.compile(
    r"\b(city|zip_code|zipcode|lat_long|latlng|latitude|longitude|"
    r"postal|address|street|geo|location)\b",
    re.I,
)

# Tranches d'ancienneté alignées sur la règle `early_onboarding` du catalogue.
_TENURE_BINS: list[float] = [0.0, 12.0, 24.0, 48.0, float("inf")]
_TENURE_LABELS: list[str] = ["0-12 mois", "12-24 mois", "24-48 mois", "48+ mois"]

# Seuils de taille d'effet (Cramér's V et |Cliff's delta| utilisent la même grille).
_EFFECT_WEAK = 0.1
_EFFECT_MEDIUM = 0.3


# ── Normalisation ────────────────────────────────────────────────────────────

def _norm(name: Any) -> str:
    """Normalise un nom de colonne : casse et caractères spéciaux ignorés."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# ── Tests statistiques ────────────────────────────────────────────────────────

def proportion_ztest(n1: int, k1: int, n2: int, k2: int) -> tuple[float, float]:
    """Z-test de deux proportions : groupe 1 (n1, k1 succès) vs groupe 2 (n2, k2 succès).

    Typiquement : segment (n1, k1 churnés) vs reste (n2, k2 churnés).

    Returns
    -------
    (z_stat, p_value)
        p_value : deux queues ; retourne (0.0, 1.0) si le test n'est pas applicable.
    """
    from scipy import stats as sp

    if n1 < 1 or n2 < 1 or k1 < 0 or k2 < 0:
        return 0.0, 1.0
    p_pool = (k1 + k2) / (n1 + n2)
    if p_pool <= 0 or p_pool >= 1:
        return 0.0, 1.0
    denom = (p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2)) ** 0.5
    if denom == 0:
        return 0.0, 1.0
    z = (k1 / n1 - k2 / n2) / denom
    p = float(2.0 * (1.0 - sp.norm.cdf(abs(z))))
    return float(z), p


def chi2_independence(obs: np.ndarray) -> tuple[float, float, float]:
    """Test du χ² d'indépendance sur table de contingence 2D.

    Returns
    -------
    (chi2, p_value, cramers_v)
        Utilisé comme indicateur d'ensemble (hors correction BH).
    """
    from scipy.stats import chi2_contingency

    obs = np.asarray(obs, dtype=float)
    if obs.shape[0] < 2 or obs.shape[1] < 2:
        return 0.0, 1.0, 0.0
    chi2, p, _, _ = chi2_contingency(obs, correction=False)
    n = float(obs.sum())
    k = float(min(obs.shape))
    cramers_v = float(np.sqrt(chi2 / (n * (k - 1)))) if n > 0 and k > 1 else 0.0
    return float(chi2), float(p), min(cramers_v, 1.0)


def mann_whitney_test(
    a: np.ndarray | pd.Series,
    b: np.ndarray | pd.Series,
) -> tuple[float, float, float]:
    """Test de Mann-Whitney U entre deux groupes (partis vs restés).

    Parameters
    ----------
    a : valeurs du groupe des clients churnés (partis)
    b : valeurs du groupe des clients retenus (restés)

    Returns
    -------
    (U, p_value, cliffs_delta)
        p_value : deux queues.
        Cliff's delta = 2U/(|a|·|b|) - 1 ∈ [-1, 1].
    """
    from scipy.stats import mannwhitneyu

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0, 0.0
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    delta = float(2.0 * U / (len(a) * len(b)) - 1.0)
    return float(U), float(p), float(np.clip(delta, -1.0, 1.0))


def benjamini_hochberg(p_values: list[float]) -> np.ndarray:
    """Correction de Benjamini-Hochberg sur une liste de p-values brutes.

    Utilise ``scipy.stats.false_discovery_control`` si disponible (scipy ≥ 1.11),
    sinon implémentation manuelle équivalente.

    Returns
    -------
    np.ndarray de même longueur avec les p-values ajustées ∈ [0, 1].
    """
    if not p_values:
        return np.array([], dtype=float)
    arr = np.asarray(p_values, dtype=float)
    try:
        from scipy.stats import false_discovery_control  # scipy ≥ 1.11
        return np.clip(false_discovery_control(arr, method="bh"), 0.0, 1.0)
    except (ImportError, AttributeError, TypeError):
        pass
    # Implémentation manuelle BH (fallback)
    n = len(arr)
    order = np.argsort(arr)
    adj = arr.copy()
    for rank, idx in enumerate(order, start=1):
        adj[idx] = min(float(arr[idx]) * n / rank, 1.0)
    # Monotonicité de droite à gauche
    for i in range(n - 2, -1, -1):
        adj[order[i]] = min(adj[order[i]], adj[order[i + 1]])
    return np.clip(adj, 0.0, 1.0)


# ── Tailles d'effet ──────────────────────────────────────────────────────────

def effect_label(v: float) -> str:
    """Libellé de taille d'effet (Cramér's V ou |Cliff's delta|).

    Seuils : faible < 0.1, moyen < 0.3, fort ≥ 0.3.
    """
    av = abs(v)
    if av < _EFFECT_WEAK:
        return "faible"
    if av < _EFFECT_MEDIUM:
        return "moyen"
    return "fort"


# ── Reconstruction des effectifs ─────────────────────────────────────────────

def reconstruct_counts(rate_pct: float, rows: int) -> int:
    """Reconstruit l'effectif churné depuis le taux (%) et le nombre de lignes.

    Exemple : reconstruct_counts(42.71, 3875) → 1655.
    """
    return int(round(rate_pct / 100.0 * rows))


# ── Détection des colonnes à exclure ─────────────────────────────────────────

def churn_columns(etl_columns: list[dict[str, Any]]) -> set[str]:
    """Colonnes à exclure en tant que fuite de cible.

    Sont exclues les colonnes dont le nom normalisé appartient à la liste
    des alias de churn connus (churn_label, churn_value, churn_score, etc.).

    Parameters
    ----------
    etl_columns : liste des dicts ``ColumnSummary`` du rapport ETL.
    """
    result: set[str] = set()
    for col in etl_columns:
        name = col.get("name", "")
        if _norm(name) in _CHURN_LEAKAGE_NORM:
            result.add(name)
    return result


def pii_columns(etl_columns: list[dict[str, Any]]) -> set[str]:
    """Colonnes à exclure en tant que PII (``is_pii=True`` dans le rapport ETL)."""
    return {col["name"] for col in etl_columns if col.get("is_pii")}


def geo_columns(col_names: list[str]) -> set[str]:
    """Colonnes à exclure en tant que géographiques (par motif de nom)."""
    return {c for c in col_names if _GEO_RE.search(c)}


def constant_columns(df: pd.DataFrame) -> set[str]:
    """Colonnes constantes (≤ 1 valeur unique), sans utilité analytique."""
    return {str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1}


# ── Garde-fou LLM ────────────────────────────────────────────────────────────

def _numbers_grounded(summary: str, source_json: str) -> bool:
    """Rejette une synthèse LLM qui contient un nombre absent des résultats calculés.

    Tout nombre ≥ 2 chiffres présent dans ``summary`` mais absent de ``source_json``
    déclenche le rejet : l'agent utilise alors la synthèse déterministe.
    """
    allowed = set(re.findall(r"\d+(?:\.\d+)?", source_json))
    for number in re.findall(r"\d+(?:[.,]\d+)?", summary):
        # Ignorer les nombres à 1 chiffre (années, petits ordinaux, etc.)
        if len(number.replace(",", ".").replace(".", "")) < 2:
            continue
        if number.replace(",", ".") not in allowed:
            return False
    return True


# ── Tranches d'ancienneté ────────────────────────────────────────────────────

def tenure_bins(series: pd.Series) -> pd.Series:
    """Découpe une série tenure_months en tranches nommées.

    Les libellés sont alignés sur la règle ``early_onboarding`` du catalogue
    d'actions : ``"0-12 mois"``, ``"12-24 mois"``, ``"24-48 mois"``, ``"48+ mois"``.

    Parameters
    ----------
    series : pd.Series de valeurs numériques (mois d'ancienneté).

    Returns
    -------
    pd.Series de même index avec les libellés de tranches (ou pd.NA si NaN).
    """
    numeric = pd.to_numeric(series, errors="coerce")
    binned = pd.cut(
        numeric,
        bins=_TENURE_BINS,
        labels=_TENURE_LABELS,
        right=False,
    )
    return binned.astype(object).where(numeric.notna(), other=pd.NA)
