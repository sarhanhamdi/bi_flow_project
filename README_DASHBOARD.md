# Dashboard Generator Agent

L'**Agent Dashboard** est la dernière étape de la chaîne BI Flow. Il agrège les résultats des agents précédents (**KPI Agent**, **BI Analyst**, **ETL**) pour construire la spécification complète et déterministe d'un tableau de bord d'aide à la décision.

```text
Profiler -> ETL -> Modèle en étoile -> KPI Agent -> BI Analyst Agent -> Dashboard Agent -> Rapport & Visuels
```

---

## 1. Principes directeurs

1. **Construction 100% déterministe** : Les visuels (cartes KPI, histogrammes, heatmaps, profils) sont construits mathématiquement via [`agents/tools/dashboard.py`](agents/tools/dashboard.py). Aucun graphique n'est inventé par un LLM.
2. **Garde-fous de conformité et gouvernance** :
   * **Fuite de cible (*Target Leakage*)** : Les métriques prédictives ou dérivées (comme `avg_churn_score`) sont automatiquement exclues des cartes de pilotage.
   * **Confidentialité & RGPD** : Aucun identifiant individuel (`CustomerID`) n'est conservé. Les micro-segments de moins de 30 personnes sont automatiquement masqués.
   * **Repli géographique automatique** : Si le dataset ne contient qu'un seul État (ex. Californie), l'agent bascule automatiquement sur l'échelle des villes (`geo_churn_by_city`).
3. **Contrôle strict de la synthèse LLM (`_numbers_grounded`)** : Le LLM ne rédige que la narration introductive. Tout résumé contenant un chiffre absent des données réelles est immédiatement rejeté au profit d'une synthèse déterministe.

---

## 2. Exécution

### Via le pipeline complet (recommandé)
```bash
# Exécute Profiler -> ETL -> KPI -> BI Analyst -> Dashboard
python run_pipeline.py data/telco.csv --preset telco

# Sans le dashboard si besoin
python run_pipeline.py data/telco.csv --preset telco --no-dashboard
```

### En ligne de commande dédiée (CLI)
```bash
python -m agents.dashboard \
  --analyst-report output/telco/analyst/analyst_report.json \
  --kpi-report output/telco/kpi/kpi_report.json \
  --etl-report output/telco/etl_report.json \
  --data output/telco/telco_clean.csv \
  --no-llm
```

### En Python (API)
```python
import json
import pandas as pd
from agents.dashboard import DashboardOptions, DashboardRequest, run

request = DashboardRequest(
    analyst_report=json.load(open("output/telco/analyst/analyst_report.json", encoding="utf-8")),
    kpi_report=json.load(open("output/telco/kpi/kpi_report.json", encoding="utf-8")),
    etl_report=json.load(open("output/telco/etl_report.json", encoding="utf-8")),
    options=DashboardOptions(output_dir="output/telco/dashboard", use_llm=False)
)
frames = {"table": pd.read_csv("output/telco/telco_clean.csv")}
report = run(request, frames=frames)
```

---

## 3. Entrées et Sorties

### Entrées
* `analyst_report.json` : Insights significatifs (tests $z$ / BH) et recommandations priorisées.
* `kpi_report.json` : Indicateurs calculés avec ventilations dimensionnelles (`breakdowns`).
* `etl_report.json` & table nettoyée (`telco_clean.csv`) : Pour les ventilations croisées (heatmaps, profils, géographie).

### Sorties
Fichier généré dans `output/<nom_dataset>/dashboard/` :
* `dashboard_report.json` : Spécification complète prête pour un rendu front-end (Streamlit, Dash, Plotly, React).

---

## 4. Les 6 sections du tableau de bord

| Section | Description | Types de visuels |
|---|---|---|
| **1. Vue d'ensemble** | Chiffres clés prioritaires du churn et des revenus | Cartes KPI (`kpi_card`) |
| **2. Segments à risque** | Facteurs majeurs de churn identifiés par l'analyste | Tableau d'insights, Heatmaps bi-variées, Bar charts triés |
| **3. Évolution et ancienneté** | Dynamique du churn selon la maturité client | Courbe de survie / ancienneté (`line`) |
| **4. Répartition géographique** | Cartographie locale du churn (État ou Ville) | Carte choroplèthe / bulles (`geo_map`) |
| **5. Profils churnés vs retenus** | Comparaison des caractéristiques (charges, usage) | Profils comparatifs (`profile_comparison`) |
| **6. Recommandations & Réserves** | Plan d'action priorisé et transparence méthodologique | Tableau d'actions, Bloc d'avertissements (`caveats`) |

---

## 5. Tests

Pour exécuter la suite de tests unitaires du Dashboard :
```bash
pytest tests/dashboard -v
```

