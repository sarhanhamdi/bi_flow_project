# SynapseBI — Autonomous Multi-Agent Intelligence Platform

Plateforme décisionnelle d'entreprise autonome pilotée par un écosystème multi-agents orchestré avec **LangGraph**.  
Elle transforme les données brutes tabulaires (CSV) en schémas décisionnels certifiés, métriques de performance métier (KPIs), analyses exploratoires et modèles d'IA explicables (**XAI** / **SHAP**) sans aucune boîte noire.

---

## 🏗️ Architecture Multi-Agents (7 Briques)

Le pipeline complet s'exécute de façon déterministe et auditable :

```
[ Données Brutes (CSV) ]
           │
           ▼
1. Data Profiler Agent      ──> Audit des types, détection PII (RGPD) et valeurs manquantes
           │
           ▼
2. ETL Agent (Kimball)      ──> Nettoyage déterministe et modélisation en étoile (Fact + Dims)
           │
           ▼
3. KPI & Semantic Agent     ──> Calcul certifié des métriques métier (Churn, ARPU, CLTV...)
           │
           ▼
4. BI Analyst Agent         ──> Détection des segments à risque et recommandations d'actions
           │
           ▼
5. Dashboard Generator      ──> Spécification visuelle complète du cockpit exécutif (23 visuels)
           │
           ▼
6. BI Auditor & XAI Agent   ──> Entraînement RandomForest, audit anti-leakage et explicabilité SHAP
           │
           ▼
7. Orchestrator Global      ──> Graphe LangGraph unifié supervisant et chronométrant l'ensemble
```

---

## 🚀 Installation & Prérequis

### Prérequis
* **Python 3.10+** (testé et validé sous Python 3.11)
* **Git**

### 1. Cloner le projet et se positionner sur la branche `WEB`
```bash
git clone https://github.com/sarhanhamdi/bi_flow_project.git
cd bi_flow_project
git checkout WEB
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
pip install shap fastapi uvicorn aiofiles jinja2 python-multipart
```

---

## 💻 Utilisation

### Mode 1 : Lancer l'Interface Web SynapseBI (Recommandé)

L'application web est servie directement par **FastAPI** (HTML5 / Tailwind CSS / Chart.js) sans configuration complexe :

```bash
python -m uvicorn web.app:app --host 127.0.0.1 --port 8000 --reload
```

Ouvrez ensuite votre navigateur à l'adresse : **`http://127.0.0.1:8000`**

#### Fonctionnalités de l'interface :
* **🏠 Welcome :** Synthèse exécutive et indicateurs d'impact économique.
* **📁 Dataset Upload :** Téléversement dynamique par glisser-déposer de n'importe quel fichier `.csv`.
* **⚡ Pipeline Execution :** Suivi live de l'orchestration des 6 agents avec statut, chronomètre et logs.
* **📊 Executive Dashboard :** Cockpit décisionnel interactif (KPIs, tendance du churn, graphiques par contrat).
* **🛡️ Data Governance :** Traçabilité des règles de transformation et de conformité appliquées par l'ETL.
* **🔍 Audit and XAI :** Métriques d'évaluation (Accuracy 78.8%, Recall 67.1%, F1 62.7%, AUC 0.841), benchmark baseline (+62.7% gain net), graphique SHAP global et fiches explicatives par client.

---

### Mode 2 : Exécution en Ligne de Commande (CLI)

Pour lancer le pipeline complet sur le jeu de données de référence :

```bash
python run_pipeline.py data/telco.csv --preset telco
```

Pour exécuter directement via l'Orchestrateur LangGraph :

```bash
python -c "from agents.orchestrator import run, OrchestratorRequest; res = run(OrchestratorRequest(source_path='data/telco.csv')); print('STATUS:', res.status); print('DURÉE TOTALE:', res.total_duration_seconds, 's')"
```

---

## 🧪 Tests Automatisés

L'intégralité du pipeline et de chaque agent est couverte par une suite de tests unitaires et d'intégration :

```bash
pytest -v
```
*(123/123 tests passés avec succès)*

---

## 📁 Structure des Artefacts de Sortie

Tous les résultats sont automatiquement enregistrés dans le dossier `output/<dataset_name>/` :
* `profiler_output.json` : Rapport d'audit qualité initial.
* `etl_report.json` & `<name>_clean.csv` : Données nettoyées et rapport de transformation.
* `model/<name>/` : Schéma en étoile (`fact_customer.csv` et dimensions).
* `kpi/kpi_report.json` : Indicateurs de performance calculés.
* `analyst/analyst_report.json` : Insights et recommandations.
* `dashboard/dashboard_report.json` : Spécifications visuelles du tableau de bord.
* `xai/xai_report.json` : Métriques ML, comparaison baseline et valeurs SHAP.
* `orchestrator_summary.json` : Synthèse globale d'exécution consolidée.
