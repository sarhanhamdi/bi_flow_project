"""Prompts LLM du BI Auditor / XAI Agent."""

SYSTEM_PROMPT = """Tu es un expert en Data Science et en explicabilité des modèles IA (XAI).
Tu analyses les résultats d'un audit ML sur des données télécom pour prédire le churn client.
Tes conclusions doivent être claires, chiffrées et orientées décision métier.
Réponds en français. Sois concis (max 250 mots)."""

SUMMARY_PROMPT = """Voici les résultats de l'audit ML :

Métriques du modèle RandomForest :
- Accuracy : {accuracy:.1%}
- Précision : {precision:.1%}
- Recall : {recall:.1%}
- F1 : {f1:.1%}
- AUC-ROC : {auc_roc:.3f}

Comparaison baseline :
- Gain d'accuracy vs DummyClassifier : +{delta_accuracy:.1%}
- Gain de F1 vs DummyClassifier : +{delta_f1:.1%}
- Verdict : {verdict}

Top 5 features SHAP (facteurs de churn) :
{top_features}

Constats d'audit ({n_findings} au total) :
{findings_summary}

Rédigez un résumé exécutif de l'audit en 3 paragraphes :
1. Performance du modèle et fiabilité
2. Facteurs explicatifs du churn (SHAP)
3. Recommandations prioritaires
"""
