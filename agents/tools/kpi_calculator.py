"""Calcul déterministe des indicateurs de performance clés (KPIs) télécoms."""

from __future__ import annotations

import pandas as pd

from agents.kpi.models import KPIResult


def calculate_telecom_kpis(df: pd.DataFrame) -> list[KPIResult]:
    """Calcule l'ensemble des métriques métier télécoms et leurs ventilations par dimension."""
    kpis: list[KPIResult] = []
    total_customers = len(df)
    if total_customers == 0:
        return kpis

    # Identification des clients churn vs actifs
    churn_mask = df["Churn Value"].isin([1, "1", "Yes", True])
    active_mask = ~churn_mask

    churned_count = int(churn_mask.sum())
    active_count = int(active_mask.sum())

    # 1. KPI : Taux de Churn (Churn Rate)
    churn_rate = (churned_count / total_customers) * 100.0
    status = "critical" if churn_rate >= 20.0 else ("warning" if churn_rate >= 10.0 else "healthy")

    # Ventilations par dimensions clés
    breakdowns_churn: dict[str, dict[str, float]] = {}
    for dim in ["Contract", "Internet Service", "Tech Support", "Payment Method"]:
        if dim in df.columns:
            group = df.groupby(dim)["Churn Value"].apply(
                lambda s: round(float((s.isin([1, "1", "Yes", True]).sum() / len(s)) * 100.0), 2)
            ).to_dict()
            breakdowns_churn[dim] = group

    kpis.append(KPIResult(
        kpi_id="KPI_01_CHURN_RATE",
        name="Taux d'Attrition Client (Churn Rate)",
        value=round(churn_rate, 2),
        formatted_value=f"{churn_rate:.2f} %",
        unit="%",
        category="customer_retention",
        formula="(Clients Résiliés / Total Clients) * 100",
        target_direction="minimize",
        status_alert=status,
        commentary=f"{churned_count} clients ont résilié sur un parc total de {total_customers}.",
        breakdowns=breakdowns_churn,
    ))

    # 2. KPI : Revenu Récurrent Mensuel (MRR - Monthly Recurring Revenue)
    monthly_col = "Monthly Charges" if "Monthly Charges" in df.columns else None
    if monthly_col:
        mrr = float(df.loc[active_mask, monthly_col].sum())
        kpis.append(KPIResult(
            kpi_id="KPI_02_MRR",
            name="Revenu Récurrent Mensuel (MRR)",
            value=round(mrr, 2),
            formatted_value=f"{mrr:,.2f} $",
            unit="$",
            category="financial",
            formula="SUM(Monthly Charges pour les clients actifs)",
            target_direction="maximize",
            status_alert="healthy",
            commentary="Revenu mensuel récurrent généré par les clients sous contrat actif.",
            breakdowns={},
        ))

        # 3. KPI : Revenu Récurrent Annuel (ARR - Annual Recurring Revenue)
        arr = mrr * 12.0
        kpis.append(KPIResult(
            kpi_id="KPI_03_ARR",
            name="Revenu Récurrent Annuelisé (ARR)",
            value=round(arr, 2),
            formatted_value=f"{arr:,.2f} $",
            unit="$",
            category="financial",
            formula="MRR * 12",
            target_direction="maximize",
            status_alert="healthy",
            commentary="Projection annuelle du chiffre d'affaires récurrent sur la base du parc actif actuel.",
            breakdowns={},
        ))

        # 4. KPI : Revenu Moyen Par Utilisateur (ARPU - Average Revenue Per User)
        arpu = float(df[monthly_col].mean())
        breakdowns_arpu: dict[str, dict[str, float]] = {}
        for dim in ["Contract", "Internet Service"]:
            if dim in df.columns:
                group = df.groupby(dim)[monthly_col].mean().round(2).to_dict()
                breakdowns_arpu[dim] = group

        kpis.append(KPIResult(
            kpi_id="KPI_04_ARPU",
            name="Revenu Moyen Par Client (ARPU)",
            value=round(arpu, 2),
            formatted_value=f"{arpu:.2f} $ / mois",
            unit="$/mois",
            category="financial",
            formula="Total Monthly Charges / Total Clients",
            target_direction="maximize",
            status_alert="healthy",
            commentary="Montant moyen facturé par client et par mois.",
            breakdowns=breakdowns_arpu,
        ))

        # 5. KPI : Chiffre d'Affaires Mensuel Perdu (Lost MRR)
        lost_mrr = float(df.loc[churn_mask, monthly_col].sum())
        pct_lost = (lost_mrr / (mrr + lost_mrr) * 100.0) if (mrr + lost_mrr) else 0.0
        kpis.append(KPIResult(
            kpi_id="KPI_05_LOST_MRR",
            name="Revenu Mensuel Perdu (Churn MRR)",
            value=round(lost_mrr, 2),
            formatted_value=f"{lost_mrr:,.2f} $ / mois",
            unit="$/mois",
            category="financial",
            formula="SUM(Monthly Charges pour les clients résiliés)",
            target_direction="minimize",
            status_alert="critical" if lost_mrr > 100000 else "warning",
            commentary=f"Perte sèche mensuelle représentant {pct_lost:.1f}% du potentiel de facturation global.",
            breakdowns={},
        ))

    # 6. KPI : Valeur Vie Client Moyenne (CLTV)
    if "CLTV" in df.columns:
        avg_cltv = float(df["CLTV"].mean())
        cltv_active = float(df.loc[active_mask, "CLTV"].mean())
        cltv_churn = float(df.loc[churn_mask, "CLTV"].mean())
        kpis.append(KPIResult(
            kpi_id="KPI_06_AVG_CLTV",
            name="Valeur Vie Client Moyenne (CLTV)",
            value=round(avg_cltv, 2),
            formatted_value=f"{avg_cltv:,.2f} $",
            unit="$",
            category="customer_retention",
            formula="AVG(Customer Lifetime Value)",
            target_direction="maximize",
            status_alert="healthy",
            commentary=f"CLTV actif: {cltv_active:,.0f} $ vs CLTV résilié: {cltv_churn:,.0f} $.",
            breakdowns={"Statut": {"Actifs": round(cltv_active, 2), "Résiliés": round(cltv_churn, 2)}},
        ))

    # 7. KPI : Durée de fidélité moyenne (Ancienneté)
    tenure_col = "Tenure Months" if "Tenure Months" in df.columns else ("tenure" if "tenure" in df.columns else None)
    if tenure_col and tenure_col in df.columns:
        avg_tenure_active = float(df.loc[active_mask, tenure_col].mean())
        avg_tenure_churn = float(df.loc[churn_mask, tenure_col].mean())
        kpis.append(KPIResult(
            kpi_id="KPI_07_AVG_TENURE",
            name="Ancienneté Moyenne Avant Départ",
            value=round(avg_tenure_churn, 1),
            formatted_value=f"{avg_tenure_churn:.1f} mois",
            unit="mois",
            category="operational",
            formula="AVG(Tenure Months pour clients partis)",
            target_direction="monitor",
            status_alert="warning" if avg_tenure_churn < 24 else "healthy",
            commentary=f"Les clients qui partent le font en moyenne après {avg_tenure_churn:.1f} mois, contre {avg_tenure_active:.1f} mois pour les actifs.",
            breakdowns={"Ancienneté": {"Moyenne Actifs": round(avg_tenure_active, 1), "Moyenne Résiliés": round(avg_tenure_churn, 1)}},
        ))

    # 8. KPI : Volume de Clients à Très Haut Risque
    if "Churn Score" in df.columns:
        high_risk_count = int((df["Churn Score"] >= 75).sum())
        high_risk_pct = (high_risk_count / total_customers) * 100.0
        kpis.append(KPIResult(
            kpi_id="KPI_08_HIGH_RISK_CUSTOMERS",
            name="Clients à Très Haut Risque (Score >= 75)",
            value=high_risk_count,
            formatted_value=f"{high_risk_count} clients ({high_risk_pct:.1f} %)",
            unit="clients",
            category="risk",
            formula="COUNT(Clients avec Churn Score >= 75)",
            target_direction="minimize",
            status_alert="critical" if high_risk_pct >= 20 else "warning",
            commentary="Volume de clients prioritaires nécessitant une action marketing/rétention immédiate.",
            breakdowns={},
        ))

    return kpis
