"""Interface en ligne de commande du BI Analyst Agent.

Usage
-----
python -m agents.bi_analyst --kpi-report output/telco/kpi/kpi_report.json --no-llm
python -m agents.bi_analyst --kpi-report output/telco/kpi/kpi_report.json \\
                             --etl-report output/telco/etl_report.json \\
                             --output-dir output/telco/analyst
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agents.bi_analyst",
        description="BI Analyst Agent — analyse des insights churn depuis le KPIReport.",
    )
    parser.add_argument(
        "--kpi-report", required=True,
        help="Chemin vers kpi_report.json (produit par le KPI Agent).",
    )
    parser.add_argument(
        "--etl-report", default=None,
        help="Chemin vers etl_report.json (optionnel ; requis pour les métadonnées PII "
             "et les segmentations supplémentaires).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Répertoire de sortie. Défaut : même répertoire parent que kpi_report.",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Désactive la narration LLM (synthèse déterministe uniquement).",
    )
    parser.add_argument(
        "--min-excess-rate", type=float, default=3.0,
        help="Écart minimal en points de %% pour retenir un insight (défaut : 3.0).",
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Nombre maximal d'insights retenus (défaut : 10).",
    )
    args = parser.parse_args()

    kpi_path = Path(args.kpi_report)
    if not kpi_path.exists():
        print(f"[ERREUR] Fichier introuvable : {kpi_path}", file=sys.stderr)
        return 1

    kpi_raw = json.loads(kpi_path.read_text(encoding="utf-8"))

    etl_raw = None
    if args.etl_report:
        etl_path = Path(args.etl_report)
        if not etl_path.exists():
            print(f"[AVERTISSEMENT] ETL report introuvable : {etl_path}", file=sys.stderr)
        else:
            etl_raw = json.loads(etl_path.read_text(encoding="utf-8"))

    out_dir = args.output_dir or str(kpi_path.parent.parent / "analyst")

    from agents.bi_analyst import AnalystOptions, AnalystRequest, run as analyst_run

    request = AnalystRequest(
        kpi_report=kpi_raw,
        etl_report=etl_raw,
        options=AnalystOptions(
            output_dir=out_dir,
            use_llm=not args.no_llm,
            min_excess_rate=args.min_excess_rate,
            top_n_insights=args.top_n,
        ),
    )

    print(f"[BI Analyst] Lecture : {kpi_path}")
    report = analyst_run(request)

    # Affichage résumé
    print(f"[BI Analyst] Statut : {report.status}")
    print(f"[BI Analyst] Tests : {report.n_tests} z-tests (correction BH)")
    if report.null_result:
        print("[BI Analyst] Aucun insight significatif après filtrage.")
    else:
        print(f"[BI Analyst] {len(report.insights)} insight(s) :")
        for ins in report.insights:
            print(f"  #{ins.rank} [{ins.confidence}] {ins.title}")
            print(f"       Départs en excès : {ins.impact_churners:.0f} | "
                  f"Revenu perdu excès : {ins.revenue_monthly_lost_excess:,.0f} /mois")
        print(f"[BI Analyst] {len(report.recommendations)} recommandation(s) :")
        for rec in report.recommendations:
            print(f"  [{rec.priority}] {rec.action[:80]}…")
    if report.errors:
        print("[BI Analyst] Erreurs :")
        for e in report.errors:
            print(f"  ⚠ {e}")
    for p, v in report.output_files.items():
        print(f"[BI Analyst] Fichier : {p} → {v}")
    return 0 if report.status != "error" else 1


if __name__ == "__main__":
    sys.exit(main())