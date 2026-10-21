"""Teste le pipeline complet Profiler -> ETL -> Étoile -> KPI -> BI Analyst -> Dashboard.

Usage :
    python run_pipeline.py data/telco.csv --preset telco
    python run_pipeline.py chemin/vers/mon_fichier.csv
    python run_pipeline.py mon_fichier.csv --key subscriber_id --sep ";"
    python run_pipeline.py data/telco.csv --preset telco --no-dashboard
    python run_pipeline.py data/telco.csv --preset telco --no-analyst
    python run_pipeline.py data/telco.csv --preset telco --no-kpi
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from agents.bi_analyst import AnalystOptions, AnalystRequest
from agents.bi_analyst import run as run_analyst
from agents.dashboard import DashboardOptions, DashboardRequest
from agents.dashboard import run as run_dashboard
from agents.etl import ETLOptions, ETLRequest, run_with_data
from agents.kpi import KPIOptions, KPIRequest
from agents.kpi import run as compute_kpis
from agents.profiler.agent import run as profile
from agents.profiler.models import ProfileOptions, ProfileRequest, SourceConfig
from agents.xai import XAIOptions, XAIRequest
from agents.xai import run as run_xai


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pipeline complet : Profiler -> ETL -> Étoile -> KPI -> BI Analyst -> Dashboard"
    )
    parser.add_argument("csv", help="Chemin vers le fichier CSV source")
    parser.add_argument("--name", help="Nom logique de la table (défaut : nom du fichier)")
    parser.add_argument("--goal", default="Analyse BI telecom", help="Objectif métier")
    parser.add_argument("--preset", choices=["telco"], default=None, help="Preset de règles de nettoyage")
    parser.add_argument("--key", help="Colonne clé métier (ex. customer_id)")
    parser.add_argument("--sep", default=",", help="Séparateur CSV (défaut : ,)")
    parser.add_argument("--encoding", default="utf-8", help="Encodage du CSV (défaut : utf-8)")
    parser.add_argument("--out", default="output", help="Répertoire racine de sortie (défaut : output)")
    parser.add_argument("--no-kpi", action="store_true", help="S'arrêter après l'ETL (pas de KPI)")
    parser.add_argument("--no-analyst", action="store_true", help="S'arrêter après les KPIs (pas de BI Analyst)")
    parser.add_argument("--no-dashboard", action="store_true", help="Ne pas générer le Dashboard")
    parser.add_argument("--no-xai", action="store_true", help="Ne pas exécuter l'audit XAI")
    parser.add_argument("--llm", action="store_true", help="Activer le LLM pour les synthèses narratives")
    args = parser.parse_args()

    path = Path(args.csv)
    name = args.name or re.sub(r"\W+", "_", path.stem).strip("_").lower()
    source = SourceConfig(
        name=name,
        type="csv",
        uri=str(path),
        options={"sep": args.sep, "encoding": args.encoding},
    )
    keys = {name: [args.key]} if args.key else {}

    print(f"1) PROFILER sur {path} ...")
    profiling = profile(
        ProfileRequest(
            business_goal=args.goal,
            sources=[source],
            options=ProfileOptions(business_keys=keys),
        )
    )
    print(f"   statut : {profiling.status} | {len(profiling.issues)} problème(s) signalé(s)")
    for issue in profiling.issues:
        print(f"   - [{issue.severity}] {issue.table}.{issue.column or ''} : {issue.rule} "
              f"({issue.affected_rows} ligne(s))")

    out_dir = Path(args.out) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "profiler_output.json").write_text(
        profiling.model_dump_json(indent=2), encoding="utf-8"
    )

    print("\n2) ETL ...")
    report, frames = run_with_data(
        ETLRequest(
            profiling_report=profiling,
            sources=[source],
            options=ETLOptions(
                preset=args.preset,
                business_keys=keys,
                use_llm=False,
                output_dir=str(out_dir),
            ),
        )
    )
    print(f"   statut : {report.status}")
    for table in report.tables:
        print(f"   {table.name} : {table.rows_in} -> {table.rows_out} lignes | qualité "
              f"{table.quality_before.overall} -> {table.quality_after.overall}")
    print(f"   {len(report.actions)} action(s) :")
    for action in report.actions:
        if action.rows_affected:
            print(f"   - {action.rule}"
                  f"{'.' + action.column if action.column else ''} : {action.rows_affected} ligne(s)")
    for error in report.errors:
        print(f"   ! {error}")

    for star in report.star_schemas:
        print(f"\n3) MODÈLE EN ÉTOILE : {star.fact_name}  ({star.grain})")
        print(f"   mesures    : {[m.column for m in star.measures]}")
        for dim in star.dimensions:
            print(f"   {dim.name:<16}{dim.rows:>6} lignes  {dim.columns}")
        if star.date_columns:
            print(f"   dates      : {star.date_columns} -> dim_date")
        print(f"   exclues    : {[e['column'] + ' (' + e['reason'] + ')' for e in star.excluded_columns]}")
        ok = sum(1 for c in report.validation_checks if c.check.startswith("star") and c.passed)
        total = sum(1 for c in report.validation_checks if c.check.startswith("star"))
        print(f"   contrôles étoile : {ok}/{total} OK")

    if report.remaining_issues:
        print("\n4) À VALIDER PAR UN HUMAIN :")
        for issue in report.remaining_issues:
            print(f"   - [{issue.severity}] {issue.rule} : {issue.evidence}")

    kpi_status = "skipped"
    kpi_report = None
    if not args.no_kpi and report.status != "error":
        print("\n5) KPI ...")
        kpi_report = compute_kpis(
            KPIRequest(
                etl_report=report.model_dump(mode="json"),
                options=KPIOptions(output_dir=str(out_dir / "kpi"), use_llm=False),
            ),
            frames=frames,
        )
        kpi_status = kpi_report.status
        print(f"   statut : {kpi_report.status}")
        for kpi in kpi_report.kpis:
            if kpi.status != "not_computable":
                flag = "  [SUSPECT]" if kpi.status == "suspect" else ""
                print(f"   - {kpi.name:<34}{kpi.value:>12} {kpi.unit}{flag}")
        skipped = [k.id for k in kpi_report.kpis if k.status == "not_computable"]
        if skipped:
            print(f"   non calculables avec ces données : {skipped}")
        for caveat in kpi_report.caveats:
            print(f"   ? {caveat}")
        for error in kpi_report.errors:
            print(f"   ! {error}")

    analyst_status = "skipped"
    analyst_report = None
    if (
        not args.no_analyst
        and not args.no_kpi
        and kpi_status != "error"
        and report.status != "error"
        and kpi_report is not None
    ):
        print("\n6) BI ANALYST ...")
        analyst_report = run_analyst(
            AnalystRequest(
                kpi_report=kpi_report.model_dump(mode="json"),
                etl_report=report.model_dump(mode="json"),
                options=AnalystOptions(
                    output_dir=str(out_dir / "analyst"),
                    use_llm=args.llm,
                ),
            ),
            frames=frames,
        )
        analyst_status = analyst_report.status
        print(f"   statut : {analyst_report.status} | {len(analyst_report.insights)} insight(s) | {len(analyst_report.recommendations)} recommandation(s)")
        for ins in analyst_report.insights[:5]:
            print(f"   - #{ins.rank} [{ins.confidence}] {ins.title}")
        if len(analyst_report.insights) > 5:
            print(f"     ... (+{len(analyst_report.insights) - 5} autres)")
        for rec in analyst_report.recommendations[:3]:
            print(f"   * [{rec.priority}] {rec.action[:75]}...")
        for err in analyst_report.errors:
            print(f"   ! {err}")

    dashboard_status = "skipped"
    if (
        not args.no_dashboard
        and analyst_status not in {"skipped", "error"}
        and report.status != "error"
        and analyst_report is not None
        and kpi_report is not None
    ):
        print("\n7) DASHBOARD GENERATOR ...")
        dashboard_report = run_dashboard(
            DashboardRequest(
                analyst_report=analyst_report.model_dump(mode="json"),
                kpi_report=kpi_report.model_dump(mode="json"),
                etl_report=report.model_dump(mode="json"),
                options=DashboardOptions(
                    output_dir=str(out_dir / "dashboard"),
                    use_llm=args.llm,
                ),
            ),
            frames=frames,
        )
        dashboard_status = dashboard_report.status
        print(f"   statut : {dashboard_report.status} | {len(dashboard_report.visuals)} visuel(s) | {len(dashboard_report.excluded_visuals)} exclu(s)")
        for s in dashboard_report.sections:
            print(f"   - [{s.title}] {len(s.visual_ids)} visuel(s)")
        for err in dashboard_report.errors:
            print(f"   ! {err}")

    xai_status = "skipped"
    if (
        not args.no_xai
        and report.status != "error"
        and (out_dir / f"{name}_clean.csv").exists()
    ):
        print("\n8) BI AUDITOR / XAI ...")
        xai_report = run_xai(
            XAIRequest(
                clean_data_path=str(out_dir / f"{name}_clean.csv"),
                business_goal=args.goal,
                options=XAIOptions(
                    output_dir=str(out_dir / "xai"),
                    use_llm=args.llm,
                ),
            )
        )
        xai_status = xai_report.status
        print(f"   statut : {xai_report.status}")
        if xai_report.model_metrics:
            m = xai_report.model_metrics
            print(f"   RandomForest : Accuracy {m.accuracy:.1%} | Recall {m.recall:.1%} | F1 {m.f1:.1%} | AUC-ROC {m.auc_roc:.3f}")
        if xai_report.baseline_comparison:
            b = xai_report.baseline_comparison
            print(f"   Baseline     : Dummy F1 {b.baseline_f1:.1%} -> Gain F1 +{b.delta_f1:.1%} ({b.verdict})")
        if xai_report.shap_summary:
            top_3 = [f.feature for f in xai_report.shap_summary.top_features[:3]]
            print(f"   Top SHAP     : {', '.join(top_3)}")
        for err in xai_report.errors:
            print(f"   ! {err}")

    print(f"\nTous les artefacts sont disponibles dans : {out_dir}")
    success = (
        report.status != "error"
        and kpi_status != "error"
        and analyst_status != "error"
        and dashboard_status != "error"
        and xai_status != "error"
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

    sys.exit(main())