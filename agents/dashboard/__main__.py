"""CLI : python -m agents.dashboard --analyst-report output/telco/analyst/analyst_report.json
        --kpi-report output/telco/kpi/kpi_report.json --data data/telco.csv --no-llm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from agents.dashboard.agent import run
from agents.dashboard.models import DashboardOptions, DashboardRequest


def _load(path: str | None) -> dict | None:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def main() -> int:
    p = argparse.ArgumentParser(description="Dashboard Generator Agent")
    p.add_argument("--analyst-report", required=True)
    p.add_argument("--kpi-report")
    p.add_argument("--etl-report")
    p.add_argument("--data", help="table nettoyée (csv/parquet) si absente de l'etl_report")
    p.add_argument("--out", default="output/telco/dashboard")
    p.add_argument("--geo-level", choices=["state", "city"], default="state")
    p.add_argument("--max-visuals", type=int, default=20, help="nombre max de visuels (hors tables et textes)")
    p.add_argument("--no-llm", action="store_true")
    a = p.parse_args()

    frames = None
    if a.data:
        frames = {"table": pd.read_parquet(a.data) if a.data.endswith(".parquet") else pd.read_csv(a.data)}
    report = run(DashboardRequest(
        analyst_report=_load(a.analyst_report) or {}, kpi_report=_load(a.kpi_report),
        etl_report=_load(a.etl_report),
        options=DashboardOptions(output_dir=a.out, use_llm=not a.no_llm, geo_level=a.geo_level,
                         max_visuals=a.max_visuals)), frames=frames)

    print(f"statut : {report.status} | {len(report.visuals)} visuel(s) | {len(report.excluded_visuals)} exclu(s)")
    for s in report.sections:
        print(f"  [{s.title}] {', '.join(s.visual_ids)}")
    for e in report.excluded_visuals:
        print(f"  - exclu : {e.candidate} ({e.reason})")
    for err in report.errors:
        print(f"  ! {err}")
    return 0 if report.status != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
