"""Serveur FastAPI pour la plateforme SynapseBI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from agents.orchestrator import OrchestratorOptions, OrchestratorRequest
from agents.orchestrator import run as run_orchestrator

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data"

app = FastAPI(title="SynapseBI — Unified Multi-Agent Intelligence", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

pipeline_state: dict[str, Any] = {
    "is_running": False,
    "last_result": None,
    "error": None,
}


def _load_json(path: Path) -> dict[str, Any] | None:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = Path(__file__).parent / "templates" / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Template index.html introuvable")
    return index_file.read_text(encoding="utf-8")


@app.get("/api/status")
async def get_status():
    available_files: list[dict[str, str]] = []
    if DATA_DIR.exists():
        for f in sorted(DATA_DIR.rglob("*.csv")):
            rel = f.relative_to(DATA_DIR).as_posix()
            if "holdout" in rel.lower():
                continue
            available_files.append({
                "id": f.stem,
                "label": rel,
                "filename": rel,
            })

    output_datasets = [d.name for d in OUTPUT_DIR.iterdir() if d.is_dir()] if OUTPUT_DIR.exists() else []

    return {
        "pipeline_state": pipeline_state,
        "available_files": available_files,
        "output_datasets": output_datasets,
        "default_dataset": None,
    }


@app.get("/api/reports/{dataset_name}/{report_name}")
async def get_report(dataset_name: str, report_name: str):
    target = OUTPUT_DIR / dataset_name
    mapping = {
        "orchestrator": target / "orchestrator_summary.json",
        "profiler": target / "profiler_output.json",
        "etl": target / "etl_report.json",
        "kpi": target / "kpi" / "kpi_report.json",
        "analyst": target / "analyst" / "analyst_report.json",
        "dashboard": target / "dashboard" / "dashboard_report.json",
        "xai": target / "xai" / "xai_report.json",
    }

    if report_name not in mapping:
        raise HTTPException(status_code=400, detail="Nom de rapport inconnu")

    path = mapping[report_name]
    data = _load_json(path)
    if not data:
        raise HTTPException(status_code=404, detail="Rapport non encore généré")
    return data


def _background_run_pipeline(source_path: str, dataset_name: str, preset: str | None, skip_xai: bool):
    global pipeline_state
    pipeline_state["is_running"] = True
    pipeline_state["error"] = None
    try:
        req = OrchestratorRequest(
            source_path=source_path,
            dataset_name=dataset_name,
            business_goal="Analyse décisionnelle et prédiction comportementale",
            options=OrchestratorOptions(
                preset=preset,
                output_dir=str(OUTPUT_DIR),
                skip_xai=skip_xai,
            ),
        )
        res = run_orchestrator(req)
        pipeline_state["last_result"] = res.model_dump(mode="json")
    except Exception as exc:
        pipeline_state["error"] = str(exc)
    finally:
        pipeline_state["is_running"] = False


@app.post("/api/pipeline/run")
async def trigger_pipeline(
    background_tasks: BackgroundTasks,
    source_filename: str = "churn/telco.csv",
    preset: str = "none",
    skip_xai: bool = False,
):
    source_path = DATA_DIR / source_filename
    if not source_path.exists():
        matches = list(DATA_DIR.rglob(source_filename)) or list(DATA_DIR.rglob(Path(source_filename).name))
        if matches:
            source_path = matches[0]
        else:
            raise HTTPException(status_code=404, detail=f"Fichier {source_filename} introuvable dans data/")

    if pipeline_state["is_running"]:
        return {"status": "busy", "message": "Un pipeline est déjà en cours d'exécution."}

    dataset_name = Path(source_filename).stem
    background_tasks.add_task(
        _background_run_pipeline,
        str(source_path),
        dataset_name,
        preset if preset != "none" else None,
        skip_xai,
    )
    return {"status": "started", "dataset": dataset_name, "message": "Pipeline SynapseBI lancé."}


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_path = DATA_DIR / file.filename
    content = await file.read()
    file_path.write_bytes(content)
    return {"filename": file.filename, "size_bytes": len(content), "status": "saved"}
