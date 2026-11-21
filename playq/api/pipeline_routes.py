"""Real pipeline endpoints (the older /api/v1/analyze in main.py still returns demo data)."""
import json
import os
import shutil
import time
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from playq.agents.schemas import BioVision, MatchInsights
from playq.pipeline import analyze

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE = os.path.join(ROOT, "data", "cache")
CALIB_DIR = os.path.join(ROOT, "config", "calibration")

router = APIRouter(prefix="/api/v1/pipeline", tags=["Pipeline (real)"])


class Funnel(BaseModel):
    source_seconds: float
    rallies: int
    active_seconds: float
    frames_sampled: int
    reel_seconds: float
    video_seconds_reduction_pct: float


class PipelineResponse(BaseModel):
    run_id: str
    timings: dict[str, float]
    funnel: Funnel
    knee_flexion_deg: Optional[float] = Field(default=None, description="Peak planted-foot knee flexion")
    knee_level: Optional[str] = None
    players: list[dict]
    ai_sources: dict[str, str] = Field(description="gemini | template | fallback-say | mock | error | off — what was real")
    reel_url: str
    result_url: str


def _profiles() -> dict[str, str]:
    return {f[:-5]: os.path.join(CALIB_DIR, f) for f in os.listdir(CALIB_DIR) if f.endswith(".json")}


@router.get("/venues", summary="Calibrated venue/camera profiles")
def venues() -> list[str]:
    return sorted(_profiles())


@router.get("/schema/insights", summary="Pydantic contract Gemini must satisfy")
def insights_schema() -> dict:
    return {"match_insights": MatchInsights.model_json_schema(), "bio_vision": BioVision.model_json_schema()}


@router.post("/analyze", response_model=PipelineResponse, summary="Upload a clip and run the full pipeline")
async def run(video: UploadFile = File(...), venue: str = Form("the_lab"), use_gemini: bool = Form(True)) -> PipelineResponse:
    profiles = _profiles()
    if venue not in profiles:
        raise HTTPException(422, f"Unknown venue '{venue}'. Available: {sorted(profiles)}")
    run_id = time.strftime("api_%Y%m%d_%H%M%S")
    out = os.path.join(CACHE, run_id)
    os.makedirs(out, exist_ok=True)
    src = os.path.join(out, "upload.mp4")
    with open(src, "wb") as fh:
        shutil.copyfileobj(video.file, fh)
    try:
        res = analyze(src, profiles[venue], out, use_gemini=use_gemini)
    except Exception as e:   # surface pipeline problems as a clear 500 instead of a stack trace
        raise HTTPException(500, f"Pipeline failed: {type(e).__name__}: {e}")
    knee = res.get("knee") or {}
    return PipelineResponse(
        run_id=run_id, timings=res["timings"], funnel=Funnel(**{k: res["funnel"][k] for k in Funnel.model_fields}),
        knee_flexion_deg=knee.get("flexion"), knee_level=knee.get("level"),
        players=[{"track_id": p["track_id"], "overall": p["card"]["overall"], "stats": p["card"]["stats"],
                  "placeholder_stats": p["card"]["placeholder_stats"]} for p in res["players"]],
        ai_sources=res["ai"]["sources"],
        reel_url=f"/api/v1/pipeline/runs/{run_id}/{res['reel']['file']}", result_url=f"/api/v1/pipeline/runs/{run_id}/result.json",
    )


@router.get("/runs/{run_id}/{filename}", summary="Fetch an output file of a run")
def get_file(run_id: str, filename: str):
    path = os.path.realpath(os.path.join(CACHE, run_id, filename))
    if not path.startswith(os.path.realpath(CACHE) + os.sep) or not os.path.isfile(path):   # block path traversal
        raise HTTPException(404, "Not found")
    return FileResponse(path)
