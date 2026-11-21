#!/usr/bin/env python3
"""CLI for the full pipeline.  python scripts/run_pipeline.py <video> <calibration.json> <out_dir> [--no-gemini]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from playq.pipeline import analyze, finalize

ap = argparse.ArgumentParser()
ap.add_argument("video", nargs="?"); ap.add_argument("calib", nargs="?"); ap.add_argument("out", nargs="?")
ap.add_argument("--finalize", metavar="OUT_DIR", help="re-run only the knee-frame/reel/Gemini/cards stage from an existing stage2.json")
ap.add_argument("--no-gemini", action="store_true")
a = ap.parse_args()
res = finalize(a.finalize, use_gemini=not a.no_gemini) if a.finalize else analyze(a.video, a.calib, a.out, use_gemini=not a.no_gemini,
              progress=lambda s, f, m: print(f"  [{s:<6}] {f:4.0%}  {m}", flush=True) if s != "cv" or f > 0.95 or f == 0 else None)
print(json.dumps({"funnel": res["funnel"], "timings": res["timings"], "knee": res["knee"],
                  "players": [(p["track_id"], p["card"]["overall"], p["card"]["stats"]) for p in res["players"]],
                  "ai_sources": res["ai"]["sources"]}, indent=2, default=str))
