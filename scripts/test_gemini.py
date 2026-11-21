#!/usr/bin/env python3
"""
Staged Gemini connectivity check. Run after putting GEMINI_API_KEY in .env.

  python scripts/test_gemini.py                 # key + model list + one structured call
  python scripts/test_gemini.py --full          # run all agents on data/match_report.json
  python scripts/test_gemini.py --full --video VID.mp4   # also verify rallies by watching the video
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from playq import config
from playq.agents.gemini_agents import (BiomechanicsAgent, PlayQOrchestrator, _GENAI_AVAILABLE)

G, R, Y, X = "\033[92m", "\033[91m", "\033[93m", "\033[0m"
ap = argparse.ArgumentParser()
ap.add_argument("--full", action="store_true")
ap.add_argument("--video")
args = ap.parse_args()
root = os.path.join(os.path.dirname(__file__), "..")

print("1) API key")
if not _GENAI_AVAILABLE:
    sys.exit(f"{R}google-genai not installed: pip install google-genai{X}")
if not config.api_key():
    sys.exit(f"{R}No GEMINI_API_KEY found.{X}\n   1. Create a key: https://aistudio.google.com/apikey\n"
             f"   2. cp .env.example .env  and paste it after GEMINI_API_KEY=\n   3. re-run this script")
print(f"   {G}key found{X} (ends …{config.api_key()[-4:]})")

print("2) Models available to this key")
from google import genai
client = genai.Client(api_key=config.api_key())
try:
    names = sorted(m.name.replace("models/", "") for m in client.models.list())
except Exception as e:
    sys.exit(f"   {R}list failed: {type(e).__name__}: {e}{X}\n   (invalid key, no network, or API not enabled for the project)")
print(f"   {len(names)} models. Configured: text={config.TEXT_MODEL} video={config.VIDEO_MODEL}")
for label, m in (("text", config.TEXT_MODEL), ("video", config.VIDEO_MODEL)):
    print(f"   {G+'✓' if m in names else R+'✗ NOT FOUND — set PLAYQ_GEMINI_'+('MODEL' if label=='text' else 'VIDEO_MODEL')+' in .env'}{X} {label}: {m}")
for kw in ("flash", "image", "live", "lyria"):
    print(f"   {kw:<6}: {', '.join(n for n in names if kw in n)[:200] or '(none visible to this key)'}")

print("3) Structured output (Biomechanics agent, sample data)")
res = BiomechanicsAgent().analyze({"sport": "pickleball", "peak_angles_deg": {"lumbar_flexion": 41.0, "shoulder_abduction": 118.0},
                                   "high_risk_events": 2, "valgus_assessable_frames": 0, "recommendations": []})
print(f"   _source = {res.get('_source')}")
if res.get("_source") != "gemini":
    sys.exit(f"   {R}{res}{X}")
print(f"   {G}schema-valid{X}: severity={res['severity']} | {res['diagnosis'][:120]}")
print(f"   what it says the data can't show: {res['confidence_note'][:120]}")

if args.full:
    rp = os.path.join(root, "data", "match_report.json")
    if not os.path.exists(rp):
        sys.exit(f"{Y}Run scripts/analyze_video.py first to create {rp}{X}")
    print("4) Full agent run on the real match report")
    out = PlayQOrchestrator().process_report(json.load(open(rp)), video_path=args.video, verify_video=bool(args.video))
    dest = os.path.join(root, "data", "ai_report.json")
    json.dump(out, open(dest, "w"), indent=2, ensure_ascii=False)
    print(f"   {G}wrote {dest}{X}")
    if "rally_verification" in out:
        print(f"   rally verification: {out['rally_verification'].get('_source')} — {str(out['rally_verification'].get('summary', out['rally_verification']))[:200]}")
    print(f"   caption: {out['broadcast'].get('instagram_caption', out['broadcast'])}")
