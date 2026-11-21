#!/usr/bin/env python3
"""Export pre-processed runs into web/demo/ for the static site (no backend needed).
   python scripts/export_web.py            # re-run after regenerating data/cache/*"""
import json, os, shutil, subprocess, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from playq.cv.rally_detector import _ffmpeg_exe

RUNS = [  # (cache dir, title, subtitle, footage note)
    ("night_court_66s", "Night court", "Phone footage · 66 s · 2 players", "Own footage"),
    ("wpt_full_match", "WPT Cincinnati · Court 17", "Overhead broadcast cam · 17 min", "Third-party test footage"),
    ("full_match", "The Lab · indoor", "Broadcast cam · 9 min doubles", "Third-party test footage"),
]
OUT = os.path.join(ROOT, "web", "demo")
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
index = []
for rid, title, sub, note in RUNS:
    src = os.path.join(ROOT, "data", "cache", rid)
    if not os.path.exists(os.path.join(src, "result.json")):
        print("skip (no result):", rid); continue
    r = json.load(open(os.path.join(src, "result.json")))
    dst = os.path.join(OUT, rid); os.makedirs(dst)
    reel = os.path.join(src, r["reel"]["file"])
    subprocess.run([_ffmpeg_exe(), "-y", "-i", reel, "-vf", "scale=-2:720", "-c:v", "libx264", "-preset", "slow", "-crf", "27",
                    "-c:a", "aac", "-b:a", "112k", "-movflags", "+faststart", os.path.join(dst, "reel.mp4")], check=True, capture_output=True)
    files = ["heatmap_all.png", r["knee"]["image"] if r.get("knee") else None] + [p["heatmap"] for p in r["players"]] + [p["thumb"] for p in r["players"] if p.get("thumb")]
    for f in filter(None, files):
        shutil.copy(os.path.join(src, f), dst)
    for p in r["players"]:
        p.pop("points", None); p.pop("bio", None)
    slim = {k: r[k] for k in ("meta", "funnel", "timings", "players", "knee", "team_heatmap")}
    slim["reel"] = {"seconds": r["reel"]["seconds"], "timeline": r["reel"].get("timeline", []), "segments": r["reel"]["segments"]}
    ai = r["ai"]
    slim["ai"] = {"sources": ai["sources"], "vision": ai.get("vision"), "insights": ai.get("insights"), "usage": ai.get("usage"),
                  "voice": {k: v for k, v in (ai.get("voice") or {}).items() if k in ("_source", "voices", "timing", "dropped")}}
    slim["meta"] = {k: v for k, v in r["meta"].items() if k != "video"}
    slim.update({"id": rid, "title": title, "subtitle": sub, "note": note})
    json.dump(slim, open(os.path.join(dst, "data.json"), "w"), ensure_ascii=False)
    size = sum(os.path.getsize(os.path.join(dst, f)) for f in os.listdir(dst)) / 1e6
    index.append({"id": rid, "title": title, "subtitle": sub, "note": note, "duration_s": r["meta"]["duration_s"]})
    print(f"{rid}: {size:.1f} MB")
json.dump(index, open(os.path.join(OUT, "index.json"), "w"), ensure_ascii=False)
print("total", round(sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(OUT) for f in fs) / 1e6, 1), "MB")
