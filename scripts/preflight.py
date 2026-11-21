#!/usr/bin/env python3
"""
Demo-day preflight: run this before going on stage.   python scripts/preflight.py
Every line is PASS / WARN / FAIL with what to do about it. Costs well under a cent (one tiny Gemini call).
"""
import glob, json, os, shutil, socket, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from playq import config  # loads .env

G, Y, R, X = "\033[92m", "\033[93m", "\033[91m", "\033[0m"
fails = warns = 0
def ok(m): print(f"{G}PASS{X}  {m}")
def warn(m):
    global warns; warns += 1; print(f"{Y}WARN{X}  {m}")
def fail(m):
    global fails; fails += 1; print(f"{R}FAIL{X}  {m}")

# 1 tools
try:
    from playq.cv.rally_detector import _ffmpeg_exe
    ok(f"ffmpeg available ({os.path.basename(_ffmpeg_exe() or '')})") if _ffmpeg_exe() else fail("ffmpeg missing")
except Exception as e:
    fail(f"ffmpeg: {e}")
try:
    import streamlit, mediapipe, cv2  # noqa
    ok(f"streamlit {streamlit.__version__}, opencv {cv2.__version__}, mediapipe import OK")
except Exception as e:
    fail(f"python deps: {e}")

# 2 cached results the demo depends on
runs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, "data", "cache", "*", "result.json")) if "/live_" not in p)
(ok if runs else fail)(f"pre-processed results: {runs or 'NONE - run scripts/run_pipeline.py first'}")
for r in runs:
    res = json.load(open(os.path.join(ROOT, "data", "cache", r, "result.json")))
    src = res["ai"]["sources"]
    bad = [k for k, v in src.items() if v in ("error", "mock", "off")]
    vid = os.path.join(ROOT, "data", "cache", r, res["reel"]["file"])
    (ok if not bad and os.path.exists(vid) else warn)(f"  {r}: AI {src}  reel {'present' if os.path.exists(vid) else 'MISSING'}")
demo = os.path.join(ROOT, "data", "demo", "live_15s.mp4")
(ok if os.path.exists(demo) else warn)(f"live-run demo clip {'present' if os.path.exists(demo) else 'missing: ' + demo}")

# 3 Gemini
if not config.api_key():
    fail("GEMINI_API_KEY missing in .env")
else:
    try:
        from google import genai
        from google.genai import types
        c = genai.Client(api_key=config.api_key(), http_options=types.HttpOptions(timeout=25000))
        results = {}
        for m in config.MODEL_CHAIN:
            t = time.time()
            try:
                c.models.generate_content(model=m, contents="Reply OK"); results[m] = f"{time.time() - t:.1f}s"
            except Exception as e:
                results[m] = "429 quota" if "429" in str(e) else ("timeout" if "504" in str(e) or "imed out" in str(e) else "error")
        good = [m for m, v in results.items() if v.endswith("s")]
        (ok if good else fail)(f"Gemini models: {results}")
        if good and good[0] != config.MODEL_CHAIN[0]:
            warn(f"best model {config.MODEL_CHAIN[0]} is unavailable; the demo will use {good[0]} (slightly lower quality)")
    except Exception as e:
        fail(f"Gemini: {e}")

# 4 voice
cred = os.path.expanduser(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""))
if cred and os.path.exists(cred):
    try:
        from playq.agents.prototype import _cloud_token, _cloud_voices
        tok = _cloud_token(); v = _cloud_voices(*tok); ok(f"Google Cloud TTS ready, voices {v}")
    except Exception as e:
        warn(f"Cloud TTS credentials found but failing ({str(e)[:90]}); edge-tts will be used")
else:
    warn("no service account -> voice = edge-tts (still neural, but an unofficial service that needs internet)")
try:
    socket.create_connection(("speech.platform.bing.com", 443), timeout=5).close(); ok("edge-tts endpoint reachable")
except Exception:
    warn("edge-tts endpoint not reachable -> voiceover falls back to the local macOS voice (robotic)")

# 5 network + port
try:
    socket.create_connection(("generativelanguage.googleapis.com", 443), timeout=5).close(); ok("internet to Google OK")
except Exception:
    fail("no route to generativelanguage.googleapis.com: use the pre-processed results and turn 'Use Gemini' off")
s = socket.socket()
try:
    s.bind(("127.0.0.1", 8501)); ok("port 8501 free")
except OSError:
    warn("port 8501 in use (is the app already running? fine)")
finally:
    s.close()
print(f"\n{('READY' if not fails else 'NOT READY')}: {fails} fail, {warns} warn")
sys.exit(1 if fails else 0)
