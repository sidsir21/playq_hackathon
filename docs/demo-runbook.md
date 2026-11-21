# PlayQ — demo runbook (jury demo)

## 30 minutes before
1. `cd ~/Desktop/PlayQ && .venv/bin/python scripts/preflight.py`  → want **READY** (warnings are explained on screen).
2. Start the app: `.venv/bin/streamlit run app/streamlit_app.py` (open http://localhost:8501, full-screen the browser, sound ON).
3. Plug in power, disable sleep, close other apps (CPU is used by the live run), use a phone hotspot as backup Wi-Fi.
4. Click through once: **Pre-processed** → pick each result → press play on the reel (checks audio).

## The story (≈ 3 min)
1. **Hook (20 s):** "Every week players prop a phone on a fence, never edit the footage, and risk knee injuries."
2. **Live run (60 s)** — *Analyze → Live run → 15 s demo rally → Analyze match.* Narrate the stages as they tick:
   audio → court homography + MediaPipe → reel → Gemini (vision + video + script) → two-voice commentary.
3. **Show the four outputs** on the result: reel with commentary (sound!), heatmap (each player a colour), annotated knee frame, rating card (click through players).
4. **Pre-processed full match (45 s)** — *the 17-min WPT match*: "same pipeline, run once ahead of time (≈ 6 min)". Open **Architecture & cost**:
   - the funnel (17 min → 454 s of real live play → 40 s reel → Gemini),
   - the **visual check**: audio alone called 86 % of the match active because of other courts and crowd,
   - cost per run in dollars, Pydantic schema.
5. **Share & unlock:** QR on the projector, mock ₹99 UPI → WhatsApp share link.
6. **Close (15 s):** ₹5,000 pod, ₹99 per match, Delhi-NCR venues.

## If something breaks
| Symptom | Do this |
|---|---|
| Gemini slow / quota / no Wi-Fi | Toggle **Use Gemini** off for the live run (pure CV, ~7 s) and show the pre-processed results (already contain the AI output). |
| Voice sounds robotic | `edge-tts` unreachable → local voice. Say "voice is a fallback"; or switch to a pre-processed result. |
| Live run crashes | Switch to **Pre-processed**; it is just reading JSON + MP4 from disk. |
| A judge asks "is this real?" | Upload any clip from the venue profile; show `data/cache/<run>/stage2.json` / the Swagger page `http://localhost:8000/docs` (`uvicorn playq.api.main:app`). |

## Be upfront about (judges find these anyway)
- Test footage is third-party broadcast video, not the PlayQ pod; the ₹5,000 pod and 60 fps capture are not demonstrated by it.
- Far-court players are small: tracked less reliably; far-side positions ±0.5–1 m.
- Knee-risk bands, rating scales are heuristics, not medical advice. Clutch/consistency/shot-style are Gemini estimates from a ~40 s reel.
- Valgus is only reported when the player faces/turns away from the camera.
