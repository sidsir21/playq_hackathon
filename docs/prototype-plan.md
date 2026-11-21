# PlayQ — 24-hour prototype plan (scoring-driven)

Weights: Functional product 30 · Architecture/engineering 20 · Google AI depth 20 · Problem fit 15 · UX/accessibility 10 · Innovation 5.

## What the judges will see (one web app, `streamlit run app/streamlit_app.py`)
| Output | Where it comes from | Status |
|---|---|---|
| Auto-trimmed highlight reel **with Hinglish voiceover** | scipy audio spikes → FFmpeg cut → Gemini script → Gemini TTS → FFmpeg mux | built; TTS needs API key |
| 2D movement heatmap (all players + per player) | stabilised 4/5-point homography + tracker | built |
| Annotated skeleton frame with knee angle | MediaPipe 33-pt pose, planted-foot lunge filter, Gemini vision second opinion | built |
| FIFA-style PlayQ Rating Card (per player) | 3 measured stats (coverage, aggression, stamina) + 2 Gemini-estimated (consistency, clutch) | built |
| Live-run vs pre-cached toggle | 15 s clip live (~10 s) · 9:43 match from cache | built (cache needs a run) |
| Architecture & cost tab, measured funnel, Pydantic schema | read from each run's `result.json` | built |
| QR + mocked ₹99 UPI + WhatsApp share | QR → LAN URL; payment is a labelled demo | built |
| FastAPI + Swagger `/docs` | `uvicorn playq.api.main:app` → `/api/v1/pipeline/*` | built |

## Pipeline (cost & latency funnel)
1. **Audio filter (CPU)** — ball-pop transients over a rolling noise floor → rallies. 
2. **Deterministic CV** — ORB camera stabiliser (skips replay/cut frames) → homography → MediaPipe on full frame + 3 far-court strips → tracker (colour + court position + no-net-crossing) → knee flexion / valgus (valgus only when the player's back/front faces the camera).
3. **Reel** — best 4×7 s windows by ball-impact density, FFmpeg.
4. **Gemini chain** (not one prompt): vision on the skeleton frame → video + metrics → Pydantic `MatchInsights` → TTS.
5. Mux voiceover over ducked match audio.

## Measured results (this machine, 2026-11-20)
| Run | CV | Gemini | Total |
|---|---|---|---|
| Live 15 s clip | ~5 s | ~21 s | **~28 s** (≈7 s with Gemini off) |
| Full 9:43 match (25 fps) | 308 s | ~28 s | **~5.7 min**, run once and cached |

Gemini model latency on the video+schema call: 3.5-flash-lite 3.6 s · 3.6-flash 9.8 s · 3.5-flash 11.6 s · 3.8-flash ~60 s → default `gemini-3.6-flash`, fallback `gemini-3.5-flash-lite`.
Voiceover: this key lists no TTS/Live models, so the voice is the local macOS en-IN voice (`fallback-say`). Better option: Google Cloud Text-to-Speech (needs a Cloud project).

## Honest limits (say these before a judge finds them)
- Test footage `Tes.mp4` is **25 fps** third-party broadcast footage, not our 60 fps pod. Pitch claims about 60 fps are not demonstrated by it.
- Far-court players are small/occluded: tracked sparsely; far-side positions are extrapolated from near-side landmarks.
- The audio filter keeps ~90 % of `Tes.mp4` because it is pre-edited; the saving comes from sending a ~28 s reel to Gemini.
- Knee risk bands are heuristic, not medical. "Clutch" and "consistency" are Gemini estimates from a ~28 s reel.
- ID-lock accuracy, angle accuracy: not yet measured against labels.
- Gemini free tier may use content to improve Google products; use billing-enabled keys for customer video.

## Demo run-sheet
1. Key in `.env` (`GEMINI_API_KEY`). `python scripts/test_gemini.py` → green.
2. Pre-cache is already built in `data/cache/full_match` (re-run AI only: `python scripts/run_pipeline.py --finalize data/cache/full_match`; full redo: `python scripts/run_pipeline.py Tes.mp4 config/calibration/the_lab.json data/cache/full_match`).
3. `streamlit run app/streamlit_app.py` → **Live run** with the 15 s clip (judges watch it execute) → then **Pre-processed full match**.
4. Open **Architecture** tab, then `http://localhost:8000/docs` if asked.
5. **Share & unlock** tab → QR on projector.
Fallback if Wi-Fi dies: toggle "Use Gemini" off (labelled template commentary + local voice) or use cached result.

## Cut list
Real Razorpay/UPI, RTSP camera, playq.in hosting, Nano Banana/Lyria.


## Update 2026-11-20 (round 2 feedback)
- **Voice:** two neural Hindi voices (Madhur + Swara, via `edge-tts`, no API key, needs internet). Each line is synthesised separately and placed at its timestamp; lines never overlap (pushed later, then sped up ≤1.3x, then dropped). Match audio is ducked only while someone speaks. Fallback: local macOS en-IN voice.
- **Script:** Gemini receives the reel's real timeline (segments, ball impacts) and the tracked players (shirt colour, side, distance to net); lines must sit inside the segment they describe; video sampled at 4 fps.
- **Reel:** length scales with the source (25-45 s for clips under 2.5 min, 35-40 s for long matches); cuts snap between shots; small gaps inside a rally are played through.
- **Heatmap:** fixed wrong kitchen-line positions (they were 2.13 m from the baselines instead of from the net); one colour per player + legend. Accuracy ~±0.2 m near half, ±0.5-1 m far half.
- **Aggression:** continuous (mean distance to net + time in 3 m zone) blended 50/50 with Gemini's shot-style score. Two baseliners legitimately score similarly.
- **Identity:** each player has a fixed colour, shirt-colour name, and a photo. Tracker bugs fixed: bystander accepted 0.9 m outside the sideline; orange player split across two tracks because depth jitters at the net (now a 0.8 m dead-band).
- **Reliability:** model chain (3.6-flash → 3.5-flash → 3.5-flash-lite → 3.1-flash-lite), hedged video call, 429s skip to the next model. Free-tier quota ran out on the two best models today.
