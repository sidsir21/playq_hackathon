# 🏓 PlayQ

**AI-powered Full HD 60fps broadcast highlights, spatial analytics & biomechanical intelligence for recreational sports**

> _Pickleball · Padel · Turf Football · Box Cricket · Badminton_
> Built for **Google DeepMind / Google Cloud — Startup Buildathon India 2026**

---

## What is PlayQ?

PlayQ is a **B2B2C AI Sports Intelligence & Social Highlight platform** powered by a low-cost ₹5,000 plug-and-play Full HD 1080p/60fps smart camera kit ("PlayQ Pod") installed at partner courts. After each match, players scan a QR code and unlock a complete post-match intelligence bundle for ₹99:

| Feature | What you get |
|---|---|
| 🦴 Biomechanical Audit | Knee valgus, shoulder abduction, lumbar flexion flags + correction cards |
| 📐 2D Court Heatmap | Bird's-eye court coverage, dead zones, kitchen discipline |
| ⭐ PlayQ Rating Card | FIFA-style 5-stat card (PWR / ACC / AGI / IQ / END) |
| 🎬 Highlight Reel | Hinglish-commentated 30s 1080p vertical reel + Lyria beat |
| 📊 Leaderboard | Live venue ranking across all matched players |

---

## Architecture (5 Stages)

```
Stage 1  │ Edge Ingestion   │ 1080p/60fps MP4 → audio spike detect (scipy) → FFmpeg trim → Cloud Storage
Stage 2  │ Court Mapping    │ OpenCV 4-corner homography → 2D bird's-eye grid + MediaPipe 33-keypoint pose
Stage 3  │ Multi-Agent AI   │ Gemini Omni (video) · Gemini Flash (tactics JSON) · Gemini Live (commentary)
Stage 4  │ DB Persistence   │ Cloud Firestore (profiles/leaderboard) · BigQuery (historical metrics)
Stage 5  │ Delivery         │ FastAPI on Cloud Run → Next.js on playq.in (Vercel)
```

---

## Tech Stack

| Layer | Technology | Cost |
|---|---|---|
| CV Pipeline | OpenCV · MediaPipe · scipy · FFmpeg | Free (OSS) |
| AI Reasoning | Gemini Flash / Omni (AI Studio) | Free tier |
| Agent Framework | Google Antigravity + Agent Skills | Included in AI Pro |
| Backend | FastAPI on Cloud Run | Free (2 M req/mo) |
| Database | Cloud Firestore | Free (1 GiB) |
| Storage | Google Cloud Storage | Free (5 GB) |
| Frontend | Next.js on Vercel | Free tier |
| Music | Lyria (AI music gen) | ~₹100 for demos |

---

## Project Structure

```
PlayQ/
├── main.py                        # uvicorn entry point
├── pyproject.toml
├── scripts/
│   └── demo_pipeline.py           # 🎮 Run this first — no video/API key needed
├── playq/
│   ├── cv/
│   │   ├── homography.py          # OpenCV 4-corner → 2D bird's-eye + heatmap
│   │   ├── pose_analyzer.py       # MediaPipe Pose → joint angles → risk flags
│   │   └── rally_detector.py      # Audio energy → rally segments → FFmpeg trim
│   ├── models/
│   │   └── rating.py              # PlayQStats, PlayerProfile, RatingEngine
│   ├── agents/
│   │   └── gemini_agents.py       # BiomechanicsAgent + TacticsAgent + BroadcastAgent
│   ├── skills/                    # Antigravity Agent Skills (Day 4-5)
│   └── api/
│       └── main.py                # FastAPI endpoints
└── frontend/                      # Next.js dashboard (Day 6-7)
```

---

## Quick Start

```bash
# 1. Clone & enter
git clone https://github.com/your-handle/playq.git
cd playq

# 2. Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Run the synthetic demo (no video or API key needed)
python scripts/demo_pipeline.py

# 5. Start the API server
python main.py
# → http://localhost:8000/docs
```

**Optional — enable real Gemini AI:**
```bash
export GOOGLE_API_KEY="your_key_from_aistudio.google.com"
python scripts/demo_pipeline.py
```

**Optional — install FFmpeg for video trimming:**
```bash
brew install ffmpeg        # macOS
sudo apt install ffmpeg    # Ubuntu/Debian
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/api/v1/analyze` | Submit video for analysis |
| GET | `/api/v1/player/{id}` | Get player profile & rating |
| GET | `/api/v1/leaderboard/{sport}` | Venue leaderboard |
| POST | `/api/v1/calibrate` | Submit 4 court corner points |

Full interactive docs at **http://localhost:8000/docs** after `python main.py`.

---

## 7-Day Sprint Plan

| Day | Goal |
|---|---|
| **1** ✅ | Scaffold · CV modules · FastAPI · Gemini agents · demo pipeline |
| **2–3** | Record 3 real clips · tune homography · validate MediaPipe on 60fps |
| **4–5** | Antigravity skills · Gemini API integration · end-to-end pipeline |
| **6–7** | Next.js dashboard · Cloud Run deploy · Vercel frontend |

---

## Business Model

- **B2B Venue SaaS** — ₹4,999/court/month → "PlayQ Smart Arena"
- **B2C Match Pass** — ₹99/player (70:30 split with venue; ~84% gross margin)
- **Zero-CapEx Pod** — ₹6,000 refundable deposit covers ₹5,000 camera kit
- **Viral Loop** — every Nano Banana highlight reel carries PlayQ watermark

---

## License

MIT © 2026 PlayQ
