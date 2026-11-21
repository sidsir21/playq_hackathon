# PlayQ web (static)

Plain HTML/CSS/JS, **no build step and no backend**. It plays the pre-processed pipeline outputs in `demo/`.

```
web/
  index.html  styles.css  app.js
  assets/logo.png
  demo/<run>/  reel.mp4  data.json  heatmap_*.png  knee_frame.jpg  player_*.jpg
```

## Run locally
`python -m http.server 8620 -d web`  →  http://localhost:8620

## Refresh the demo data
After regenerating results in `data/cache/`: `python scripts/export_web.py` (re-encodes reels to 720p, strips raw
position tracks, copies only what the page needs; ~12 MB total).

## Host it
Upload the **contents of `web/`** to any static host: AI Studio / Cloud Run static, Netlify (drag-and-drop the folder),
GitHub Pages, Firebase Hosting. Open `index.html` through a server, not `file://` (the page fetches JSON).

## Before making the repo public
- `demo/` contains clips of third-party broadcast matches. Use a **private** repo or remove those runs (edit `RUNS` in `scripts/export_web.py`).
- Never commit `.env` or the service-account JSON (both are git-ignored / kept outside the repo).
- Live analysis (upload your own clip) needs the Python app (`streamlit run app/streamlit_app.py`); this site is the showcase.
