"""PlayQ web app — upload a match clip, get the reel, heatmap, skeleton frame and rating card."""
import base64
import glob
import io
import json
import os
import socket
import sys
import time
import urllib.parse

import qrcode
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from playq import config  # noqa: E402  (loads .env)
from playq.agents.schemas import MatchInsights  # noqa: E402
from playq.pipeline import analyze  # noqa: E402

CACHE = os.path.join(ROOT, "data", "cache")
DEMO_CLIP = os.path.join(ROOT, "data", "demo", "live_15s.mp4")
CALIBS = {
    "The Lab — indoor, fixed broadcast camera": os.path.join(ROOT, "config", "calibration", "the_lab.json"),
    "Night court — phone tripod": os.path.join(ROOT, "config", "calibration", "night_court.json"),
}
STAT_NAMES = {"COV": "Court Coverage", "CON": "Shot Consistency", "AGG": "Attack Aggression", "STA": "Stamina", "CLU": "Clutch Factor"}

st.set_page_config(page_title="PlayQ — AI sports intelligence", page_icon="🏓", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');
:root{--ball:#C6FF3D;--teal:#22D3A5;--ink:#0A0F1C;--panel:rgba(255,255,255,.035);--line:rgba(255,255,255,.09);--mute:#9FB0C7}
html,body,[class*="css"]{font-family:'Inter',system-ui,sans-serif}
header[data-testid="stHeader"],#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
.stApp{background:radial-gradient(1200px 500px at 85% -10%,rgba(34,211,165,.13),transparent 60%),radial-gradient(900px 500px at -10% 10%,rgba(198,255,61,.07),transparent 55%),#0A0F1C}
.block-container{padding-top:1.2rem;max-width:1380px}
/* hero */
.hero{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:22px;padding:26px 30px 22px;margin-bottom:18px;
 background:linear-gradient(120deg,rgba(34,211,165,.10),rgba(10,15,28,.6) 45%,rgba(198,255,61,.07)),#0d1424}
.hero:before{content:"";position:absolute;inset:0;opacity:.35;pointer-events:none;
 background:linear-gradient(var(--line) 1px,transparent 1px) 0 0/100% 46px,linear-gradient(90deg,var(--line) 1px,transparent 1px) 0 0/70px 100%;
 mask-image:linear-gradient(90deg,transparent,#000 40%,#000 60%,transparent)}
.brand{font-family:'Rajdhani',sans-serif;font-weight:700;font-size:3.3rem;letter-spacing:.04em;line-height:1;position:relative}
.brand b{color:var(--ball)}
.tag{color:#cfe0f5;font-size:1.08rem;margin:.35rem 0 .9rem;position:relative;max-width:820px}
.badges{position:relative;display:flex;flex-wrap:wrap;gap:8px}
.badge{border:1px solid var(--line);background:rgba(255,255,255,.05);border-radius:99px;padding:.22rem .75rem;font-size:.78rem;font-weight:600;color:#dbe7f7}
.badge i{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--ball);margin-right:7px;box-shadow:0 0 10px var(--ball)}
.live{position:absolute;right:26px;top:22px;font-family:'Rajdhani';font-weight:700;letter-spacing:.2em;font-size:.85rem;color:#0A0F1C;background:var(--ball);padding:.25rem .8rem;border-radius:6px}
/* section headings */
h2,h3,[data-testid="stHeading"] h3{font-family:'Rajdhani',sans-serif!important;letter-spacing:.03em;font-weight:700!important}
h3:after{content:"";display:block;width:46px;height:3px;border-radius:3px;margin-top:4px;background:linear-gradient(90deg,var(--ball),var(--teal))}
/* chips */
.pq-chip{display:inline-block;padding:.12rem .65rem;border-radius:99px;font-size:.76rem;font-weight:600;margin-right:.4rem;border:1px solid #2B3A52}
.ok{background:rgba(34,211,165,.12);color:#6EF3C5;border-color:rgba(34,211,165,.45)}
.warn{background:rgba(255,209,102,.1);color:#FFD166;border-color:rgba(255,209,102,.4)}
.off{background:#1a2232;color:#9FB0C7}
/* scoreboard tiles */
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:12px 16px;position:relative;overflow:hidden}
.kpi:after{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:linear-gradient(var(--ball),var(--teal))}
.kpi small{color:var(--mute);text-transform:uppercase;letter-spacing:.12em;font-size:.68rem;font-weight:600}
.kpi div{font-family:'Rajdhani',sans-serif;font-size:2.1rem;font-weight:700;line-height:1.05;color:#f4f8ff}
/* rating card */
.card{border-radius:20px;padding:18px 20px;background:linear-gradient(155deg,#1c2a44 0%,#0f1626 55%,#2a2140 100%);box-shadow:0 18px 44px #0009,inset 0 1px 0 rgba(255,255,255,.08)}
.card .ov{font-family:'Rajdhani',sans-serif;font-size:4.2rem;font-weight:700;line-height:.9}
.card .tier{letter-spacing:.2em;font-size:.7rem;font-weight:700}
.stat{display:flex;align-items:center;gap:10px;margin:8px 0}.stat b{width:44px;font-family:'Rajdhani';font-size:1.05rem;letter-spacing:.06em}
.bar{flex:1;height:10px;background:#1b2740;border-radius:6px;overflow:hidden}.bar i{display:block;height:100%;border-radius:6px}
.stat span{width:34px;text-align:right;font-family:'Rajdhani';font-weight:700;font-size:1.1rem}
/* player picker */
.pp{border:1px solid var(--line);border-radius:14px;overflow:hidden;background:var(--panel)}
.pp img{width:100%;height:128px;object-fit:cover;object-position:50% 12%;display:block}
.pp div{padding:6px 8px 7px;font-size:.78rem;text-align:center;border-bottom:3px solid var(--c,#888)}
/* media framing */
[data-testid="stVideo"] video,[data-testid="stImage"] img{border-radius:14px;border:1px solid var(--line)}
[data-testid="stExpander"]{border:1px solid var(--line)!important;border-radius:14px!important;background:var(--panel)}
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{font-family:'Rajdhani';font-weight:700;font-size:1.05rem;letter-spacing:.05em;padding:8px 16px;border-radius:10px 10px 0 0}
.stTabs [aria-selected="true"]{color:var(--ball)!important}
.stButton>button[kind="primary"]{background:linear-gradient(90deg,var(--ball),var(--teal));color:#07101a;font-weight:700;border:0;border-radius:12px;padding:.55rem 1.3rem}
</style>""", unsafe_allow_html=True)


# ── helpers ─────────────────────────────────────────────────────────────────
def chip(label: str, state: str) -> str:
    cls = {"gemini": "ok", "real": "ok", "google-cloud-tts": "ok", "edge-tts": "ok", "template": "warn", "fallback-say": "warn", "mock": "warn", "error": "warn"}.get(state, "off")
    icon = {"ok": "✓", "warn": "!", "off": "–"}[cls]
    return f'<span class="pq-chip {cls}">{icon} {label}: {state}</span>'


def lan_url() -> str:
    pub = os.environ.get("PLAYQ_PUBLIC_URL", "")
    f = os.path.join(ROOT, "data", "public_url.txt")          # written by scripts/share.sh while a tunnel is up
    if not pub and os.path.exists(f):
        pub = open(f).read().strip()
    if pub:
        return pub
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = "localhost"
    return f"http://{ip}:8501"


def qr_png(url: str) -> bytes:
    buf = io.BytesIO()
    qrcode.make(url).save(buf, format="PNG")
    return buf.getvalue()


def thumb_b64(d: str, name: str | None) -> str:
    if not name or not os.path.exists(os.path.join(d, name)):
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(open(os.path.join(d, name), "rb").read()).decode()


def card_html(card: dict, d: str = "") -> str:
    est = set(card.get("gemini_estimated", []))
    rows = ""
    for k, v in card["stats"].items():
        ph = k in card.get("placeholder_stats", [])
        colour = "#6b7a93" if ph else ("#22D3A5" if v >= 65 else "#FFD166" if v >= 40 else "#FF6B6B")
        tag = " (not measured)" if ph else " (AI est.)" if k in est else ""
        rows += (f'<div class="stat" title="{STAT_NAMES[k]}{tag}"><b>{k}</b><div class="bar"><i style="width:{v}%;background:{colour}"></i></div>'
                 f'<span>{v}{"*" if ph or k in est else ""}</span></div>')
    accent = card.get("color_hex", "#3b4a6b")
    img = thumb_b64(d, card.get("thumb"))
    photo = f'<img src="{img}" alt="{card["player_name"]}" style="height:96px;border-radius:10px;border:2px solid {accent};margin-right:14px">' if img else ""
    return (f'<div class="card" role="group" aria-label="PlayQ rating card for {card["player_name"]}" style="border:2px solid {accent}">'
            f'<div class="tier" style="color:{accent}">PLAYQ · {card["tier"].upper()} · {str(card.get("side") or "").upper()} COURT</div>'
            f'<div style="display:flex;align-items:center;margin-top:6px">{photo}<div style="flex:1"><div style="font-size:1.35rem;font-weight:800">'
            f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{accent};margin-right:8px"></span>{card["player_name"]}</div></div>'
            f'<div class="ov" style="color:{card["tier_color"]}">{card["overall"]}</div></div><hr style="border-color:#2B3A52">{rows}'
            f'<div style="color:#9FB0C7;font-size:.75rem;margin-top:8px">* AI-estimated or not yet measurable · COV, AGG, STA measured from tracking</div></div>')


def render_result(res: dict, d: str) -> None:
    f, tm, ai = res["funnel"], res["timings"], res["ai"]
    k = st.columns(5)
    for col, (label, val) in zip(k, [
        ("Source video", f'{f["source_seconds"]:.0f}s'), ("Rallies found", f'{f["rallies"]}'),
        ("Reel length", f'{res["reel"].get("seconds", f["reel_seconds"]):.0f}s'),
        ("Pipeline time", f'{tm["total_s"]:.0f}s'), ("Players tracked", f'{len(res["players"])}')]):
        col.markdown(f'<div class="kpi"><small>{label}</small><div>{val}</div></div>', unsafe_allow_html=True)
    src = ai["sources"]
    st.markdown(chip("vision", src["vision"]) + chip("scores & script", src["insights"]) + chip("voice", src["voice"]), unsafe_allow_html=True)
    st.write("")

    players = res["players"]
    key = "sel_" + os.path.basename(d)
    if players and st.session_state.get(key) not in [p["track_id"] for p in players]:
        st.session_state[key] = players[0]["track_id"]

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("🎬 Auto-trimmed highlight reel")
        st.video(os.path.join(d, res["reel"]["file"]))
        tl = res["reel"].get("timeline") or []
        segs = " · ".join(f'{t["source_start_s"]:.0f}–{t["source_end_s"]:.0f}s' for t in tl) or ", ".join(f"{a:.0f}–{b:.0f}s" for a, b in res["reel"]["segments"])
        st.caption(f"Best action from the source ({segs}); cuts land between shots. 🔊 Turn sound on: two commentators over the match audio.")
        ins = ai.get("insights") or {}
        lines = ins.get("hinglish_commentary") or []
        voice = ai.get("voice") or {}
        if lines:
            origin = "Gemini" if ins.get("_source") == "gemini" else "template (Gemini unavailable)"
            with st.expander(f"🎙️ Commentary script — {origin}", expanded=True):
                placed = {round(t["start_s"], 1): t for t in voice.get("timing", [])}
                st.caption("🅰️ Madhur (hype) · 🅱️ Swara (witty) — times are on the reel; a line shown struck-through did not fit and was not spoken.")
                dropped = set(voice.get("dropped") or [])
                for l in lines:
                    who = "🅰️" if l.get("speaker") == "A" else "🅱️"
                    txt = l.get("text_roman", l.get("text", ""))
                    shown = f"~~{txt}~~" if txt in dropped else txt
                    st.markdown(f"`{l['start_s']:4.1f}s` {who} {shown}")
                if ins.get("instagram_caption"):
                    st.markdown(f"**Caption:** {ins['instagram_caption']}")
                if ins.get("roast_line"):
                    st.markdown(f"**Roast:** {ins['roast_line']}")
    with right:
        st.subheader("⭐ PlayQ Rating Card")
        if players:
            cols = st.columns(len(players))
            for col, p in zip(cols, players):
                with col:
                    sel = st.session_state[key] == p["track_id"]
                    img = thumb_b64(d, p.get("thumb"))
                    st.markdown(f'<div class="pp" style="--c:{p.get("color_hex", "#888")}">' + (f'<img src="{img}" alt="{p.get("label", "player")}">' if img else "")
                                + f'<div><b>{p.get("label", "Player")}</b><br><span style="color:#9FB0C7">{p["movement"].get("side", "")} court</span></div></div>', unsafe_allow_html=True)
                    if st.button("✓ Selected" if sel else "Select", key=f"b_{key}_{p['track_id']}", disabled=sel, width="stretch"):
                        st.session_state[key] = p["track_id"]
                        st.rerun()
            cur = next(p for p in players if p["track_id"] == st.session_state[key])
            st.markdown(card_html(cur["card"], d), unsafe_allow_html=True)
            if cur["card"].get("reason"):
                st.caption(f'AI note: {cur["card"]["reason"]}')
            mv = cur["movement"]
            st.caption(f'Measured: plays on average {mv.get("mean_dist_to_net_m", "–")} m from the net · {mv.get("net_zone_share", 0):.0%} of the time within 3 m of it · '
                       f'avg speed {mv.get("avg_speed_mps", "–")} m/s.')
        else:
            st.info("No player tracked long enough for a card.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("🗺️ 2D movement heatmap")
        show = st.radio("Show", ["Everyone"] + [p["label"] for p in players], horizontal=True, label_visibility="collapsed", key="hm_" + key)
        fn = res["team_heatmap"] if show == "Everyone" else next(p["heatmap"] for p in players if p["label"] == show)
        st.image(os.path.join(d, fn), width=400)
        st.caption("Top-down court, net at the middle line; each player has their own colour. Positions come from a 4-point homography "
                   "plus camera-drift correction. Accuracy: roughly ±0.2 m on the near half, ±0.5–1 m on the far half "
                   "(farther from the camera = fewer pixels per metre).")
    with c2:
        st.subheader("🦴 Biomechanics — knee load")
        kn = res.get("knee")
        if kn:
            st.image(os.path.join(d, kn["image"]), caption="Deepest planted-foot knee flexion during live play, from MediaPipe's 33-point pose.")
            v = (ai.get("vision") or {})
            if v.get("_source") == "gemini":
                st.markdown(f"**Gemini review:** risk **{v['injury_risk_level']}** — {v['coaching_cue']}")
                if not v.get("agrees_with_measurement", True):
                    st.warning("Gemini thinks the pose may be mis-detected in this frame — treat the angle with caution.")
            st.caption("Valgus is only reported when the player faces/turns away from the camera; otherwise it's shown as not visible. "
                       "Risk bands are heuristic, not medical advice.")
        else:
            st.info("No loaded-lunge moment found in this clip.")


# ── layout ──────────────────────────────────────────────────────────────────
# ── optional access control for a shared/public link ───────────────────────────
PUBLIC = os.environ.get("PLAYQ_PUBLIC_MODE", "") == "1"      # no uploads, capped live runs
_PW = os.environ.get("PLAYQ_DEMO_PASSWORD", "")
if _PW and not st.session_state.get("authed"):
    st.markdown('<div class="hero"><div class="brand">PLAY<b>Q</b></div><div class="tag">Private demo. Enter the access code you were given.</div></div>', unsafe_allow_html=True)
    code = st.text_input("Access code", type="password", label_visibility="collapsed", placeholder="Access code")
    if code:
        if code == _PW:
            st.session_state["authed"] = True
            st.rerun()
        st.error("Wrong code.")
    st.stop()

st.markdown(f'''<div class="hero">
  <div class="live">● LIVE DEMO</div>
  <div class="brand">PLAY<b>Q</b></div>
  <div class="tag">Every rally, analysed. Broadcast-style highlights with Hinglish commentary, 2D movement heatmaps, injury-risk biomechanics
  and a FIFA-style rating, from a single match video.</div>
  <div class="badges"><span class="badge"><i></i>Gemini multimodal</span><span class="badge"><i></i>MediaPipe pose</span>
  <span class="badge"><i></i>OpenCV homography</span><span class="badge"><i></i>Google Cloud TTS</span><span class="badge"><i></i>Court-side QR · ₹99</span></div>
</div>''', unsafe_allow_html=True)
tab_run, tab_arch, tab_share = st.tabs(["Analyze", "Architecture & cost", "Share & unlock"])

with tab_run:
    mode = st.radio("Mode", ["⚡ Live run (real-time)", "📦 Pre-processed full match"], horizontal=True)
    if mode.startswith("⚡"):
        c1, c2 = st.columns([2, 1])
        with c1:
            up = None if PUBLIC else st.file_uploader("Upload a rally clip (mp4, 1080p)", type=["mp4", "mov"], help="Fixed-camera footage of a venue we have calibrated.")
            if PUBLIC:
                st.caption("Public demo: uploads are disabled. Run the built-in rally, or open a pre-processed match.")
            use_demo = st.checkbox("Use the built-in 15-second demo rally", value=up is None and os.path.exists(DEMO_CLIP))
        with c2:
            venue = st.selectbox("Venue / camera profile", list(CALIBS))
            use_ai = st.toggle("Use Gemini", value=bool(config.api_key()), help="Off = pure computer vision, no API calls.")
            if use_ai and not config.api_key():
                st.warning("No GEMINI_API_KEY in .env — AI steps will fall back and be labelled as such.")
        if PUBLIC and st.session_state.get("runs", 0) >= 5:
            st.warning("Live-run limit reached for this session. Pre-processed matches are unlimited.")
        if st.button("▶ Analyze match", type="primary", disabled=not (up or use_demo) or (PUBLIC and st.session_state.get("runs", 0) >= 5)):
            st.session_state["runs"] = st.session_state.get("runs", 0) + 1
            tag = time.strftime("live_%Y%m%d_%H%M%S")
            out = os.path.join(CACHE, tag)
            os.makedirs(out, exist_ok=True)
            if up:
                src = os.path.join(out, "upload.mp4")
                with open(src, "wb") as fh:
                    fh.write(up.getbuffer())
            else:
                src = DEMO_CLIP
            labels = {"audio": "1/5 · Audio spike filter", "cv": "2/5 · Court homography + MediaPipe pose", "render": "3/5 · Heatmaps",
                      "reel": "4/5 · Cutting highlight reel", "ai": "5/5 · Gemini analysis + voiceover", "done": "Done"}
            with st.status("Analyzing…", expanded=True) as status:
                bar = st.progress(0.0)
                def cb(stage, frac, msg):
                    status.update(label=labels.get(stage, stage))
                    bar.progress(min(1.0, frac))
                    st.write(msg) if frac in (0.0, 1.0) or "Gemini" in msg else None
                try:
                    res = analyze(src, CALIBS[venue], out, use_gemini=use_ai, progress=cb)
                    status.update(label=f"Done in {res['timings']['total_s']:.0f}s", state="complete")
                    st.session_state["live"] = (res, out)
                except Exception as e:
                    status.update(label="Failed", state="error")
                    st.error(f"Analysis failed: {type(e).__name__}: {e}")
        if "live" in st.session_state:
            render_result(*st.session_state["live"])
    else:
        runs = sorted(p for p in glob.glob(os.path.join(CACHE, "*", "result.json")) if "/live_" not in p)
        if not runs:
            st.info("No pre-processed runs yet. Run `python scripts/run_pipeline.py Tes.mp4 config/calibration/the_lab.json data/cache/full_match`.")
        else:
            labels = {os.path.basename(os.path.dirname(p)): p for p in runs}
            pick = st.selectbox("Pre-processed result", list(labels))
            d = os.path.dirname(labels[pick])
            res = json.load(open(labels[pick]))
            st.caption(f"Processed {res['meta']['analyzed_at']} · {res['meta']['video']} · {res['meta']['duration_s']/60:.1f} min · "
                       f"{res['timings']['total_s']:.0f}s of compute. Loaded instantly from cache.")
            st.session_state["live"] = (res, d)
            render_result(res, d)

with tab_arch:
    last = st.session_state.get("live")
    st.subheader("Edge-to-cloud funnel: Gemini never sees the raw match")
    if last:
        f, tm = last[0]["funnel"], last[0]["timings"]
        st.graphviz_chart(f'''digraph G {{ rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#131A26", color="#2B3A52", fontcolor="#E8EDF5", fontname="Helvetica"]; edge [color="#22D3A5", fontcolor="#9FB0C7", fontname="Helvetica"];
 A [label="Match video\\n{f["source_seconds"]:.0f}s · 1080p"];
 B [label="1 · Audio spike filter (CPU)\\n{f["rallies"]} rallies · {f["impacts"]} ball impacts\\n{tm["audio_filter_s"]}s"];
 C [label="2 · OpenCV homography + MediaPipe\\n{f["frames_sampled"]} frames sampled of {f["frames_total"]}\\n{tm["cv_s"]}s · deterministic"];
 D [label="3 · FFmpeg highlight reel\\n{f["reel_seconds"]:.0f}s\\n{tm["reel_s"]}s"];
 E [label="4 · Gemini (multimodal)\\n1 frame + {f["sent_to_gemini"]["video_seconds"]:.0f}s @480p + JSON\\n{tm["gemini_s"]}s"];
 F [label="5 · TTS voiceover + FFmpeg mux\\nPlayable Hinglish reel"];
 A -> B -> C -> D -> E -> F; }}''')
        st.markdown(f"""
| Stage | What runs | Measured |
|---|---|---|
| Audio filter | scipy band-pass + transient detector, FFmpeg audio extract | **{tm['audio_filter_s']}s** → {f['rallies']} rallies, {f['active_seconds']:.0f}s active |
| Computer vision | OpenCV ORB stabiliser + 4-point homography, MediaPipe Pose, tracker | **{tm['cv_s']}s** on {f['frames_sampled']} frames ({f['frames_skipped_replay']} replay/cut frames skipped) |
| Reel | FFmpeg cut + concat | **{tm['reel_s']}s** → {f['reel_seconds']:.0f}s reel |
| Gemini | vision + video + TTS | **{tm['gemini_s']}s** |
| **Total** | | **{tm['total_s']}s** |

**Video seconds reaching Gemini: {f['sent_to_gemini']['video_seconds']:.0f}s of {f['source_seconds']:.0f}s ({f['video_seconds_reduction_pct']}% fewer).**
""")
        usage = (last[0]["ai"].get("usage") or {})
        if usage.get("calls"):
            st.markdown(f"**Gemini cost for this run: ≈ ${usage['est_cost_usd']:.3f}** "
                        f"({usage['input_tokens']:,} tokens in, {usage['output_tokens']:,} out). "
                        + " · ".join(f"{c['step']}: {c['model']} ${c['usd']:.4f}" for c in usage["calls"])
                        + ". Prices from the AI Studio pricing page; the racing of two models on the video call is counted twice.")
        vis = f.get("visual_check")
        if vis:
            st.info(f"**Why a visual check?** Tournament audio (other courts, crowd) made the audio filter call "
                    f"{f.get('audio_only_active_seconds', 0):.0f}s of this match 'active'. Checking that a ball is actually *flying* "
                    f"inside this court cut that to {f['active_seconds']:.0f}s of real live play "
                    f"(ball-colour detection inside the calibrated court region, {vis['profile_s']:.0f}s of compute).")
        if f["active_seconds"] / max(f["source_seconds"], 1) > 0.8 and f["source_seconds"] > 120:
            st.info("The audio filter keeps most of this footage as active play because the source is already a pre-edited broadcast. "
                    "On raw venue footage (with dead time between points) it removes far more; the biggest saving here comes from "
                    "sending only a ~28s reel instead of the full match.")
    else:
        st.info("Run or load an analysis to see the measured funnel for that run.")
    st.subheader("Pydantic contracts for the Gemini step")
    st.caption("Gemini must return JSON matching this schema (validated server-side).")
    st.json(MatchInsights.model_json_schema(), expanded=False)
    st.markdown(f"**Models:** `{config.VIDEO_MODEL}` (vision + video) · TTS model auto-detected · **MediaPipe Pose** on-device · "
                "FastAPI service with Swagger at `/docs` (`uvicorn playq.api.main:app`).")

with tab_share:
    st.subheader("Court-side QR → match pack")
    url = lan_url()
    a, b = st.columns([1, 2], gap="large")
    with a:
        st.image(qr_png(url), caption=f"Scan on the same Wi-Fi → {url}", width=220)
    with b:
        st.markdown("**Match Pack · ₹99** — HD highlight reel with Hinglish commentary · 2D heatmap · knee-load audit · PlayQ Rating Card")
        if st.button("Pay ₹99 with UPI (demo)", type="primary"):
            with st.spinner("Contacting UPI app…"):
                time.sleep(1.2)
            st.session_state["paid"] = True
        if st.session_state.get("paid"):
            st.success("✅ Demo payment received — no real transaction took place.")
            last = st.session_state.get("live")
            ovr = last[0]["players"][0]["card"]["overall"] if last and last[0]["players"] else "—"
            msg = f"🏓 My PlayQ match pack is ready! Rating {ovr}. Watch my highlights 🔥 #PlayQ"
            st.link_button("Share on WhatsApp", "https://wa.me/?text=" + urllib.parse.quote(msg))
            st.code(msg, language=None)
        st.caption("Prototype: the QR opens this app on your local network; playq.in hosting and real UPI are not wired up.")
