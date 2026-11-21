"""
PlayQ end-to-end pipeline (the one function the web app and the CLI both call).

Cost/latency funnel — Gemini never sees the raw match:
  Stage 1  Audio filter (librosa-free: scipy + FFmpeg)  : whole video -> rally windows
  Stage 2  Deterministic CV (OpenCV + MediaPipe)         : homography, tracking, joint angles, heatmaps
  Stage 3  Highlight reel (FFmpeg)                       : ~28 s of the best rallies
  Stage 4  Gemini chain                                  : 1 frame + 480p reel + JSON metrics only
  Stage 5  Voiceover (Gemini TTS) muxed onto the reel

Everything measured here is deterministic; Gemini only judges what needs vision/language and the
result says which parts were real (`ai.sources`).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import time
from datetime import datetime
from typing import Callable, Optional

import cv2
import numpy as np

from playq import config
from playq.agents.gemini_agents import RallyAgent
from playq.agents.prototype import PrototypeAI, mux_voiceover, synthesize_dialogue
from playq.cv.annotate import annotate_knee_frame
from playq.cv.homography import CameraStabilizer, CourtHomography, PIXELS_PER_METER
from playq.cv.identity import accent, color_name, player_crop, torso_color
from playq.cv.knee import knee_metrics, plausible_leg
from playq.cv.movement import movement_metrics
from playq.cv.pose_analyzer import PoseAnalyzer
from playq.cv.rally_detector import RallyDetector, _ffmpeg_exe
from playq.cv.tracker import PlayerTracker
from playq.models.rating import PlayerProfile, RatingEngine, Sport

Progress = Callable[[str, float, str], None]
MIN_STABILISER_INLIERS = 40   # below this the frame is a replay / cut / dissolve and is skipped


def _noop(stage: str, frac: float, msg: str) -> None:
    pass


# ── helpers ─────────────────────────────────────────────────────────────────
def far_half_roi(court: CourtHomography, to_ref, W: int, H: int) -> tuple[int, int, int, int]:
    """Image-space box around the far half of the court (+ headroom) in the CURRENT frame."""
    wm, hm = court.court_dims
    ground = np.float32([[x * 100, y * 100] for x, y in ((0, 0), (wm, 0), (wm, hm / 2), (0, hm / 2))])
    pts = cv2.perspectiveTransform(ground.reshape(-1, 1, 2), court._inv_matrix).reshape(-1, 2)
    if to_ref is not None:
        pts = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), np.linalg.inv(to_ref)).reshape(-1, 2)
    x0, y0, x1, y1 = pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()
    pad = 0.15 * (x1 - x0)
    return int(x0 - pad), int(y0 - 0.9 * (y1 - y0) - 120), int(x1 + pad), int(y1 + 20)


def _run_ffmpeg(args: list[str]) -> None:
    r = subprocess.run([_ffmpeg_exe(), "-y", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-400:]}")


def reel_budget(duration: float, active: float) -> float:
    """Target reel length: most of the action for short sources, a ~35 s teaser for long matches."""
    if duration <= 150:
        return float(min(45.0, max(25.0, 0.85 * active)))
    return 35.0


def plan_reel(hits: list[tuple[float, float]], rallies, duration: float, budget: float, seg_len: float = 10.0):
    """
    Choose the most action-dense windows of the footage until `budget` seconds are filled.
    - windows of seg_len seconds slide (1 s step) inside each rally and are ranked by ball impacts;
    - each chosen window is trimmed to 'first impact - 1.2 s' .. 'last impact + 1.6 s' so cuts land between
      shots, never mid-swing;
    - back-to-back windows are merged so a long rally plays continuously.
    """
    htimes = [h for h, _ in hits]
    cands = []
    for r in rallies:
        L = min(seg_len, r.end_time - r.start_time)
        t = r.start_time
        while t + L <= r.end_time + 1e-6:
            cands.append((sum(1 for h in htimes if t <= h < t + L), t, t + L))
            t += 1.0
    cands.sort(key=lambda c: -c[0])
    chosen, total = [], 0.0
    for n, a, b in cands:
        if total >= budget:
            break
        if n == 0 or any(a < cb and b > ca for _, ca, cb in chosen):
            continue
        inside = [h for h in htimes if a <= h < b]
        a2, b2 = max(a, min(inside) - 1.2), min(b, duration, max(inside) + 1.6)
        chosen.append((n, a2, b2))
        total += b2 - a2
    segs = sorted((a, b) for _, a, b in chosen)
    merged: list[list[float]] = []
    for a, b in segs:
        in_same_rally = merged and any(r.start_time <= merged[-1][1] and r.end_time >= a for r in rallies)
        if merged and (a - merged[-1][1] < 1.0 or (a - merged[-1][1] < 4.0 and in_same_rally)):
            merged[-1][1] = max(merged[-1][1], b)          # play straight through small gaps inside one rally
        else:
            merged.append([a, b])
    # spend any leftover budget by extending segments toward the edges of the rally they sit in
    used = sum(b - a for a, b in merged)
    for seg in merged:
        rally = next((r for r in rallies if r.start_time - 0.5 <= seg[0] and seg[1] <= r.end_time + 0.5), None)
        if rally is None or used >= budget:
            continue
        for_end = min(budget - used, rally.end_time - seg[1])
        if for_end > 0:
            seg[1] += for_end
            used += for_end
        for_start = min(budget - used, seg[0] - rally.start_time)
        if for_start > 0:
            seg[0] -= for_start
            used += for_start
    # extension may make neighbours touch: re-merge
    out: list[list[float]] = []
    for a, b in sorted(merged):
        if out and a <= out[-1][1] + 0.3:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out] or [(0.0, min(duration, budget))]


def build_reel(video: str, segments: list[tuple[float, float]], out: str) -> str:
    """Cut + concatenate segments at 1080p (re-encoded identically so concat can stream-copy)."""
    tmp = os.path.dirname(out)
    parts = []
    for i, (a, b) in enumerate(segments):
        p = os.path.join(tmp, f"_seg{i}.mp4")
        _run_ffmpeg(["-ss", f"{a:.2f}", "-i", video, "-t", f"{b - a:.2f}", "-vf", "scale=1920:1080,fps=30",
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
                     "-c:a", "aac", "-ar", "44100", "-ac", "2", p])
        parts.append(p)
    lst = os.path.join(tmp, "_concat.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{os.path.abspath(p)}'\n" for p in parts)
    _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", out])
    for p in parts + [lst]:
        os.remove(p)
    return out


def knee_level(flexion: float, valgus: Optional[float]) -> str:
    """Heuristic bands for a loaded pickleball lunge (needs physio validation)."""
    if flexion >= 105 or (valgus is not None and valgus >= 20):
        return "high"
    if flexion >= 80 or (valgus is not None and valgus >= 12):
        return "elevated"
    return "normal"


def _blend_aggression(position_score, shot_style_0_99):
    """Aggression = where you play (measured: distance to the net) + how you play (Gemini: drives/smashes vs dinks)."""
    if position_score is None:
        return None if shot_style_0_99 is None else shot_style_0_99 / 100
    if shot_style_0_99 is None:
        return position_score
    return 0.5 * position_score + 0.5 * (shot_style_0_99 / 100)


def template_commentary(top: dict, reel_s: float, timeline: list[dict] | None = None) -> list[dict]:
    """Rule-based Hinglish used only when Gemini is unavailable (flagged _commentary_source='template')."""
    lines = [
        ("A", "Arre wah! Kya rally shuru hui hai!", "अरे वाह! क्या rally शुरू हुई है!"),
        ("B", "Dheere dheere, abhi toh intezaar baaki hai.", "धीरे धीरे, अभी तो इंतज़ार बाकी है."),
        ("A", "Kitchen pe dink pe dink, kya mukabla hai!", "kitchen पे dink पे dink, क्या मुक़ाबला है!"),
        ("B", "Dono taraf se full josh, point kiska hoga?", "दोनों तरफ़ से full जोश, point किसका होगा?"),
        ("A", "PlayQ pe poora match dekho, tez aur sahi!", "PlayQ पे पूरा match देखो, तेज़ और सही!"),
    ]
    n_seg = max(1, len(timeline or [1]))
    out = []
    for i, (spk, roman, speech) in enumerate(lines):
        start = 0.4 + i * max(0.0, reel_s - 6.0) / max(1, len(lines) - 1)         # spread evenly across the reel
        seg = next((t["segment"] for t in (timeline or []) if t["reel_start_s"] <= start < t["reel_end_s"]), n_seg - 1)
        out.append({"start_s": round(start, 1), "speaker": spk, "text_roman": roman, "speech": speech, "emotion": "excited", "segment": seg})
    out.sort(key=lambda l: l["start_s"])
    return out


# ── main entry ─────────────────────────────────────────────────────────────
def analyze(
    video_path: str,
    calib_path: str,
    out_dir: str,
    *,
    sport: str = "pickleball",
    sample_hz: float = 6.0,
    reel_seconds: float | None = None,
    use_gemini: bool = True,
    progress: Progress = _noop,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    t_all = time.time()
    timings: dict[str, float] = {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    cfg = json.load(open(calib_path))

    # Court calibration + stabiliser (needed by the visual play detector too) ----------
    court = CourtHomography(sport=sport)
    court.calibrate_landmarks([l["px"] for l in cfg["landmarks"]], [l["court_m"] for l in cfg["landmarks"]])
    ref_idx = cfg.get("reference_frame", int(round(cfg.get("reference_time_s", 0.0) * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
    ok, ref = cap.read()
    stabilizer = CameraStabilizer(ref) if ok else None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Optional: blank out everything outside this court before pose detection. At tournaments the frame also holds
    # spectators and other courts; with a cap of N people per frame they crowd out the real players.
    court_mask = None
    if cfg.get("mask_outside_court"):
        from playq.cv.ball_activity import court_roi_mask
        court_mask = court_roi_mask(court, W, H, margin_m=1.2, headroom_px=int(cfg.get("headroom_px", 80)))

    # Stage 1 — when is the ball in play on OUR court? --------------------------------
    t0 = time.time()
    progress("audio", 0.0, "Listening for ball impacts (audio spike detection)…")
    from playq.cv.rally_detector import RallySegment
    detector = RallyDetector(sport=sport)
    hits = detector.detect_hits(video_path)
    audio_rallies = detector.detect_rallies(video_path)
    audio_active_s = sum(r.duration for r in audio_rallies)
    timings["audio_filter_s"] = round(time.time() - t0, 1)
    visual = None
    rallies = audio_rallies
    if cfg.get("visual_play_detection"):
        # Tournament audio (other courts, crowd) makes audio-only detection call ~85 % of a match "active".
        # Confirm with the picture: is a ball-coloured object FLYING inside this court's region?
        t0 = time.time()
        progress("audio", 0.5, "Checking the ball is actually in play on this court (vision)…")
        from playq.cv.ball_activity import ball_motion_profile, live_play_segments
        starts, act = ball_motion_profile(video_path, court)
        segs_v = live_play_segments(starts, act)
        rallies = [RallySegment(a_, b_, b_ - a_, 0.0, float(min(1.0, np.mean(act[(starts >= a_) & (starts < b_)]) * 4)), "rally")
                   for a_, b_ in segs_v]
        # reel scoring must not use audio pops (other courts): use ball-flight density instead
        hits = [(float(t_ + 1.0), float(a)) for t_, a in zip(starts, act) for _ in range(int(round(a * 10)))]
        visual = {"profile_s": round(time.time() - t0, 1), "segments": len(rallies)}
        timings["ball_activity_s"] = visual["profile_s"]
    if not rallies:   # short / quiet clip: treat everything as live play
        rallies = [RallySegment(0.0, duration, duration, 0.0, 0.3, "rally")]
    active_s = sum(r.duration for r in rallies)

    def in_rally(t: float) -> bool:
        return any(r.start_time <= t <= r.end_time for r in rallies)

    # Stage 2 — deterministic CV ---------------------------------------------
    t0 = time.time()
    analyzer = PoseAnalyzer(sport=sport)
    tracker = PlayerTracker(court, max_players=4)
    stride = max(1, int(round(fps / sample_hz)))
    net_y = court.court_dims[1] / 2
    cands: list[dict] = []   # loaded-knee candidates (top-K kept in stage2.json)
    appearance: dict[int, dict] = {}   # per track: shirt colours seen + best player crop
    snapshots: list[np.ndarray] = []   # contact sheet to eyeball ID assignment
    n_assigned = 0
    n_samples = n_skipped = 0
    est_samples = max(1, int(active_s * sample_hz))
    i = 0
    while True:
        t = i / fps
        if i % stride != 0 or not in_rally(t):
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        n_samples += 1
        to_ref = stabilizer.to_reference(frame) if stabilizer else None
        if stabilizer and stabilizer.last_inliers < MIN_STABILISER_INLIERS:
            n_skipped += 1          # replay / cut / dissolve: positions would be wrong
            i += 1
            continue
        pose_frame_img = cv2.bitwise_and(frame, frame, mask=court_mask) if court_mask is not None else frame
        poses = analyzer.analyze_frame_all(pose_frame_img, i, t * 1000.0)
        if n_samples % 2 == 0:   # far-court pass on alternate samples (players there move slowly)
            x0, y0, x1, y1 = far_half_roi(court, to_ref, W, H)
            w3 = x1 - x0
            for k in range(3):   # overlapping strips: one crop finds ~1 person, strips find 3-4
                strip = (x0 + int(k * w3 / 3) - int(0.08 * w3), y0, x0 + int((k + 1) * w3 / 3) + int(0.08 * w3), y1)
                for fp in analyzer.analyze_roi_all(pose_frame_img, strip, scale=3.0, frame_index=i, timestamp_ms=t * 1000.0):
                    if fp.torso_px > 0.11 * H:   # far-court players are small; a big person here is a NEAR-court player
                        continue                  # seen with his feet cut off by the strip (position would be wrong)
                    c = np.mean([(fp.landmarks[q][0], fp.landmarks[q][1]) for q in (11, 12, 23, 24)], axis=0)
                    if all(np.hypot(*(c - np.mean([(o.landmarks[q][0], o.landmarks[q][1]) for q in (11, 12, 23, 24)], axis=0))) > 0.03
                           for o in poses):   # skip people already found
                        poses.append(fp)
        for tid, pose, xy in tracker.update(frame, poses, t, to_ref):
            app = appearance.setdefault(tid, {"colors": [], "thumb": None, "score": 0.0})
            if abs(xy[1] - net_y) > 1.8:     # only learn what a player looks like when they are clearly on their own side
                col = torso_color(frame, pose.landmarks)
                if col and len(app["colors"]) < 400:
                    app["colors"].append(col)
                pc = player_crop(frame, pose.landmarks)
                if pc and pc[1] > app["score"]:
                    app["thumb"], app["score"] = pc
            if n_assigned % 40 == 0 and len(snapshots) < 8:
                pass  # marker: snapshot drawn after the loop over assigned people (below)
            km = knee_metrics(pose.landmarks, W, H)
            if km["torso_px"] < 110 or not km["legs"]:
                continue
            side, leg = max(km["legs"].items(), key=lambda kv: kv[1]["flexion"])
            if leg["flexion"] > 150 or not plausible_leg(pose.landmarks, W, H, side):
                continue                  # implausible skeleton => pose glitch, not a lunge
            # A LUNGE, not a stride: the flexed leg's foot must be planted (not kicked up behind) and the
            # player must be nearly stationary. Otherwise we'd flag a running knee-flick as "knee load".
            lm = pose.landmarks
            ank, other = (27, 28) if side == "left" else (28, 27)
            if lm[ank][1] < lm[other][1] - 0.25 * km["torso_px"] / H:
                continue
            tr_ = next(x for x in tracker.tracks if x.id == tid)
            if len(tr_.samples) >= 2:
                (ta, xa, ya), (tb, xb, yb) = tr_.samples[-2], tr_.samples[-1]
                if tb - ta > 0 and math.hypot(xb - xa, yb - ya) / (tb - ta) > 1.2:
                    continue
            near_net = abs(xy[1] - net_y) <= 3.5
            cands.append({"score": leg["flexion"] + (12 if near_net else 0), "flexion": leg["flexion"], "valgus": leg["valgus"],
                          "side": side, "track_id": tid, "frame_idx": i, "t": round(t, 2),
                          "court_xy_m": [round(xy[0], 2), round(xy[1], 2)], "near_kitchen": bool(near_net),
                          "frontal": bool(km["frontal"]), "landmarks": [list(map(float, l)) for l in pose.landmarks]})
            cands = sorted(cands, key=lambda c: -c["score"])[:60]
        assigned_now = [(tid, pose, xy) for tid, pose, xy in tracker.last_assigned]
        if assigned_now:
            n_assigned += 1
            if n_assigned % 45 == 1 and len(snapshots) < 8:
                snap = frame.copy()
                for tid, pose, xy in assigned_now:
                    ax = int((pose.landmarks[27][0] + pose.landmarks[28][0]) / 2 * W)
                    ay = int((pose.landmarks[27][1] + pose.landmarks[28][1]) / 2 * H)
                    col = accent(tid)[1]
                    cv2.circle(snap, (ax, ay), 12, col, -1)
                    cv2.putText(snap, f"P{tid} ({xy[0]:.1f},{xy[1]:.1f})", (ax - 60, ay + 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
                cv2.putText(snap, f"t={t:.1f}s", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
                snapshots.append(cv2.resize(snap, (960, 540)))
        if n_samples % 15 == 0:
            progress("cv", min(0.99, n_samples / est_samples), f"Tracking players & joints… {n_samples}/{est_samples}")
        i += 1
    cap.release()
    timings["cv_s"] = round(time.time() - t0, 1)

    # Players: heatmaps, movement, biomechanics ---------------------------------
    progress("render", 0.0, "Rendering heatmaps…")
    merged_tracks = tracker.merge_fragments()   # one person split over several IDs -> one track
    players = []
    wm, hm = court.court_dims
    windows = [(r.start_time, r.end_time) for r in rallies]
    for tr in [t for t in tracker.tracks if t.n >= 12]:
        mm = movement_metrics(tr.samples, wm, hm, rally_windows=windows)
        bio = analyzer.generate_biomechanics_report(tr.poses)
        app = appearance.get(tr.id, {"colors": [], "thumb": None})
        votes: dict[str, int] = {}
        for c_ in app["colors"]:
            votes[color_name(c_)] = votes.get(color_name(c_), 0) + 1
        shirt = max(votes, key=votes.get) if votes else "unknown"
        # ID purity: how many of this track's samples look like its majority shirt colour. A track that mixes two
        # people (an ID swap) scores low. Measured proxy for "ID lock", not ground truth.
        base_ = shirt.replace("dark ", "")
        purity = (sum(n for k_, n in votes.items() if k_.replace("dark ", "") == base_) / sum(votes.values())) if votes else 0.0
        hexc, _ = accent(tr.id)
        thumb = None
        if app["thumb"] is not None:
            thumb = f"player_{tr.id}.jpg"
            cv2.imwrite(os.path.join(out_dir, thumb), app["thumb"], [cv2.IMWRITE_JPEG_QUALITY, 90])
        players.append({"track_id": tr.id, "samples": tr.n, "movement": mm, "shirt": shirt, "id_purity": round(purity, 3), "color_hex": hexc, "thumb": thumb,
                        "points": [[round(t_, 2), round(x_, 2), round(y_, 2)] for t_, x_, y_ in tr.samples],
                        "bio": {k: bio[k] for k in ("high_risk_count", "worst_angles", "valgus_assessable_frames", "recommendations")}})
    # human labels: "Black shirt"; disambiguate duplicates with the court side
    for p in players:
        same = [q for q in players if q["shirt"] == p["shirt"]]
        base = f'{p["shirt"].capitalize()} shirt'
        p["label"] = base if len(same) == 1 else f'{base} ({p["movement"].get("side", "?")})'
    colors = {p["track_id"]: accent(p["track_id"])[1] for p in players}
    labels = {p["track_id"]: p["label"] for p in players}
    for p in players:
        pts = {p["track_id"]: [(x * 100, y * 100) for _, x, y in p["points"]]}
        hp = f"heatmap_p{p['track_id']}.png"
        cv2.imwrite(os.path.join(out_dir, hp), court.generate_player_heatmap(pts, colors, {p["track_id"]: p["label"]}))
        p["heatmap"] = hp
    allpts = {p["track_id"]: [(x * 100, y * 100) for _, x, y in p["points"]] for p in players}
    cv2.imwrite(os.path.join(out_dir, "heatmap_all.png"), court.generate_player_heatmap(allpts, colors, labels))

    if snapshots:
        rows = [np.hstack(snapshots[i:i + 2] + [np.zeros_like(snapshots[0])] * (2 - len(snapshots[i:i + 2]))) for i in range(0, len(snapshots), 2)]
        cv2.imwrite(os.path.join(out_dir, "tracking_check.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    stage2 = {
        "video": os.path.abspath(video_path), "sport": sport, "fps": fps, "W": W, "H": H, "duration": duration,
        "total_frames": total_frames, "rallies": [r.__dict__ for r in rallies], "hits": hits,
        "n_samples": n_samples, "n_skipped": n_skipped, "players": players, "tracking": tracker.summary(),
        "estimated_feet": tracker.estimated_feet, "merged_tracks": merged_tracks, "audio_active_s": audio_active_s, "visual": visual,
        "cands": cands, "timings": {**timings}, "t_start": t_all,
    }
    with open(os.path.join(out_dir, "stage2.json"), "w") as f:
        json.dump(stage2, f, default=str)
    return finalize(out_dir, reel_seconds=reel_seconds, use_gemini=use_gemini, progress=progress)


def finalize(
    out_dir: str, *, reel_seconds: float | None = None, use_gemini: bool = True,
    rebuild_reel: bool = True, progress: Progress = _noop,
) -> dict:
    """
    Everything after the expensive CV pass: pick the knee frame, cut the reel, run the Gemini chain,
    build rating cards, write result.json. Re-runnable from stage2.json (no video re-processing),
    e.g. after adding an API key: `python scripts/run_pipeline.py --finalize <out_dir>`.
    """
    t_fin = time.time()
    s2 = json.load(open(os.path.join(out_dir, "stage2.json")))
    video_path, sport, fps, W, H, duration = s2["video"], s2["sport"], s2["fps"], s2["W"], s2["H"], s2["duration"]
    timings = dict(s2["timings"])
    players, cands = s2["players"], s2["cands"]
    from playq.cv.rally_detector import RallySegment
    rallies = [RallySegment(**{k: r[k] for k in ("start_time", "end_time", "duration", "peak_db", "confidence", "sport_event")}) for r in s2["rallies"]]
    hits = [tuple(h) for h in s2["hits"]]
    engine = RatingEngine()

    # Annotated knee frame -------------------------------------------------------
    knee, best = None, (cands[0] if cands else None)
    if best:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, best["frame_idx"])
        ok, fr = cap.read()
        cap.release()
        if ok:
            level = knee_level(best["flexion"], best["valgus"])
            verdict = {"high": "HIGH knee load", "elevated": "ELEVATED knee load", "normal": "Normal knee load"}[level]
            img = annotate_knee_frame(fr, [tuple(l) for l in best["landmarks"]], best["side"],
                                      {"flexion": best["flexion"], "valgus": best["valgus"]}, verdict, level,
                                      label_prefix=next((q["label"] for q in players if q["track_id"] == best["track_id"]), f"P{best['track_id']}") + "  ·  ")
            cv2.imwrite(os.path.join(out_dir, "knee_frame.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            knee = {k: best[k] for k in ("flexion", "valgus", "side", "track_id", "t", "court_xy_m", "near_kitchen", "frontal")}
            knee.update({"level": level, "verdict": verdict, "image": "knee_frame.jpg"})

    # Highlight reel -------------------------------------------------------------
    t0 = time.time()
    progress("reel", 0.0, "Cutting the highlight reel…")
    reel_raw = os.path.join(out_dir, "highlight_raw.mp4")
    active_total = sum(r.duration for r in rallies)
    budget = reel_budget(duration, active_total) if reel_seconds is None else reel_seconds
    if duration <= budget + 3:
        segs = [(0.0, duration)]
    else:
        segs = plan_reel(hits, rallies, duration, budget)
    timeline, cur = [], 0.0
    for k, (a, b) in enumerate(segs):
        n_in = sum(1 for h, _ in hits if a <= h < b)
        entry = {"segment": k, "reel_start_s": round(cur, 1), "reel_end_s": round(cur + (b - a), 1),
                 "source_start_s": round(a, 1), "source_end_s": round(b, 1)}
        if s2.get("visual"):   # `hits` are derived from ball-flight density here, not real audio impacts
            entry["ball_in_play_activity"] = round(min(1.0, n_in / max(1.0, (b - a) * 5)), 2)
        else:
            entry["ball_impacts"] = n_in
        timeline.append(entry)
        cur += b - a
    if rebuild_reel or not os.path.exists(reel_raw):
        build_reel(video_path, segs, reel_raw)
    reel_s = sum(b - a for a, b in segs)
    timings["reel_s"] = round(time.time() - t0, 1)

    # Gemini chain ---------------------------------------------------------------
    t0 = time.time()
    sources = {"vision": "off", "insights": "off", "voice": "off"}
    vision = insights = None
    ai = PrototypeAI() if use_gemini else None
    reel_voiced = reel_raw
    voice_meta = None
    if os.path.exists(os.path.join(out_dir, "highlight_voiced.mp4")):
        os.remove(os.path.join(out_dir, "highlight_voiced.mp4"))
    if ai is not None:
        # The vision check (1 frame) and the video call (reel + metrics) are independent -> run in parallel.
        from concurrent.futures import ThreadPoolExecutor
        progress("ai", 0.1, "Gemini inspecting the skeleton frame and watching the reel (parallel)…")
        proxy = RallyAgent._make_proxy(reel_raw, height=480, fps=10)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_vis = pool.submit(ai.vision_check, os.path.join(out_dir, "knee_frame.jpg"),
                                    {k: knee[k] for k in ("flexion", "valgus", "side", "near_kitchen", "frontal")}) if knee else None
                f_ins = pool.submit(ai.match_insights, proxy, players, {"knee": knee}, sport, timeline)
                vision = f_vis.result() if f_vis else None
                insights = f_ins.result()
        finally:
            if os.path.exists(proxy):
                os.remove(proxy)
        sources["vision"] = (vision or {}).get("_source", "off")
        sources["insights"] = insights.get("_source", "?")

        script = insights.get("hinglish_commentary", []) if insights.get("_source") == "gemini" else []
        if not script:
            script = template_commentary(knee or {"flexion": 0}, reel_s, timeline)
            insights = {**(insights or {}), "hinglish_commentary": script, "_commentary_source": "template"}
        progress("ai", 0.8, "Recording the two commentators…")
        wav = os.path.join(out_dir, "voice.wav")
        voice = synthesize_dialogue(script, reel_s, wav, ai)
        sources["voice"] = voice.get("_source", "?")
        voice_meta = {k: v for k, v in voice.items() if k != "path"}
        if voice.get("path") and os.path.exists(wav):
            reel_voiced = mux_voiceover(reel_raw, wav, os.path.join(out_dir, "highlight_voiced.mp4"))
    timings["gemini_s"] = round(time.time() - t0, 1)
    usage = {"input_tokens": 0, "output_tokens": 0, "est_cost_usd": 0.0, "calls": []}
    for name_, part_ in (("vision", vision), ("video+script", insights)):
        if part_ and part_.get("_usage"):
            u_ = part_["_usage"]; mult_ = part_.get("_raced", 1)
            c_ = config.cost_usd(part_.get("_model", ""), u_["in"], u_["out"]) * mult_
            usage["input_tokens"] += u_["in"]; usage["output_tokens"] += u_["out"]; usage["est_cost_usd"] += c_
            usage["calls"].append({"step": name_, "model": part_.get("_model"), "in": u_["in"], "out": u_["out"], "raced_x": mult_, "usd": round(c_, 5)})
    usage["est_cost_usd"] = round(usage["est_cost_usd"], 5)

    # Rating cards ---------------------------------------------------------------
    scores = {s["track_id"]: s for s in (insights or {}).get("player_scores", [])} if insights and insights.get("_source") == "gemini" else {}
    for p in players:
        sc = scores.get(p["track_id"], {})
        stats = engine.compute_match_stats({
            "court_coverage": p["movement"].get("court_coverage"),
            "attack_aggression": _blend_aggression(p["movement"].get("attack_aggression"), sc.get("shot_aggression_0_99")),
            "stamina": p["movement"].get("stamina"),
            "shot_consistency": (sc.get("shot_consistency_0_99") / 100) if sc.get("shot_consistency_0_99") is not None else None,
            "clutch_factor": (sc.get("clutch_factor_0_99") / 100) if sc.get("clutch_factor_0_99") is not None else None,
        })
        prof = PlayerProfile(player_id=f"track_{p['track_id']}", display_name=p["label"],
                             sport=Sport(sport), current_rating=stats, total_matches=1)
        card = engine.generate_card_data(prof)
        card["side"] = p["movement"].get("side")
        card.update({"label": p["label"], "color_hex": p["color_hex"], "thumb": p["thumb"], "shirt": p["shirt"], "track_id": p["track_id"]})
        card["reason"] = sc.get("reason")
        card["gemini_estimated"] = [k for k in ("CON", "CLU") if k not in card["placeholder_stats"]] + (["AGG"] if sc.get("shot_aggression_0_99") is not None else [])
        p["card"] = card

    timings["finalize_s"] = round(time.time() - t_fin, 1)
    timings["total_s"] = round(timings.get("audio_filter_s", 0) + timings.get("cv_s", 0) + timings["finalize_s"], 1)
    active_s = sum(r.duration for r in rallies)
    result = {
        "meta": {"video": os.path.basename(video_path), "sport": sport, "analyzed_at": datetime.now().isoformat(timespec="seconds"),
                 "duration_s": round(duration, 1), "fps": round(fps, 2), "resolution": [W, H]},
        "funnel": {
            "source_seconds": round(duration, 1), "rallies": len(rallies), "impacts": len(hits),
            "audio_only_active_seconds": round(s2.get("audio_active_s", 0), 1), "visual_check": s2.get("visual"),
            "active_seconds": round(active_s, 1), "frames_sampled": s2["n_samples"], "frames_skipped_replay": s2["n_skipped"],
            "frames_total": s2["total_frames"], "reel_seconds": round(reel_s, 1),
            "sent_to_gemini": {"video_seconds": round(reel_s, 1), "video_height_px": 480, "images": 1 if knee else 0},
            "video_seconds_reduction_pct": round(100 * (1 - reel_s / duration), 1),
        },
        "timings": timings,
        "rallies": s2["rallies"],
        "reel": {"segments": [[round(a, 1), round(b, 1)] for a, b in segs], "timeline": timeline, "seconds": round(reel_s, 1), "file": os.path.basename(reel_voiced), "raw": "highlight_raw.mp4"},
        "players": players,
        "team_heatmap": "heatmap_all.png",
        "tracking": s2["tracking"],
        "knee": knee,
        "ai": {"sources": sources, "vision": vision, "insights": insights, "voice": voice_meta, "model": config.VIDEO_MODEL, "usage": usage},
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2, default=str, ensure_ascii=False)
    progress("done", 1.0, "Done")
    return result
