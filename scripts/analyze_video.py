#!/usr/bin/env python3
"""
PlayQ Video Analysis Pipeline
=============================
Processes real match footage (MP4) through:
1. MediaPipe Pose Skeletal Extraction & Biomechanics Audit
2. Footwork Trajectory & 2D Perspective Court Heatmap
3. 5-Attribute PlayQ FIFA-Style Rating Calculation
4. Annotated High-Risk Posture Keyframe Export

Usage:
    python scripts/analyze_video.py [video_path] [sport]
"""

import sys
import os
import time
import json
import numpy as np
import cv2
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from playq.cv.pose_analyzer import PoseAnalyzer, PoseFrame
from playq.cv.homography import CourtHomography, CameraStabilizer
from playq.cv.tracker import PlayerTracker
from playq.cv.movement import movement_metrics
from playq.cv.rally_detector import RallyDetector
from playq.models.rating import (
    RatingEngine, PlayerProfile, MatchResult, BiomechanicsReport,
    PlayQStats, Sport, RiskLevel
)

# Terminal colors
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
WHITE = "\033[97m"
RESET = "\033[0m"


def print_banner():
    print()
    print(f"{CYAN}{BOLD}╔══════════════════════════════════════════════════════════╗{RESET}")
    print(f"{CYAN}{BOLD}║         🏓  PlayQ Match Video Intelligence Pipeline       ║{RESET}")
    print(f"{CYAN}{BOLD}║         Full HD 60fps Biomechanical & Spatial Engine     ║{RESET}")
    print(f"{CYAN}{BOLD}╚══════════════════════════════════════════════════════════╝{RESET}")
    print()


def print_fifa_card(card: dict):
    w = 42
    sep = "─" * w
    print()
    print(f"{CYAN}┌{sep}┐{RESET}")
    title = f" {card['sport_icon']}  PLAYQ RATING CARD"
    print(f"{CYAN}│{title:<{w}}│{RESET}")
    print(f"{CYAN}├{sep}┤{RESET}")
    name = f"  {card['player_name'].upper()}"
    print(f"{WHITE}{BOLD}│{name:<{w}}│{RESET}")
    sub = f"  {card['sport'].replace('_', ' ').upper()} · {card['tier'].upper()}"
    print(f"{YELLOW}│{sub:<{w}}│{RESET}")
    print(f"{CYAN}├{sep}┤{RESET}")

    overall_val = f"  OVERALL  {card['overall']:>3}"
    stars = "★" * (card['overall'] // 20) + "☆" * (5 - (card['overall'] // 20))
    rating_line = f"{overall_val}   {stars}"
    print(f"{GREEN}{BOLD}│{rating_line:<{w}}│{RESET}")
    print(f"{CYAN}├{sep}┤{RESET}")

    for stat, val in card["stats"].items():
        mark = "*" if stat in card.get("placeholder_stats", []) else " "
        filled = int(val / 99 * 20)
        bar = "█" * filled + "░" * (20 - filled)
        line = f"  {stat:<4} {val:>2}{mark} {bar}"
        c = RED if val < 45 else (YELLOW if val < 70 else GREEN)
        print(f"{c}│{line:<{w}}│{RESET}")

    print(f"{CYAN}├{sep}┤{RESET}")
    if card.get("placeholder_stats"):
        print(f"{WHITE}│{'  * = not yet measured (neutral 50)':<{w}}│{RESET}")
    matches_str = f"  Matches Logged: {card['total_matches']}"
    print(f"{WHITE}│{matches_str:<{w}}│{RESET}")
    ver_str = f"  {card['card_version']}"
    print(f"{MAGENTA}│{ver_str:<{w}}│{RESET}")
    print(f"{CYAN}└{sep}┘{RESET}")
    print()


def far_half_roi(court, to_ref, width, height):
    """Image-space box around the far half of the court (+ headroom), in the CURRENT frame."""
    wm, hm = court.court_dims
    ground = np.float32([[x * 100, y * 100] for x, y in ((0, 0), (wm, 0), (wm, hm / 2), (0, hm / 2))])
    pts = cv2.perspectiveTransform(ground.reshape(-1, 1, 2), court._inv_matrix).reshape(-1, 2)   # reference frame px
    if to_ref is not None:
        pts = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), np.linalg.inv(to_ref)).reshape(-1, 2)
    x0, y0, x1, y1 = pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()
    pad = 0.15 * (x1 - x0)
    return int(x0 - pad), int(y0 - 0.9 * (y1 - y0) - 120), int(x1 + pad), int(y1 + 20)


def analyze_match(video_path: str, sport: str = "pickleball", calib_path: str | None = None):
    print_banner()
    root = os.path.join(os.path.dirname(__file__), "..")
    calib_path = calib_path or os.path.join(root, "config", "court_calibration.json")
    data_dir = os.path.abspath(os.path.join(root, "data"))
    os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(video_path):
        sys.exit(f"{RED}Error: Video not found at {video_path}{RESET}")
    if not os.path.exists(calib_path):
        sys.exit(f"{RED}Error: no court calibration at {calib_path} (tracking needs it){RESET}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"{RED}Error: Cannot open video {video_path}{RESET}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / fps
    print(f"{BOLD}📹 {width}x{height} @ {fps:.2f} FPS | {duration_s:.1f}s | {total_frames} frames{RESET}\n")

    # Step 1: rallies from audio
    print(f"{BOLD}▶ Step 1: Detecting rallies from audio (ball-pop transients)...{RESET}")
    detector = RallyDetector(sport=sport)
    rallies = detector.detect_rallies(video_path)
    hits = detector.detect_hits(video_path)
    for r in rallies:
        print(f"  • {r.start_time:6.1f}s → {r.end_time:6.1f}s  ({r.duration:4.1f}s)  conf {r.confidence:.2f}")
    active_s = sum(r.duration for r in rallies)
    print(f"  {GREEN}{len(rallies)} rallies, {len(hits)} impacts, {active_s:.0f}s active ({active_s / duration_s:.0%}){RESET}\n")

    def in_rally(t: float) -> bool:
        return any(r.start_time <= t <= r.end_time for r in rallies)

    # Court calibration + camera stabiliser
    cfg = json.load(open(calib_path))
    court = CourtHomography(sport=sport)
    court.calibrate_landmarks([l["px"] for l in cfg["landmarks"]], [l["court_m"] for l in cfg["landmarks"]])
    cap.set(cv2.CAP_PROP_POS_FRAMES, cfg.get("reference_frame", 80))
    ok, ref = cap.read()
    stabilizer = CameraStabilizer(ref) if ok else None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Step 2: pose for everyone + player tracking (live play only)
    print(f"{BOLD}▶ Step 2: Pose extraction + player tracking (ID lock)...{RESET}")
    sample_stride = 10  # ~6 samples/s. Use 1-2 for impact windows in production.
    analyzer = PoseAnalyzer(sport=sport)
    tracker = PlayerTracker(court, max_players=4)
    snapshots: list[np.ndarray] = []
    accepted_frames = 0
    frame_idx, t0 = 0, time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        t_s = frame_idx / fps
        if frame_idx % sample_stride == 0 and in_rally(t_s):
            poses = analyzer.analyze_frame_all(frame, frame_idx, t_s * 1000.0)
            to_ref = stabilizer.to_reference(frame) if stabilizer else None
            # Second pass: upscaled crop of the far half, where players are too small for full-frame detection
            roi = far_half_roi(court, to_ref, width, height)
            for fp in analyzer.analyze_roi_all(frame, roi, frame_index=frame_idx, timestamp_ms=t_s * 1000.0):
                c = np.mean([(fp.landmarks[i][0], fp.landmarks[i][1]) for i in (11, 12, 23, 24)], axis=0)
                if all(np.hypot(*(c - np.mean([(q.landmarks[i][0], q.landmarks[i][1]) for i in (11, 12, 23, 24)], axis=0))) > 0.03
                       for q in poses):   # skip duplicates of people already found
                    poses.append(fp)
            assigned = tracker.update(frame, poses, t_s, to_ref)
            if assigned:
                accepted_frames += 1
                if accepted_frames % 20 == 1 and len(snapshots) < 6:
                    snap = frame.copy()
                    for tid, pose, xy in assigned:
                        ax = int((pose.landmarks[27][0] + pose.landmarks[28][0]) / 2 * width)
                        ay = int((pose.landmarks[27][1] + pose.landmarks[28][1]) / 2 * height)
                        cv2.circle(snap, (ax, ay), 10, (0, 255, 255), -1)
                        cv2.putText(snap, f"P{tid} ({xy[0]:.1f},{xy[1]:.1f})m", (ax - 40, ay + 45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
                    cv2.putText(snap, f"t={t_s:.1f}s", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
                    snapshots.append(cv2.resize(snap, (960, 540)))
        frame_idx += 1
    cap.release()
    print(f"  {GREEN}Processed in {time.time() - t0:.1f}s{RESET}")
    print(f"  Rejected as bystanders (off-court): {tracker.rejected_outside}   |   no visible ankles: {tracker.rejected_no_feet}")
    players = [t for t in tracker.tracks if t.n >= 15]
    for row in tracker.summary():
        print(f"  • Player {row['track_id']}: {row['samples']:>3} samples, {row['side']} side, median court pos {row['median_xy_m']} m")
    if snapshots:
        rows = [np.hstack(snapshots[i:i + 2] + [np.zeros_like(snapshots[0])] * (2 - len(snapshots[i:i + 2])))
                for i in range(0, len(snapshots), 2)]
        sheet = os.path.join(data_dir, "tracking_check.jpg")
        cv2.imwrite(sheet, np.vstack(rows))
        print(f"  Tracking contact sheet → {CYAN}{sheet}{RESET}   (check that IDs follow the same person)")
    print()

    engine = RatingEngine()
    packs = []
    all_events = []
    wm, hm = court.court_dims
    for tr in players:
        # Step 3: per-player biomechanics
        print(f"{BOLD}▶ Player {tr.id}: biomechanics, heatmap, rating{RESET}")
        bio = analyzer.generate_biomechanics_report(tr.poses)
        events = analyzer.group_events(tr.poses)
        for ev in events:
            all_events.append((ev["severity"], tr.id, ev))
        print(f"  • Poses {bio['total_frames_analyzed']} | valgus-assessable {bio['valgus_assessable_frames']} | "
              f"high-risk events {RED if events else GREEN}{len(events)}{RESET}")
        for rec in bio["recommendations"]:
            print(f"    {rec}")

        # Step 4: heatmap in court space
        pos = [(x * 100, y * 100) for _, x, y in tr.samples]
        hm_path = os.path.join(data_dir, f"heatmap_player_{tr.id}.png")
        cv2.imwrite(hm_path, court.generate_heatmap(pos, resolution=(700, 350)))

        # Step 5: movement-derived rating attributes
        mm = movement_metrics(tr.samples, wm, hm)
        stats = engine.compute_match_stats({
            "court_coverage": mm["court_coverage"],
            "attack_aggression": mm["attack_aggression"],
            "stamina": mm["stamina"],
            "shot_consistency": None,   # needs shot outcomes (Gemini video reasoning)
            "clutch_factor": None,      # needs score tracking
        })
        print(f"  • Movement: coverage {mm.get('coverage_ratio')} | net-zone {mm.get('net_zone_share')} | "
              f"{mm.get('avg_speed_mps')} m/s | late/early {mm.get('late_vs_early_speed')}")
        player = PlayerProfile(player_id=f"track_{tr.id}", display_name=f"Player {tr.id} ({mm.get('side', '?')} side)",
                               sport=Sport(sport))
        risk = RiskLevel.DANGER if len(events) >= 5 else (RiskLevel.CAUTION if events else RiskLevel.SAFE)
        wa = bio["worst_angles"]
        result = MatchResult(
            match_id=f"match_{int(time.time())}_p{tr.id}", player_id=player.player_id, sport=Sport(sport),
            date=datetime.now(), stats=stats,
            biomechanics=BiomechanicsReport(
                risk_level=risk, knee_valgus_max=wa.get("knee_valgus"),
                shoulder_abduction_max=wa.get("shoulder_abduction"), lumbar_flexion_max=wa.get("lumbar_flexion"),
                high_risk_moments=len(events), recommendations=bio["recommendations"]),
            heatmap_url=hm_path, rally_count=len(rallies), duration_minutes=round(duration_s / 60, 2))
        player.update_rating(result)
        card = engine.generate_card_data(player)
        print_fifa_card(card)
        packs.append({
            "player": player.display_name, "track_id": tr.id, "playq_overall": card["overall"],
            "stats": card["stats"], "placeholder_stats": card["placeholder_stats"], "movement": mm,
            "risk_level": risk.value, "high_risk_events": len(events),
            "valgus_assessable_frames": bio["valgus_assessable_frames"], "peak_angles_deg": wa,
            "recommendations": bio["recommendations"], "heatmap": hm_path,
        })

    # Annotated risk keyframes across all players (worst first, >=3 s apart)
    all_events.sort(key=lambda e: -e[0])
    chosen = []
    for sev, tid, ev in all_events:
        if all(abs(ev["start_ms"] - c[2]["start_ms"]) > 3000 for c in chosen):
            chosen.append((sev, tid, ev))
        if len(chosen) == 3:
            break
    kf_paths = []
    cap = cv2.VideoCapture(video_path)
    for i, (_, tid, ev) in enumerate(chosen):
        wf = ev["worst"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, wf.frame_index)
        ok, img = cap.read()
        if not ok:
            continue
        for x, y, _, v in wf.landmarks:
            if v > 0.5:
                cv2.circle(img, (int(x * width), int(y * height)), 5, (0, 255, 0), -1)
        bad = {ja.joint_name: ja for ja in wf.joint_angles if ja.risk_level == "danger"}
        cv2.rectangle(img, (0, 0), (width, 120), (0, 0, 0), -1)
        cv2.putText(img, f"PlayQ risk moment  P{tid} @ {wf.timestamp_ms / 1000:.1f}s", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        cv2.putText(img, ", ".join(f"{n}={ja.angle_degrees:.0f}deg" for n, ja in bad.items()), (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        out_k = os.path.join(data_dir, f"risk_moment_{i + 1}.jpg")
        cv2.imwrite(out_k, img)
        kf_paths.append(out_k)
    cap.release()

    out_json = os.path.join(data_dir, "match_report.json")
    with open(out_json, "w") as f:
        json.dump({"video": os.path.basename(video_path), "sport": sport, "rallies": [r.__dict__ for r in rallies],
                   "players": packs, "risk_frames": kf_paths}, f, indent=2, default=str)
    print(f"{GREEN}{BOLD}✅ Done. Report → {out_json}{RESET}\n")


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "/Users/sidsirohi/Desktop/PlayQ/VID_20261116_213210350.mp4"
    sport_arg = sys.argv[2] if len(sys.argv) > 2 else "pickleball"
    analyze_match(video, sport=sport_arg)
