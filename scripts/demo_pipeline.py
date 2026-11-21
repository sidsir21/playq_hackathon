#!/usr/bin/env python3
"""
PlayQ Demo Pipeline
====================
Demonstrates the full PlayQ analysis pipeline using **synthetic data only**.
No real video file or API key required.

Run from project root:
    python scripts/demo_pipeline.py
"""

import sys
import os
import math
import random
import json
from datetime import datetime

# Allow running from scripts/ dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2

from playq.cv.homography import CourtHomography
from playq.models.rating import (
    PlayQStats, STAT_KEYS, PlayerProfile, MatchResult,
    BiomechanicsReport, RatingEngine, Sport, RiskLevel
)

# ─── Colour helpers ───────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
MAGENTA= "\033[95m"
WHITE  = "\033[97m"

def colour(text, c): return f"{c}{text}{RESET}"

# ─── 1. Synthetic Pose Data ────────────────────────────────────────────────────

def make_synthetic_pose_frames(n_frames: int = 300, sport: str = "pickleball") -> list[dict]:
    """Generate fake PoseFrame-like dicts with realistic joint angle distributions."""
    rng = random.Random(42)
    frames = []
    for i in range(n_frames):
        t = i / 30.0 * 1000  # ms at 30fps

        # Base angles with natural variation
        knee_valgus = rng.gauss(8, 3)                    # mostly safe
        shoulder_abduction = rng.gauss(70, 15)
        lumbar_flexion = rng.gauss(20, 8)

        # Inject 10 "danger" moments (deep lunges, smashes)
        if i in {30, 60, 90, 120, 150, 180, 210, 240, 270, 290}:
            knee_valgus = rng.gauss(24, 2)               # danger > 20°
            shoulder_abduction = rng.gauss(128, 5)       # danger > 120°

        # Build angle dicts
        angles = [
            {"joint_name": "left_knee_valgus",          "angle_degrees": abs(knee_valgus),
             "risk_level": _risk(abs(knee_valgus), 12, 20)},
            {"joint_name": "right_knee_valgus",         "angle_degrees": abs(knee_valgus - rng.gauss(0,1)),
             "risk_level": _risk(abs(knee_valgus), 12, 20)},
            {"joint_name": "left_shoulder_abduction",   "angle_degrees": abs(shoulder_abduction),
             "risk_level": _risk(abs(shoulder_abduction), 90, 120)},
            {"joint_name": "right_shoulder_abduction",  "angle_degrees": abs(shoulder_abduction),
             "risk_level": _risk(abs(shoulder_abduction), 90, 120)},
            {"joint_name": "left_lumbar_flexion",       "angle_degrees": abs(lumbar_flexion),
             "risk_level": _risk(abs(lumbar_flexion), 30, 45)},
        ]
        risk_flags = [a["joint_name"] for a in angles if a["risk_level"] == "danger"]
        frames.append({
            "frame_index": i,
            "timestamp_ms": t,
            "joint_angles": angles,
            "risk_flags": risk_flags,
        })
    return frames


def _risk(val: float, safe: float, danger: float) -> str:
    if val <= safe:   return "safe"
    if val >= danger: return "danger"
    return "caution"


def build_pose_report(frames: list[dict]) -> dict:
    """Aggregate synthetic frames into a biomechanics report."""
    high_risk = [f for f in frames if f["risk_flags"]]
    risk_dist = {"safe": 0, "caution": 0, "danger": 0}
    worst: dict[str, float] = {}

    for f in frames:
        for ja in f["joint_angles"]:
            risk_dist[ja["risk_level"]] = risk_dist.get(ja["risk_level"], 0) + 1
            base = ja["joint_name"].split("_", 1)[-1]
            if base not in worst or ja["angle_degrees"] > worst[base]:
                worst[base] = round(ja["angle_degrees"], 1)

    recs = []
    if worst.get("knee_valgus", 0) > 15:
        recs.append("⚠️  Knee valgus detected — strengthen VMO with single-leg squats.")
    if worst.get("shoulder_abduction", 0) > 100:
        recs.append("⚠️  Shoulder overload on overheads — rotator cuff band exercises.")
    if not recs:
        recs.append("✅ Biomechanics look healthy — maintain good form!")

    return {
        "total_frames_analyzed": len(frames),
        "high_risk_count": len(high_risk),
        "risk_distribution": risk_dist,
        "worst_angles": worst,
        "recommendations": recs,
    }


# ─── 2. Synthetic Court Positions ─────────────────────────────────────────────

def make_court_positions(n: int = 500, court_width: int = 610, court_height: int = 1341) -> list[tuple]:
    """Generate synthetic footwork positions biased towards kitchen/net area."""
    rng = np.random.default_rng(7)
    positions = []

    # 60% near the kitchen zone (non-volley zone)
    kitchen_y_lo = int(court_height * 0.35)
    kitchen_y_hi = int(court_height * 0.65)
    for _ in range(int(n * 0.6)):
        x = rng.uniform(0, court_width)
        y = rng.uniform(kitchen_y_lo, kitchen_y_hi)
        positions.append((float(x), float(y)))

    # 25% baseline
    for _ in range(int(n * 0.25)):
        positions.append((float(rng.uniform(0, court_width)),
                          float(rng.choice([rng.uniform(0, int(court_height * 0.2)),
                                            rng.uniform(int(court_height * 0.8), court_height)]))))
    # 15% random
    for _ in range(n - len(positions)):
        positions.append((float(rng.uniform(0, court_width)),
                          float(rng.uniform(0, court_height))))

    return positions


# ─── 3. Heatmap Generation ────────────────────────────────────────────────────

def generate_and_save_heatmap(positions: list[tuple], output_path: str) -> str:
    """Create a pickleball heatmap from synthetic positions and save it."""
    court = CourtHomography("pickleball")

    # Manual calibration: synthetic flat court (no perspective needed)
    w_px, h_px = court.court_pixels           # 610 x 1341
    src = np.float32([[0,0],[w_px,0],[w_px,h_px],[0,h_px]])
    court.calibrate(src)

    heatmap = court.generate_heatmap(positions, resolution=(670, 305))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, heatmap)
    return output_path


# ─── 4. Rating Computation ────────────────────────────────────────────────────

def compute_rating(pose_report: dict, positions: list[tuple]) -> PlayQStats:
    engine = RatingEngine()
    court_width, court_height = 610, 1341
    total_cells = 20 * 20
    occupied = set()
    for x, y in positions:
        cx = int(x / court_width * 20)
        cy = int(y / court_height * 20)
        occupied.add((min(cx, 19), min(cy, 19)))
    coverage_pct = len(occupied) / total_cells

    return engine.compute_match_stats({
        "court_coverage": min(1.0, coverage_pct * 1.5),
        "attack_aggression": 0.55,
        "stamina": 0.7,
        "shot_consistency": None,  # needs shot outcomes
        "clutch_factor": None,     # needs score tracking
    })


# ─── 5. ASCII PlayQ Rating Card ───────────────────────────────────────────────

def print_rating_card(profile: PlayerProfile):
    card_w = 38
    sep = "─" * card_w
    card = RatingEngine().generate_card_data(profile)

    print()
    print(colour(f"┌{sep}┐", CYAN))
    title = f" {card['sport_icon']}  PLAYQ RATING CARD"
    print(colour(f"│{title:<{card_w}}│", CYAN))
    print(colour(f"├{sep}┤", CYAN))

    name_line = f"  {card['player_name'].upper()}"
    print(colour(f"│{name_line:<{card_w}}│", WHITE + BOLD))

    sport_line = f"  {card['sport'].replace('_',' ').upper()} · {card['tier'].upper()}"
    print(colour(f"│{sport_line:<{card_w}}│", YELLOW))
    print(colour(f"├{sep}┤", CYAN))

    overall_line = f"  OVERALL  {card['overall']:>3}"
    stars = "★" * (card['overall'] // 20) + "☆" * (5 - card['overall'] // 20)
    overall_full = f"{overall_line}   {stars}"
    print(colour(f"│{overall_full:<{card_w}}│", GREEN + BOLD))
    print(colour(f"├{sep}┤", CYAN))

    for stat, val in card["stats"].items():
        filled = int(val / 99 * 18)
        bar = "█" * filled + "░" * (18 - filled)
        line = f"  {stat:<4} {val:>2}  {bar}"
        c = RED if val < 40 else (YELLOW if val < 65 else GREEN)
        print(colour(f"│{line:<{card_w}}│", c))

    print(colour(f"├{sep}┤", CYAN))
    matches_line = f"  Matches played: {card['total_matches']}"
    print(colour(f"│{matches_line:<{card_w}}│", WHITE))
    version_line = f"  {card['card_version']}"
    print(colour(f"│{version_line:<{card_w}}│", MAGENTA))
    print(colour(f"└{sep}┘", CYAN))
    print()


# ─── 6. Main Demo ─────────────────────────────────────────────────────────────

def main():
    print()
    print(colour("╔══════════════════════════════════════════╗", CYAN + BOLD))
    print(colour("║      🏓  PlayQ MVP Demo Pipeline         ║", CYAN + BOLD))
    print(colour("║   Full pipeline · synthetic data only    ║", CYAN + BOLD))
    print(colour("╚══════════════════════════════════════════╝", CYAN + BOLD))
    print()

    # ── Step 1: Pose data ──────────────────────────────────────────────────────
    print(colour("▶ Step 1: Generating synthetic pose frames ...", BOLD))
    frames = make_synthetic_pose_frames(300)
    pose_report = build_pose_report(frames)
    high_risk = [f for f in frames if f["risk_flags"]]

    print(f"  Frames analysed  : {pose_report['total_frames_analyzed']}")
    print(f"  High-risk moments: {colour(str(pose_report['high_risk_count']), RED if pose_report['high_risk_count'] > 5 else GREEN)}")
    print(f"  Risk distribution: {pose_report['risk_distribution']}")
    print(f"  Worst angles     : {pose_report['worst_angles']}")

    # ── Step 2: Court heatmap ──────────────────────────────────────────────────
    print()
    print(colour("▶ Step 2: Generating 2D court heatmap ...", BOLD))
    positions = make_court_positions(600)
    heatmap_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "demo_heatmap.png"
    )
    saved = generate_and_save_heatmap(positions, heatmap_path)
    print(f"  Court positions  : {len(positions)} points")
    print(f"  Heatmap saved    : {colour(os.path.abspath(saved), CYAN)}")

    # ── Step 3: Compute PlayQ rating ───────────────────────────────────────────
    print()
    print(colour("▶ Step 3: Computing PlayQ Rating ...", BOLD))
    stats = compute_rating(pose_report, positions)
    for k in STAT_KEYS:
        flag = "  (placeholder)" if k in stats.placeholders else ""
        print(f"  {k.replace('_', ' ').title():<18}: {getattr(stats, k)}{flag}")
    print(f"  Overall  : {colour(str(stats.overall), GREEN + BOLD)}")

    # ── Step 4: Build player profile & update rating ──────────────────────────
    print()
    print(colour("▶ Step 4: Updating player profile ...", BOLD))
    player = PlayerProfile(
        player_id="demo_player_001",
        display_name="Sid Sirohi",
        sport=Sport.PICKLEBALL,
    )
    bio = BiomechanicsReport(
        risk_level=RiskLevel.CAUTION if pose_report["high_risk_count"] > 5 else RiskLevel.SAFE,
        knee_valgus_max=pose_report["worst_angles"].get("knee_valgus"),
        shoulder_abduction_max=pose_report["worst_angles"].get("shoulder_abduction"),
        high_risk_moments=pose_report["high_risk_count"],
        recommendations=pose_report["recommendations"],
    )
    match = MatchResult(
        match_id="demo_match_001",
        player_id=player.player_id,
        sport=Sport.PICKLEBALL,
        date=datetime.now(),
        stats=stats,
        biomechanics=bio,
        rally_count=18,
        duration_minutes=24.5,
    )
    player.update_rating(match)
    print(f"  Player : {player.display_name}")
    print(f"  Matches: {player.total_matches}")

    # ── Step 5: Print rating card ──────────────────────────────────────────────
    print(colour("▶ Step 5: PlayQ Rating Card", BOLD))
    print_rating_card(player)

    # ── Step 6: Biomechanics summary ──────────────────────────────────────────
    print(colour("▶ Step 6: Biomechanics Recommendations", BOLD))
    for rec in pose_report["recommendations"]:
        print(f"  {rec}")

    # ── Step 7: Pipeline summary JSON ─────────────────────────────────────────
    print()
    print(colour("▶ Step 7: Pipeline output summary (JSON)", BOLD))
    summary = {
        "player_id": player.player_id,
        "sport": "pickleball",
        "overall_rating": stats.overall,
        "high_risk_moments": pose_report["high_risk_count"],
        "heatmap": os.path.abspath(heatmap_path),
        "timestamp": datetime.now().isoformat(),
    }
    print(json.dumps(summary, indent=2))

    print()
    print(colour("✅  Demo complete! All pipeline stages executed successfully.", GREEN + BOLD))
    print(colour("   Next: record real clips → wire in Gemini API key → deploy to Cloud Run.", WHITE))
    print()


if __name__ == "__main__":
    main()
