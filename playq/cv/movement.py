"""
Movement metrics from a tracked player's court positions.

All three outputs are normalised to 0-1 and feed PlayQStats. The scaling constants are
HEURISTICS chosen for recreational pickleball; they need calibration against many real
matches before the scores mean anything comparable across players.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

# Recreational pickleball reference points (tune with real data)
COVERAGE_FULL_AT = 0.45      # visiting 45 % of the 1 m cells on your half *within a single rally* = max score
AGGRESSION_FULL_AT = 0.80    # spending 80 % of rally time within NET_ZONE_M of the net = max (kitchen-line doubles)
NET_ZONE_M = 3.0             # kitchen is 2.13 m; +0.9 m for the transition step
NET_FAR_M, NET_NEAR_M = 7.0, 2.0   # mean distance to the net: >=7 m (baseline) -> 0, <=2 m (kitchen line) -> 1
INTENSITY_FULL_AT = 1.6      # m/s average tracked movement (includes some pose jitter) = max


def _smooth(samples: list[tuple[float, float, float]]) -> np.ndarray:
    a = np.array(samples, dtype=float)            # columns: t, x, y
    if len(a) >= 3:
        a[:, 1] = median_filter(a[:, 1], size=3, mode="nearest")   # kill single-frame pose jitter
        a[:, 2] = median_filter(a[:, 2], size=3, mode="nearest")
    return a


def movement_metrics(
    samples: list[tuple[float, float, float]],
    court_w_m: float,
    court_len_m: float,
    max_gap_s: float = 0.5,
    rally_windows: list[tuple[float, float]] | None = None,
) -> dict:
    """
    Args:
        samples: [(t_s, x_m, y_m)] for one player, live-play only.
    Returns:
        dict with raw values and the three normalised scores
        (court_coverage, attack_aggression, stamina), or None scores if too little data.
    """
    if len(samples) < 8:
        return {"court_coverage": None, "attack_aggression": None, "stamina": None, "n_samples": len(samples)}

    a = _smooth(samples)
    t, x, y = a[:, 0], a[:, 1], a[:, 2]
    net_y = court_len_m / 2
    near = np.median(y) > net_y
    half = (net_y, court_len_m) if near else (0.0, net_y)

    # coverage: 1 m x 1 m cells visited on the player's own half
    cols, rows = int(np.ceil(court_w_m)), int(np.ceil(half[1] - half[0]))
    cells = {(int(np.clip(xi, 0, court_w_m - 1e-6)), int(np.clip(yi - half[0], 0, half[1] - half[0] - 1e-6)))
             for xi, yi in zip(x, y)}
    cov_ratio = len(cells) / (cols * rows)           # whole-match (reported, but it saturates)
    if rally_windows:                                  # per-rally coverage is what discriminates players
        per = []
        for a0, b0 in rally_windows:
            m = (t >= a0) & (t <= b0)
            if m.sum() >= 6:
                c = {(int(np.clip(xi, 0, court_w_m - 1e-6)), int(np.clip(yi - half[0], 0, half[1] - half[0] - 1e-6)))
                     for xi, yi in zip(x[m], y[m])}
                per.append(len(c) / (cols * rows))
        rally_cov = float(np.mean(per)) if per else cov_ratio
    else:
        rally_cov = cov_ratio

    # aggression: share of time within NET_ZONE_M of the net
    dist_net = np.abs(y - net_y)
    aggr_ratio = float(np.mean(dist_net <= NET_ZONE_M))
    mean_dist = float(np.mean(dist_net))
    # Continuous "how far forward do you play" blended with time-in-zone, so two baseliners are not both pinned at 0
    forward = float(np.clip((NET_FAR_M - mean_dist) / (NET_FAR_M - NET_NEAR_M), 0, 1))
    aggression = 0.6 * forward + 0.4 * min(1.0, aggr_ratio / AGGRESSION_FULL_AT)

    # stamina: average speed while sampled continuously, and how well it holds up late
    dt, step = np.diff(t), np.hypot(np.diff(x), np.diff(y))
    ok = (dt > 0) & (dt <= max_gap_s) & (step / np.maximum(dt, 1e-6) < 6.0)   # drop teleports
    speeds = (step[ok] / dt[ok]) if ok.any() else np.array([0.0])
    intensity = float(speeds.mean())
    half_i = len(speeds) // 2
    early, late = (speeds[:half_i].mean(), speeds[half_i:].mean()) if half_i >= 3 else (1.0, 1.0)
    sustain = float(np.clip(late / early, 0.0, 1.0)) if early > 0 else 0.5   # 1.0 = no fade-out
    stamina = 0.6 * min(1.0, intensity / INTENSITY_FULL_AT) + 0.4 * sustain

    return {
        "n_samples": len(samples),
        "side": "near" if near else "far",
        "cells_visited": len(cells),
        "coverage_ratio": round(cov_ratio, 3),
        "coverage_per_rally": round(rally_cov, 3),
        "net_zone_share": round(aggr_ratio, 3),
        "mean_dist_to_net_m": round(mean_dist, 2),
        "avg_speed_mps": round(intensity, 2),
        "late_vs_early_speed": round(float(late / early), 2) if early > 0 else None,
        "court_coverage": float(min(1.0, rally_cov / COVERAGE_FULL_AT)),
        "attack_aggression": float(aggression),
        "stamina": float(min(1.0, stamina)),
    }
