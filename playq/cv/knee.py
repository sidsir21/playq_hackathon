"""Knee metrics for the flagged-lunge frame: flexion (always) and valgus (only when frontal)."""
from __future__ import annotations

import numpy as np

L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24
LEGS = {"left": (23, 25, 27), "right": (24, 26, 28)}  # hip, knee, ankle


def _angle(a, b, c) -> float:
    ba, bc = np.asarray(a, float) - b, np.asarray(c, float) - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def torso_px(lm, W, H) -> float:
    sh = np.array([(lm[L_SH][0] + lm[R_SH][0]) / 2 * W, (lm[L_SH][1] + lm[R_SH][1]) / 2 * H])
    hp = np.array([(lm[L_HIP][0] + lm[R_HIP][0]) / 2 * W, (lm[L_HIP][1] + lm[R_HIP][1]) / 2 * H])
    return float(np.linalg.norm(sh - hp))


def facing_ratio(lm, W, H) -> float:
    """Shoulder span / torso length. >0.8 => player faces (or turns their back to) the camera."""
    span = np.hypot((lm[L_SH][0] - lm[R_SH][0]) * W, (lm[L_SH][1] - lm[R_SH][1]) * H)
    t = torso_px(lm, W, H)
    return float(span / t) if t > 0 else 0.0


def knee_metrics(lm, W: int, H: int, min_vis: float = 0.6) -> dict:
    """
    lm: 33 landmark tuples (x, y, z, visibility), normalised.
    Returns {"torso_px", "frontal", "legs": {side: {"flexion", "valgus"|None}}}.
    flexion = 180 - interior hip-knee-ankle angle (0 = straight leg; ~90 = deep lunge).
    valgus  = medial knee deviation from the hip->ankle line, only if the player is frontal.
    """
    out = {"torso_px": torso_px(lm, W, H), "frontal": facing_ratio(lm, W, H) >= 0.8, "legs": {}}
    mid_hip_x = (lm[L_HIP][0] + lm[R_HIP][0]) / 2 * W
    for side, (h, k, a) in LEGS.items():
        if min(lm[h][3], lm[k][3], lm[a][3]) < min_vis:
            continue
        hip, knee, ank = (np.array([lm[i][0] * W, lm[i][1] * H]) for i in (h, k, a))
        flex = 180.0 - _angle(hip, knee, ank)
        valgus = None
        if out["frontal"] and abs(ank[1] - hip[1]) > 1e-6:
            x_line = hip[0] + (knee[1] - hip[1]) / (ank[1] - hip[1]) * (ank[0] - hip[0])
            medial = np.sign(knee[0] - x_line) == (np.sign(mid_hip_x - hip[0]) or 1.0)
            dev = 180.0 - _angle(hip, knee, ank)   # frontal-plane deviation when viewed head-on
            valgus = round(dev, 1) if (medial and dev <= 45.0) else 0.0
        out["legs"][side] = {"flexion": round(flex, 1), "valgus": valgus}
    return out


def plausible_leg(lm, W: int, H: int, side: str) -> bool:
    """
    Reject skeletons whose leg geometry is anatomically implausible (MediaPipe sometimes snaps
    landmarks onto a second person or a hand when the player is bent over / occluded).
    """
    h, k, a = LEGS[side]
    hip, knee, ank = (np.array([lm[i][0] * W, lm[i][1] * H]) for i in (h, k, a))
    thigh, shank = np.linalg.norm(knee - hip), np.linalg.norm(ank - knee)
    t = torso_px(lm, W, H)
    if min(thigh, shank, t) < 20:
        return False
    if not 0.6 <= thigh / shank <= 1.7:        # thigh ~ shank in length
        return False
    if not 0.6 <= t / thigh <= 2.2:            # torso ~ 1.3x thigh (less when bent forward)
        return False
    if knee[1] < hip[1] - 0.1 * thigh or ank[1] < knee[1] - 0.15 * shank:   # knee above hip / ankle above knee
        return False
    return True
