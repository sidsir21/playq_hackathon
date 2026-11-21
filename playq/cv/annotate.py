"""Draw the annotated skeleton frame (the 'proof' image for the biomechanics claim)."""
from __future__ import annotations

import cv2
import numpy as np

BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
         (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32), (27, 29), (28, 30), (0, 11), (0, 12)]
LEG_IDX = {"left": (23, 25, 27), "right": (24, 26, 28)}
RED, GREEN, WHITE, AMBER = (60, 60, 255), (120, 230, 120), (255, 255, 255), (0, 190, 255)


def annotate_knee_frame(frame, lm, side: str, metrics: dict, verdict: str, level: str, label_prefix: str = "") -> np.ndarray:
    """Skeleton over the player (crop), flagged leg in red, knee-angle arc + degrees, verdict banner."""
    H, W = frame.shape[:2]
    pts = {i: (int(lm[i][0] * W), int(lm[i][1] * H)) for i in range(33) if lm[i][3] > 0.4}
    xs, ys = [p[0] for p in pts.values()], [p[1] for p in pts.values()]
    pad_x, pad_y = int(0.9 * (max(xs) - min(xs)) + 60), int(0.35 * (max(ys) - min(ys)) + 50)
    x0, x1 = max(0, min(xs) - pad_x), min(W, max(xs) + pad_x)
    y0, y1 = max(0, min(ys) - pad_y), min(H, max(ys) + pad_y)
    img = frame.copy()
    hip, knee, ank = (pts.get(i) for i in LEG_IDX[side])
    upper_ok = True
    if all(i in pts for i in (11, 12, 23, 24)) and hip and knee:
        torso = np.linalg.norm(np.array(pts[11]) / 2 + np.array(pts[12]) / 2 - np.array(pts[23]) / 2 - np.array(pts[24]) / 2)
        thigh = np.linalg.norm(np.array(hip) - np.array(knee)) + 1e-6
        upper_ok = 0.6 <= torso / thigh <= 2.2   # otherwise the upper-body landmarks are likely on someone else
    for a, b in BONES:
        if not upper_ok and (a < 23 or b < 23):
            continue
        if a in pts and b in pts:
            flagged = a in LEG_IDX[side] and b in LEG_IDX[side]
            cv2.line(img, pts[a], pts[b], RED if flagged else GREEN, 6 if flagged else 3, cv2.LINE_AA)
    for i, p in pts.items():
        if upper_ok or i >= 23:
            cv2.circle(img, p, 5, WHITE, -1, cv2.LINE_AA)
    if hip and knee and ank:
        # arc at the knee between the thigh and shank directions
        a1 = np.degrees(np.arctan2(hip[1] - knee[1], hip[0] - knee[0]))
        a2 = np.degrees(np.arctan2(ank[1] - knee[1], ank[0] - knee[0]))
        cv2.ellipse(img, knee, (46, 46), 0, min(a1, a2), max(a1, a2) if abs(a1 - a2) < 180 else min(a1, a2) + 360 - abs(a1 - a2), AMBER, 4, cv2.LINE_AA)
        cv2.circle(img, knee, 11, RED, -1, cv2.LINE_AA)
    crop = img[y0:y1, x0:x1]
    scale = 900 / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # text overlays (positions in crop space)
    if knee:
        kx, ky = int((knee[0] - x0) * scale), int((knee[1] - y0) * scale)
        txt = f"KNEE {metrics['flexion']:.0f} deg"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
        tx = min(max(10, kx + 60), crop.shape[1] - tw - 10)
        cv2.rectangle(crop, (tx - 8, ky - th - 10), (tx + tw + 8, ky + 12), (0, 0, 0), -1)
        cv2.putText(crop, txt, (tx, ky), cv2.FONT_HERSHEY_DUPLEX, 1.1, AMBER, 2, cv2.LINE_AA)
    banner_h = 86
    out = np.zeros((crop.shape[0] + banner_h, crop.shape[1], 3), np.uint8)
    out[:crop.shape[0]] = crop
    colour = {"high": RED, "elevated": AMBER, "normal": GREEN}.get(level, WHITE)
    cv2.putText(out, f"{label_prefix}{verdict}", (14, crop.shape[0] + 34), cv2.FONT_HERSHEY_DUPLEX, 0.9, colour, 2, cv2.LINE_AA)
    val = metrics.get("valgus")
    sub = f"flexion {metrics['flexion']:.0f} deg" + (f"  |  valgus {val:.0f} deg" if val is not None else "  |  valgus: not visible from this angle")
    cv2.putText(out, sub, (14, crop.shape[0] + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 1, cv2.LINE_AA)
    return out
