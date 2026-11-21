"""Human-readable player identity: a fixed colour, a shirt-colour name and a cropped photo."""
from __future__ import annotations

import cv2
import numpy as np

# id -> (hex, BGR). Chosen to stay distinguishable on a dark UI and for common colour-blindness.
PALETTE = {
    1: ("#22D3A5", (165, 211, 34)),    # teal
    2: ("#FF6B9D", (157, 107, 255)),   # pink
    3: ("#FFD166", (102, 209, 255)),   # amber
    4: ("#6EA8FF", (255, 168, 110)),   # sky blue
}


def accent(track_id: int) -> tuple[str, tuple]:
    return PALETTE[((track_id - 1) % len(PALETTE)) + 1]


def color_name(bgr) -> str:
    """Rough human colour word for a shirt from its median BGR."""
    b, g, r = (int(v) for v in bgr)
    h, s, v = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
    if v < 55 or (s < 60 and v < 110):      # charcoal under court lights reads as black
        return "black"
    if s < 45:
        return "white" if v > 185 else "grey"
    if v > 150 and s < 85:                 # white shirts under warm indoor lighting read as pale orange/yellow
        return "white"
    hue = int(h)
    if hue < 4 or hue >= 172:
        base = "red"
    elif hue < 20:
        base = "orange"
    elif hue < 34:
        base = "yellow"
    elif hue < 85:
        base = "green"
    elif hue < 130:
        base = "blue"
    elif hue < 160:
        base = "purple"
    else:
        base = "pink"
    return ("dark " + base) if v < 105 else base


def torso_color(frame: np.ndarray, lm) -> tuple | None:
    """Median BGR of the torso region (between shoulders and hips, edges trimmed)."""
    H, W = frame.shape[:2]
    xs = [lm[i][0] * W for i in (11, 12, 23, 24)]
    ys = [lm[i][1] * H for i in (11, 12, 23, 24)]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    dx, dy = (x1 - x0) * 0.25, (y1 - y0) * 0.2
    x0, x1, y0, y1 = int(max(0, x0 + dx)), int(min(W, x1 - dx)), int(max(0, y0 + dy)), int(min(H, y1 - dy))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    return tuple(float(v) for v in np.median(frame[y0:y1, x0:x1].reshape(-1, 3), axis=0))


def player_crop(frame: np.ndarray, lm, height: int = 300) -> tuple[np.ndarray, float] | None:
    """Head-to-toe crop of one player and a quality score (bigger + more visible = better)."""
    H, W = frame.shape[:2]
    pts = [(l[0] * W, l[1] * H) for l in lm if l[3] > 0.4]
    if len(pts) < 14:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if y1 - y0 < 50 or x0 < 2 or y0 < 2 or x1 > W - 2 or y1 > H - 2:   # skip tiny or edge-clipped players
        return None
    pw, ph = (x1 - x0) * 0.35 + 12, (y1 - y0) * 0.12 + 10
    crop = frame[int(max(0, y0 - ph)):int(min(H, y1 + ph)), int(max(0, x0 - pw)):int(min(W, x1 + pw))]
    sc = height / crop.shape[0]
    crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    return crop, float((y1 - y0) * np.mean([l[3] for l in lm]))
