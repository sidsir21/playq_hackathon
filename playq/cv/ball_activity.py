"""
Visual "is the ball in play on OUR court?" detector.

Audio alone fails at tournaments: pops from neighbouring courts and the crowd make 80-90 % of the match
look "active". A resting ball is stationary; a ball in play keeps appearing at new positions. So we find
ball-coloured blobs (optic-yellow) inside the court region and measure how often one shows up somewhere
new between consecutive samples. Players' kit, spectators and other courts are excluded by the court mask.

Limits: needs a calibrated court (homography) and a yellow/green ball that contrasts with the surface.
"""
from __future__ import annotations

import cv2
import numpy as np

from playq.cv.homography import CourtHomography

FAST_MIN_PX, FAST_MAX_PX = 22.0, 300.0   # displacement per sample (10 Hz) that counts as a ball in flight


def court_roi_mask(court: CourtHomography, W: int, H: int, margin_m: float = 1.5, headroom_px: int = 180) -> np.ndarray:
    """Image mask of the court + margin, lifted at the far edge so lobs are inside."""
    wm, hm = court.court_dims
    pts = [(-margin_m, -margin_m), (wm + margin_m, -margin_m), (wm + margin_m, hm + margin_m), (-margin_m, hm + margin_m)]
    P = cv2.perspectiveTransform(np.float32([[[x * 100, y * 100] for x, y in pts]]), court._inv_matrix)[0]
    P[:2, 1] -= headroom_px
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [P.astype(np.int32)], 255)
    return mask


def ball_blobs(frame: np.ndarray, roi: np.ndarray) -> list[np.ndarray]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.bitwise_and(cv2.inRange(hsv, (22, 100, 140), (38, 255, 255)), roi)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    out = []
    for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        a = cv2.contourArea(c)
        if 6 <= a <= 300:
            per = cv2.arcLength(c, True)
            if 4 * np.pi * a / (per * per + 1e-6) > 0.55:     # round
                (x, y), _ = cv2.minEnclosingCircle(c)
                out.append(np.array([x, y]))
    return out


def ball_motion_profile(video_path: str, court: CourtHomography, sample_hz: float = 10.0, bin_s: float = 2.0,
                        progress=None, margin_m: float = 0.8, headroom_px: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (bin_start_times, activity) where activity in [0, 1] is the share of consecutive samples in each
    `bin_s` bin at which a ball-coloured blob appeared at a new position.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi = court_roi_mask(court, W, H, margin_m=margin_m, headroom_px=headroom_px)   # tight: a neighbouring court's ball must not count
    stride = max(1, int(round(fps / sample_hz)))
    nbins = int(np.ceil(total / fps / bin_s)) + 1
    moved, count = np.zeros(nbins), np.zeros(nbins)
    prev, i = None, 0
    while True:
        if i % stride:
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        cur = ball_blobs(frame, roi)
        b = int(i / fps / bin_s)
        if prev is not None:
            count[b] += 1
            # a ball IN PLAY jumps tens of pixels between samples; a resting or hand-carried ball does not
            if prev and any(FAST_MIN_PX <= min(np.linalg.norm(c - q) for q in prev) <= FAST_MAX_PX for c in cur):
                moved[b] += 1
        prev = cur
        if progress and i % (stride * 300) == 0:
            progress(i / max(total, 1))
        i += 1
    cap.release()
    return np.arange(nbins) * bin_s, np.where(count > 0, moved / np.maximum(count, 1), 0.0)


def live_play_segments(starts: np.ndarray, activity: np.ndarray, bin_s: float = 2.0, thresh: float = 0.10,
                       max_gap_s: float = 3.0, min_len_s: float = 6.0) -> list[tuple[float, float]]:
    """Active bins -> merged (start, end) segments; short gaps are bridged, short blips dropped."""
    act = activity >= thresh
    segs: list[list[float]] = []
    for t, a in zip(starts, act):
        if not a:
            continue
        if segs and t - segs[-1][1] <= max_gap_s:
            segs[-1][1] = t + bin_s
        else:
            segs.append([t, t + bin_s])
    return [(s, e) for s, e in segs if e - s >= min_len_s]
