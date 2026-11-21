#!/usr/bin/env python3
"""Draw the calibrated court lines over a video frame so you can eyeball the homography.

Usage: python scripts/verify_calibration.py <video> [config.json] [out.jpg] [frame_idx]
If the yellow lines don't sit on the painted lines, re-pick the landmark pixels.
"""
import json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from playq.cv.homography import CourtHomography, PIXELS_PER_METER, COURT_DIMENSIONS


def load_court(cfg_path: str) -> CourtHomography:
    cfg = json.load(open(cfg_path))
    court = CourtHomography(sport=cfg["sport"])
    court.calibrate_landmarks([l["px"] for l in cfg["landmarks"]], [l["court_m"] for l in cfg["landmarks"]])
    return court


if __name__ == "__main__":
    video = sys.argv[1]
    cfg = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "config", "court_calibration.json")
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), "..", "data", "calibration_check.jpg")
    idx = int(sys.argv[4]) if len(sys.argv) > 4 else 80
    cap = cv2.VideoCapture(video); cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, frame = cap.read()
    if not ok:
        sys.exit("cannot read frame")
    court = load_court(cfg)
    wm, hm = COURT_DIMENSIONS[court.sport]
    def px(x, y):
        p = cv2.perspectiveTransform(np.float32([[[x * PIXELS_PER_METER, y * PIXELS_PER_METER]]]), court._inv_matrix)[0, 0]
        return int(p[0]), int(p[1])
    segs = [((0, hm), (wm, hm)), ((wm, hm), (wm, hm / 2)), ((wm, hm / 2), (0, hm / 2)), ((0, hm / 2), (0, hm)),
            ((0, hm - 4.57), (wm, hm - 4.57)), ((wm / 2, hm), (wm / 2, hm - 4.57)), ((wm / 2, hm - 4.57), (wm / 2, hm / 2))]
    for a, b in segs:
        cv2.line(frame, px(*a), px(*b), (0, 255, 255), 3)
    cv2.imwrite(out, frame); print("wrote", out)
