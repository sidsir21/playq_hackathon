#!/usr/bin/env python3
"""Sanity-check the heatmap: draw every tracked foot position + the calibrated court lines on the
calibration reference frame.  Usage: python scripts/verify_positions.py <run_dir> <calibration.json> [out.jpg]"""
import json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from playq.cv.homography import CourtHomography, PIXELS_PER_METER

run, calib = sys.argv[1], sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(run, "positions_check.jpg")
s2 = json.load(open(os.path.join(run, "stage2.json"))); cfg = json.load(open(calib))
court = CourtHomography(cfg["sport"])
court.calibrate_landmarks([l["px"] for l in cfg["landmarks"]], [l["court_m"] for l in cfg["landmarks"]])
cap = cv2.VideoCapture(s2["video"]); fps = s2["fps"]
idx = cfg.get("reference_frame", int(round(cfg.get("reference_time_s", 0) * fps)))
cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, img = cap.read()
inv = court._inv_matrix
def px(x, y):
    p = cv2.perspectiveTransform(np.float32([[[x * PIXELS_PER_METER, y * PIXELS_PER_METER]]]), inv)[0, 0]
    return int(p[0]), int(p[1])
W, L = court.court_dims
for a, b in [((0,0),(W,0)),((W,0),(W,L)),((W,L),(0,L)),((0,L),(0,0)),((0,L/2),(W,L/2)),((0,L/2-2.13),(W,L/2-2.13)),((0,L/2+2.13),(W,L/2+2.13))]:
    cv2.line(img, px(*a), px(*b), (0, 255, 255), 2)
cols = [(255,211,34),(157,107,255),(102,209,255),(255,168,110)]
for p in s2["players"]:
    c = cols[(p["track_id"] - 1) % 4]
    for t, x, y in p["points"]:
        cv2.circle(img, px(x, y), 5, c, -1)
    cv2.putText(img, p.get("label", f'P{p["track_id"]}'), (20, 40 + 34 * p["track_id"]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)
cv2.imwrite(out, img); print("wrote", out)
