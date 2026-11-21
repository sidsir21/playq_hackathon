"""
PlayQ Player Tracker
====================
Gives every on-court player a stable ID across frames (the pitch's "ID lock").

Per sampled frame:
  1. Each detected pose is projected to court coordinates (metres) via the stabilised
     homography. People standing outside the court (bystanders, bags) are rejected.
  2. Each remaining pose gets an appearance signature: an HSV colour histogram of the torso.
  3. Detections are matched to existing tracks with the Hungarian algorithm, using a blend of
     appearance distance and court-position distance. Tracks are never deleted, so a player who
     leaves the frame or is occluded re-acquires the same ID when they come back.

Limitations (be honest in the demo):
  - Teammates in near-identical kit rely mostly on position continuity; ID swaps are possible
    when they cross paths. Sampling every ~0.17 s keeps motion between samples small.
  - Not yet validated against hand-labelled ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from playq.cv.homography import CourtHomography, PIXELS_PER_METER
from playq.cv.pose_analyzer import PoseFrame

L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK = 11, 12, 23, 24, 27, 28


@dataclass
class Track:
    id: int
    hist: np.ndarray                     # running appearance signature
    court_m: tuple[float, float]         # last known (x, y) in metres
    last_t: float
    samples: list[tuple[float, float, float]] = field(default_factory=list)   # (t_s, x_m, y_m)
    poses: list[PoseFrame] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.samples)

    def side(self, net_y: float) -> int:
        """+1 = near half (y > net), -1 = far half. Long-run median once established, so a player standing at the
        net (whose measured depth jitters across the net line) does not flip sides."""
        ys = [smp[2] for smp in (self.samples if len(self.samples) >= 8 else self.samples[-5:])] or [self.court_m[1]]
        return 1 if float(np.median(ys)) > net_y else -1


class PlayerTracker:
    def __init__(
        self,
        court: CourtHomography,
        max_players: int = 4,
        margin_side_m: float = 0.6,    # people standing >0.6 m outside the sidelines are bystanders
        margin_end_m: float = 1.5,     # players do retreat behind the baselines
        w_appearance: float = 0.5,
        w_position: float = 0.5,
        max_cost: float = 0.8,
        enforce_sides: bool = True,
    ):
        self.court = court
        self.max_players = max_players
        self.margin_x, self.margin_y = margin_side_m, margin_end_m
        self.w_app, self.w_pos, self.max_cost = w_appearance, w_position, max_cost
        # In net sports (pickleball, padel, badminton) nobody crosses to the other half mid-rally,
        # so a detection on the opposite side can never be the same player. This resolves
        # look-alike kits where appearance alone swaps IDs. Turn off for football/cricket.
        self.enforce_sides = enforce_sides
        self.net_band_m = 0.8   # |y - net| below this: side is ambiguous (measurement noise at the net)
        self.tracks: list[Track] = []
        self.rejected_outside = 0   # detections discarded as bystanders
        self.estimated_feet = 0     # positions extrapolated from the knees (feet out of frame)
        self.rejected_no_feet = 0   # detections discarded because ankles weren't visible
        self._next_id = 1
        self.last_assigned: list = []   # (id, pose, xy) accepted in the most recent update()

    # ── appearance ────────────────────────────────────────────────────────────
    @staticmethod
    def _signature(frame: np.ndarray, lm: list) -> np.ndarray | None:
        """HSV histogram of the torso box (between shoulders and hips), insets trimmed."""
        h, w = frame.shape[:2]
        xs = [lm[i][0] * w for i in (L_SH, R_SH, L_HIP, R_HIP)]
        ys = [lm[i][1] * h for i in (L_SH, R_SH, L_HIP, R_HIP)]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        dx, dy = (x1 - x0) * 0.2, (y1 - y0) * 0.15
        x0, x1, y0, y1 = int(max(0, x0 + dx)), int(min(w, x1 - dx)), int(max(0, y0 + dy)), int(min(h, y1 - dy))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        return cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1).flatten().astype(np.float32)

    @staticmethod
    def _app_dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))  # 0 identical .. 1 disjoint

    # ── geometry ──────────────────────────────────────────────────────────────
    def _court_position(self, pose: PoseFrame, to_ref: np.ndarray | None, w: int, h: int):
        lm = pose.landmarks
        if lm[L_ANK][3] >= 0.5 and lm[R_ANK][3] >= 0.5:
            foot = np.float32([[((lm[L_ANK][0] + lm[R_ANK][0]) / 2 * w, (lm[L_ANK][1] + lm[R_ANK][1]) / 2 * h)]])
        elif all(lm[i][3] >= 0.5 for i in (23, 24, 25, 26)) and max(lm[25][1], lm[26][1]) > 0.72:
            # Feet cut off by the BOTTOM of the frame (player close to the camera; knees low in the image): extrapolate the ankle from hip->knee
            # (shank ~ thigh length). Less precise than a visible ankle, but better than losing the player.
            fx = (lm[25][0] + lm[26][0]) / 2 + ((lm[25][0] + lm[26][0]) / 2 - (lm[23][0] + lm[24][0]) / 2) * 0.95
            fy = (lm[25][1] + lm[26][1]) / 2 + ((lm[25][1] + lm[26][1]) / 2 - (lm[23][1] + lm[24][1]) / 2) * 0.95
            foot = np.float32([[(fx * w, fy * h)]])
            self.estimated_feet += 1
        else:
            return None
        if to_ref is not None:
            foot = cv2.perspectiveTransform(foot.reshape(1, 1, 2), to_ref).reshape(1, 2)
        x, y = self.court.transform_point(tuple(foot[0]))
        return x / PIXELS_PER_METER, y / PIXELS_PER_METER

    def _on_court(self, xy: tuple[float, float]) -> bool:
        wm, hm = self.court.court_dims
        return -self.margin_x <= xy[0] <= wm + self.margin_x and -self.margin_y <= xy[1] <= hm + self.margin_y

    # ── main entry ────────────────────────────────────────────────────────────
    def update(
        self,
        frame: np.ndarray,
        poses: list[PoseFrame],
        t_s: float,
        to_ref: np.ndarray | None = None,
    ) -> list[tuple[int, PoseFrame, tuple[float, float]]]:
        """
        Assign IDs for one frame.

        Args:
            frame:  BGR frame (for the appearance signature).
            poses:  all people detected in this frame (PoseAnalyzer.analyze_frame_all).
            t_s:    timestamp in seconds.
            to_ref: 3x3 homography current-frame -> calibration reference frame (CameraStabilizer).

        Returns:
            [(track_id, pose, (x_m, y_m))] for people accepted this frame.
        """
        h, w = frame.shape[:2]
        self.last_assigned = []
        dets = []
        for pose in poses:
            xy = self._court_position(pose, to_ref, w, h)
            if xy is None:
                self.rejected_no_feet += 1
                continue
            if not self._on_court(xy):
                self.rejected_outside += 1
                continue
            sig = self._signature(frame, pose.landmarks)
            if sig is None:
                continue
            dets.append((pose, xy, sig))
        if not dets:
            return []

        # One person is often detected twice (full-frame pass + a tiled pass) with slightly different landmarks. Two
        # real players are never within 0.6 m of each other, so keep only the biggest detection of each such pair;
        # otherwise the duplicate opens a second track for the same person.
        dets.sort(key=lambda d_: -d_[0].torso_px)
        kept = []
        for d_ in dets:
            if all(np.hypot(d_[1][0] - k_[1][0], d_[1][1] - k_[1][1]) > 0.6 for k_ in kept):
                kept.append(d_)
        dets = kept

        net_y = self.court.court_dims[1] / 2
        assigned: dict[int, int] = {}  # det index -> track index
        if self.tracks:
            cost = np.ones((len(dets), len(self.tracks)), dtype=np.float64)
            for i, (_, xy, sig) in enumerate(dets):
                for j, tr in enumerate(self.tracks):
                    dt = max(0.05, t_s - tr.last_t)
                    allowed = 1.0 + 5.0 * dt   # metres a person could plausibly cover (+ slack)
                    dist = float(np.hypot(xy[0] - tr.court_m[0], xy[1] - tr.court_m[1]))
                    in_net_band = abs(xy[1] - net_y) <= self.net_band_m   # too close to the net to know the side
                    if self.enforce_sides and not in_net_band and (1 if xy[1] > net_y else -1) != tr.side(net_y):
                        cost[i, j] = 2.0   # impossible: would require crossing the net
                        continue
                    cost[i, j] = self.w_app * self._app_dist(sig, tr.hist) + self.w_pos * min(1.0, dist / allowed)
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] <= self.max_cost:
                    assigned[i] = j

        out = []
        for i, (pose, xy, sig) in enumerate(dets):
            if i in assigned:
                tr = self.tracks[assigned[i]]
                tr.hist = (0.9 * tr.hist + 0.1 * sig).astype(np.float32)  # slow EMA keeps ID stable
            elif len(self.tracks) < self.max_players and abs(xy[1] - net_y) > (self.net_band_m if self.enforce_sides else 0):
                tr = Track(id=self._next_id, hist=sig, court_m=xy, last_t=t_s)
                self._next_id += 1
                self.tracks.append(tr)
            else:
                continue  # more people than expected and no good match -> ignore
            tr.court_m, tr.last_t = xy, t_s
            tr.samples.append((t_s, xy[0], xy[1]))
            tr.poses.append(pose)
            out.append((tr.id, pose, xy))
        self.last_assigned = out
        return out

    # ── post-processing ───────────────────────────────────────────────────────
    def merge_fragments(self, max_overlap: float = 0.05, max_app_dist: float = 0.5) -> int:
        """
        Merge tracks that are really one person. Two tracks on the same side are merged when
          (a) they almost never exist at the same instant (one person cannot be in two places), and
          (b) their shirt-colour signatures are similar.
        Returns the number of merges. Fragmentation happens when a player's detection flickers (feet at the frame
        edge, occlusion) and the matcher opens a new track instead of re-using the old one.
        """
        net_y = self.court.court_dims[1] / 2
        merges = 0
        changed = True
        while changed:
            changed = False
            for i in range(len(self.tracks)):
                for j in range(i + 1, len(self.tracks)):
                    a, b = self.tracks[i], self.tracks[j]
                    if a.n < 5 or b.n < 5:
                        continue
                    if (np.median([x[2] for x in a.samples]) > net_y) != (np.median([x[2] for x in b.samples]) > net_y):
                        continue
                    ta, tb = {round(x[0], 1) for x in a.samples}, {round(x[0], 1) for x in b.samples}
                    overlap = len(ta & tb) / max(1, min(len(ta), len(tb)))
                    if overlap > max_overlap or self._app_dist(a.hist, b.hist) > max_app_dist:
                        continue
                    pairs = sorted(zip(a.samples + b.samples, a.poses + b.poses), key=lambda sp: sp[0][0])
                    a.samples, a.poses = [sp[0] for sp in pairs], [sp[1] for sp in pairs]
                    a.hist = ((a.hist * a.n + b.hist * b.n) / (a.n + b.n)).astype(np.float32)
                    self.tracks.pop(j)
                    merges += 1
                    changed = True
                    break
                if changed:
                    break
        return merges

    # ── reporting ─────────────────────────────────────────────────────────────
    def summary(self) -> list[dict]:
        wm, hm = self.court.court_dims
        net_y = hm / 2
        rows = []
        for tr in self.tracks:
            ys = np.array([s[2] for s in tr.samples])
            rows.append({
                "track_id": tr.id,
                "samples": tr.n,
                "side": "near" if np.median(ys) > net_y else "far",
                "median_xy_m": (round(float(np.median([s[1] for s in tr.samples])), 2), round(float(np.median(ys)), 2)),
            })
        return rows
