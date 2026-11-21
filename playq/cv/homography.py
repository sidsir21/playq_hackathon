"""
PlayQ Court Homography Module
=============================
Perspective transformation of sports court footage to a bird's-eye 2D view.
Supports: pickleball, padel, turf_football, box_cricket, badminton.
"""

import cv2
import numpy as np
from typing import Optional

# Court dimensions in meters (width, height)
COURT_DIMENSIONS: dict[str, tuple[float, float]] = {
    "pickleball": (6.10, 13.41),    # 20ft x 44ft
    "padel": (10.0, 20.0),
    "turf_football": (30.0, 50.0),  # 7v7
    "box_cricket": (18.0, 22.0),
    "badminton": (6.1, 13.4),
}

# Court-specific line features (normalized 0-1 relative to dimensions)
COURT_LINES: dict[str, list[tuple[float, float, float, float]]] = {
    "pickleball": [
        # Boundary
        (0, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 1), (0, 1, 0, 0),
        # Net at the mid-line; non-volley zone ("kitchen") lines are 7 ft = 2.13 m EITHER SIDE OF THE NET
        (0, 0.5, 1, 0.5),
        (0, 0.5 - 0.1588, 1, 0.5 - 0.1588), (0, 0.5 + 0.1588, 1, 0.5 + 0.1588),
        # Centre service lines run baseline -> kitchen line only (not through the kitchen)
        (0.5, 0, 0.5, 0.5 - 0.1588), (0.5, 0.5 + 0.1588, 0.5, 1),
    ],
    "padel": [
        (0, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 1), (0, 1, 0, 0),
        (0, 0.5, 1, 0.5),  # Net
        (0, 0.35, 1, 0.35), (0, 0.65, 1, 0.65),  # Service lines
    ],
    "turf_football": [
        (0, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 1), (0, 1, 0, 0),
        (0, 0.5, 1, 0.5),  # Halfway line
        (0.35, 0, 0.65, 0),  # Goal area top
        (0.35, 1, 0.65, 1),  # Goal area bottom
    ],
    "box_cricket": [
        (0, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 1), (0, 1, 0, 0),
        (0, 0.5, 1, 0.5),  # Pitch center
    ],
    "badminton": [
        (0, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 1), (0, 1, 0, 0),
        (0, 0.5, 1, 0.5),  # Net
        (0, 0.097, 1, 0.097), (0, 0.903, 1, 0.903),  # Short service
    ],
}

PIXELS_PER_METER = 100


class CourtHomography:
    """Handles perspective calibration and point transformation for sports courts."""

    def __init__(self, sport: str = "pickleball"):
        if sport not in COURT_DIMENSIONS:
            raise ValueError(f"Unknown sport '{sport}'. Choose from: {list(COURT_DIMENSIONS.keys())}")
        self.sport = sport
        self._width_m, self._height_m = COURT_DIMENSIONS[sport]
        self._matrix: Optional[np.ndarray] = None
        self._inv_matrix: Optional[np.ndarray] = None

    @property
    def court_dims(self) -> tuple[float, float]:
        """Returns (width_m, height_m) of the court."""
        return (self._width_m, self._height_m)

    @property
    def court_pixels(self) -> tuple[int, int]:
        """Returns (width_px, height_px) of the court in the 2D map."""
        return (int(self._width_m * PIXELS_PER_METER), int(self._height_m * PIXELS_PER_METER))

    def calibrate(self, src_points: np.ndarray) -> np.ndarray:
        """
        Compute the perspective transform from 4 camera-frame corner points
        to the top-down court coordinate system.

        Args:
            src_points: Shape (4, 2) float32 — corners in camera frame
                        ordered: top-left, top-right, bottom-right, bottom-left.

        Returns:
            The 3x3 homography matrix.
        """
        src = np.float32(src_points).reshape(4, 2)
        w_px, h_px = self.court_pixels
        dst = np.float32([
            [0, 0],
            [w_px, 0],
            [w_px, h_px],
            [0, h_px],
        ])
        self._matrix = cv2.getPerspectiveTransform(src, dst)
        self._inv_matrix = cv2.getPerspectiveTransform(dst, src)
        return self._matrix

    def calibrate_landmarks(self, src_px: np.ndarray, court_m: np.ndarray) -> np.ndarray:
        """
        Calibrate from >=4 known court landmarks (does not need the four outer corners).

        Args:
            src_px:  (N, 2) pixel coordinates of landmarks in the camera frame.
            court_m: (N, 2) the same landmarks in metres: x across the court (0..width),
                     y along the court (0..length, 0 = far baseline).

        Returns:
            The 3x3 homography (camera px -> court-map px at PIXELS_PER_METER).
        """
        src = np.float32(src_px).reshape(-1, 2)
        dst = np.float32(court_m).reshape(-1, 2) * PIXELS_PER_METER
        if len(src) < 4 or len(src) != len(dst):
            raise ValueError("Need >=4 matching landmark pairs")
        self._matrix, _ = cv2.findHomography(src, dst, method=0)
        self._inv_matrix = np.linalg.inv(self._matrix)
        return self._matrix

    def _ensure_calibrated(self):
        if self._matrix is None:
            raise RuntimeError("Call calibrate() with 4 court corners before transforming points.")

    def transform_point(self, point: tuple[float, float]) -> tuple[float, float]:
        """Transform a single (x, y) from camera space to court space."""
        self._ensure_calibrated()
        pts = np.float32([[point]]).reshape(1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self._matrix)
        x, y = transformed[0, 0]
        return (float(x), float(y))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Batch-transform an (N, 2) array of camera-space points to court space."""
        self._ensure_calibrated()
        pts = np.float32(points).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self._matrix).reshape(-1, 2)

    def generate_heatmap(
        self,
        positions: list[tuple[float, float]],
        resolution: tuple[int, int] = (500, 1000),
    ) -> np.ndarray:
        """
        Generate a 2D heatmap image from court-space positions.

        Args:
            positions: List of (x, y) in court-space pixel coordinates.
            resolution: Output image (height, width).

        Returns:
            BGR numpy array ready for cv2.imwrite.
        """
        h, w = resolution
        w_px, h_px = self.court_pixels

        # Accumulation map
        acc = np.zeros((h, w), dtype=np.float32)
        for px, py in positions:
            # Scale court coords to resolution
            ix = int(px / w_px * w) if w_px > 0 else 0
            iy = int(py / h_px * h) if h_px > 0 else 0
            if 0 <= ix < w and 0 <= iy < h:
                acc[iy, ix] += 1.0

        # Gaussian blur for smooth heatmap
        ksize = max(31, int(min(h, w) * 0.08) | 1)  # ensure odd
        acc = cv2.GaussianBlur(acc, (ksize, ksize), 0)

        # Normalize to 0-255
        if acc.max() > 0:
            acc = (acc / acc.max() * 255).astype(np.uint8)
        else:
            acc = acc.astype(np.uint8)

        # Apply color map
        heatmap = cv2.applyColorMap(acc, cv2.COLORMAP_JET)

        # Draw court lines
        lines = COURT_LINES.get(self.sport, COURT_LINES["pickleball"])
        for x1_n, y1_n, x2_n, y2_n in lines:
            pt1 = (int(x1_n * w), int(y1_n * h))
            pt2 = (int(x2_n * w), int(y2_n * h))
            cv2.line(heatmap, pt1, pt2, (255, 255, 255), 2)

        # Sport label
        cv2.putText(
            heatmap,
            f"PlayQ | {self.sport.replace('_', ' ').title()}",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        return heatmap

    def generate_player_heatmap(
        self,
        tracks: dict,
        colors: dict,
        labels: dict | None = None,
        resolution: tuple[int, int] = (700, 350),
    ) -> np.ndarray:
        """
        Top-down court with one colour per player (so players are easy to tell apart).

        Args:
            tracks: {player_id: [(x, y) in court-map px]}  (px = metres * PIXELS_PER_METER)
            colors: {player_id: (B, G, R)}
            labels: {player_id: text} for the legend.
        """
        h, w = resolution
        w_px, h_px = self.court_pixels
        canvas = np.full((h, w, 3), (38, 30, 22), np.uint8)               # dark court surface
        zone = np.zeros((h, w), np.uint8)
        lines = COURT_LINES.get(self.sport, COURT_LINES["pickleball"])
        if self.sport == "pickleball":                                     # tint the kitchens
            for y0n, y1n in ((0.5 - 0.1588, 0.5), (0.5, 0.5 + 0.1588)):
                canvas[int(y0n * h):int(y1n * h)] = (52, 44, 30)
        canvas = canvas.astype(np.float32)
        for pid, pts in tracks.items():
            acc = np.zeros((h, w), np.float32)
            for px, py in pts:
                ix = int(px / w_px * w) if w_px else 0
                iy = int(py / h_px * h) if h_px else 0
                if 0 <= ix < w and 0 <= iy < h:
                    acc[iy, ix] += 1.0
            k = max(31, int(min(h, w) * 0.09) | 1)
            acc = cv2.GaussianBlur(acc, (k, k), 0)
            if acc.max() <= 0:
                continue
            alpha = np.clip(acc / acc.max(), 0, 1) ** 0.6 * 0.92          # gamma lifts faint areas
            col = np.array(colors.get(pid, (255, 255, 255)), np.float32)
            canvas = canvas * (1 - alpha[..., None]) + col * alpha[..., None]
        out = canvas.astype(np.uint8)
        for x1n, y1n, x2n, y2n in lines:
            cv2.line(out, (int(x1n * w), int(y1n * h)), (int(x2n * w), int(y2n * h)), (235, 235, 235), 2)
        cv2.putText(out, "NET", (w - 48, int(0.5 * h) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        if labels:   # legend strip
            strip = np.full((34 * ((len(labels) + 1) // 2) + 8, w, 3), (20, 16, 12), np.uint8)
            for n, (pid, text) in enumerate(labels.items()):
                cx, cy = 10 + (n % 2) * (w // 2), 8 + (n // 2) * 34
                cv2.rectangle(strip, (cx, cy), (cx + 22, cy + 22), colors.get(pid, (255, 255, 255)), -1)
                cv2.putText(strip, text[:22], (cx + 30, cy + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
            out = np.vstack([out, strip])
        return out

    def draw_court_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw the court boundary and key lines on a camera-frame image."""
        self._ensure_calibrated()
        out = frame.copy()
        h, w_frame = out.shape[:2]
        w_px, h_px = self.court_pixels

        lines = COURT_LINES.get(self.sport, COURT_LINES["pickleball"])
        for x1_n, y1_n, x2_n, y2_n in lines:
            # Court-space points
            pt1_court = np.float32([[x1_n * w_px, y1_n * h_px]]).reshape(1, 1, 2)
            pt2_court = np.float32([[x2_n * w_px, y2_n * h_px]]).reshape(1, 1, 2)
            # Transform back to camera space
            pt1_cam = cv2.perspectiveTransform(pt1_court, self._inv_matrix)[0, 0]
            pt2_cam = cv2.perspectiveTransform(pt2_court, self._inv_matrix)[0, 0]
            cv2.line(out, tuple(pt1_cam.astype(int)), tuple(pt2_cam.astype(int)), (0, 255, 0), 2)

        return out


def detect_court_corners(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Attempt automatic court corner detection using Canny + HoughLinesP.

    Returns:
        4 corner points as (4, 2) float32 ordered TL, TR, BR, BL — or None if detection fails.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=10)
    if lines is None or len(lines) < 4:
        return None

    # Collect all line endpoints
    points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        points.append((x1, y1))
        points.append((x2, y2))

    points = np.array(points, dtype=np.float32)

    # Find the convex hull and approximate to 4 points
    hull = cv2.convexHull(points)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    if len(approx) != 4:
        # Try harder: find the 4 extreme points
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        approx = np.int32(box).reshape(-1, 1, 2)

    corners = approx.reshape(4, 2).astype(np.float32)

    # Sort: top-left, top-right, bottom-right, bottom-left
    s = corners.sum(axis=1)
    d = np.diff(corners, axis=1).flatten()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = corners[np.argmin(s)]   # top-left
    ordered[2] = corners[np.argmax(s)]   # bottom-right
    ordered[1] = corners[np.argmin(d)]   # top-right
    ordered[3] = corners[np.argmax(d)]   # bottom-left

    return ordered


if __name__ == "__main__":
    import os

    print("🏓 PlayQ Court Homography Demo")
    print("=" * 40)

    # Create a synthetic 1280x720 frame with a trapezoid "court"
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Simulated court corners as seen by an angled camera
    src_corners = np.float32([
        [300, 200],   # top-left
        [980, 200],   # top-right
        [1100, 600],  # bottom-right
        [180, 600],   # bottom-left
    ])

    # Draw the "court" on the frame
    cv2.polylines(frame, [src_corners.astype(int)], True, (0, 255, 0), 2)

    # Calibrate
    court = CourtHomography("pickleball")
    matrix = court.calibrate(src_corners)
    print(f"Sport: {court.sport}")
    print(f"Court dims: {court.court_dims[0]}m x {court.court_dims[1]}m")
    print(f"Court pixels: {court.court_pixels}")
    print(f"Homography matrix:\n{matrix}")

    # Transform some test points
    test_point = (640, 400)
    court_point = court.transform_point(test_point)
    print(f"\nCamera point {test_point} → Court point ({court_point[0]:.1f}, {court_point[1]:.1f})")

    # Generate heatmap with random positions
    np.random.seed(42)
    w_px, h_px = court.court_pixels
    positions = [
        (np.random.uniform(0, w_px), np.random.uniform(0, h_px))
        for _ in range(500)
    ]
    # Bias toward kitchen area
    for _ in range(200):
        positions.append((np.random.uniform(0, w_px), np.random.uniform(h_px * 0.35, h_px * 0.65)))

    heatmap = court.generate_heatmap(positions)
    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_heatmap.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, heatmap)
    print(f"\n✅ Heatmap saved to {os.path.abspath(out_path)}")
    print(f"   Resolution: {heatmap.shape[1]}x{heatmap.shape[0]}")


class CameraStabilizer:
    """
    Tracks camera motion relative to a reference frame so one court calibration stays valid
    when the camera is handheld / vibrating. Matches ORB features between each frame and the
    reference (players are rejected as RANSAC outliers) and returns H mapping current-frame
    pixels -> reference-frame pixels.
    """

    def __init__(self, reference_frame: np.ndarray, n_features: int = 3000):
        self._orb = cv2.ORB_create(n_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        self._kp_ref, self._des_ref = self._orb.detectAndCompute(gray, None)
        self._last = np.eye(3)
        self.last_inliers = 0   # RANSAC inliers of the latest match; low => replay / cut / dissolve

    def to_reference(self, frame: np.ndarray) -> np.ndarray:
        """3x3 homography current->reference. Falls back to the last good estimate."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self._orb.detectAndCompute(gray, None)
        self.last_inliers = 0
        if des is None or self._des_ref is None:
            return self._last
        matches = self._bf.match(des, self._des_ref)
        if len(matches) < 30:
            return self._last
        src = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([self._kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None or mask is None or int(mask.sum()) < 25:
            return self._last
        self._last = H
        self.last_inliers = int(mask.sum())
        return H

    @staticmethod
    def map_point(H: np.ndarray, pt: tuple[float, float]) -> tuple[float, float]:
        out = cv2.perspectiveTransform(np.float32([[pt]]), H)[0, 0]
        return float(out[0]), float(out[1])
