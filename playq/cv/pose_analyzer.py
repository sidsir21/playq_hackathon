"""
PlayQ Pose Analyzer Module
===========================
Extracts skeletal keypoints using MediaPipe Pose and computes
biomechanical angles for sport-specific injury risk assessment.
"""

import cv2
import numpy as np
import math
import sys
from dataclasses import dataclass, field

try:
    import mediapipe as mp
    # MediaPipe ≥ 0.10 uses the Tasks API; < 0.10 used .solutions
    _MP_VERSION = tuple(int(x) for x in mp.__version__.split(".")[:2])
    _MP_TASKS_API = _MP_VERSION >= (0, 10)
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    _MP_TASKS_API = False
    _MP_VERSION = (0, 0)


@dataclass
class JointAngle:
    """A measured joint angle with risk assessment."""
    joint_name: str
    angle_degrees: float
    risk_level: str          # "safe", "caution", "danger"
    threshold_safe: float
    threshold_danger: float


@dataclass
class PoseFrame:
    """Analysis results for a single video frame."""
    frame_index: int
    timestamp_ms: float
    landmarks: list          # Raw MediaPipe landmark data
    joint_angles: list       # list[JointAngle]
    risk_flags: list         # list[str] — names of joints at "danger"
    torso_px: float = 0.0            # shoulder-mid -> hip-mid length (size of the player in frame)
    valgus_assessable: bool = False  # True only if the player faces the camera (frontal plane visible)


# Sport-specific injury thresholds: {joint_name: (safe_max, danger_min)}
SPORT_THRESHOLDS: dict[str, dict[str, tuple[float, float]]] = {
    "pickleball": {
        "knee_valgus": (12.0, 20.0),
        "shoulder_abduction": (90.0, 120.0),
        "lumbar_flexion": (30.0, 45.0),
        "wrist_extension": (30.0, 60.0),
    },
    "padel": {
        "knee_valgus": (12.0, 20.0),
        "shoulder_abduction": (90.0, 120.0),
        "lumbar_flexion": (30.0, 45.0),
        "wrist_extension": (30.0, 60.0),
    },
    "turf_football": {
        "knee_valgus": (10.0, 18.0),
        "hip_flexion": (90.0, 120.0),
        "ankle_dorsiflexion": (20.0, 35.0),
    },
    "box_cricket": {
        "knee_valgus": (12.0, 20.0),
        "shoulder_abduction": (85.0, 115.0),
        "lumbar_flexion": (25.0, 40.0),
    },
    "badminton": {
        "knee_valgus": (12.0, 20.0),
        "shoulder_abduction": (90.0, 130.0),
        "lumbar_flexion": (30.0, 45.0),
        "wrist_extension": (35.0, 65.0),
    },
}

# MediaPipe landmark indices — works with both legacy .solutions and Tasks API
if _MP_AVAILABLE and not _MP_TASKS_API:
    PoseLM = mp.solutions.pose.PoseLandmark
    LM_MAP = {
        "left_shoulder": PoseLM.LEFT_SHOULDER,
        "right_shoulder": PoseLM.RIGHT_SHOULDER,
        "left_elbow": PoseLM.LEFT_ELBOW,
        "right_elbow": PoseLM.RIGHT_ELBOW,
        "left_hip": PoseLM.LEFT_HIP,
        "right_hip": PoseLM.RIGHT_HIP,
        "left_knee": PoseLM.LEFT_KNEE,
        "right_knee": PoseLM.RIGHT_KNEE,
        "left_ankle": PoseLM.LEFT_ANKLE,
        "right_ankle": PoseLM.RIGHT_ANKLE,
        "left_wrist": PoseLM.LEFT_WRIST,
        "right_wrist": PoseLM.RIGHT_WRIST,
    }
else:
    # MediaPipe Tasks API (≥0.10) — integer indices for BlazePose 33-landmark model
    # https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    LM_MAP = {
        "left_shoulder":  11, "right_shoulder": 12,
        "left_elbow":     13, "right_elbow":    14,
        "left_hip":       23, "right_hip":      24,
        "left_knee":      25, "right_knee":     26,
        "left_ankle":     27, "right_ankle":    28,
        "left_wrist":     15, "right_wrist":    16,
    }


class PoseAnalyzer:
    """Biomechanical analysis using MediaPipe Pose."""

    def __init__(
        self,
        sport: str = "pickleball",
        min_detection_confidence: float = 0.5,  # 0.7 dropped 58% of side-on frames
        min_tracking_confidence: float = 0.5,
        model_variant: str = "lite",  # "full" is ~35x slower on CPU (547s vs 15s for a 66s clip)
    ):
        if not _MP_AVAILABLE:
            raise ImportError(
                "mediapipe is required. Install with: pip install mediapipe"
            )
        self.sport = sport
        self.thresholds = SPORT_THRESHOLDS.get(sport, SPORT_THRESHOLDS["pickleball"])

        if _MP_TASKS_API:
            # MediaPipe ≥ 0.10 Tasks API
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            import urllib.request, tempfile, os
            # Download lite model if not cached
            model_path = os.path.join(tempfile.gettempdir(), f"pose_landmarker_{model_variant}.task")
            if not os.path.exists(model_path):
                url = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                       f"pose_landmarker_{model_variant}/float16/latest/pose_landmarker_{model_variant}.task")
                try:
                    urllib.request.urlretrieve(url, model_path)
                except Exception:
                    model_path = None  # will fall through to no-pose path

            self._pose = None
            self._mp_landmarker = None
            if model_path and os.path.exists(model_path):
                base_opts = mp_python.BaseOptions(model_asset_path=model_path)
                opts = mp_vision.PoseLandmarkerOptions(
                    base_options=base_opts,
                    min_pose_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_poses=6,  # 4 players + far-court people: a lower cap silently drops a player
                )
                self._mp_landmarker = mp_vision.PoseLandmarker.create_from_options(opts)
            self._mp_drawing = None
            self._mp_pose = None
        else:
            # Legacy .solutions API
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._mp_drawing = mp.solutions.drawing_utils
            self._mp_pose = mp.solutions.pose
            self._mp_landmarker = None

    @staticmethod
    def _calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        Calculate the angle at point b formed by vectors ba and bc.
        Returns degrees in range [0, 180].
        """
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cosine = np.clip(cosine, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    def _assess_risk(self, angle: float, safe_threshold: float, danger_threshold: float) -> str:
        """Classify angle into safe / caution / danger."""
        if angle <= safe_threshold:
            return "safe"
        elif angle >= danger_threshold:
            return "danger"
        return "caution"

    def _get_point(self, landmarks, idx) -> np.ndarray:
        """Extract (x, y, z) from a landmark by index (int or enum)."""
        if isinstance(idx, int):
            lm = landmarks[idx]
        else:
            lm = landmarks[idx]
        return np.array([lm.x, lm.y, lm.z])

    def analyze_frame_all(
        self, frame: np.ndarray, frame_index: int = 0, timestamp_ms: float = 0.0,
    ) -> list["PoseFrame"]:
        """Pose + joint angles for EVERY person detected (up to num_poses)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        if _MP_TASKS_API:
            if self._mp_landmarker is None:
                return []
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            people = list(self._mp_landmarker.detect(mp_image).pose_landmarks or [])
        else:
            res = self._pose.process(rgb)
            people = [res.pose_landmarks.landmark] if res.pose_landmarks else []
        return [self._build_pose_frame(l, img_w, img_h, frame_index, timestamp_ms) for l in people]

    def analyze_roi_all(
        self, frame: np.ndarray, roi: tuple[int, int, int, int], scale: float = 2.5,
        frame_index: int = 0, timestamp_ms: float = 0.0,
    ) -> list["PoseFrame"]:
        """
        Detect people inside roi=(x0, y0, x1, y1) after upscaling it — catches small, distant
        players (e.g. the far side of the court) that full-frame detection misses. Landmarks are
        mapped back to full-frame normalised coordinates, so results are interchangeable with
        analyze_frame_all().
        """
        from types import SimpleNamespace
        H, W = frame.shape[:2]
        x0, y0, x1, y1 = max(0, roi[0]), max(0, roi[1]), min(W, roi[2]), min(H, roi[3])
        if x1 - x0 < 16 or y1 - y0 < 16 or self._mp_landmarker is None:
            return []
        crop = cv2.resize(frame[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res = self._mp_landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        rw, rh = x1 - x0, y1 - y0
        out = []
        for lms in (res.pose_landmarks or []):
            # Feet cut off by the crop's bottom edge => MediaPipe invents ankles on that edge, which would place a
            # person standing further down (e.g. a near-court player at the kitchen) at the wrong court position.
            if max(lms[i].y for i in (27, 28, 29, 30, 31, 32)) > 0.93 or min(lms[i].visibility for i in (27, 28)) < 0.5:
                continue
            mapped = [SimpleNamespace(x=(x0 + l.x * rw) / W, y=(y0 + l.y * rh) / H,
                                      z=l.z * rw / W, visibility=l.visibility) for l in lms]
            out.append(self._build_pose_frame(mapped, W, H, frame_index, timestamp_ms))
        return out

    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        timestamp_ms: float = 0.0,
    ) -> PoseFrame | None:
        """
        Process a single BGR frame; returns the NEAREST (largest-torso) person, or None.
        Use analyze_frame_all() + PlayerTracker when identities matter.
        """
        people = self.analyze_frame_all(frame, frame_index, timestamp_ms)
        return max(people, key=lambda p: p.torso_px) if people else None

    def _build_pose_frame(self, lms, img_w: int, img_h: int, frame_index: int, timestamp_ms: float) -> "PoseFrame":
        W, H = img_w, img_h
        min_vis = 0.5
        angles: list[JointAngle] = []
        risk_flags: list[str] = []

        def vis(*names) -> bool:
            return all(lms[LM_MAP[n]].visibility >= min_vis for n in names)

        def pt(name) -> np.ndarray:
            # Pixel space (z is scaled like x). Using normalized 0-1 coords distorts every
            # angle by the 16:9 aspect ratio.
            lm = lms[LM_MAP[name]]
            return np.array([lm.x * W, lm.y * H, lm.z * W])

        def add(name: str, value: float, thr: tuple[float, float]) -> None:
            risk = self._assess_risk(value, *thr)
            angles.append(JointAngle(name, round(value, 1), risk, thr[0], thr[1]))
            if risk == "danger":
                risk_flags.append(name)

        torso = self._torso_len(lms, W, H)
        sh_mid = (pt("left_shoulder") + pt("right_shoulder")) / 2
        hip_mid = (pt("left_hip") + pt("right_hip")) / 2
        shoulder_span = float(np.linalg.norm((pt("left_shoulder") - pt("right_shoulder"))[:2]))
        # Facing the camera => shoulders look wide relative to torso height (>0.8);
        # side-on => narrow (~0.1-0.4). Measured on real footage: median 0.19.
        facing_camera = torso > 0 and (shoulder_span / torso) >= 0.8

        # --- Knee valgus: FRONTAL-plane medial collapse of the knee. Only measurable
        # when the player faces the camera; from a side view it would just be knee flexion.
        if "knee_valgus" in self.thresholds and facing_camera:
            for side in ("left", "right"):
                if not vis(f"{side}_hip", f"{side}_knee", f"{side}_ankle"):
                    continue
                hip, knee, ankle = (pt(f"{side}_{j}")[:2] for j in ("hip", "knee", "ankle"))
                deviation = 180.0 - self._calculate_angle(hip, knee, ankle)
                # Medial = knee sits on the body-midline side of the hip->ankle line
                dy = ankle[1] - hip[1]
                if abs(dy) < 1e-6:
                    continue
                x_on_line = hip[0] + (knee[1] - hip[1]) / dy * (ankle[0] - hip[0])
                offset = knee[0] - x_on_line
                medial = np.sign(offset) == (np.sign(hip_mid[0] - hip[0]) or 1.0)
                if deviation > 45.0:  # physiologically implausible in the frontal plane => pose glitch
                    continue
                add(f"{side}_knee_valgus", deviation if medial else 0.0, self.thresholds["knee_valgus"])

        # --- Shoulder abduction: angle between trunk axis and upper arm
        if "shoulder_abduction" in self.thresholds:
            for side in ("left", "right"):
                if not vis(f"{side}_elbow", f"{side}_shoulder", f"{side}_hip"):
                    continue
                add(
                    f"{side}_shoulder_abduction",
                    self._calculate_angle(pt(f"{side}_elbow"), pt(f"{side}_shoulder"), pt(f"{side}_hip")),
                    self.thresholds["shoulder_abduction"],
                )

        # --- Trunk lean (proxy for lumbar flexion): angle of shoulder-mid -> hip-mid from vertical.
        # Works from any camera angle; an athletic ready stance sits ~10-25 deg.
        trunk_ids = [LM_MAP[n] for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
        trunk_reliable = all(
            lms[i].visibility >= 0.7 and 0.03 <= lms[i].x <= 0.97 and 0.03 <= lms[i].y <= 0.97 for i in trunk_ids
        )  # near-edge / low-confidence landmarks scatter and fake a huge lean
        if "lumbar_flexion" in self.thresholds and trunk_reliable:
            trunk = sh_mid[:2] - hip_mid[:2]
            lean = float(np.degrees(np.arctan2(abs(trunk[0]), abs(trunk[1]) + 1e-6)))
            add("trunk_lumbar_flexion", lean, self.thresholds["lumbar_flexion"])

        # --- Hip flexion (football)
        if "hip_flexion" in self.thresholds:
            for side in ("left", "right"):
                if not vis(f"{side}_shoulder", f"{side}_hip", f"{side}_knee"):
                    continue
                add(
                    f"{side}_hip_flexion",
                    self._calculate_angle(pt(f"{side}_shoulder"), pt(f"{side}_hip"), pt(f"{side}_knee")),
                    self.thresholds["hip_flexion"],
                )

        return PoseFrame(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            landmarks=[(lm.x, lm.y, lm.z, lm.visibility) for lm in lms],
            joint_angles=angles,
            risk_flags=risk_flags,
            torso_px=torso,
            valgus_assessable=facing_camera,
        )

    @staticmethod
    def _torso_len(lms, W: int, H: int) -> float:
        """Pixel length shoulder-mid -> hip-mid; used to pick the nearest / largest person."""
        sh = np.array([(lms[11].x + lms[12].x) / 2 * W, (lms[11].y + lms[12].y) / 2 * H])
        hp = np.array([(lms[23].x + lms[24].x) / 2 * W, (lms[23].y + lms[24].y) / 2 * H])
        return float(np.linalg.norm(sh - hp))

    def analyze_video(self, video_path: str, sample_rate: int = 2) -> list[PoseFrame]:
        """
        Analyze a video file, processing every Nth frame.

        Args:
            video_path: Path to video file.
            sample_rate: Process every Nth frame (default 2 = every other frame).

        Returns:
            List of PoseFrame results for frames where pose was detected.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        results: list[PoseFrame] = []
        frame_idx = 0

        print(f"📹 Analyzing video: {video_path}")
        print(f"   FPS: {fps:.1f}, Total frames: {total}, Sample rate: 1/{sample_rate}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                ts = frame_idx / fps * 1000.0
                pf = self.analyze_frame(frame, frame_idx, ts)
                if pf:
                    results.append(pf)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"   Processed {frame_idx}/{total} frames ({len(results)} poses detected)")

        cap.release()
        print(f"✅ Done: {len(results)} poses from {frame_idx} frames")
        return results

    def get_high_risk_moments(self, frames: list[PoseFrame]) -> list[PoseFrame]:
        """Filter frames that have at least one 'danger' risk flag."""
        return [f for f in frames if f.risk_flags]

    def group_events(self, frames: list[PoseFrame], max_gap_ms: float = 600.0, min_samples: int = 2) -> list[dict]:
        """
        Merge consecutive danger frames into discrete events. An event needs >= min_samples
        danger samples (one jittery landmark frame is not an injury risk). Returns one dict per
        event with start/end time, the worst frame (by severity) and the flags involved.
        """
        def severity(f: PoseFrame) -> float:
            return sum(
                (ja.angle_degrees - ja.threshold_danger) for ja in f.joint_angles if ja.risk_level == "danger"
            )

        events: list[dict] = []
        for f in frames:
            if not f.risk_flags:
                continue
            if events and f.timestamp_ms - events[-1]["end_ms"] <= max_gap_ms:
                ev = events[-1]
                ev["end_ms"] = f.timestamp_ms
                ev["flags"] |= set(f.risk_flags)
                if severity(f) > ev["severity"]:
                    ev["worst"], ev["severity"] = f, severity(f)
            else:
                events.append({"start_ms": f.timestamp_ms, "end_ms": f.timestamp_ms, "n": 0,
                               "flags": set(f.risk_flags), "worst": f, "severity": severity(f)})
            events[-1]["n"] += 1
        return [e for e in events if e["n"] >= min_samples]

    def generate_biomechanics_report(self, frames: list[PoseFrame]) -> dict:
        """
        Summary report: risk distribution, discrete high-risk EVENTS (not raw frames),
        worst angles, and recommendations. Valgus is only reported when the camera could see it.
        """
        if not frames:
            return {
                "total_frames_analyzed": 0,
                "high_risk_count": 0,
                "high_risk_frames": 0,
                "valgus_assessable_frames": 0,
                "risk_distribution": {"safe": 0, "caution": 0, "danger": 0},
                "worst_angles": {},
                "recommendations": ["No frames analyzed — ensure video has visible player."],
            }

        risk_dist = {"safe": 0, "caution": 0, "danger": 0}
        worst: dict[str, float] = {}
        high_risk_frames = 0
        valgus_frames = sum(1 for f in frames if f.valgus_assessable)

        for f in frames:
            for ja in f.joint_angles:
                risk_dist[ja.risk_level] = risk_dist.get(ja.risk_level, 0) + 1
                base = ja.joint_name.split("_", 1)[-1] if "_" in ja.joint_name else ja.joint_name
                if base not in worst or ja.angle_degrees > worst[base]:
                    worst[base] = ja.angle_degrees
            if f.risk_flags:
                high_risk_frames += 1

        events = self.group_events(frames)
        # Frequency of danger per joint family (so one noisy frame can't drive a recommendation)
        n = len(frames)
        def frac(key: str) -> float:
            return sum(1 for f in frames if any(key in x for x in f.risk_flags)) / n

        recs: list[str] = []
        if valgus_frames == 0 and "knee_valgus" in self.thresholds:
            recs.append(
                "ℹ️ Knee valgus not assessable: player never faced the camera. "
                "Mount the pod behind the baseline, centred on the court."
            )
        elif frac("knee_valgus") > 0.05:
            recs.append("⚠️ Knee valgus detected — strengthen VMO with single-leg squats and lateral band walks.")
        if frac("shoulder_abduction") > 0.10:
            recs.append("⚠️ Frequent high shoulder elevation — work on rotator-cuff strengthening (band external rotation).")
        if frac("lumbar_flexion") > 0.10:
            recs.append("⚠️ Excessive forward trunk lean on low shots — practise the hip-hinge (RDLs, bird-dogs) and bend the knees more.")
        if not any(r.startswith("⚠️") for r in recs):
            recs.append("✅ No sustained high-risk mechanics detected.")

        return {
            "total_frames_analyzed": n,
            "high_risk_count": len(events),
            "high_risk_frames": high_risk_frames,
            "valgus_assessable_frames": valgus_frames,
            "risk_distribution": risk_dist,
            "worst_angles": worst,
            "recommendations": recs,
        }

    def draw_skeleton(
        self, frame: np.ndarray, landmarks, highlight_danger: bool = True
    ) -> np.ndarray:
        """Draw the MediaPipe skeleton on a frame. Red for danger joints."""
        out = frame.copy()
        if not _MP_AVAILABLE or _MP_TASKS_API or self._mp_drawing is None:
            return out

        self._mp_drawing.draw_landmarks(
            out,
            landmarks,
            self._mp_pose.POSE_CONNECTIONS,
            self._mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            self._mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
        )
        return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pose_analyzer.py <video_path>")
        print("  Analyzes a video file for biomechanical risk factors.")
        sys.exit(1)

    analyzer = PoseAnalyzer(sport="pickleball")
    frames = analyzer.analyze_video(sys.argv[1])
    high_risk = analyzer.get_high_risk_moments(frames)
    report = analyzer.generate_biomechanics_report(frames)

    print("\n" + "=" * 50)
    print("📊 BIOMECHANICS REPORT")
    print("=" * 50)
    for k, v in report.items():
        if k != "recommendations":
            print(f"  {k}: {v}")
    print("\nRecommendations:")
    for r in report["recommendations"]:
        print(f"  {r}")
