"""
PlayQ Rally Detector Module
============================
Detects rally boundaries in sports footage using audio energy analysis
and manages clip trimming via FFmpeg.
"""

import numpy as np
import subprocess
import shutil
import tempfile
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

def _ffmpeg_exe() -> str | None:
    """System ffmpeg if on PATH, else the binary bundled with imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


try:
    from scipy.io import wavfile
    from scipy.signal import medfilt
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


@dataclass
class RallySegment:
    """A detected rally/active-play segment in a video."""
    start_time: float       # seconds
    end_time: float         # seconds
    duration: float         # seconds
    peak_db: float          # peak decibel level
    confidence: float       # 0.0 - 1.0
    sport_event: str        # "rally", "goal", "wicket", "point", etc.


# Sport-specific defaults
SPORT_DEFAULTS: dict[str, dict] = {
    "pickleball": {"db_threshold": -20.0, "event": "rally", "min_duration": 2.0},
    "padel": {"db_threshold": -22.0, "event": "rally", "min_duration": 2.0},
    "turf_football": {"db_threshold": -25.0, "event": "play", "min_duration": 3.0},
    "box_cricket": {"db_threshold": -18.0, "event": "delivery", "min_duration": 1.5},
    "badminton": {"db_threshold": -21.0, "event": "rally", "min_duration": 2.0},
}


class RallyDetector:
    """
    Detects active rally segments in sports video using audio energy analysis.
    Requires FFmpeg for audio extraction and clip trimming.
    """

    def __init__(
        self,
        sport: str = "pickleball",
        db_threshold: Optional[float] = None,
        min_rally_duration: Optional[float] = None,
        max_gap: float = 1.5,
        padding: float = 1.0,
    ):
        defaults = SPORT_DEFAULTS.get(sport, SPORT_DEFAULTS["pickleball"])
        self.sport = sport
        self.db_threshold = db_threshold or defaults["db_threshold"]
        self.min_rally_duration = min_rally_duration or defaults["min_duration"]
        self.max_gap = max_gap
        self.padding = padding
        self._event_type = defaults.get("event", "rally")
        self._temp_files: list[str] = []

    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available on PATH."""
        return _ffmpeg_exe() is not None

    def _extract_audio_wav(self, video_path: str) -> str:
        """
        Extract mono 16kHz WAV audio from video using FFmpeg.
        Returns path to temporary WAV file.
        """
        if not self._check_ffmpeg():
            raise RuntimeError(
                "FFmpeg not found! Install it:\n"
                "  macOS:  brew install ffmpeg\n"
                "  Ubuntu: sudo apt install ffmpeg\n"
                "  Windows: choco install ffmpeg"
            )

        tmp_wav = tempfile.mktemp(suffix=".wav", prefix="playq_audio_")
        self._temp_files.append(tmp_wav)

        cmd = [
            _ffmpeg_exe(), "-y",
            "-i", video_path,
            "-vn",                    # No video
            "-acodec", "pcm_s16le",   # 16-bit PCM
            "-ar", "16000",           # 16kHz sample rate
            "-ac", "1",               # Mono
            tmp_wav,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg audio extraction failed:\n{result.stderr[:500]}")

        return tmp_wav

    def detect_rallies_energy(self, video_path: str) -> list[RallySegment]:
        """
        Detect active rally segments in a video based on audio energy.

        Args:
            video_path: Path to the input video file.

        Returns:
            List of RallySegment with timestamps of active play.
        """
        if not _SCIPY_AVAILABLE:
            raise ImportError("scipy is required. Install with: pip install scipy")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Extract audio
        wav_path = self._extract_audio_wav(video_path)

        # Load WAV
        sample_rate, audio = wavfile.read(wav_path)
        audio = audio.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

        # Compute RMS energy in 50ms windows
        window_samples = int(sample_rate * 0.05)  # 50ms
        hop = window_samples // 2  # 25ms hop
        n_windows = (len(audio) - window_samples) // hop + 1

        rms = np.zeros(n_windows)
        for i in range(n_windows):
            start = i * hop
            window = audio[start : start + window_samples]
            rms[i] = np.sqrt(np.mean(window ** 2) + 1e-10)

        # Convert to dB
        db = 20 * np.log10(rms + 1e-10)

        # Smooth with median filter
        if len(db) > 5:
            db = medfilt(db, kernel_size=5)

        # Find regions above threshold
        active = db > self.db_threshold
        window_duration = hop / sample_rate  # seconds per window

        # Find contiguous active regions
        segments: list[RallySegment] = []
        in_segment = False
        seg_start = 0
        seg_peak = -100.0

        for i, is_active in enumerate(active):
            if is_active and not in_segment:
                seg_start = i
                seg_peak = db[i]
                in_segment = True
            elif is_active and in_segment:
                seg_peak = max(seg_peak, db[i])
            elif not is_active and in_segment:
                in_segment = False
                start_t = seg_start * window_duration
                end_t = i * window_duration
                segments.append(RallySegment(
                    start_time=start_t,
                    end_time=end_t,
                    duration=end_t - start_t,
                    peak_db=float(seg_peak),
                    confidence=0.5,
                    sport_event=self._event_type,
                ))

        # Handle segment still open at end
        if in_segment:
            start_t = seg_start * window_duration
            end_t = len(active) * window_duration
            segments.append(RallySegment(
                start_time=start_t, end_time=end_t,
                duration=end_t - start_t, peak_db=float(seg_peak),
                confidence=0.5, sport_event=self._event_type,
            ))

        # Merge segments closer than max_gap
        merged: list[RallySegment] = []
        for seg in segments:
            if merged and (seg.start_time - merged[-1].end_time) < self.max_gap:
                merged[-1] = RallySegment(
                    start_time=merged[-1].start_time,
                    end_time=seg.end_time,
                    duration=seg.end_time - merged[-1].start_time,
                    peak_db=max(merged[-1].peak_db, seg.peak_db),
                    confidence=(merged[-1].confidence + seg.confidence) / 2,
                    sport_event=self._event_type,
                )
            else:
                merged.append(seg)

        # Filter by minimum duration
        filtered = [s for s in merged if s.duration >= self.min_rally_duration]

        # Add padding and compute confidence
        result: list[RallySegment] = []
        max_db = max((s.peak_db for s in filtered), default=-100)
        for seg in filtered:
            padded_start = max(0, seg.start_time - self.padding)
            padded_end = seg.end_time + self.padding
            conf = min(1.0, (seg.peak_db - self.db_threshold) / (max_db - self.db_threshold + 1e-6))
            result.append(RallySegment(
                start_time=round(padded_start, 2),
                end_time=round(padded_end, 2),
                duration=round(padded_end - padded_start, 2),
                peak_db=round(seg.peak_db, 1),
                confidence=round(max(0.1, conf), 2),
                sport_event=self._event_type,
            ))

        return result

    def detect_hits(self, video_path: str, k: float = 4.0, min_gap: float = 0.15) -> list[tuple[float, float]]:
        """
        Detect ball-pop transients (paddle/ball impacts) in the audio.

        Works on the 1-6 kHz band and measures each frame's rise above a rolling 1 s
        noise floor, so it adapts to venue loudness / mic distance instead of using a
        fixed dB threshold.

        Returns:
            List of (time_seconds, onset_strength_db).
        """
        from scipy.signal import butter, sosfilt, find_peaks
        from scipy.ndimage import median_filter

        wav_path = self._extract_audio_wav(video_path)
        sr, audio = wavfile.read(wav_path)
        audio = audio.astype(np.float32) / 32768.0
        sos = butter(4, [1000, 6000], btype="bandpass", fs=sr, output="sos")
        x = sosfilt(sos, audio)
        hop = int(sr * 0.005)  # 5 ms resolution
        n = len(x) // hop
        db = 20 * np.log10(np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
        onset = db - median_filter(db, size=200)  # rise over 1 s rolling floor
        med = np.median(onset)
        mad = 1.4826 * np.median(np.abs(onset - med)) + 1e-6
        peaks, _ = find_peaks(onset, height=med + k * mad, distance=int(min_gap / 0.005))
        return [(float(p * 0.005), float(onset[p])) for p in peaks]

    def detect_rallies(self, video_path: str, max_hit_gap: float = 3.5, min_hits: int = 3) -> list[RallySegment]:
        """
        Detect rallies as clusters of ball-pop hits separated by < max_hit_gap seconds.
        Falls back to nothing (empty list) if the audio has no clear impacts.
        """
        hits = self.detect_hits(video_path)
        clusters: list[list[tuple[float, float]]] = []
        for h in hits:
            if clusters and h[0] - clusters[-1][-1][0] <= max_hit_gap:
                clusters[-1].append(h)
            else:
                clusters.append([h])

        segments: list[RallySegment] = []
        for c in clusters:
            if len(c) < min_hits:
                continue
            start = max(0.0, c[0][0] - self.padding)
            end = c[-1][0] + self.padding
            strength = float(np.mean([o for _, o in c]))
            segments.append(RallySegment(
                start_time=round(start, 2),
                end_time=round(end, 2),
                duration=round(end - start, 2),
                peak_db=round(max(o for _, o in c), 1),  # onset strength above floor, not absolute dB
                confidence=round(min(1.0, len(c) / 12.0), 2),
                sport_event=self._event_type,
            ))
        return segments

    def trim_clip(
        self,
        video_path: str,
        segment: RallySegment,
        output_path: str,
        vertical: bool = False,
    ) -> str:
        """
        Extract a segment from video using FFmpeg.

        Args:
            video_path: Source video.
            segment: RallySegment with timestamps.
            output_path: Destination file path.
            vertical: If True, crop to 9:16 aspect ratio for social media.

        Returns:
            output_path on success.
        """
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg not found!")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        cmd = [
            _ffmpeg_exe(), "-y",
            "-ss", str(segment.start_time),
            "-i", video_path,
            "-t", str(segment.duration),
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac",
        ]

        if vertical:
            cmd.extend(["-vf", "crop=ih*9/16:ih,scale=1080:1920"])

        cmd.append(output_path)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg trim failed:\n{result.stderr[:500]}")

        return output_path

    def trim_all_rallies(
        self, video_path: str, output_dir: str, vertical: bool = False
    ) -> list[str]:
        """Detect rallies and trim each one to a separate file."""
        segments = self.detect_rallies(video_path)
        os.makedirs(output_dir, exist_ok=True)

        outputs: list[str] = []
        for i, seg in enumerate(segments):
            ext = ".mp4"
            fname = f"rally_{i+1:03d}_{seg.start_time:.0f}s{ext}"
            out_path = os.path.join(output_dir, fname)
            self.trim_clip(video_path, seg, out_path, vertical=vertical)
            outputs.append(out_path)
            print(f"  ✂️ Trimmed rally {i+1}: {seg.start_time:.1f}s → {seg.end_time:.1f}s")

        return outputs

    def create_highlight_reel(
        self, clips: list[str], output_path: str, max_duration: float = 30.0
    ) -> str:
        """
        Concatenate clips into a highlight reel using FFmpeg concat demuxer.

        Args:
            clips: List of video file paths.
            output_path: Destination file.
            max_duration: Max total duration in seconds.

        Returns:
            output_path on success.
        """
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg not found!")
        if not clips:
            raise ValueError("No clips provided for highlight reel.")

        # Create concat list file
        concat_file = tempfile.mktemp(suffix=".txt", prefix="playq_concat_")
        self._temp_files.append(concat_file)

        with open(concat_file, "w") as f:
            for clip in clips:
                f.write(f"file '{os.path.abspath(clip)}'\n")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        cmd = [
            _ffmpeg_exe(), "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-t", str(max_duration),
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg concat failed:\n{result.stderr[:500]}")

        return output_path

    def cleanup(self):
        """Remove temporary files."""
        for f in self._temp_files:
            if os.path.exists(f):
                os.remove(f)
        self._temp_files.clear()

    def __del__(self):
        self.cleanup()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("🎵 PlayQ Rally Detector")
        print("=" * 40)
        print(f"FFmpeg available: {RallyDetector()._check_ffmpeg()}")
        print(f"scipy available: {_SCIPY_AVAILABLE}")
        print()
        print("Usage: python rally_detector.py <video_path> [sport]")
        print("  Sports: pickleball, padel, turf_football, box_cricket, badminton")
        sys.exit(0)

    video = sys.argv[1]
    sport = sys.argv[2] if len(sys.argv) > 2 else "pickleball"

    detector = RallyDetector(sport=sport)
    print(f"🎵 Detecting rallies in: {video}")
    print(f"   Sport: {sport}, Threshold: {detector.db_threshold} dB")

    segments = detector.detect_rallies(video)
    print(f"\n📍 Found {len(segments)} rally segments:")
    for i, seg in enumerate(segments):
        print(
            f"  {i+1}. {seg.start_time:.1f}s → {seg.end_time:.1f}s "
            f"({seg.duration:.1f}s) peak={seg.peak_db:.1f}dB conf={seg.confidence:.2f}"
        )
