"""
PlayQ Gemini Multi-Agent System
================================
Four agents (matching the pitch deck), all returning schema-validated JSON:
  1. RallyAgent        — watches the video and verifies / corrects the audio-detected rallies
  2. TacticsAgent      — court-positioning insights + the two attributes only Gemini can judge
  3. BiomechanicsAgent — injury-risk diagnosis & corrections from the measured angles
  4. BroadcastAgent    — Hinglish commentary, caption and music prompt

Every result carries `_source`: "gemini" | "mock" (no API key) | "error" (call failed), so a
failure is never mistaken for a real analysis.

Setup: put GEMINI_API_KEY in .env (see .env.example), then run scripts/test_gemini.py.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from typing import Optional, Type

from pydantic import BaseModel

from playq import config
from playq.agents.schemas import (
    BiomechanicsDiagnosis, BroadcastScript, RallyVerification, TacticsReport,
)

logger = logging.getLogger("playq.agents")

try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GENAI_AVAILABLE = False
    logger.warning("google-genai not installed. Agents will return mock responses.")


class PlayQAgent(ABC):
    """Base class: one place for client setup, structured output and error handling."""

    def __init__(self, name: str, model: str | None = None):
        self.name = name
        self.model = model or config.TEXT_MODEL
        self._client = None
        key = config.api_key()
        if _GENAI_AVAILABLE and key:
            self._client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=config.REQUEST_TIMEOUT_S * 1000))
        elif _GENAI_AVAILABLE:
            logger.warning("[%s] No GEMINI_API_KEY set — returning mock responses.", name)

    # -- core call ---------------------------------------------------------------
    def _generate(
        self,
        contents,
        schema: Type[BaseModel],
        system_instruction: str,
        model: str | None = None,
        temperature: float = 0.4,
    ) -> dict:
        """Call Gemini with a Pydantic response schema and return the validated dict."""
        if not self._client:
            return {"_source": "mock", "agent": self.name,
                    "note": "Set GEMINI_API_KEY in .env to enable real Gemini responses."}
        cfg = types.GenerateContentConfig(
            system_instruction=system_instruction, response_mime_type="application/json",
            response_schema=schema, temperature=temperature,
        )
        last_err = ""
        first = model or self.model
        models = [first] + [m for m in config.MODEL_CHAIN if m != first]
        for m in models:
            for attempt in range(1):   # one try per model: a 504 already burned the full timeout, so fail over fast
                try:
                    resp = self._client.models.generate_content(model=m, contents=contents, config=cfg)
                    parsed = resp.parsed if isinstance(resp.parsed, BaseModel) else schema.model_validate_json(resp.text)
                    out = parsed.model_dump()
                    out["_source"], out["_model"] = "gemini", m
                    out["_usage"] = self._usage(resp)
                    return out
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"[:400]
                    transient = any(k in last_err for k in ("503", "500", "429", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE", "Timeout", "timed out", "overloaded"))
                    logger.warning("[%s] %s attempt %d failed: %s", self.name, m, attempt + 1, last_err[:120])
                    if not transient:
                        return {"_source": "error", "agent": self.name, "error": last_err}   # schema/safety/auth: retrying won't help
                    time.sleep(1.5 * (attempt + 1))
        return {"_source": "error", "agent": self.name, "error": last_err}

    @staticmethod
    def _usage(resp) -> dict:
        u = getattr(resp, "usage_metadata", None)
        if not u:
            return {"in": 0, "out": 0}
        return {"in": int(getattr(u, "prompt_token_count", 0) or 0),
                "out": int((getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0))}

    def _generate_hedged(self, contents, schema: Type[BaseModel], system_instruction: str,
                         models: list[str], temperature: float = 0.4) -> dict:
        """
        Send the SAME request to several models in parallel and return the first valid answer.
        Gemini tail latency is dominated by server load (a model that answers in 20 s can time out at
        100 s on the next call), so racing two models bounds the wait at about the faster one.
        """
        if not self._client:
            return {"_source": "mock", "agent": self.name, "note": "Set GEMINI_API_KEY in .env"}
        from concurrent.futures import ThreadPoolExecutor, as_completed
        cfg = types.GenerateContentConfig(system_instruction=system_instruction, response_mime_type="application/json",
                                          response_schema=schema, temperature=temperature)

        def call(m: str) -> dict:
            resp = self._client.models.generate_content(model=m, contents=contents, config=cfg)
            parsed = resp.parsed if isinstance(resp.parsed, BaseModel) else schema.model_validate_json(resp.text)
            out = parsed.model_dump()
            out["_source"], out["_model"] = "gemini", m
            out["_usage"] = self._usage(resp)
            out["_raced"] = len(models[:2])    # the slower model is still billed even though we ignore its answer
            return out

        errors = []
        for i in range(0, len(models), 2):            # race a pair; if both fail (quota/outage) try the next pair
            pair = models[i:i + 2]
            pool = ThreadPoolExecutor(max_workers=len(pair))
            futs = {pool.submit(call, m): m for m in pair}
            try:
                for fut in as_completed(futs):
                    try:
                        return fut.result()
                    except Exception as e:
                        errors.append(f"{futs[fut]}: {type(e).__name__}: {e}"[:160])
                        logger.warning("[%s] hedged call failed on %s", self.name, errors[-1])
            finally:
                pool.shutdown(wait=False, cancel_futures=True)   # don't wait for the slower model
        return {"_source": "error", "agent": self.name, "error": " | ".join(errors)[:500]}

    @abstractmethod
    def analyze(self, data: dict) -> dict: ...


_GROUNDING = (
    "Use ONLY the numbers provided. Never invent measurements. If a measurement is missing or "
    "marked not assessable, say so instead of guessing. You are not a doctor: frame advice as "
    "general injury-prevention guidance and suggest seeing a physiotherapist for pain."
)


class RallyAgent(PlayQAgent):
    """Gemini video reasoning: confirm the audio-detected rallies and catch missed ones."""

    SYSTEM = (
        "You are PlayQ's rally analyst. You watch recreational racquet-sport footage and report "
        "when real rallies happen (continuous play with the ball being hit back and forth), "
        "ignoring warm-ups, ball pickups, walking between points, and neighbouring courts. "
        "Timestamps are seconds from the start of the video you are given. " + _GROUNDING
    )

    def __init__(self):
        super().__init__("rally_agent", config.VIDEO_MODEL)

    @staticmethod
    def _make_proxy(video_path: str, height: int = 480, fps: int = 10) -> str:
        """Downscale for upload: Gemini samples frames anyway, so 1080p60 only costs bandwidth."""
        from playq.cv.rally_detector import _ffmpeg_exe
        out = os.path.join(tempfile.gettempdir(), f"playq_proxy_{os.getpid()}.mp4")
        subprocess.run(
            [_ffmpeg_exe(), "-y", "-i", video_path, "-vf", f"scale=-2:{height}", "-r", str(fps),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-c:a", "aac", "-b:a", "64k", out],
            check=True, capture_output=True,
        )
        return out

    def verify(self, video_path: str, candidate_rallies: list[dict], sport: str = "pickleball") -> dict:
        """
        Args:
            video_path: original video (a 480p proxy is uploaded, not the 1080p file).
            candidate_rallies: [{"start_time": s, "end_time": s}, ...] from the audio detector.
        """
        if not self._client:
            return self._generate("", RallyVerification, self.SYSTEM)  # returns the mock marker
        proxy = None
        try:
            proxy = self._make_proxy(video_path)
            uploaded = self._client.files.upload(file=proxy)
            for _ in range(60):  # wait until Google finishes processing the video
                uploaded = self._client.files.get(name=uploaded.name)
                if uploaded.state and uploaded.state.name == "ACTIVE":
                    break
                if uploaded.state and uploaded.state.name == "FAILED":
                    return {"_source": "error", "agent": self.name, "error": "video processing failed"}
                time.sleep(2)
            cands = [{"start_s": r["start_time"], "end_s": r["end_time"]} for r in candidate_rallies]
            prompt = (
                f"This is {sport} footage. An audio detector (ball-impact sounds) proposed these rally "
                f"windows: {json.dumps(cands)}.\n"
                "For each candidate, judge whether it is a real rally, count the shots you can see, and "
                "refine nothing you cannot see. Then list any real rallies that are NOT covered by a "
                "candidate (missed_rallies). Report how many players you can see."
            )
            result = self._generate([uploaded, prompt], RallyVerification, self.SYSTEM)
            try:
                self._client.files.delete(name=uploaded.name)  # don't leave customer video on Google's servers
            except Exception:
                pass
            return result
        except Exception as e:
            return {"_source": "error", "agent": self.name, "error": f"{type(e).__name__}: {e}"[:500]}
        finally:
            if proxy and os.path.exists(proxy):
                os.remove(proxy)

    def analyze(self, data: dict) -> dict:
        return self.verify(data["video_path"], data.get("candidate_rallies", []), data.get("sport", "pickleball"))


class BiomechanicsAgent(PlayQAgent):
    SYSTEM = (
        "You are PlayQ's biomechanics coach for recreational pickleball, padel, badminton, turf "
        "football and box cricket. You interpret joint angles measured by pose estimation and give "
        "prioritised, practical injury-prevention corrections. Pose estimation is noisy: treat single "
        "extreme values with suspicion and prefer patterns. " + _GROUNDING
    )

    def __init__(self):
        super().__init__("biomechanics_agent")

    def analyze(self, data: dict) -> dict:
        prompt = (
            f"Sport: {data.get('sport', 'pickleball')}\n"
            f"Peak angles (degrees): {json.dumps(data.get('peak_angles_deg', {}))}\n"
            f"High-risk events (sustained danger-level readings): {data.get('high_risk_events', 0)}\n"
            f"Frames where knee valgus was assessable (player facing camera): {data.get('valgus_assessable_frames', 0)}\n"
            f"Rule-based flags already raised: {json.dumps(data.get('recommendations', []))}\n"
            "Give the diagnosis, what the data could NOT show, and corrections."
        )
        return self._generate(prompt, BiomechanicsDiagnosis, self.SYSTEM)

    @staticmethod
    def posture_card_prompt(diagnosis: dict) -> str:
        """Prompt for Nano Banana (image model) to draw the side-by-side posture card."""
        areas = ", ".join(f"{r['joint']} ({r['risk_level']})" for r in diagnosis.get("risk_areas", [])[:3]) or "general posture"
        return (
            "Sports-medicine posture correction card, split screen. LEFT: incorrect form with red "
            f"highlights on {areas}. RIGHT: corrected form with green highlights. Clean flat medical "
            "illustration, dark background, neon accents, small 'PlayQ' wordmark."
        )


class TacticsAgent(PlayQAgent):
    SYSTEM = (
        "You are PlayQ's tactics analyst for recreational racquet sports. You read tracked court "
        "positions (metres; x across the court, y along it, net at the midpoint) and rally data. "
        "You may only judge shot_consistency_0_99 and clutch_factor_0_99 if the input contains "
        "evidence for them (shot outcomes / score); otherwise return null. " + _GROUNDING
    )

    def __init__(self):
        super().__init__("tactics_agent")

    def analyze(self, data: dict) -> dict:
        prompt = (
            f"Sport: {data.get('sport', 'pickleball')}\n"
            f"Player: {data.get('player', 'Player')}\n"
            f"Movement metrics: {json.dumps(data.get('movement', {}))}\n"
            f"Rallies: {json.dumps(data.get('rallies', [])[:20])}\n"
            "Describe positioning habits, dead zones (as % of time away from useful areas), strengths "
            "and weaknesses."
        )
        return self._generate(prompt, TacticsReport, self.SYSTEM)


class BroadcastAgent(PlayQAgent):
    SYSTEM = (
        "You are PlayQ's Hinglish sports commentator — energetic, witty, shareable on Instagram, in "
        "the spirit of Indian cricket commentary. Roasts must be playful and never target body, "
        "appearance, religion, caste or anything personal; roast the shot, not the person. " + _GROUNDING
    )

    def __init__(self):
        super().__init__("broadcast_agent")

    def analyze(self, data: dict) -> dict:
        prompt = (
            f"Sport: {data.get('sport', 'pickleball')}. Players: {', '.join(data.get('player_names', ['Player']))}.\n"
            f"Highlight moments: {json.dumps(data.get('highlights', [])[:10])}\n"
            f"Language: {data.get('language', 'hinglish')}\n"
            "Write one short commentary line per highlight (timestamps from the list), an Instagram "
            "caption with hashtags (#PlayQ), a music prompt for an AI music model, and one playful roast."
        )
        return self._generate(prompt, BroadcastScript, self.SYSTEM, temperature=0.9)


class PlayQOrchestrator:
    """Runs the four agents on a match_report.json produced by scripts/analyze_video.py."""

    def __init__(self):
        self.rally = RallyAgent()
        self.biomechanics = BiomechanicsAgent()
        self.tactics = TacticsAgent()
        self.broadcast = BroadcastAgent()

    def process_report(self, report: dict, video_path: Optional[str] = None, verify_video: bool = False) -> dict:
        sport = report.get("sport", "pickleball")
        out: dict = {"sport": sport, "players": []}

        if verify_video and video_path:
            out["rally_verification"] = self.rally.verify(video_path, report.get("rallies", []), sport)

        for p in report.get("players", []):
            out["players"].append({
                "player": p["player"],
                "biomechanics": self.biomechanics.analyze({
                    "sport": sport,
                    "peak_angles_deg": p.get("peak_angles_deg", {}),
                    "high_risk_events": p.get("high_risk_events", 0),
                    "valgus_assessable_frames": p.get("valgus_assessable_frames", 0),
                    "recommendations": p.get("recommendations", []),
                }),
                "tactics": self.tactics.analyze({
                    "sport": sport, "player": p["player"], "movement": p.get("movement", {}),
                    "rallies": report.get("rallies", []),
                }),
            })

        rallies = sorted(report.get("rallies", []), key=lambda r: -r.get("duration", 0))[:3]
        out["broadcast"] = self.broadcast.analyze({
            "sport": sport,
            "player_names": [p["player"] for p in report.get("players", [])] or ["Player"],
            "highlights": [{"timestamp_s": r["start_time"], "event": f"{r['duration']:.0f}s rally"} for r in rallies],
        })
        if out["players"] and out["players"][0]["biomechanics"].get("_source") == "gemini":
            out["posture_card_prompt"] = BiomechanicsAgent.posture_card_prompt(out["players"][0]["biomechanics"])
        return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"google-genai available: {_GENAI_AVAILABLE} | key set: {bool(config.api_key())} | model: {config.TEXT_MODEL}")
