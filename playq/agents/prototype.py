"""
Prototype AI layer — a 3-step Gemini chain (not a single prompt):

  1. vision_check()    : Gemini LOOKS at the flagged skeleton frame and judges the form independently
  2. match_insights()  : Gemini WATCHES the highlight reel (480p proxy) with the measured metrics
                         as ground truth -> rating inputs + 15 s Hinglish script (Pydantic-validated)
  3. synthesize_voice(): Gemini TTS -> WAV, then ffmpeg muxes it onto the reel
                         (falls back to the macOS `say` voice if TTS is unavailable)

Every step returns a dict with `_source` ("gemini" | "mock" | "error" | "fallback-say") so the UI
can show exactly what was real.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
import wave

from playq import config
from playq.agents.gemini_agents import PlayQAgent, _GENAI_AVAILABLE
from playq.agents.schemas import BioVision, MatchInsights
from playq.cv.rally_detector import _ffmpeg_exe

if _GENAI_AVAILABLE:
    from google.genai import types

_GROUND = (
    "Measured numbers come from deterministic computer vision and are ground truth: never contradict or "
    "invent them. If something is not visible, say so or return null. Not medical advice."
)


class PrototypeAI(PlayQAgent):
    def __init__(self):
        super().__init__("prototype_ai", config.VIDEO_MODEL)

    def analyze(self, data: dict) -> dict:  # ABC requirement; use the step methods directly
        raise NotImplementedError

    # 1 ── vision on the flagged frame ─────────────────────────────────────────
    def vision_check(self, frame_jpg: str, metrics: dict) -> dict:
        if not self._client:
            return {"_source": "mock", "note": "no GEMINI_API_KEY"}
        data = open(frame_jpg, "rb").read()
        prompt = (
            "This is the annotated frame of a pickleball player's deepest knee load during live play "
            f"(skeleton drawn by MediaPipe). Measured: {json.dumps(metrics)}.\n"
            "Judge injury-risk level for the lower body from the posture you see, give one coaching cue, "
            "and say whether the image supports the measured angle."
        )
        return self._generate(
            [types.Part.from_bytes(data=data, mime_type="image/jpeg"), prompt],
            BioVision, "You are a sports physiotherapist reviewing a still frame. " + _GROUND,
        )

    # 2 ── video + metrics -> rating inputs + commentary ───────────────────────
    def _upload_video(self, path: str):
        up = self._client.files.upload(file=path)
        for _ in range(60):
            up = self._client.files.get(name=up.name)
            if up.state and up.state.name == "ACTIVE":
                return up
            if up.state and up.state.name == "FAILED":
                raise RuntimeError("Gemini video processing failed")
            time.sleep(2)
        raise TimeoutError("Gemini video processing timed out")

    def match_insights(self, reel_proxy: str, players: list[dict], bio: dict, sport: str = "pickleball",
                       timeline: list[dict] | None = None) -> dict:
        if not self._client:
            return {"_source": "mock", "note": "no GEMINI_API_KEY"}
        up = None
        try:
            up = self._upload_video(reel_proxy)
            compact = [{"track_id": p["track_id"], "who": f"{p['label']} ({p['movement'].get('side')} court)",
                        "mean_distance_to_net_m": p["movement"].get("mean_dist_to_net_m"),
                        "time_within_3m_of_net": p["movement"].get("net_zone_share"),
                        "avg_speed_mps": p["movement"].get("avg_speed_mps")} for p in players]
            total = timeline[-1]["reel_end_s"] if timeline else None
            prompt = (
                f"This is a {total:.0f}-second {sport} highlight reel made of {len(timeline or [])} segments joined together. "
                f"Timeline (reel seconds, ground truth): {json.dumps(timeline)}.\n"
                f"Tracked players (measured, ground truth): {json.dumps(compact)}.\nBiomechanics: {json.dumps(bio)}.\n\n"
                "TASKS\n1) Read the scoreboard if legible.\n"
                "2) For each track id give shot_consistency_0_99, clutch_factor_0_99 and shot_aggression_0_99 (shot style: soft dinks/resets vs hard drives/smashes) "
                "ONLY if the footage supports it, else null, with a one-sentence reason. Be discriminating: if two players play differently, their scores must differ.\n"
                "3) Write a two-person commentary: speaker A (hype, male) and speaker B (witty, female) trading short lines.\n"
                "RULES for the commentary:\n"
                "- Describe what ACTUALLY happens in each segment (shot types: dink, drive, lob, smash, volley, reset; who wins the point; "
                "unforced errors). Refer to players by shirt colour ('blue shirt wala'), never invent names.\n"
                "- Every line's start_s must fall inside the reel segment it describes, and `segment` must be that segment's index. "
                "At least one line per segment; first line starts at 0.3s or later.\n"
                "- Speaking pace is ~2.5 words/second: a line of N words needs N/2.5 seconds plus a 0.4s gap before the next line starts. "
                "Never let lines overlap and never run past the end of the reel.\n"
                "- The LAST line must start at least 4.5 s before the end of the reel so it can finish.\n"
                "- Keep the commentary continuous: start a new line roughly every 4-5 seconds so there is never more than ~6 s of silence, but never overlap.\n"
                "- Max 13 words per line. Be FUNNY: filmy dialogues, cricket-commentary similes, playful teasing of a SHOT or an error "
                "(never a person's looks, body or background). Keep it clean and shareable.\n"
                "- `speech` = the same line prepared for a Hindi TTS voice (Hindi words in Devanagari, English words in Latin).\n"
                "4) Instagram caption with #PlayQ and one roast_line about a shot."
            )
            video_part = types.Part(file_data=types.FileData(file_uri=up.uri, mime_type=up.mime_type or "video/mp4"),
                                    video_metadata=types.VideoMetadata(fps=4))   # default is 1 fps: too coarse to see shots
            race = [config.VIDEO_MODEL] + [m for m in config.MODEL_CHAIN if m != config.VIDEO_MODEL]
            return self._generate_hedged(
                [video_part, prompt], MatchInsights,
                "You are PlayQ's analyst and a hilarious Indian sports commentary duo. " + _GROUND, race, temperature=0.9,
            )
        except Exception as e:
            return {"_source": "error", "error": f"{type(e).__name__}: {e}"[:400]}
        finally:
            try:
                if up is not None:
                    self._client.files.delete(name=up.name)  # don't leave footage on Google's servers
            except Exception:
                pass

    # 3 ── voiceover ───────────────────────────────────────────────────────────
    def _pick_tts_model(self) -> str | None:
        if config.TTS_MODEL:
            return config.TTS_MODEL
        try:
            names = [m.name.replace("models/", "") for m in self._client.models.list()]
            tts = sorted(n for n in names if "tts" in n)
            return tts[0] if tts else None
        except Exception:
            return None

    def synthesize_voice(self, lines: list[str], out_wav: str) -> dict:
        text = " ".join(l.strip() for l in lines if l.strip())
        if not text:
            return {"_source": "error", "error": "no commentary text"}
        if self._client:
            model = self._pick_tts_model()
            if model:
                try:
                    resp = self._client.models.generate_content(
                        model=model,
                        contents="Say this like an excited Indian sports commentator, fast and punchy, "
                                 f"natural Hinglish pronunciation: {text}",
                        config=types.GenerateContentConfig(
                            response_modalities=["AUDIO"],
                            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=config.TTS_VOICE))),
                        ),
                    )
                    pcm = resp.candidates[0].content.parts[0].inline_data.data
                    with wave.open(out_wav, "wb") as w:   # Gemini TTS returns 24 kHz mono 16-bit PCM
                        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(pcm)
                    return {"_source": "gemini", "_model": model, "path": out_wav}
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"[:300]
            else:
                err = "no TTS model visible to this key"
        else:
            err = "no GEMINI_API_KEY"
        # Fallback so the demo never goes silent: macOS en-IN voice (reads Roman-script Hinglish far better than the hi-IN voice).
        if shutil.which("say"):
            aiff = out_wav.replace(".wav", ".aiff")
            subprocess.run(["say", "-v", os.environ.get("PLAYQ_FALLBACK_VOICE", "Rishi"), "-r", "185", "-o", aiff, text], check=False, capture_output=True)
            if os.path.exists(aiff):
                subprocess.run([_ffmpeg_exe(), "-y", "-i", aiff, "-ar", "24000", "-ac", "1", out_wav], check=True, capture_output=True)
                os.remove(aiff)
                return {"_source": "fallback-say", "path": out_wav, "why": err}
        return {"_source": "error", "error": err}


EDGE_VOICES = {"A": ("hi-IN-MadhurNeural", "+12%", "+3Hz"), "B": ("hi-IN-SwaraNeural", "+10%", "+2Hz")}


def _wav_len(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


# ── text-to-speech providers ────────────────────────────────────────────────────────
_CLOUD_VOICE_CACHE: dict = {}


def _cloud_token() -> tuple[str, str] | None:
    """OAuth token from a service-account JSON (GOOGLE_APPLICATION_CREDENTIALS). Cloud TTS rejects plain API keys."""
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not path or not os.path.exists(os.path.expanduser(path)):
        return None
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account
    creds = service_account.Credentials.from_service_account_file(
        os.path.expanduser(path), scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(Request())
    return creds.token, creds.project_id


def _cloud_voices(token: str, project: str) -> dict:
    """Pick one male + one female Hindi voice, best tier first: Chirp3-HD > Neural2 > Wavenet > Standard."""
    if _CLOUD_VOICE_CACHE:
        return _CLOUD_VOICE_CACHE
    import requests
    r = requests.get("https://texttospeech.googleapis.com/v1/voices", params={"languageCode": "hi-IN"},
                     headers={"Authorization": f"Bearer {token}", "x-goog-user-project": project}, timeout=20)
    r.raise_for_status()
    voices = r.json().get("voices", [])
    rank = lambda n: 0 if "Chirp3-HD" in n else 1 if "Neural2" in n else 2 if "Wavenet" in n else 3
    for gender, spk in (("MALE", "A"), ("FEMALE", "B")):
        cands = sorted((v for v in voices if v.get("ssmlGender") == gender), key=lambda v: (rank(v["name"]), v["name"]))
        if cands:
            _CLOUD_VOICE_CACHE[spk] = cands[0]["name"]
    return _CLOUD_VOICE_CACHE


def _synthesise_lines(lines: list[dict], tmp: str) -> tuple[list[str], str, dict]:
    """One WAV per line. Order: Google Cloud TTS (if a service account is set up) -> edge-tts neural voices."""
    import asyncio, base64
    # 1) Google Cloud Text-to-Speech
    tok = None
    try:
        tok = _cloud_token()
    except Exception as e:  # bad credential file etc.: fall through to edge-tts, remember why
        logging.getLogger("playq.agents").warning("Cloud TTS credentials unusable: %s", e)
    if tok:
        import requests
        token, project = tok
        try:
            voices = _cloud_voices(token, project)
            wavs = []
            for i, l in enumerate(lines):
                spk = l.get("speaker", "A")
                body = {"input": {"text": l.get("speech") or l["text_roman"]},
                        "voice": {"languageCode": "hi-IN", "name": voices.get(spk) or next(iter(voices.values()))},
                        "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000, "speakingRate": 1.08}}
                r = requests.post("https://texttospeech.googleapis.com/v1/text:synthesize", json=body, timeout=30,
                                  headers={"Authorization": f"Bearer {token}", "x-goog-user-project": project})
                r.raise_for_status()
                w = os.path.join(tmp, f"_line{i}.wav")
                with open(w, "wb") as fh:
                    fh.write(base64.b64decode(r.json()["audioContent"]))
                wavs.append(w)
            return wavs, "google-cloud-tts", voices
        except Exception as e:
            logging.getLogger("playq.agents").warning("Cloud TTS failed, using edge-tts: %s", str(e)[:200])
    # 2) edge-tts
    import edge_tts

    async def synth_all():
        async def one(i, l):
            voice, rate, pitch = EDGE_VOICES[l.get("speaker", "A")]
            mp3 = os.path.join(tmp, f"_line{i}.mp3")
            await edge_tts.Communicate(l.get("speech") or l["text_roman"], voice, rate=rate, pitch=pitch).save(mp3)
            return mp3
        return await asyncio.gather(*(one(i, l) for i, l in enumerate(lines)))

    mp3s = asyncio.run(asyncio.wait_for(synth_all(), timeout=45))
    wavs = []
    for m in mp3s:
        w = m.replace(".mp3", ".wav")
        subprocess.run([_ffmpeg_exe(), "-y", "-i", m, "-ar", "24000", "-ac", "1", w], check=True, capture_output=True)
        os.remove(m)
        wavs.append(w)
    return wavs, "edge-tts", {k: v[0] for k, v in EDGE_VOICES.items()}


def synthesize_dialogue(lines: list[dict], reel_s: float, out_wav: str, ai: "PrototypeAI | None" = None) -> dict:
    """
    Two-voice commentary with every line placed at its own timestamp.

    1. Each line is synthesised separately with a human-sounding neural Hindi voice (Microsoft Edge neural
       voices via `edge-tts`: no API key; needs internet). Speaker A -> Madhur, B -> Swara.
    2. Real durations are measured. A line that would collide with the next one (or run past the reel) is
       sped up by up to 1.35x; if it still does not fit it is dropped, so speech never overlaps.
    3. Lines are mixed onto one WAV at their start times (adelay).
    Falls back to a single local voice if edge-tts is unavailable.
    """
    import asyncio
    lines = sorted((l for l in lines if (l.get("speech") or l.get("text_roman")) and l.get("start_s") is not None),
                   key=lambda l: l["start_s"])
    if not lines:
        return {"_source": "error", "error": "no commentary lines"}
    tmp = os.path.dirname(out_wav)
    try:
        wavs, provider, voices_used = _synthesise_lines(lines, tmp)
        durs = [_wav_len(w) for w in wavs]

        # schedule: never overlap. Prefer natural speed; push a line later if the previous one is still talking;
        # only if the whole script would overrun the reel speed everything up a little (max 1.3x); drop the rest.
        seg_end = {}
        for l in lines:
            seg_end[l.get("segment")] = max(seg_end.get(l.get("segment"), 0), l["start_s"])
        def schedule(tempo):
            out, cur = [], 0.0
            for i, l in enumerate(lines):
                d = durs[i] / tempo
                start = max(l["start_s"], cur)
                if start + d > reel_s - 0.1:
                    out.append({"i": i, "dropped": True})
                    continue
                out.append({"i": i, "start": round(start, 2), "end": round(start + d, 2), "tempo": round(tempo, 2), "dropped": False})
                cur = start + d + 0.25
            return out
        for tempo in (1.0, 1.1, 1.2, 1.3):
            placed = schedule(tempo)
            if not any(p["dropped"] for p in placed):
                break
        for p in placed:
            if not p["dropped"] and p["tempo"] > 1.01:
                f = wavs[p["i"]].replace(".wav", "_t.wav")
                subprocess.run([_ffmpeg_exe(), "-y", "-i", wavs[p["i"]], "-filter:a", f"atempo={p['tempo']}", f], check=True, capture_output=True)
                os.replace(f, wavs[p["i"]])
        keep = [p for p in placed if not p["dropped"]]
        if not keep:
            return {"_source": "error", "error": "no line fit the timeline"}
        args, parts = [], []
        for n, p in enumerate(keep):
            args += ["-i", wavs[p["i"]]]
            parts.append(f"[{n}:a]adelay={int(p['start'] * 1000)}|{int(p['start'] * 1000)}[d{n}]")
        mix = "".join(f"[d{n}]" for n in range(len(keep))) + f"amix=inputs={len(keep)}:normalize=0,apad=whole_dur={reel_s:.2f}[out]"
        subprocess.run([_ffmpeg_exe(), "-y", *args, "-filter_complex", ";".join(parts + [mix]), "-map", "[out]", "-t", f"{reel_s:.2f}", out_wav],
                       check=True, capture_output=True)
        for w in wavs:
            if os.path.exists(w):
                os.remove(w)
        timing = [{"start_s": p["start"], "end_s": p["end"], "speaker": lines[p["i"]].get("speaker"), "text": lines[p["i"]].get("text_roman"),
                   "tempo": p["tempo"]} for p in keep]
        return {"_source": provider, "path": out_wav, "voices": voices_used, "timing": timing,
                "characters": sum(len(l.get("speech") or l.get("text_roman") or "") for l in lines),
                "dropped": [lines[p["i"]].get("text_roman") for p in placed if p["dropped"]]}
    except Exception as e:
        why = f"{type(e).__name__}: {e}"[:200]
        fb = (ai or PrototypeAI()).synthesize_voice([l.get("text_roman", "") for l in lines], out_wav)
        fb["why"] = f"neural TTS failed ({why})"
        return fb


def mux_voiceover(video_in: str, wav: str, video_out: str, duck: float = 0.35) -> str:
    """Put the voiceover over the reel; match audio is ducked only while someone is talking."""
    fc_duck = ("[1:a]asplit=2[v1][v2];[0:a][v1]sidechaincompress=threshold=0.015:ratio=10:attack=15:release=350[bg];"
               "[v2]volume=1.6[vo];[bg][vo]amix=inputs=2:duration=first:normalize=0[a]")
    fc_flat = f"[0:a]volume={duck}[a0];[1:a]volume=1.7[a1];[a0][a1]amix=inputs=2:duration=first:dropout_transition=0[a]"
    for fc in (fc_duck, fc_flat):
        r = subprocess.run([_ffmpeg_exe(), "-y", "-i", video_in, "-i", wav, "-filter_complex", fc, "-map", "0:v", "-map", "[a]",
                            "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", video_out], capture_output=True, text=True)
        if r.returncode == 0:
            return video_out
    raise RuntimeError("ffmpeg mux failed: " + r.stderr[-300:])
