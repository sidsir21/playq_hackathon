"""Settings: loads a project-root .env (no extra dependency) and exposes Gemini config."""
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path = _ROOT / ".env") -> None:
    """Load KEY=VALUE lines into os.environ without overriding variables already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if v and not os.environ.get(k):      # a non-empty value always beats a blank duplicate
            os.environ[k] = v
        else:
            os.environ.setdefault(k, v)


load_env()


def api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or None


# gemini-2.0-flash is shut down and 2.5 is deprecated (per AI Studio model docs, Nov 2026).
# Benchmarked on a video+schema call: 3.5-flash-lite 3.6s, 3.6-flash 9.8s, 3.5-flash 11.6s, 3.8-flash ~60s.
TEXT_MODEL = os.environ.get("PLAYQ_GEMINI_MODEL", "gemini-3.6-flash")
VIDEO_MODEL = os.environ.get("PLAYQ_GEMINI_VIDEO_MODEL", "gemini-3.6-flash")

# Text-to-speech model: leave blank to auto-pick the first "tts" model visible to your key.
TTS_MODEL = os.environ.get("PLAYQ_GEMINI_TTS_MODEL", "")
TTS_VOICE = os.environ.get("PLAYQ_GEMINI_TTS_VOICE", "Kore")

# Ordered preference list. Quotas are PER MODEL, so when the best models return 429/5xx the next ones take over.
MODEL_CHAIN = [m.strip() for m in os.environ.get("PLAYQ_GEMINI_MODEL_CHAIN", "gemini-3.6-flash,gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite").split(",") if m.strip()]

# USD per 1M tokens (input, output incl. thinking) from the AI Studio pricing page, Nov 2026. gemini-3.5-flash is not
# listed there, so it is priced like 3.6-flash as a conservative estimate. 3.6/3.8 prices double on 2027-01-01.
PRICES = {
    "gemini-3.6-flash": (0.75, 3.75), "gemini-3.8-flash": (0.75, 3.75), "gemini-3.5-flash": (0.75, 3.75),
    "gemini-3.5-flash-lite": (0.30, 2.50), "gemini-3.1-flash-lite": (0.25, 1.50),
}


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    pin, pout = PRICES.get(model, (0.75, 3.75))
    return (tokens_in * pin + tokens_out * pout) / 1_000_000


# Used automatically if the primary model returns 5xx / 429 / times out (e.g. "high demand" 503s).
FALLBACK_MODEL = os.environ.get("PLAYQ_GEMINI_FALLBACK_MODEL", "gemini-3.5-flash")
REQUEST_TIMEOUT_S = int(os.environ.get("PLAYQ_GEMINI_TIMEOUT_S", "45"))
