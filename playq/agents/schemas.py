"""Pydantic schemas Gemini must fill in (strict structured output, as promised in the pitch)."""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class RiskArea(BaseModel):
    joint: str
    angle_deg: Optional[float] = None
    risk_level: Literal["safe", "caution", "danger"]
    explanation: str


class Correction(BaseModel):
    issue: str
    exercise: str
    sets_reps: str
    priority: int = Field(ge=1, le=3, description="1 = most urgent")


class BiomechanicsDiagnosis(BaseModel):
    diagnosis: str
    severity: int = Field(ge=1, le=10)
    confidence_note: str = Field(description="What the data could NOT show (e.g. valgus not assessable from side view)")
    risk_areas: list[RiskArea]
    corrections: list[Correction]
    prevention_tips: list[str]


class DeadZone(BaseModel):
    zone_name: str
    percentage_time: float


class TacticsReport(BaseModel):
    tactical_analysis: str
    dead_zones: list[DeadZone]
    strengths: list[str]
    weaknesses: list[str]
    shot_consistency_0_99: Optional[int] = Field(default=None, ge=0, le=99, description="null if not judgeable")
    clutch_factor_0_99: Optional[int] = Field(default=None, ge=0, le=99, description="null if not judgeable")


class CommentaryLine(BaseModel):
    timestamp_s: float
    text: str
    emotion: str
    energy_level: int = Field(ge=1, le=10)


class BroadcastScript(BaseModel):
    commentary_script: list[CommentaryLine]
    instagram_caption: str
    music_prompt: str
    roast_line: str


class VerifiedRally(BaseModel):
    start_s: float
    end_s: float
    shots_visible: int
    is_real_rally: bool = Field(description="False if it is warm-up, ball pickup, or an adjacent court")
    note: str


class RallyVerification(BaseModel):
    rallies: list[VerifiedRally]
    missed_rallies: list[VerifiedRally] = Field(description="Rallies visible in the video that were NOT in the candidate list")
    players_visible: int
    summary: str


# ── Prototype pipeline schemas ────────────────────────────────────────────────
class BioVision(BaseModel):
    """Gemini's independent look at the flagged skeleton frame."""
    injury_risk_level: Literal["low", "moderate", "high"]
    coaching_cue: str = Field(description="One actionable sentence, max 20 words, plain English")
    observed_form: str = Field(description="What you actually see in the posture (stance width, trunk, knee tracking)")
    agrees_with_measurement: bool = Field(description="Does the image support the measured angle? false if the pose looks wrong/occluded")


class PlayerScores(BaseModel):
    track_id: int
    shot_consistency_0_99: Optional[int] = Field(default=None, ge=0, le=99, description="null if you cannot judge from the video")
    clutch_factor_0_99: Optional[int] = Field(default=None, ge=0, le=99, description="null if you cannot judge from the video")
    shot_aggression_0_99: Optional[int] = Field(default=None, ge=0, le=99, description="Shot STYLE: 0 = only soft dinks/resets, 99 = constant hard drives, smashes, speed-ups. null if not judgeable")
    reason: str = Field(description="One sentence grounded in what is visible (shots, errors, score situation)")


class VoiceLine(BaseModel):
    """One spoken line. Times are seconds on the REEL timeline (0 = first frame of the reel)."""
    start_s: float = Field(ge=0, description="When the line starts, in reel seconds")
    speaker: Literal["A", "B"] = Field(description="A = hype commentator (male), B = witty co-commentator (female)")
    text_roman: str = Field(description="Hinglish in Roman script, shown on screen. Max 13 words")
    speech: str = Field(description="Same line for a Hindi TTS engine: Hindi words in Devanagari, English words (dink, kitchen, smash, point, rally) in Latin script")
    emotion: str
    segment: int = Field(ge=0, description="Index of the reel segment this line describes")


class MatchInsights(BaseModel):
    score_visible: Optional[str] = Field(default=None, description="Scoreboard text if legible, e.g. 'Team A 4 - Team B 4'")
    player_scores: list[PlayerScores]
    hinglish_commentary: list[VoiceLine] = Field(description="Banter across the whole reel, at least one line per segment")
    instagram_caption: str
    roast_line: str
