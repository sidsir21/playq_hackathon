"""
PlayQ Rating System
====================
FIFA-style 5-stat player rating card and data models
for recreational sports analysis.
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Sport(str, Enum):
    """Supported sports."""
    PICKLEBALL = "pickleball"
    PADEL = "padel"
    TURF_FOOTBALL = "turf_football"
    BOX_CRICKET = "box_cricket"
    BADMINTON = "badminton"


class RiskLevel(str, Enum):
    """Biomechanical risk classification."""
    SAFE = "safe"
    CAUTION = "caution"
    DANGER = "danger"


STAT_KEYS = ("court_coverage", "shot_consistency", "attack_aggression", "stamina", "clutch_factor")
STAT_ABBR = {
    "court_coverage": "COV", "shot_consistency": "CON", "attack_aggression": "AGG",
    "stamina": "STA", "clutch_factor": "CLU",
}


class PlayQStats(BaseModel):
    """
    FIFA/2K-style 5-attribute rating on a 0-99 scale (matches the pitch deck).

    - court_coverage   : share of the player's half actually covered (measured from tracking)
    - shot_consistency : unforced-error rate / rally length (needs shot outcomes -> Gemini)
    - attack_aggression: time spent at the net / in the transition zone (measured from tracking)
    - stamina          : movement output sustained through the session (measured from tracking)
    - clutch_factor    : performance in close points (needs score tracking -> Gemini)

    `placeholders` lists the attributes that had no real measurement and sit at neutral 50.
    """
    court_coverage: int = Field(ge=0, le=99, default=50)
    shot_consistency: int = Field(ge=0, le=99, default=50)
    attack_aggression: int = Field(ge=0, le=99, default=50)
    stamina: int = Field(ge=0, le=99, default=50)
    clutch_factor: int = Field(ge=0, le=99, default=50)
    placeholders: list[str] = Field(default_factory=list)

    @property
    def overall(self) -> int:
        """Equal-weighted overall rating."""
        return round(sum(getattr(self, k) for k in STAT_KEYS) / len(STAT_KEYS))


class BiomechanicsReport(BaseModel):
    """Summary of biomechanical analysis from pose data."""
    risk_level: RiskLevel = RiskLevel.SAFE
    knee_valgus_max: Optional[float] = None
    shoulder_abduction_max: Optional[float] = None
    lumbar_flexion_max: Optional[float] = None
    high_risk_moments: int = 0
    recommendations: list[str] = Field(default_factory=list)


class MatchResult(BaseModel):
    """Complete result from a single match analysis."""
    match_id: str
    player_id: str
    sport: Sport
    date: datetime
    opponent_id: Optional[str] = None
    stats: PlayQStats
    biomechanics: BiomechanicsReport
    heatmap_url: Optional[str] = None
    highlight_reel_url: Optional[str] = None
    rally_count: int = 0
    duration_minutes: float = 0.0


class PlayerProfile(BaseModel):
    """Persistent player profile with longitudinal rating tracking."""
    player_id: str
    display_name: str
    sport: Sport
    current_rating: PlayQStats = Field(default_factory=PlayQStats)
    match_history: list[str] = Field(default_factory=list)  # match_ids
    total_matches: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def update_rating(self, new_match: MatchResult, decay: float = 0.3) -> None:
        """
        Update player rating using exponential moving average.
        new_stat = old_stat * (1 - decay) + match_stat * decay
        """
        old = self.current_rating
        new = new_match.stats
        # First match seeds the rating directly; later matches blend via EMA.
        if self.total_matches == 0:
            self.current_rating = new.model_copy()
        else:
            blended = {k: round(getattr(old, k) * (1 - decay) + getattr(new, k) * decay) for k in STAT_KEYS}
            self.current_rating = PlayQStats(**blended, placeholders=new.placeholders)
        self.match_history.append(new_match.match_id)
        self.total_matches += 1
        self.updated_at = datetime.now()


class RatingEngine:
    """
    Computes PlayQ stats from CV analysis outputs and generates
    FIFA-style card data for rendering.
    """

    @staticmethod
    def _to_stat(x: Optional[float]) -> tuple[int, bool]:
        """0-1 measurement -> 1-99 stat. None -> neutral 50, flagged as placeholder."""
        if x is None:
            return 50, True
        return int(min(99, max(1, round(x * 100)))), False

    def compute_match_stats(self, measurements: dict[str, Optional[float]]) -> PlayQStats:
        """
        Map normalised (0-1) measurements to 0-99 stats.

        Args:
            measurements: any of STAT_KEYS -> float in [0, 1], or None / missing when the
                          attribute could not be measured (it becomes a flagged placeholder).
        """
        stats, placeholders = {}, []
        for k in STAT_KEYS:
            stats[k], is_placeholder = self._to_stat(measurements.get(k))
            if is_placeholder:
                placeholders.append(k)
        return PlayQStats(**stats, placeholders=placeholders)

    def generate_card_data(self, profile: PlayerProfile) -> dict:
        """
        Generate data for rendering a FIFA-style PlayQ rating card.

        Returns:
            Dict with all fields needed for the card UI.
        """
        stats = profile.current_rating
        overall = stats.overall

        # Determine card tier based on overall rating
        if overall >= 85:
            tier = "legendary"
            color = "#FFD700"  # Gold
        elif overall >= 75:
            tier = "elite"
            color = "#C0C0C0"  # Silver
        elif overall >= 60:
            tier = "pro"
            color = "#CD7F32"  # Bronze
        else:
            tier = "amateur"
            color = "#4A90D9"  # Blue

        # Sport icon mapping
        sport_icons = {
            Sport.PICKLEBALL: "🏓",
            Sport.PADEL: "🎾",
            Sport.TURF_FOOTBALL: "⚽",
            Sport.BOX_CRICKET: "🏏",
            Sport.BADMINTON: "🏸",
        }

        return {
            "player_name": profile.display_name,
            "player_id": profile.player_id,
            "overall": overall,
            "tier": tier,
            "tier_color": color,
            "sport": profile.sport.value,
            "sport_icon": sport_icons.get(profile.sport, "🎯"),
            "stats": {STAT_ABBR[k]: getattr(stats, k) for k in STAT_KEYS},
            "placeholder_stats": [STAT_ABBR[k] for k in stats.placeholders],
            "total_matches": profile.total_matches,
            "card_version": "PlayQ v0.2",
        }


if __name__ == "__main__":
    # Demo: Create a player and simulate matches
    print("⭐ PlayQ Rating Engine Demo")
    print("=" * 40)

    engine = RatingEngine()

    # Create player
    player = PlayerProfile(
        player_id="sid_001",
        display_name="Sid Sirohi",
        sport=Sport.PICKLEBALL,
    )

    # Simulate a match (None = not measured yet)
    stats = engine.compute_match_stats({
        "court_coverage": 0.68, "attack_aggression": 0.55, "stamina": 0.72,
        "shot_consistency": None, "clutch_factor": None,
    })

    match = MatchResult(
        match_id="match_001",
        player_id=player.player_id,
        sport=Sport.PICKLEBALL,
        date=datetime.now(),
        stats=stats,
        biomechanics=BiomechanicsReport(risk_level=RiskLevel.CAUTION, knee_valgus_max=15.0),
        rally_count=12,
        duration_minutes=22.5,
    )

    player.update_rating(match)
    card = engine.generate_card_data(player)

    print(f"\n{'─' * 30}")
    print(f"  {card['sport_icon']} {card['player_name']}")
    print(f"  OVERALL: {card['overall']}  [{card['tier'].upper()}]")
    print(f"{'─' * 30}")
    for stat, val in card["stats"].items():
        bar = "█" * (val // 5) + "░" * (20 - val // 5)
        print(f"  {stat:>3}: {val:>2}  {bar}")
    print(f"{'─' * 30}")
    print(f"  Matches: {card['total_matches']}  |  {card['card_version']}")
