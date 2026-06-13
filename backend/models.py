"""
models.py — Pydantic schemas para FastAPI
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class MatchRequest(BaseModel):
    team_home:  str = Field(..., example="Argentina")
    team_away:  str = Field(..., example="France")
    date:       str = Field(..., example="2026-06-14")
    city:       str = Field("", example="East Rutherford")
    country:    str = Field("United States", example="United States")
    neutral:    bool = Field(True)
    n_sim:      int  = Field(50_000, ge=1000, le=200_000)


class ScoreProbability(BaseModel):
    score:       str
    probability: float


class PredictionResponse(BaseModel):
    team_home:      str
    team_away:      str
    date:           str
    p_home:         float
    p_draw:         float
    p_away:         float
    lambda_home:    float
    lambda_away:    float
    prob_over25:    float
    mc_p_home:      float
    mc_p_draw:      float
    mc_p_away:      float
    mc_p_over25:    float
    mc_p_both_score: float
    mc_avg_goals:   float
    top_scores:     List[ScoreProbability]
    elo_home:       float
    elo_away:       float
    favorite:       str
    confidence:     str


class TeamStatsResponse(BaseModel):
    team:               str
    elo_rating:         float
    avg_goals_for:      float
    avg_goals_against:  float
    win_rate:           float
    draw_rate:          float
    loss_rate:          float
    clean_sheet_rate:   float
    over25_rate:        float
    recent_form:        str
    total_matches:      int
    last_match_date:    Optional[str]


class SimulationRequest(BaseModel):
    team_home:     str = Field(..., example="Brazil")
    team_away:     str = Field(..., example="Germany")
    n_simulations: int = Field(100_000, ge=1000, le=500_000)


class GoalsDistribution(BaseModel):
    home: Dict[str, float]
    away: Dict[str, float]


class SimulationResponse(BaseModel):
    team_home:          str
    team_away:          str
    simulations:        int
    lambda_home:        float
    lambda_away:        float
    p_home:             float
    p_draw:             float
    p_away:             float
    p_over25:           float
    p_both_score:       float
    p_clean_sheet_home: float
    p_clean_sheet_away: float
    avg_total_goals:    float
    goals_distribution: GoalsDistribution
    top_scores:         List[ScoreProbability]
    execution_time_ms:  float


class HealthResponse(BaseModel):
    status:        str
    models_loaded: bool
    n_features:    int
    historical_matches: int
    version:       str = "1.0.0"


class PredictionSummary(BaseModel):
    match_id:   str
    team_home:  str
    team_away:  str
    date:       str
    p_home:     float
    p_draw:     float
    p_away:     float
    lambda_home: float
    lambda_away: float
    favorite:   str


class PredictionsListResponse(BaseModel):
    predictions: List[PredictionSummary]
    count:       int
