"""
utils.py — Funciones auxiliares para FastAPI
"""
from typing import Dict, List


def build_prediction_response(result: Dict, date: str) -> Dict:
    """Construir respuesta estandarizada desde resultado de Predictor."""
    pred = result["prediction"]
    sim  = result["simulation"]

    ph, pd_, pa = pred["p_home"], pred["p_draw"], pred["p_away"]
    if ph >= pa and ph >= pd_:
        fav = result["team_home"]
    elif pa >= ph and pa >= pd_:
        fav = result["team_away"]
    else:
        fav = "Draw"

    max_p = max(ph, pd_, pa)
    if max_p > 0.60:
        confidence = "HIGH"
    elif max_p > 0.45:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    top_scores = [
        {"score": s["score"], "probability": s["prob"]}
        if "prob" in s else s
        for s in sim.get("top_scores", [])[:10]
    ]

    return {
        "team_home":        result["team_home"],
        "team_away":        result["team_away"],
        "date":             date,
        "p_home":           round(ph, 4),
        "p_draw":           round(pd_, 4),
        "p_away":           round(pa, 4),
        "lambda_home":      round(pred["lambda_home"], 3),
        "lambda_away":      round(pred["lambda_away"], 3),
        "prob_over25":      round(pred.get("prob_over25", sim.get("p_over25", 0.5)), 4),
        "mc_p_home":        round(sim.get("p_home_sim", ph), 4),
        "mc_p_draw":        round(sim.get("p_draw_sim", pd_), 4),
        "mc_p_away":        round(sim.get("p_away_sim", pa), 4),
        "mc_p_over25":      round(sim.get("p_over25", 0.5), 4),
        "mc_p_both_score":  round(sim.get("p_both_score", 0.5), 4),
        "mc_avg_goals":     round(sim.get("avg_total_goals", 2.5), 3),
        "top_scores":       top_scores,
        "elo_home":         round(result.get("elo_home", 1500.0), 1),
        "elo_away":         round(result.get("elo_away", 1500.0), 1),
        "favorite":         fav,
        "confidence":       confidence,
    }


def build_simulation_response(sim: Dict, team_home: str, team_away: str,
                               n_sim: int, elapsed_ms: float) -> Dict:
    """Construir respuesta de simulacion."""
    top_scores = []
    for s in sim.get("top_scores", [])[:10]:
        if "prob" in s:
            top_scores.append({"score": s["score"], "probability": s["prob"]})
        elif "probability" in s:
            top_scores.append(s)

    return {
        "team_home":          team_home,
        "team_away":          team_away,
        "simulations":        n_sim,
        "lambda_home":        round(sim.get("lambda_home", 1.2), 3),
        "lambda_away":        round(sim.get("lambda_away", 1.0), 3),
        "p_home":             round(sim.get("p_home_sim", 0.33), 4),
        "p_draw":             round(sim.get("p_draw_sim", 0.33), 4),
        "p_away":             round(sim.get("p_away_sim", 0.33), 4),
        "p_over25":           round(sim.get("p_over25", 0.5), 4),
        "p_both_score":       round(sim.get("p_both_score", 0.5), 4),
        "p_clean_sheet_home": round(sim.get("p_clean_sheet_h", 0.2), 4),
        "p_clean_sheet_away": round(sim.get("p_clean_sheet_a", 0.2), 4),
        "avg_total_goals":    round(sim.get("avg_total_goals", 2.5), 3),
        "goals_distribution": {
            "home": sim.get("dist_goals_home", {}),
            "away": sim.get("dist_goals_away", {}),
        },
        "top_scores":         top_scores,
        "execution_time_ms":  elapsed_ms,
    }
