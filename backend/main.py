"""
main.py — FastAPI backend para el sistema de prediccion Mundial 2026
"""
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    HealthResponse,
    MatchRequest,
    PredictionResponse,
    PredictionsListResponse,
    SimulationRequest,
    SimulationResponse,
    TeamStatsResponse,
)
from .predictor import Predictor
from .utils import build_prediction_response, build_simulation_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api.main")

predictor = Predictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando FastAPI — cargando modelos...")
    predictor.initialize()
    logger.info("Backend listo")
    yield
    logger.info("Apagando backend")


app = FastAPI(
    title="Mundial 2026 Predictor API",
    description="API de prediccion para partidos del Mundial 2026 usando CatBoost + Monte Carlo",
    version="1.0.0",
    lifespan=lifespan,
)

_ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if _ENVIRONMENT == "production":
    _ALLOWED_ORIGINS = [
        "https://mundial-predictor.streamlit.app",
        "https://mundial-predictor-api.onrender.com",
    ]
else:
    _ALLOWED_ORIGINS = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

logger.info("CORS origins: %s", _ALLOWED_ORIGINS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def health():
    """Estado del servidor y modelos cargados."""
    return {
        "status":              "ok",
        "models_loaded":       predictor.models_loaded,
        "n_features":          predictor.n_features,
        "historical_matches":  predictor.n_historical,
        "version":             "1.0.0",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediccion"])
def predict(req: MatchRequest):
    """
    Prediccion completa para un partido.
    Incluye probabilidades ML + goles esperados + simulacion Monte Carlo.
    """
    t0 = time.perf_counter()

    if req.team_home == req.team_away:
        raise HTTPException(400, "Los equipos no pueden ser iguales")

    try:
        result = predictor.predict(
            team_home=req.team_home,
            team_away=req.team_away,
            date=req.date,
            city=req.city,
            country=req.country,
            neutral=req.neutral,
            n_sim=req.n_sim,
        )
    except Exception as e:
        logger.error("Error en predict(%s vs %s): %s", req.team_home, req.team_away, e)
        raise HTTPException(500, f"Error en prediccion: {e}")

    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    logger.info("predict %s vs %s — %.0f ms", req.team_home, req.team_away, elapsed)

    return build_prediction_response(result, req.date)


@app.get("/stats/{team}", response_model=TeamStatsResponse, tags=["Estadisticas"])
def team_stats(team: str):
    """
    Estadisticas historicas de un equipo.
    ELO, tasa de victorias, goles promedio, forma reciente.
    """
    stats = predictor.get_team_stats(team)
    if stats is None:
        raise HTTPException(404, f"Equipo no encontrado: {team}")
    return stats


@app.post("/simulate", response_model=SimulationResponse, tags=["Simulacion"])
def simulate(req: SimulationRequest):
    """
    Simulacion Monte Carlo pura.
    Distribucion de goles, marcadores probables, partidos ambos marcan.
    """
    t0 = time.perf_counter()

    if req.team_home == req.team_away:
        raise HTTPException(400, "Los equipos no pueden ser iguales")

    try:
        sim = predictor.simulate(req.team_home, req.team_away, req.n_simulations)
    except Exception as e:
        logger.error("Error en simulate(%s vs %s): %s",
                     req.team_home, req.team_away, e)
        raise HTTPException(500, f"Error en simulacion: {e}")

    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    return build_simulation_response(
        sim, req.team_home, req.team_away, req.n_simulations, elapsed
    )


@app.get("/predictions/latest", response_model=PredictionsListResponse,
         tags=["Prediccion"])
def latest_predictions(limit: int = 10):
    """
    Ultimas N predicciones generadas.
    Lee de data/predictions_detailed.csv.
    """
    if limit < 1 or limit > 100:
        raise HTTPException(400, "limit debe estar entre 1 y 100")
    rows = predictor.get_latest_predictions(limit=limit)
    return {"predictions": rows, "count": len(rows)}


@app.get("/teams", tags=["Estadisticas"])
def list_teams():
    """Lista todos los equipos con historial."""
    return {"teams": predictor.get_all_teams()}
