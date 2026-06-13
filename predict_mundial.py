#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_mundial.py
==================
Fase 4 - Prediccion + Monte Carlo
Modelo Predictivo Mundial 2026

Uso:
    python predict_mundial.py                    # partidos WC2026 desde API
    python predict_mundial.py --demo             # demo con ultimos partidos historicos
    python predict_mundial.py --partido "ARG" "FRA" 2026-07-14

Salida:
    reports/predictions_EQUIPO_vs_EQUIPO.txt
    reports/all_predictions.txt
    data/predictions_detailed.csv
    data/simulations_results.json
    plots/*.png
"""

import argparse
import json
import logging
import pickle
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ─── CONFIG ─────────────────────────────────────────────────────────────────

DATA_DIR    = Path("data")
MODELS_DIR  = Path("models")
REPORTS_DIR = Path("reports")
PLOTS_DIR   = Path("plots")

for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR, PLOTS_DIR]:
    d.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(DATA_DIR / "log_predict.txt", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("predict")

N_SIM = 100_000
RANDOM_SEED = 42

# Importar el FeatureEngineer de Fase 2 para re-usar logica identica
sys.path.insert(0, str(Path(__file__).parent))
try:
    from feature_engineering import FeatureEngineer, CITY_ALT_M, COUNTRY_CLIMATE
    FEAT_ENG_AVAILABLE = True
except ImportError:
    FEAT_ENG_AVAILABLE = False
    logger.warning("feature_engineering.py no encontrado — usando modo simplificado")

# ELO por defecto para equipos nuevos
DEFAULT_ELO = 1500.0

# Mapa de codigos FIFA / abreviaturas → nombre normalizado
TEAM_ALIASES = {
    "ARG": "Argentina", "FRA": "France", "BRA": "Brazil",
    "ENG": "England",   "ESP": "Spain",  "GER": "Germany",
    "POR": "Portugal",  "NED": "Netherlands", "BEL": "Belgium",
    "ITA": "Italy",     "CRO": "Croatia","MAR": "Morocco",
    "MEX": "Mexico",    "USA": "United States", "CAN": "Canada",
    "JPN": "Japan",     "KOR": "South Korea",   "URU": "Uruguay",
    "COL": "Colombia",  "SEN": "Senegal","AUS": "Australia",
    "DEN": "Denmark",   "POL": "Poland", "SUI": "Switzerland",
    "ECU": "Ecuador",   "QAT": "Qatar",  "IRI": "Iran",
    "SRB": "Serbia",    "GHA": "Ghana",  "CMR": "Cameroon",
    "TUN": "Tunisia",   "CRC": "Costa Rica",
    "Czechia": "Czech Republic",
}


# ─── CLASE PRINCIPAL ─────────────────────────────────────────────────────────

class PredictorMundial:
    """
    Predictor completo con Monte Carlo para partidos del Mundial 2026.

    Pipeline:
      1. load_models()            — carga los 4 modelos .pkl
      2. load_historical_data()   — carga matches_cleaned.csv
      3. load_next_matches()      — descarga fixtures desde Football-Data API
      4. build_features()         — computa las 118 features via FeatureEngineer
      5. predict_all()            — predicciones + Monte Carlo por partido
      6. generate_reports()       — reportes TXT + CSV + JSON + PNG
    """

    # ── INICIALIZACION ───────────────────────────────────────────────────────

    def __init__(self, models_dir: str = "models", api_key: str = ""):
        self.models_dir  = Path(models_dir)
        self.api_key     = api_key
        self.rng         = np.random.default_rng(RANDOM_SEED)

        self.model_resultado  = None
        self.model_goals_home = None
        self.model_goals_away = None
        self.model_over25     = None
        self.features_list: List[str] = []

        self.df_hist: Optional[pd.DataFrame] = None   # historical matches
        self.df_upcoming: Optional[pd.DataFrame] = None  # matches to predict
        self.df_features: Optional[pd.DataFrame] = None  # engineered features

        self._predictions: List[Dict] = []
        self._simulations: List[Dict] = []

        self.load_models()
        self.load_historical_data()

    def load_models(self):
        """Cargar los 4 modelos CatBoost y la lista de features seleccionadas."""
        logger.info("Cargando modelos...")
        for name, attr in [
            ("model_resultado.pkl",  "model_resultado"),
            ("model_goals_home.pkl", "model_goals_home"),
            ("model_goals_away.pkl", "model_goals_away"),
            ("model_over25.pkl",     "model_over25"),
        ]:
            path = self.models_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Modelo no encontrado: {path}")
            with open(path, "rb") as f:
                setattr(self, attr, pickle.load(f))
            logger.info("  [OK] %s", name)

        feat_path = self.models_dir / "selected_features.json"
        if feat_path.exists():
            with open(feat_path, encoding="utf-8") as f:
                self.features_list = json.load(f)["features"]
            logger.info("  [OK] %d features seleccionadas", len(self.features_list))
        else:
            raise FileNotFoundError("selected_features.json no encontrado en models/")

    def load_historical_data(self):
        """Cargar matches_cleaned.csv como base historica para features."""
        path = DATA_DIR / "matches_cleaned.csv"
        if not path.exists():
            raise FileNotFoundError(f"No encontrado: {path}")
        self.df_hist = pd.read_csv(path)
        self.df_hist["date"] = pd.to_datetime(self.df_hist["date"])
        logger.info("Historial cargado: %d partidos", len(self.df_hist))

    # ── DESCARGA DE PARTIDOS ─────────────────────────────────────────────────

    def load_next_matches(self, status: str = "SCHEDULED") -> pd.DataFrame:
        """
        Descarga partidos WC2026 desde Football-Data.org.
        status: SCHEDULED (proximos) | FINISHED (jugados) | LIVE | IN_PLAY
        """
        logger.info("Descargando partidos WC2026 (status=%s)...", status)

        if not self.api_key:
            logger.warning("Sin API key — usando demo con ultimos partidos historicos")
            return self._demo_matches()

        try:
            url = f"https://api.football-data.org/v4/competitions/WC/matches?status={status}"
            resp = requests.get(url,
                                headers={"X-Auth-Token": self.api_key},
                                timeout=15)
            resp.raise_for_status()
            data = resp.json()
            matches_raw = data.get("matches", [])

            rows = []
            for m in matches_raw:
                rows.append({
                    "match_id":  f"fd_{m['id']}",
                    "date":      m["utcDate"].split("T")[0],
                    "team_home": m.get("homeTeam", {}).get("name", ""),
                    "team_away": m.get("awayTeam", {}).get("name", ""),
                    "goals_home": m.get("score", {}).get("fullTime", {}).get("home"),
                    "goals_away": m.get("score", {}).get("fullTime", {}).get("away"),
                    "competition": "FIFA World Cup",
                    "neutral": True,
                    "city":    m.get("venue", ""),
                    "country": "USA",
                    "status":  m.get("status", status),
                })

            df = pd.DataFrame(rows)
            logger.info("  Encontrados: %d partidos", len(df))
            return df

        except Exception as e:
            logger.warning("Error API: %s — usando demo", str(e)[:80])
            return self._demo_matches()

    def _demo_matches(self) -> pd.DataFrame:
        """
        Modo demo: usar los 12 partidos mas recientes del historial
        + 4 partidos hipoteticos WC2026 de ejemplo.
        """
        logger.info("  Modo DEMO: usando partidos recientes + fixtures ejemplo")

        # Partidos ya jugados del historial (los ultimos 8)
        hist_recent = self.df_hist.nlargest(8, "date")[
            ["match_id", "date", "team_home", "team_away",
             "goals_home", "goals_away", "competition", "neutral",
             "city", "country"]
        ].copy()
        hist_recent["status"] = "FINISHED"

        # Fixtures hipoteticos WC2026
        fixtures_wc26 = pd.DataFrame([
            {"match_id": "wc26_001", "date": "2026-06-14",
             "team_home": "Argentina",     "team_away": "France",
             "goals_home": None, "goals_away": None,
             "competition": "FIFA World Cup", "neutral": True,
             "city": "East Rutherford", "country": "United States", "status": "SCHEDULED"},
            {"match_id": "wc26_002", "date": "2026-06-15",
             "team_home": "Brazil",        "team_away": "Germany",
             "goals_home": None, "goals_away": None,
             "competition": "FIFA World Cup", "neutral": True,
             "city": "Inglewood", "country": "United States", "status": "SCHEDULED"},
            {"match_id": "wc26_003", "date": "2026-06-16",
             "team_home": "Spain",         "team_away": "England",
             "goals_home": None, "goals_away": None,
             "competition": "FIFA World Cup", "neutral": True,
             "city": "Dallas", "country": "United States", "status": "SCHEDULED"},
            {"match_id": "wc26_004", "date": "2026-06-17",
             "team_home": "Morocco",       "team_away": "Japan",
             "goals_home": None, "goals_away": None,
             "competition": "FIFA World Cup", "neutral": True,
             "city": "Vancouver", "country": "Canada", "status": "SCHEDULED"},
            {"match_id": "wc26_005", "date": "2026-06-18",
             "team_home": "United States", "team_away": "Mexico",
             "goals_home": None, "goals_away": None,
             "competition": "FIFA World Cup", "neutral": True,
             "city": "Kansas City", "country": "United States", "status": "SCHEDULED"},
            {"match_id": "wc26_006", "date": "2026-06-19",
             "team_home": "Portugal",      "team_away": "Netherlands",
             "goals_home": None, "goals_away": None,
             "competition": "FIFA World Cup", "neutral": True,
             "city": "Philadelphia", "country": "United States", "status": "SCHEDULED"},
        ])

        df = pd.concat([hist_recent, fixtures_wc26], ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ── CONSTRUCCION DE FEATURES ─────────────────────────────────────────────

    def build_features(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Construye las 118 features para los partidos en df_new.
        Metodo: agrega df_new al historial, corre FeatureEngineer,
        extrae las filas de los nuevos partidos.
        Garantia anti-leakage: las features de cada nuevo partido
        usan SOLO datos historicos anteriores a su fecha.
        """
        logger.info("Construyendo features para %d partidos...", len(df_new))

        # Preparar df_new con columnas requeridas
        df_new = df_new.copy()
        df_new["date"]       = pd.to_datetime(df_new["date"])
        df_new["resultado"]  = df_new.apply(
            lambda r: (1 if r["goals_home"] > r["goals_away"]
                       else -1 if r["goals_home"] < r["goals_away"] else 0)
            if pd.notna(r.get("goals_home")) and pd.notna(r.get("goals_away"))
            else 0,
            axis=1
        )
        df_new["goals_home"] = df_new["goals_home"].fillna(0)
        df_new["goals_away"] = df_new["goals_away"].fillna(0)
        df_new["neutral"]    = df_new.get("neutral", True)

        # ELO para los equipos nuevos: usar el ultimo ELO conocido del historial
        elo_map = self._last_elo_per_team()
        df_new["elo_home"] = df_new["team_home"].map(elo_map).fillna(DEFAULT_ELO)
        df_new["elo_away"] = df_new["team_away"].map(elo_map).fillna(DEFAULT_ELO)
        df_new["diff_elo"] = df_new["elo_home"] - df_new["elo_away"]

        # Asegurar todas las columnas del historial esten en df_new
        required_cols = list(self.df_hist.columns)
        for col in required_cols:
            if col not in df_new.columns:
                df_new[col] = np.nan

        new_ids = set(df_new["match_id"].astype(str))

        if FEAT_ENG_AVAILABLE:
            # Combinar historial + nuevos, ordenados por fecha
            df_combined = pd.concat(
                [self.df_hist, df_new[self.df_hist.columns]],
                ignore_index=True
            )
            df_combined = df_combined.drop_duplicates(subset=["match_id"])

            fe = FeatureEngineer(df_combined)
            df_all_feat = fe.generate_all()

            # Extraer solo las filas de los partidos nuevos
            mask = df_all_feat["match_id"].astype(str).isin(new_ids)
            df_feat = df_all_feat[mask].copy()
        else:
            # Modo simplificado si FeatureEngineer no esta disponible
            df_feat = self._build_features_simple(df_new)

        logger.info("  Features construidas: %d filas x %d cols",
                    len(df_feat), len(df_feat.columns))
        return df_feat

    def _last_elo_per_team(self) -> Dict[str, float]:
        """Ultimo ELO conocido por equipo del historial."""
        elo_home = (self.df_hist.sort_values("date")
                    .groupby("team_home")["elo_home"].last())
        elo_away = (self.df_hist.sort_values("date")
                    .groupby("team_away")["elo_away"].last())
        elo = elo_home.combine_first(elo_away).to_dict()
        return {str(k): float(v) for k, v in elo.items()}

    def _build_features_simple(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Fallback: construye un subconjunto de features sin FeatureEngineer.
        Suficiente para las features mas importantes (elo_ratio, forma, h2h).
        """
        rows = []
        for _, row in df_new.iterrows():
            h = row["team_home"]
            a = row["team_away"]
            d = pd.to_datetime(row["date"])

            elo_h = float(row.get("elo_home", DEFAULT_ELO))
            elo_a = float(row.get("elo_away", DEFAULT_ELO))
            elo_d = elo_h - elo_a

            # Historial de cada equipo
            def _team_hist(team, before):
                mask = (
                    ((self.df_hist["team_home"] == team) |
                     (self.df_hist["team_away"] == team)) &
                    (self.df_hist["date"] < before)
                )
                h = self.df_hist[mask].sort_values("date").tail(10)
                gf, ga, wins, draws = [], [], [], []
                for _, r in h.iterrows():
                    is_home = r["team_home"] == team
                    gf.append(r["goals_home"] if is_home else r["goals_away"])
                    ga.append(r["goals_away"] if is_home else r["goals_home"])
                    res = r.get("resultado", 0)
                    wins.append(1 if (is_home and res == 1) or (not is_home and res == -1) else 0)
                    draws.append(1 if res == 0 else 0)
                return gf, ga, wins, draws

            gf_h, ga_h, w_h, d_h = _team_hist(h, d)
            gf_a, ga_a, w_a, d_a = _team_hist(a, d)

            def _avg(lst, n): return float(np.mean(lst[-n:])) if lst else 1.2

            feat = {
                "match_id": row["match_id"],
                "elo_home": elo_h, "elo_away": elo_a,
                "elo_diff": elo_d, "elo_ratio": elo_h / max(elo_a, 1),
                "elo_win_prob_home": 1 / (1 + 10 ** (-elo_d / 400)),
                "elo_win_prob_away": 1 / (1 + 10 ** (elo_d / 400)),
                "match_competitiveness": 1 / (1 + abs(elo_d) / 400),
                "underdog_strength": max(0, elo_a - elo_h),
                "goals_for_avg5_home":    _avg(gf_h, 5),
                "goals_against_avg5_home": _avg(ga_h, 5),
                "goals_for_avg5_away":    _avg(gf_a, 5),
                "goals_against_avg5_away": _avg(ga_a, 5),
                "win_avg5_home":   _avg(w_h, 5),
                "win_avg5_away":   _avg(w_a, 5),
                "draw_avg5_home":  _avg(d_h, 5),
                "draw_avg5_away":  _avg(d_a, 5),
                "avg_goals_team_home": _avg(gf_h, 10),
                "avg_goals_team_away": _avg(gf_a, 10),
                "goal_balance_diff": (_avg(gf_h, 5) - _avg(ga_h, 5)) -
                                     (_avg(gf_a, 5) - _avg(ga_a, 5)),
                "is_neutral": int(row.get("neutral", True)),
                "is_world_cup": 1,
                "competition_level": 1.0,
            }
            # Rellenar el resto con 0
            for f in self.features_list:
                if f not in feat:
                    feat[f] = 0.0
            rows.append(feat)

        return pd.DataFrame(rows)

    # ── PREDICCION PUNTUAL ───────────────────────────────────────────────────

    def predict_partido(self, X: pd.DataFrame) -> Dict:
        """
        Prediccion puntual para un partido.
        X: DataFrame de 1 fila con las 118 features.
        """
        X_arr = X[self.features_list].fillna(0).astype(np.float32)

        # Probabilidades resultado (clases: 0=-1visita, 1=empate, 2=local)
        probs_raw = self.model_resultado.predict_proba(X_arr)[0]
        # CatBoost ordena clases por clase label (0,1,2)
        p_away, p_draw, p_home = float(probs_raw[0]), float(probs_raw[1]), float(probs_raw[2])

        # Goles esperados (Poisson lambda)
        lambda_home = float(max(0.05, self.model_goals_home.predict(X_arr)[0]))
        lambda_away = float(max(0.05, self.model_goals_away.predict(X_arr)[0]))

        # Over 2.5
        prob_over25 = float(self.model_over25.predict_proba(X_arr)[0][1])

        return {
            "p_home":      round(p_home, 4),
            "p_draw":      round(p_draw, 4),
            "p_away":      round(p_away, 4),
            "lambda_home": round(lambda_home, 3),
            "lambda_away": round(lambda_away, 3),
            "prob_over25": round(prob_over25, 4),
        }

    # ── MONTE CARLO ──────────────────────────────────────────────────────────

    def simulate_monte_carlo(
        self,
        lambda_home: float,
        lambda_away: float,
        n_sim: int = N_SIM
    ) -> Dict:
        """
        Simulacion Monte Carlo usando distribucion Poisson.
        n_sim=100,000 tarda ~10ms con numpy vectorizado.
        """
        lh = max(0.05, lambda_home)
        la = max(0.05, lambda_away)

        goals_h = self.rng.poisson(lh, n_sim).clip(0, 10)
        goals_a = self.rng.poisson(la, n_sim).clip(0, 10)

        total = goals_h + goals_a

        # Frecuencias resultado
        p_home_sim = float((goals_h > goals_a).mean())
        p_draw_sim = float((goals_h == goals_a).mean())
        p_away_sim = float((goals_h < goals_a).mean())

        # Top marcadores
        scores_cnt = Counter(zip(goals_h.tolist(), goals_a.tolist()))
        top_scores = [
            {"score": f"{h}-{a}", "prob": round(cnt / n_sim, 4)}
            for (h, a), cnt in scores_cnt.most_common(15)
        ]

        # Distribuciones de goles
        def _dist(arr):
            c = Counter(arr.tolist())
            return {str(k): round(v / n_sim, 4) for k, v in sorted(c.items()) if k <= 7}

        return {
            "n_sim": n_sim,
            "p_home_sim":        round(p_home_sim, 4),
            "p_draw_sim":        round(p_draw_sim, 4),
            "p_away_sim":        round(p_away_sim, 4),
            "p_over25":          round(float((total >= 3).mean()), 4),
            "p_both_score":      round(float(((goals_h > 0) & (goals_a > 0)).mean()), 4),
            "p_clean_sheet_h":   round(float((goals_a == 0).mean()), 4),
            "p_clean_sheet_a":   round(float((goals_h == 0).mean()), 4),
            "p_home_2plus":      round(float((goals_h >= 2).mean()), 4),
            "p_away_2plus":      round(float((goals_a >= 2).mean()), 4),
            "avg_total_goals":   round(float(total.mean()), 3),
            "std_total_goals":   round(float(total.std()), 3),
            "top_scores":        top_scores,
            "dist_goals_home":   _dist(goals_h),
            "dist_goals_away":   _dist(goals_a),
        }

    # ── PIPELINE COMPLETO ─────────────────────────────────────────────────────

    def predict_all(self, df_matches: pd.DataFrame) -> List[Dict]:
        """Predecir todos los partidos en df_matches. Retorna lista de resultados."""
        logger.info("Construyendo features para %d partidos...", len(df_matches))
        df_feat = self.build_features(df_matches)

        results = []
        for _, row in df_matches.iterrows():
            mid = str(row["match_id"])
            mask = df_feat["match_id"].astype(str) == mid
            rows_feat = df_feat[mask]

            if rows_feat.empty:
                logger.warning("Sin features para match_id=%s — omitido", mid)
                continue

            # Columnas disponibles de las 118 seleccionadas
            available = [f for f in self.features_list if f in rows_feat.columns]
            missing   = [f for f in self.features_list if f not in rows_feat.columns]
            if missing:
                for mf in missing:
                    rows_feat = rows_feat.copy()
                    rows_feat[mf] = 0.0

            pred = self.predict_partido(rows_feat[self.features_list])
            sim  = self.simulate_monte_carlo(pred["lambda_home"], pred["lambda_away"])

            # Resultado real (si ya se jugo)
            real_result = None
            if pd.notna(row.get("goals_home")) and pd.notna(row.get("goals_away")):
                gh_real = int(row["goals_home"]) if row["goals_home"] > 0 else 0
                ga_real = int(row["goals_away"]) if row["goals_away"] > 0 else 0
                real_result = {
                    "goals_home": gh_real,
                    "goals_away": ga_real,
                    "resultado": 1 if gh_real > ga_real else -1 if gh_real < ga_real else 0,
                }

            result = {
                "match_id":    mid,
                "date":        str(row["date"])[:10],
                "team_home":   row["team_home"],
                "team_away":   row["team_away"],
                "competition": row.get("competition", "FIFA World Cup"),
                "status":      row.get("status", "SCHEDULED"),
                "prediction":  pred,
                "simulation":  sim,
                "real_result": real_result,
                "elo_home":    float(rows_feat["elo_home"].iloc[0]) if "elo_home" in rows_feat else DEFAULT_ELO,
                "elo_away":    float(rows_feat["elo_away"].iloc[0]) if "elo_away" in rows_feat else DEFAULT_ELO,
            }
            results.append(result)

        self._predictions = results
        return results

    # ── REPORTES ─────────────────────────────────────────────────────────────

    def _bar(self, pct: float, width: int = 20) -> str:
        """Barra ASCII proporcional."""
        filled = int(pct * width)
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    def generate_report_partido(self, result: Dict) -> str:
        """Reporte detallado ASCII para un partido."""
        h = result["team_home"]
        a = result["team_away"]
        pred = result["prediction"]
        sim  = result["simulation"]

        ph = pred["p_home"]
        pd_ = pred["p_draw"]
        pa  = pred["p_away"]

        # Favorito
        if ph >= pa and ph >= pd_:
            fav, fav_p = h, ph
        elif pa >= ph and pa >= pd_:
            fav, fav_p = a, pa
        else:
            fav, fav_p = "Empate", pd_

        lines = [
            "=" * 60,
            f"  {h.upper()} vs {a.upper()}",
            f"  {result['date']}  |  {result['competition']}",
            f"  ELO: {result['elo_home']:.0f} vs {result['elo_away']:.0f}",
            "=" * 60,
            "",
            "PROBABILIDADES RESULTADO (modelo + simulacion MC):",
            f"  {h:<22} {ph*100:5.1f}%  {self._bar(ph)}",
            f"  {'Empate':<22} {pd_*100:5.1f}%  {self._bar(pd_)}",
            f"  {a:<22} {pa*100:5.1f}%  {self._bar(pa)}",
            "",
            "  [MC simul.]",
            f"  {h:<22} {sim['p_home_sim']*100:5.1f}%",
            f"  {'Empate':<22} {sim['p_draw_sim']*100:5.1f}%",
            f"  {a:<22} {sim['p_away_sim']*100:5.1f}%",
            "",
            f"  FAVORITO: {fav}  ({fav_p*100:.1f}%)",
            "",
            "GOLES ESPERADOS (lambda Poisson):",
            f"  {h}: {pred['lambda_home']:.2f} goles/partido",
            f"  {a}: {pred['lambda_away']:.2f} goles/partido",
            f"  Total estimado: {pred['lambda_home']+pred['lambda_away']:.2f}",
            "",
            "MARCADORES MAS PROBABLES (Top 10 de 100,000 sim.):",
        ]
        for i, sc in enumerate(sim["top_scores"][:10], 1):
            lines.append(f"  {i:2}. {h} {sc['score']} {a}   {sc['prob']*100:.2f}%")

        lines += [
            "",
            "ESTADISTICAS ADICIONALES:",
            f"  P(Over 2.5 goles):        {sim['p_over25']*100:.1f}%",
            f"  P(Ambos anotan):          {sim['p_both_score']*100:.1f}%",
            f"  P(Porteria en cero {h[:10]}): {sim['p_clean_sheet_h']*100:.1f}%",
            f"  P(Porteria en cero {a[:10]}): {sim['p_clean_sheet_a']*100:.1f}%",
            f"  P({h[:10]} 2+ goles):     {sim['p_home_2plus']*100:.1f}%",
            f"  P({a[:10]} 2+ goles):     {sim['p_away_2plus']*100:.1f}%",
            f"  Promedio total goles MC:  {sim['avg_total_goals']:.2f}",
            "",
            "DISTRIBUCION GOLES (Monte Carlo 100k sim.):",
        ]
        # Dist home
        dh = sim["dist_goals_home"]
        da = sim["dist_goals_away"]
        lines.append(f"  {'Goles':<6} {'Local%':>7}  {'Visita%':>7}")
        for g in range(8):
            ph_g = float(dh.get(str(g), 0)) * 100
            pa_g = float(da.get(str(g), 0)) * 100
            bar_h = "#" * int(ph_g / 3)
            bar_a = "#" * int(pa_g / 3)
            lines.append(f"  {g:<6} {ph_g:6.1f}%  {bar_h:<12}   {pa_g:6.1f}%  {bar_a}")

        # Si ya se jugo
        if result.get("real_result"):
            r = result["real_result"]
            pred_res = 1 if ph > pa and ph > pd_ else -1 if pa > ph and pa > pd_ else 0
            correct = "CORRECTO" if pred_res == r["resultado"] else "INCORRECTO"
            lines += [
                "",
                f"RESULTADO REAL: {h} {r['goals_home']}-{r['goals_away']} {a}",
                f"  Prediccion resultado: {correct}",
                f"  Error goles local:    {abs(r['goals_home'] - pred['lambda_home']):.2f}",
                f"  Error goles visita:   {abs(r['goals_away'] - pred['lambda_away']):.2f}",
            ]

        lines.append("=" * 60)
        return "\n".join(lines)

    def generate_report_completo(self) -> str:
        """Reporte agregado de todos los partidos."""
        if not self._predictions:
            return "Sin predicciones disponibles"

        lines = [
            "=" * 70,
            "  MUNDIAL 2026 - PREDICCIONES COMPLETAS",
            f"  Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"  Partidos: {len(self._predictions)}  |  MC simulaciones: {N_SIM:,}",
            "=" * 70,
            "",
            f"{'PARTIDO':<35} {'LOCAL%':>7} {'EMPATE%':>8} {'VISITA%':>8} {'xG':>8} {'FAVO':>10}",
            "-" * 80,
        ]

        correct_count = 0
        total_finished = 0

        for r in self._predictions:
            h = r["team_home"][:15]
            a = r["team_away"][:15]
            match_str = f"{h} vs {a}"
            pred = r["prediction"]
            sim  = r["simulation"]

            ph, pd_val, pa = pred["p_home"], pred["p_draw"], pred["p_away"]
            xg_str = f"{pred['lambda_home']:.1f}-{pred['lambda_away']:.1f}"

            if ph >= pa and ph >= pd_val:
                fav = h[:10]
            elif pa >= ph and pa >= pd_val:
                fav = a[:10]
            else:
                fav = "Empate"

            lines.append(
                f"{match_str:<35} {ph*100:6.1f}%  {pd_val*100:6.1f}%  {pa*100:6.1f}%"
                f"  {xg_str:>8}  {fav:>10}"
            )

            # Validacion si resultado real disponible
            if r.get("real_result"):
                total_finished += 1
                real_res = r["real_result"]["resultado"]
                pred_res = 1 if ph > pa and ph > pd_val else -1 if pa > ph and pa > pd_val else 0
                if pred_res == real_res:
                    correct_count += 1

        if total_finished > 0:
            acc = correct_count / total_finished * 100
            lines += [
                "",
                f"VALIDACION ({total_finished} partidos completados):",
                f"  Accuracy resultado: {acc:.1f}%  ({correct_count}/{total_finished})",
            ]

        lines += ["", "=" * 70]
        return "\n".join(lines)

    # ── VISUALIZACIONES ──────────────────────────────────────────────────────

    def generate_plots(self, results: List[Dict]):
        """Generar graficas PNG para cada partido y una comparativa global."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            logger.warning("matplotlib no disponible — omitiendo graficas")
            return

        for r in results:
            h = r["team_home"]
            a = r["team_away"]
            pred = r["prediction"]
            sim  = r["simulation"]
            fname = f"{h.replace(' ','_')}_vs_{a.replace(' ','_')}.png"

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"{h} vs {a}  |  {r['date']}", fontsize=13, fontweight="bold")

            # --- Plot 1: Probabilidades resultado ---
            ax = axes[0]
            labels = [h[:12], "Empate", a[:12]]
            probs  = [pred["p_home"]*100, pred["p_draw"]*100, pred["p_away"]*100]
            colors = ["#2196F3", "#9E9E9E", "#F44336"]
            bars = ax.bar(labels, probs, color=colors, edgecolor="white", linewidth=1.5)
            for bar, p in zip(bars, probs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{p:.1f}%", ha="center", va="bottom", fontweight="bold")
            ax.set_ylim(0, 80)
            ax.set_ylabel("Probabilidad (%)")
            ax.set_title("Probabilidades Resultado")
            ax.yaxis.grid(True, alpha=0.4)
            ax.set_axisbelow(True)

            # --- Plot 2: Distribucion goles (Poisson MC) ---
            ax = axes[1]
            max_g = 7
            goals_range = list(range(max_g + 1))
            dist_h = [float(sim["dist_goals_home"].get(str(g), 0)) * 100
                      for g in goals_range]
            dist_a = [float(sim["dist_goals_away"].get(str(g), 0)) * 100
                      for g in goals_range]
            x = np.arange(max_g + 1)
            w = 0.35
            ax.bar(x - w/2, dist_h, w, label=h[:12], color="#2196F3", alpha=0.85)
            ax.bar(x + w/2, dist_a, w, label=a[:12], color="#F44336", alpha=0.85)
            ax.set_xlabel("Goles")
            ax.set_ylabel("Frecuencia (%)")
            ax.set_title("Distribucion de Goles (MC 100k)")
            ax.set_xticks(x)
            ax.legend(fontsize=9)
            ax.yaxis.grid(True, alpha=0.4)
            ax.set_axisbelow(True)

            # --- Plot 3: Top 8 marcadores ---
            ax = axes[2]
            top8 = sim["top_scores"][:8]
            scores_labels = [s["score"] for s in top8]
            scores_probs  = [s["prob"] * 100 for s in top8]
            ax.barh(scores_labels[::-1], scores_probs[::-1],
                    color="#4CAF50", alpha=0.85, edgecolor="white")
            for i, p in enumerate(scores_probs[::-1]):
                ax.text(p + 0.1, i, f"{p:.2f}%", va="center", fontsize=9)
            ax.set_xlabel("Probabilidad (%)")
            ax.set_title("Marcadores Mas Probables")
            ax.xaxis.grid(True, alpha=0.4)
            ax.set_axisbelow(True)

            plt.tight_layout()
            out = PLOTS_DIR / fname
            plt.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            logger.info("  [OK] %s", out)

        # --- Grafica comparativa global ---
        if len(results) > 1:
            fig, ax = plt.subplots(figsize=(max(10, len(results) * 1.5), 6))
            match_labels = [f"{r['team_home'][:8]}\nvs\n{r['team_away'][:8]}"
                            for r in results]
            p_homes = [r["prediction"]["p_home"] * 100 for r in results]
            p_draws = [r["prediction"]["p_draw"] * 100 for r in results]
            p_aways = [r["prediction"]["p_away"] * 100 for r in results]

            x = np.arange(len(results))
            w = 0.25
            ax.bar(x - w, p_homes, w, label="Local gana", color="#2196F3", alpha=0.85)
            ax.bar(x,     p_draws, w, label="Empate",     color="#9E9E9E", alpha=0.85)
            ax.bar(x + w, p_aways, w, label="Visita gana",color="#F44336", alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels(match_labels, fontsize=8)
            ax.set_ylabel("Probabilidad (%)")
            ax.set_title("Mundial 2026 - Comparativa de Predicciones")
            ax.legend()
            ax.yaxis.grid(True, alpha=0.4)
            ax.set_axisbelow(True)
            plt.tight_layout()
            out = PLOTS_DIR / "comparativa_todas_predicciones.png"
            plt.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            logger.info("  [OK] %s", out)

    # ── GUARDAR SALIDAS ──────────────────────────────────────────────────────

    def save_all(self, results: List[Dict]):
        """Guardar todos los reportes, CSV, JSON y graficas."""
        logger.info("Guardando salidas...")

        # Reportes individuales TXT
        for r in results:
            txt = self.generate_report_partido(r)
            fname = f"predictions_{r['team_home'].replace(' ','_')}_vs_{r['team_away'].replace(' ','_')}.txt"
            path = REPORTS_DIR / fname
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt)

        # Reporte completo
        all_txt = self.generate_report_completo()
        with open(REPORTS_DIR / "all_predictions.txt", "w", encoding="utf-8") as f:
            f.write(all_txt)
        logger.info("[OK] reports/all_predictions.txt")

        # CSV detallado
        csv_rows = []
        for r in results:
            pred = r["prediction"]
            sim  = r["simulation"]
            row = {
                "match_id":     r["match_id"],
                "date":         r["date"],
                "team_home":    r["team_home"],
                "team_away":    r["team_away"],
                "elo_home":     r["elo_home"],
                "elo_away":     r["elo_away"],
                "p_home":       pred["p_home"],
                "p_draw":       pred["p_draw"],
                "p_away":       pred["p_away"],
                "lambda_home":  pred["lambda_home"],
                "lambda_away":  pred["lambda_away"],
                "prob_over25":  pred["prob_over25"],
                "mc_p_home":    sim["p_home_sim"],
                "mc_p_draw":    sim["p_draw_sim"],
                "mc_p_away":    sim["p_away_sim"],
                "mc_p_over25":  sim["p_over25"],
                "mc_p_btts":    sim["p_both_score"],
                "mc_avg_goals": sim["avg_total_goals"],
                "top1_score":   sim["top_scores"][0]["score"] if sim["top_scores"] else "",
                "top1_prob":    sim["top_scores"][0]["prob"] if sim["top_scores"] else 0,
                "status":       r.get("status", ""),
            }
            if r.get("real_result"):
                row["real_goals_home"] = r["real_result"]["goals_home"]
                row["real_goals_away"] = r["real_result"]["goals_away"]
                row["real_resultado"]  = r["real_result"]["resultado"]
            csv_rows.append(row)

        df_csv = pd.DataFrame(csv_rows)
        df_csv.to_csv(DATA_DIR / "predictions_detailed.csv", index=False, encoding="utf-8")
        logger.info("[OK] data/predictions_detailed.csv (%d partidos)", len(df_csv))

        # JSON simulaciones
        with open(DATA_DIR / "simulations_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        logger.info("[OK] data/simulations_results.json")

        # Graficas
        logger.info("Generando graficas...")
        self.generate_plots(results)

        # Imprimir reporte completo en consola
        print("\n" + all_txt)


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Predictor Mundial 2026")
    parser.add_argument("--demo", action="store_true",
                        help="Modo demo: sin API key, usa fixtures de ejemplo")
    parser.add_argument("--api-key", default="",
                        help="Football-Data API key")
    parser.add_argument("--n-sim", type=int, default=100_000,
                        help="Numero de simulaciones Monte Carlo")
    parser.add_argument("--status", default="SCHEDULED",
                        help="Estado partidos API: SCHEDULED|FINISHED|LIVE")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("PREDICCION MUNDIAL 2026 - Fase 4")
    logger.info("Inicio: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("MC simulaciones: %s", f"{args.n_sim:,}")
    logger.info("=" * 70)

    # Cargar API key desde .env si no se paso como argumento
    api_key = args.api_key
    if not api_key:
        try:
            from dotenv import dotenv_values
            env = dotenv_values(".env")
            api_key = env.get("FOOTBALL_DATA_API_KEY", "")
        except ImportError:
            pass

    # Inicializar predictor
    predictor = PredictorMundial(models_dir="models", api_key=api_key)

    # Actualizar n_sim si se paso como argumento
    global N_SIM
    N_SIM = args.n_sim

    # Obtener partidos
    if args.demo or not api_key:
        df_matches = predictor._demo_matches()
    else:
        df_matches = predictor.load_next_matches(status=args.status)

    if df_matches.empty:
        logger.error("Sin partidos para predecir")
        return False

    logger.info("Partidos a predecir: %d", len(df_matches))

    # Predecir + Monte Carlo
    results = predictor.predict_all(df_matches)

    if not results:
        logger.error("Sin resultados de prediccion")
        return False

    # Guardar todo
    predictor.save_all(results)

    logger.info("=" * 70)
    logger.info("[OK] FASE 4 COMPLETADA — %d predicciones", len(results))
    logger.info("     Reportes:  reports/")
    logger.info("     Datos:     data/predictions_detailed.csv")
    logger.info("     Graficas:  plots/")
    logger.info("=" * 70)
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
