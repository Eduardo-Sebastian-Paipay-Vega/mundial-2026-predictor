"""
predictor.py — Logica de prediccion para FastAPI
Reutiliza PredictorMundial de predict_mundial.py (directorio padre)
"""
import sys
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Agregar directorio padre al path para importar modulos de Fase 4
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from predict_mundial import PredictorMundial
    _PREDICTOR_AVAILABLE = True
except ImportError as e:
    _PREDICTOR_AVAILABLE = False
    logging.warning("predict_mundial no disponible: %s", e)

logger = logging.getLogger("api.predictor")

_DATA_DIR   = _ROOT / "data"
_MODELS_DIR = _ROOT / "models"


class Predictor:
    """
    Singleton de prediccion para la API.
    Se inicializa una vez en startup de FastAPI.
    """

    _instance: Optional["Predictor"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self):
        if self._initialized:
            return
        logger.info("Inicializando Predictor...")
        self._pm: Optional[PredictorMundial] = None

        if _PREDICTOR_AVAILABLE:
            try:
                api_key = self._load_api_key()
                self._pm = PredictorMundial(
                    models_dir=str(_MODELS_DIR),
                    api_key=api_key,
                )
                logger.info("PredictorMundial cargado OK")
            except Exception as e:
                logger.error("Error cargando PredictorMundial: %s", e)

        # Fallback: cargar modelos directamente
        if self._pm is None:
            self._load_models_direct()

        self._load_historical()
        self._initialized = True
        logger.info("Predictor listo — %d partidos historicos", len(self.df_hist))

    def _load_api_key(self) -> str:
        try:
            from dotenv import dotenv_values
            env = dotenv_values(str(_ROOT / ".env"))
            return env.get("FOOTBALL_DATA_API_KEY", "")
        except Exception:
            return ""

    def _load_models_direct(self):
        """Carga directa de pickles si PredictorMundial no esta disponible."""
        import pickle, json
        self._models = {}
        for name in ["model_resultado", "model_goals_home",
                     "model_goals_away", "model_over25"]:
            path = _MODELS_DIR / f"{name}.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    self._models[name] = pickle.load(f)
        feat_path = _MODELS_DIR / "selected_features.json"
        if feat_path.exists():
            with open(feat_path, encoding="utf-8") as f:
                self.features_list = json.load(f)["features"]
        else:
            self.features_list = []
        self._pm = None

    def _load_historical(self):
        path = _DATA_DIR / "matches_cleaned.csv"
        if path.exists():
            self.df_hist = pd.read_csv(path)
            self.df_hist["date"] = pd.to_datetime(self.df_hist["date"])
        else:
            self.df_hist = pd.DataFrame()
        self._load_fast_features()

    def _load_fast_features(self):
        """
        Carga features pre-computadas para prediccion rapida.

        Estrategia: para cada equipo, busca su ULTIMO partido (cualquier rol)
        y extrae las features rolling de ese partido. Si el equipo jugó como
        visitante, renombra los features _away -> _home y viceversa para poder
        usarlos en cualquier rol futuro.

        Esto da mejor precision que tomar solo partidos como local o solo como
        visitante, ya que captura el estado mas reciente del equipo.
        """
        import json
        feat_path = _DATA_DIR / "features_engineered.csv"
        sel_path  = _MODELS_DIR / "selected_features.json"

        if not feat_path.exists() or not sel_path.exists():
            self._team_latest_feats: Dict[str, Dict[str, float]] = {}
            self._selected_features  = []
            self._team_elo: Dict[str, float] = {}
            self._elo_rank: Dict[str, int] = {}
            return

        with open(sel_path, encoding="utf-8") as f:
            self._selected_features = json.load(f)["features"]

        df_feat = pd.read_csv(feat_path)
        if not self.df_hist.empty:
            # Evitar colision de columnas elo_home/elo_away que existen en ambos dfs
            df_merge = self.df_hist[["match_id", "team_home", "team_away",
                                      "date", "elo_home", "elo_away"]].rename(
                columns={"elo_home": "_elo_h", "elo_away": "_elo_a"}
            )
            df_full = df_feat.merge(df_merge, on="match_id", how="left").sort_values("date")
        else:
            df_full = df_feat

        # ELO actual por equipo (columnas renombradas para evitar conflicto)
        self._team_elo: Dict[str, float] = {}
        if "_elo_h" in df_full.columns:
            for _, r in df_full.iterrows():
                if pd.notna(r.get("_elo_h")):
                    self._team_elo[str(r["team_home"])] = float(r["_elo_h"])
                if pd.notna(r.get("_elo_a")):
                    self._team_elo[str(r["team_away"])] = float(r["_elo_a"])
        elif not self.df_hist.empty:
            df_s = self.df_hist.sort_values("date")
            for _, r in df_s.iterrows():
                if pd.notna(r.get("elo_home")):
                    self._team_elo[str(r["team_home"])] = float(r["elo_home"])
                if pd.notna(r.get("elo_away")):
                    self._team_elo[str(r["team_away"])] = float(r["elo_away"])

        sorted_teams = sorted(self._team_elo.items(), key=lambda x: x[1], reverse=True)
        self._elo_rank: Dict[str, int] = {t: i+1 for i, (t, _) in enumerate(sorted_teams)}

        # Features de sufijo _home y _away en selected
        _home_feats  = {f for f in self._selected_features if f.endswith("_home")}
        _away_feats  = {f for f in self._selected_features if f.endswith("_away")}
        _shared_feats = {f for f in self._selected_features
                         if not f.endswith("_home") and not f.endswith("_away")}

        self._team_latest_feats: Dict[str, Dict[str, float]] = {}

        if "team_home" not in df_full.columns:
            return

        for _, r in df_full.iterrows():
            th, ta = str(r["team_home"]), str(r["team_away"])
            # local stats son los _home features; visitante son los _away
            home_dict: Dict[str, float] = {}
            away_dict: Dict[str, float] = {}

            for f in self._selected_features:
                val = r.get(f, 0.0)
                val = float(val) if pd.notna(val) else 0.0
                if f in _home_feats:
                    home_dict[f] = val
                elif f in _away_feats:
                    away_dict[f] = val
                else:
                    home_dict[f] = val
                    away_dict[f] = val

            # Almacenar como si fuera local:
            # - Para el equipo home: usar home_dict directamente
            self._team_latest_feats[th] = {"role": "home", **home_dict}
            # - Para el equipo away: renombrar _away -> _home y _home -> _away
            swapped: Dict[str, float] = {}
            for f in self._selected_features:
                if f in _home_feats:
                    # Lo que era del visitante ahora es del local
                    f_away = f[:-5] + "_away"
                    swapped[f] = away_dict.get(f_away, 0.0)
                elif f in _away_feats:
                    f_home = f[:-5] + "_home"
                    swapped[f] = home_dict.get(f_home, 0.0)
                else:
                    swapped[f] = away_dict.get(f, 0.0)
            self._team_latest_feats[ta] = {"role": "away", **swapped}

        logger.info("Fast features cargadas: %d equipos / %d ELO",
                    len(self._team_latest_feats), len(self._team_elo))

    # ── PROPIEDADES ──────────────────────────────────────────────────────────

    @property
    def n_features(self) -> int:
        if self._pm:
            return len(self._pm.features_list)
        return len(getattr(self, "features_list", []))

    @property
    def models_loaded(self) -> bool:
        return self._initialized

    @property
    def n_historical(self) -> int:
        return len(self.df_hist)

    # ── PREDICCION ───────────────────────────────────────────────────────────

    def predict(self, team_home: str, team_away: str, date: str,
                city: str = "", country: str = "USA",
                neutral: bool = True, n_sim: int = 50_000) -> Dict:
        """
        Prediccion con fast path: usa features pre-computadas de features_engineered.csv
        para evitar re-ejecutar FeatureEngineer (~43s). Tiempo: <100ms.
        """
        if (self._pm is not None and
                self._selected_features and
                team_home in self._team_latest_feats and
                team_away in self._team_latest_feats):
            return self._predict_fast(team_home, team_away, neutral, n_sim)

        return self._predict_simple(team_home, team_away, n_sim)

    def _predict_fast(self, team_home: str, team_away: str,
                      neutral: bool, n_sim: int) -> Dict:
        """
        Prediccion rapida con features pre-computadas + CatBoost.

        - Features _home: del local, extraidas del ultimo partido del local.
        - Features _away: del visitante, extraidas del ultimo partido del visitante.
        - Features "other" (match-level): recomputadas desde cero usando ELO y stats
          historicas recientes de cada equipo.
        """
        feats_h = self._team_latest_feats[team_home]
        feats_a = self._team_latest_feats[team_away]

        _home_feats = {f for f in self._selected_features if f.endswith("_home")}
        _away_feats  = {f for f in self._selected_features if f.endswith("_away")}

        elo_h  = self._team_elo.get(team_home, 1500.0)
        elo_a  = self._team_elo.get(team_away, 1500.0)
        rank_h = float(self._elo_rank.get(team_home, 50))
        rank_a = float(self._elo_rank.get(team_away, 50))

        combined: Dict[str, float] = {}

        # Features _home: del equipo local (feats_h almacena stats en perspectiva home)
        for feat in _home_feats:
            combined[feat] = float(feats_h.get(feat, 0.0))

        # Features _away: del equipo visitante (feats_a almacena en perspectiva home,
        # usar clave _home para acceder a sus propias stats)
        for feat in _away_feats:
            f_as_home = feat[:-5] + "_home"
            combined[feat] = float(feats_a.get(f_as_home, 0.0))

        # Features "other" (match-level): recomputar desde stats historicas
        gf_h, ga_h = self._team_avg_goals(team_home)  # usando todos los partidos recientes
        gf_a, ga_a = self._team_avg_goals(team_away)

        stats_h = self.get_team_stats(team_home) or {}
        stats_a = self.get_team_stats(team_away) or {}

        wr_h = stats_h.get("win_rate", 0.45)
        wr_a = stats_a.get("win_rate", 0.45)
        lr_h = stats_h.get("loss_rate", 0.25)
        lr_a = stats_a.get("loss_rate", 0.25)
        bal_h = gf_h - ga_h
        bal_a = gf_a - ga_a

        combined["elo_ratio"]            = elo_h / max(elo_a, 1.0)
        combined["elo_home_rank"]        = rank_h
        combined["elo_away_rank"]        = rank_a
        combined["is_neutral"]           = float(neutral)
        combined["underdog_strength"]    = elo_a - elo_h  # positivo = visitante mas fuerte
        combined["goal_balance_diff"]    = bal_h - bal_a
        combined["home_win_rate_overall"] = wr_h
        combined["away_win_rate_overall"] = wr_a
        combined["home_loss_rate_overall"] = lr_h
        combined["away_loss_rate_overall"] = lr_a
        combined["expected_total_goals"] = gf_h + gf_a
        combined["match_competitiveness"] = 1.0 - abs(elo_h - elo_a) / 1000.0
        combined["is_world_cup"]         = 1.0
        combined["competition_level"]    = 1.0  # WC = max level
        combined["is_knockout"]          = 0.0
        combined["altitude_venue"]       = 0.0
        combined["high_altitude"]        = 0.0
        combined["rest_diff"]            = 0.0
        combined["h2h_draws"]            = self._h2h_draws(team_home, team_away)
        combined["temperature_avg"]      = 22.0
        combined["humidity_avg"]         = 60.0
        combined["precipitation_prob"]   = 0.1
        combined["weather_severity"]     = 0.0

        X = pd.DataFrame([combined])[self._selected_features].fillna(0)
        pred = self._pm.predict_partido(X)

        sim = self._pm.simulate_monte_carlo(
            pred["lambda_home"], pred["lambda_away"], n_sim
        )

        return {
            "team_home":  team_home,
            "team_away":  team_away,
            "prediction": pred,
            "simulation": sim,
            "elo_home":   round(elo_h, 1),
            "elo_away":   round(elo_a, 1),
        }

    def _h2h_draws(self, team_home: str, team_away: str) -> float:
        """Conteo de empates en enfrentamientos directos recientes."""
        if self.df_hist.empty:
            return 0.0
        df = self.df_hist
        h2h = df[
            ((df["team_home"] == team_home) & (df["team_away"] == team_away)) |
            ((df["team_home"] == team_away) & (df["team_away"] == team_home))
        ].tail(10)
        if h2h.empty:
            return 0.0
        return float((h2h["resultado"] == 0).sum())

    def _predict_simple(self, team_home: str, team_away: str,
                        n_sim: int) -> Dict:
        """Prediccion simplificada cuando el pipeline completo no esta disponible."""
        elo_map = self._last_elo_per_team()
        elo_h = elo_map.get(team_home, 1500.0)
        elo_a = elo_map.get(team_away, 1500.0)
        diff  = elo_h - elo_a

        p_home = round(1 / (1 + 10 ** (-diff / 400)), 4)
        p_away = round(1 / (1 + 10 ** (diff / 400)), 4)
        p_draw = round(1 - p_home - p_away, 4)

        # Goles basados en historial
        gf_h, ga_h = self._team_avg_goals(team_home)
        gf_a, ga_a = self._team_avg_goals(team_away)
        lam_h = round((gf_h + ga_a) / 2, 3)
        lam_a = round((gf_a + ga_h) / 2, 3)

        rng = np.random.default_rng(42)
        gh = rng.poisson(max(0.05, lam_h), n_sim).clip(0, 8)
        ga_ = rng.poisson(max(0.05, lam_a), n_sim).clip(0, 8)

        scores_cnt = Counter(zip(gh.tolist(), ga_.tolist()))
        top_scores = [
            {"score": f"{h}-{a}", "probability": round(cnt / n_sim, 4)}
            for (h, a), cnt in scores_cnt.most_common(10)
        ]

        def _dist(arr):
            c = Counter(arr.tolist())
            return {str(k): round(v / n_sim, 4) for k, v in sorted(c.items()) if k <= 7}

        return {
            "team_home": team_home,
            "team_away": team_away,
            "prediction": {
                "p_home": p_home, "p_draw": max(0.05, p_draw), "p_away": p_away,
                "lambda_home": lam_h, "lambda_away": lam_a, "prob_over25": 0.5,
            },
            "simulation": {
                "n_sim": n_sim,
                "p_home_sim": round(float((gh > ga_).mean()), 4),
                "p_draw_sim": round(float((gh == ga_).mean()), 4),
                "p_away_sim": round(float((gh < ga_).mean()), 4),
                "p_over25":   round(float(((gh + ga_) >= 3).mean()), 4),
                "p_both_score": round(float(((gh > 0) & (ga_ > 0)).mean()), 4),
                "p_clean_sheet_h": round(float((ga_ == 0).mean()), 4),
                "p_clean_sheet_a": round(float((gh == 0).mean()), 4),
                "p_home_2plus": round(float((gh >= 2).mean()), 4),
                "p_away_2plus": round(float((ga_ >= 2).mean()), 4),
                "avg_total_goals": round(float((gh + ga_).mean()), 3),
                "std_total_goals": round(float((gh + ga_).std()), 3),
                "top_scores":  top_scores,
                "dist_goals_home": _dist(gh),
                "dist_goals_away": _dist(ga_),
            },
            "elo_home": elo_h,
            "elo_away": elo_a,
        }

    def simulate(self, team_home: str, team_away: str, n_sim: int) -> Dict:
        """Simulacion Monte Carlo pura usando el mismo fast-predict path."""
        t0 = time.perf_counter()
        result = self.predict(team_home, team_away, "2026-06-14", n_sim=n_sim)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        sim = result["simulation"]
        sim["execution_time_ms"] = elapsed_ms
        sim["lambda_home"] = result["prediction"]["lambda_home"]
        sim["lambda_away"] = result["prediction"]["lambda_away"]
        return sim

    # ── ESTADISTICAS DE EQUIPO ───────────────────────────────────────────────

    def get_team_stats(self, team_name: str) -> Optional[Dict]:
        """Stats historicos de un equipo desde matches_cleaned.csv."""
        if self.df_hist.empty:
            return None

        df = self.df_hist
        mask = (df["team_home"] == team_name) | (df["team_away"] == team_name)
        team_matches = df[mask].sort_values("date")

        if team_matches.empty:
            return None

        goals_for, goals_against, wins, draws, losses, clean = [], [], [], [], [], []
        form_chars = []

        for _, r in team_matches.iterrows():
            is_home = r["team_home"] == team_name
            gf = r["goals_home"] if is_home else r["goals_away"]
            ga = r["goals_away"] if is_home else r["goals_home"]
            res = r.get("resultado", 0)
            w = (is_home and res == 1) or (not is_home and res == -1)
            d = res == 0
            goals_for.append(gf)
            goals_against.append(ga)
            wins.append(1 if w else 0)
            draws.append(1 if d else 0)
            losses.append(0 if w or d else 1)
            clean.append(1 if ga == 0 else 0)
            form_chars.append("W" if w else "D" if d else "L")

        n = len(team_matches)
        recent_form = "".join(form_chars[-5:])

        # ELO mas reciente
        elo_vals = []
        for _, r in team_matches.iterrows():
            if r["team_home"] == team_name and pd.notna(r.get("elo_home")):
                elo_vals.append(float(r["elo_home"]))
            elif r["team_away"] == team_name and pd.notna(r.get("elo_away")):
                elo_vals.append(float(r["elo_away"]))
        elo = round(elo_vals[-1], 1) if elo_vals else 1500.0

        gf_arr = np.array([g for g in goals_for if pd.notna(g)], dtype=float)
        ga_arr = np.array([g for g in goals_against if pd.notna(g)], dtype=float)

        over25 = int(((gf_arr + ga_arr) >= 3).sum())
        last_date = str(team_matches.iloc[-1]["date"])[:10] if n > 0 else None

        return {
            "team":              team_name,
            "elo_rating":        elo,
            "avg_goals_for":     round(float(gf_arr.mean()), 2) if len(gf_arr) else 0,
            "avg_goals_against": round(float(ga_arr.mean()), 2) if len(ga_arr) else 0,
            "win_rate":          round(sum(wins) / n, 3),
            "draw_rate":         round(sum(draws) / n, 3),
            "loss_rate":         round(sum(losses) / n, 3),
            "clean_sheet_rate":  round(sum(clean) / n, 3),
            "over25_rate":       round(over25 / n, 3),
            "recent_form":       recent_form,
            "total_matches":     n,
            "last_match_date":   last_date,
        }

    def get_all_teams(self) -> List[str]:
        """Lista de todos los equipos en el historial."""
        if self.df_hist.empty:
            return []
        teams = set(self.df_hist["team_home"].tolist() +
                    self.df_hist["team_away"].tolist())
        return sorted(teams)

    def get_latest_predictions(self, limit: int = 10) -> List[Dict]:
        """Ultimas predicciones del CSV de predicciones."""
        path = _DATA_DIR / "predictions_detailed.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path).sort_values("date", ascending=False).head(limit)
        rows = []
        for _, r in df.iterrows():
            ph, pd_, pa = float(r["p_home"]), float(r["p_draw"]), float(r["p_away"])
            if ph >= pa and ph >= pd_:
                fav = r["team_home"]
            elif pa >= ph and pa >= pd_:
                fav = r["team_away"]
            else:
                fav = "Draw"
            rows.append({
                "match_id":   str(r["match_id"]),
                "team_home":  r["team_home"],
                "team_away":  r["team_away"],
                "date":       str(r["date"])[:10],
                "p_home":     round(ph, 4),
                "p_draw":     round(pd_, 4),
                "p_away":     round(pa, 4),
                "lambda_home": round(float(r.get("lambda_home", 1.2)), 3),
                "lambda_away": round(float(r.get("lambda_away", 1.0)), 3),
                "favorite":   fav,
            })
        return rows

    # ── HELPERS ──────────────────────────────────────────────────────────────

    def _last_elo_per_team(self) -> Dict[str, float]:
        if self.df_hist.empty:
            return {}
        df = self.df_hist.sort_values("date")
        elo_h = df.groupby("team_home")["elo_home"].last()
        elo_a = df.groupby("team_away")["elo_away"].last()
        combined = elo_h.combine_first(elo_a)
        return {str(k): float(v) for k, v in combined.items()}

    def _team_avg_goals(self, team: str) -> Tuple[float, float]:
        if self.df_hist.empty:
            return 1.2, 1.0
        df = self.df_hist
        last = df[(df["team_home"] == team) | (df["team_away"] == team)].tail(10)
        gf, ga = [], []
        for _, r in last.iterrows():
            if r["team_home"] == team:
                gf.append(r["goals_home"]); ga.append(r["goals_away"])
            else:
                gf.append(r["goals_away"]); ga.append(r["goals_home"])
        return (
            round(float(np.mean([g for g in gf if pd.notna(g)])) if gf else 1.2, 3),
            round(float(np.mean([g for g in ga if pd.notna(g)])) if ga else 1.0, 3),
        )
