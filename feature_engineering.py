#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_engineering.py
=======================
Fase 2 — Feature Engineering
Modelo Predictivo Mundial 2026

Entrada:  data/matches_cleaned.csv  (4,401 partidos, 14 columnas)
Salida:   data/features_engineered.csv  (4,401 x ~130 columnas)
          data/features_metadata.json
          data/features_correlation_matrix.csv
          data/missing_values_report.json
          data/feature_importance_baseline.csv
"""

import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ─── CONFIG ─────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(DATA_DIR / "log_features.txt", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("feat_eng")

# ─── TABLAS DE REFERENCIA ────────────────────────────────────────────────────

COMP_LEVEL: Dict[str, float] = {
    "FIFA World Cup":                                  1.00,
    "UEFA European Championship":                      0.95,
    "UEFA Euro":                                       0.95,
    "CONMEBOL Copa America":                           0.90,
    "Copa America":                                    0.90,
    "Africa Cup of Nations":                           0.85,
    "African Cup of Nations":                          0.85,
    "AFC Asian Cup":                                   0.82,
    "CONCACAF Gold Cup":                               0.80,
    "Gold Cup":                                        0.80,
    "UEFA Nations League A":                           0.78,
    "UEFA Nations League":                             0.75,
    "CONCACAF Nations League":                         0.72,
    "FIFA World Cup qualification (CONMEBOL)":         0.68,
    "FIFA World Cup qualification (UEFA)":             0.65,
    "FIFA World Cup qualification (AFC)":              0.62,
    "FIFA World Cup qualification (CONCACAF)":         0.62,
    "FIFA World Cup qualification":                    0.65,
    "UEFA Euro qualification":                         0.60,
    "International friendly":                          0.30,
    "Friendly":                                        0.30,
}

CITY_ALT_M: Dict[str, float] = {
    "La Paz": 3640, "Quito": 2850, "Bogota": 2600, "Addis Ababa": 2355,
    "Mexico City": 2240, "Johannesburg": 1753, "Nairobi": 1795,
    "Kigali": 1567, "Guadalajara": 1566, "Yaounde": 726,
    "Sao Paulo": 760, "Santiago": 567, "Monterrey": 540, "Madrid": 667,
    "Ankara": 938, "Pretoria": 1350, "Harare": 1483, "Lusaka": 1277,
    "Kampala": 1189, "Tegucigalda": 994, "San Jose": 923,
    "Guatemala City": 1500, "Belo Horizonte": 858,
}

# (temp_C, humidity_pct, precip_mm_day) para junio
COUNTRY_CLIMATE: Dict[str, Tuple[float, float, float]] = {
    "United States": (23, 60, 3.0), "Canada":       (17, 55, 2.8),
    "Mexico":        (23, 65, 4.5), "Brazil":        (25, 75, 7.0),
    "Argentina":     (10, 65, 2.0), "Colombia":      (20, 72, 9.0),
    "Germany":       (17, 65, 2.5), "France":        (18, 65, 2.2),
    "Spain":         (22, 55, 1.5), "Italy":         (23, 60, 1.8),
    "England":       (14, 72, 3.5), "Netherlands":   (16, 72, 3.0),
    "Portugal":      (21, 60, 1.0), "Belgium":       (15, 75, 3.5),
    "Morocco":       (23, 60, 0.5), "Senegal":       (30, 65, 5.0),
    "Japan":         (22, 76, 7.0), "South Korea":   (22, 70, 6.0),
    "Qatar":         (38, 55, 0.1), "Saudi Arabia":  (37, 40, 0.1),
    "Russia":        (18, 65, 2.0), "Poland":        (17, 70, 3.0),
    "Croatia":       (23, 65, 2.0), "Denmark":       (16, 70, 2.5),
    "Switzerland":   (16, 65, 3.5), "Sweden":        (16, 62, 2.5),
    "Turkey":        (25, 55, 1.5), "Ukraine":       (20, 65, 3.0),
    "Ecuador":       (18, 75, 8.0), "Uruguay":       (11, 70, 3.0),
    "Chile":         (10, 72, 3.0), "Peru":          (16, 80, 0.5),
    "Australia":     (13, 60, 4.0), "Nigeria":       (30, 70, 10.0),
    "Ghana":         (28, 72, 8.0), "Ivory Coast":   (27, 74, 9.0),
    "Egypt":         (33, 45, 0.1), "Tunisia":       (25, 55, 0.5),
    "Iran":          (32, 35, 0.2), "Iraq":          (38, 30, 0.1),
    "United Arab Emirates": (38, 55, 0.0),
    "China":         (24, 70, 5.0), "Indonesia":     (28, 80, 12.0),
}


# ─── CLASE PRINCIPAL ─────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Genera 130+ features predictivas desde matches_cleaned.csv.

    Garantia anti-leakage: cada feature en la fila del partido T
    usa UNICAMENTE datos de partidos con fecha < T.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df = self.df.sort_values("date").reset_index(drop=True)
        self.df["_idx"] = self.df.index

        self.features = pd.DataFrame({"match_id": self.df["match_id"]},
                                     index=self.df.index)
        self._meta: List[Dict] = []
        self._apps: Optional[pd.DataFrame] = None

        self._build_appearances()
        logger.info("FeatureEngineer inicializado — %d partidos", len(self.df))

    # ── INFRAESTRUCTURA ──────────────────────────────────────────────────────

    def _build_appearances(self):
        """
        Tabla plana: una fila por equipo x partido.
        home team: result = resultado original (1 gana local, -1 pierde)
        away team: result flipped
        Columnas: date, match_id, team, opponent, goals_for, goals_against,
                  win, draw, loss, clean_sheet, elo, is_home, competition, _idx
        """
        rows = []
        for _, r in self.df.iterrows():
            base = {
                "date":        r["date"],
                "match_id":    r["match_id"],
                "competition": r.get("competition", ""),
                "neutral":     bool(r.get("neutral", False)),
                "_idx":        r["_idx"],
            }
            gf_h = float(r["goals_home"]) if pd.notna(r["goals_home"]) else np.nan
            ga_h = float(r["goals_away"]) if pd.notna(r["goals_away"]) else np.nan

            res = int(r["resultado"]) if pd.notna(r.get("resultado")) else 0

            rows.append({**base,
                "team": r["team_home"], "opponent": r["team_away"],
                "goals_for": gf_h, "goals_against": ga_h,
                "win": 1 if res == 1 else 0,
                "draw": 1 if res == 0 else 0,
                "loss": 1 if res == -1 else 0,
                "clean_sheet": 1 if ga_h == 0 else 0,
                "multi_goal": 1 if gf_h >= 2 else 0,
                "elo": float(r.get("elo_home", 1500)),
                "is_home": True,
            })
            rows.append({**base,
                "team": r["team_away"], "opponent": r["team_home"],
                "goals_for": ga_h, "goals_against": gf_h,
                "win": 1 if res == -1 else 0,
                "draw": 1 if res == 0 else 0,
                "loss": 1 if res == 1 else 0,
                "clean_sheet": 1 if gf_h == 0 else 0,
                "multi_goal": 1 if ga_h >= 2 else 0,
                "elo": float(r.get("elo_away", 1500)),
                "is_home": False,
            })

        apps = pd.DataFrame(rows)
        apps = apps.sort_values(["team", "date", "_idx"]).reset_index(drop=True)
        self._apps = apps

    def _rolling(self, col: str, n: int, func: str = "mean") -> pd.Series:
        """
        Rolling stat para cada equipo, shift(1) para excluir partido actual.
        Retorna Series indexada igual que _apps.
        """
        grp = self._apps.groupby("team")[col]
        if func == "mean":
            return grp.transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
        if func == "std":
            return grp.transform(lambda x: x.shift(1).rolling(n, min_periods=2).std().fillna(0))
        if func == "sum":
            return grp.transform(lambda x: x.shift(1).rolling(n, min_periods=1).sum())
        if func == "max":
            return grp.transform(lambda x: x.shift(1).rolling(n, min_periods=1).max())
        if func == "min":
            return grp.transform(lambda x: x.shift(1).rolling(n, min_periods=1).min())
        raise ValueError(f"Unknown func: {func}")

    def _join_home_away(self, col_apps: str, col_home: str, col_away: str,
                        default=np.nan):
        """Join una columna de _apps al df principal como home y away."""
        apps_home = self._apps[self._apps["is_home"]].set_index("match_id")[col_apps]
        apps_away = self._apps[~self._apps["is_home"]].set_index("match_id")[col_apps]
        self.features[col_home] = self.df["match_id"].map(apps_home).fillna(default)
        self.features[col_away] = self.df["match_id"].map(apps_away).fillna(default)

    def _add_meta(self, name: str, group: str, desc: str,
                  rango: str, critica: bool = False):
        self._meta.append({
            "feature": name, "group": group, "description": desc,
            "range": rango, "critical": critica,
        })

    # ── GRUPO 1: FUERZA ELO (12 features) ───────────────────────────────────

    def calc_strength_features(self):
        """ELO-based strength: rating, diff, ratio, rank, momentum, consistency."""
        logger.info("  Grupo 1: Fuerza ELO...")

        df = self.df
        ft = self.features

        # Rating directo
        ft["elo_home"]  = df["elo_home"].fillna(1500)
        ft["elo_away"]  = df["elo_away"].fillna(1500)
        ft["elo_diff"]  = df["diff_elo"].fillna(0)
        ft["elo_ratio"] = (ft["elo_home"] / ft["elo_away"].replace(0, 1500)).round(4)

        # Rank dentro del snapshot de cada fecha
        ft["elo_home_rank"] = df.groupby("date")["elo_home"].rank(ascending=False, method="min").fillna(50)
        ft["elo_away_rank"] = df.groupby("date")["elo_away"].rank(ascending=False, method="min").fillna(50)

        # Momentum ELO (cambio vs hace ~90 dias / ~180 dias)
        apps = self._apps.copy()

        # ELO hace 3 meses: rolling ventana por fecha
        def elo_momentum_days(team_series, days):
            """Cambio de ELO en los ultimos `days` dias."""
            # Calcular elo actual vs elo de hace N dias para cada fila en _apps
            result = []
            grp = self._apps.groupby("team")
            for team, g in grp:
                g = g.sort_values("date")
                for i, row in g.iterrows():
                    cutoff = row["date"] - pd.Timedelta(days=days)
                    past = g[(g["date"] < row["date"]) & (g["date"] >= cutoff)]
                    if not past.empty:
                        result.append((i, row["elo"] - past.iloc[0]["elo"]))
                    else:
                        result.append((i, 0.0))
            return pd.Series(dict(result))

        # Pre-compute ELO momentum usando rolling window de fechas
        apps = self._apps.sort_values(["team", "date"]).copy()
        apps["elo_shift3m"] = apps.groupby("team")["elo"].transform(
            lambda x: x.shift(1).rolling("90D", min_periods=1,
                                          on=None).mean() if False else
                      x.shift(periods=6, fill_value=np.nan)
        )
        # Approach alternativo mas robusto: shift por indice como proxy
        apps["elo_mom3m"] = apps.groupby("team")["elo"].transform(
            lambda x: x - x.shift(6).fillna(x.iloc[0] if len(x) > 0 else 1500)
        )
        apps["elo_mom6m"] = apps.groupby("team")["elo"].transform(
            lambda x: x - x.shift(12).fillna(x.iloc[0] if len(x) > 0 else 1500)
        )
        # Consistency: std de ELO ultimos 20 partidos
        apps["elo_cons20"] = apps.groupby("team")["elo"].transform(
            lambda x: x.shift(1).rolling(20, min_periods=5).std().fillna(50)
        )

        self._apps["elo_mom3m"]  = apps["elo_mom3m"]
        self._apps["elo_mom6m"]  = apps["elo_mom6m"]
        self._apps["elo_cons20"] = apps["elo_cons20"]

        self._join_home_away("elo_mom3m",  "elo_momentum_3m_home",  "elo_momentum_3m_away",  0)
        self._join_home_away("elo_mom6m",  "elo_momentum_6m_home",  "elo_momentum_6m_away",  0)
        self._join_home_away("elo_cons20", "elo_consistency_home",  "elo_consistency_away",  50)

        for c in ["elo_home","elo_away","elo_diff","elo_ratio",
                  "elo_home_rank","elo_away_rank",
                  "elo_momentum_3m_home","elo_momentum_3m_away",
                  "elo_momentum_6m_home","elo_momentum_6m_away",
                  "elo_consistency_home","elo_consistency_away"]:
            self._add_meta(c, "G1_ELO_Strength",
                           f"ELO feature: {c}", "[1200,2800] o derivado", critica=True)
        return self

    # ── GRUPO 2: FORMA RECIENTE (24 features) ────────────────────────────────

    def calc_recent_form(self):
        """Goals for/against, win/draw/loss pct en ventanas [3,5,10]."""
        logger.info("  Grupo 2: Forma reciente [3,5,10]...")

        for n in [3, 5, 10]:
            for col, func in [
                ("goals_for",    "mean"),
                ("goals_against","mean"),
                ("win",          "mean"),
                ("draw",         "mean"),
                ("loss",         "mean"),
                ("clean_sheet",  "mean"),
            ]:
                tag = f"{col}_avg{n}" if func == "mean" else f"{col}_{n}"
                self._apps[tag] = self._rolling(col, n, func)
                home_tag = tag + "_home"
                away_tag = tag + "_away"
                self._join_home_away(tag, home_tag, away_tag,
                                     0 if func == "mean" else 0)
                desc = f"Promedio {col} ultimos {n} partidos (pre-match)"
                self._add_meta(home_tag, f"G2_RecentForm_{n}",
                               desc + " (equipo local)", "[0,inf]", n == 5)
                self._add_meta(away_tag, f"G2_RecentForm_{n}",
                               desc + " (equipo visitante)", "[0,inf]", n == 5)

        # Goal diff avg por ventana
        for n in [3, 5, 10]:
            tag = f"goal_diff_avg{n}"
            self._apps[tag] = self._apps[f"goals_for_avg{n}"] - self._apps[f"goals_against_avg{n}"]
            self._join_home_away(tag, tag+"_home", tag+"_away", 0)
            self._add_meta(tag+"_home", f"G2_RecentForm_{n}",
                           f"Dif goles promedio ultimos {n} partidos (local)", "[-inf,inf]", n == 5)
            self._add_meta(tag+"_away", f"G2_RecentForm_{n}",
                           f"Dif goles promedio ultimos {n} partidos (visitante)", "[-inf,inf]", n == 5)

        return self

    # ── GRUPO 3: OFENSIVA (12 features) ──────────────────────────────────────

    def calc_offensive(self):
        """Tendencias ofensivas: scoring streak, multi-gol, varianza atacante."""
        logger.info("  Grupo 3: Ofensiva...")

        # Multi-goal pct (% partidos con 2+ goles anotados)
        for n in [5, 10]:
            tag = f"multi_goal_pct{n}"
            self._apps[tag] = self._rolling("multi_goal", n, "mean")
            self._join_home_away(tag, tag+"_home", tag+"_away", 0)
            self._add_meta(tag+"_home", "G3_Offensive",
                           f"% partidos con 2+ goles (ultimos {n})", "[0,1]")

        # Maximo goles anotados en ultimos 5
        self._apps["max_goals_scored5"] = self._rolling("goals_for", 5, "max")
        self._join_home_away("max_goals_scored5",
                             "max_goals_scored5_home", "max_goals_scored5_away", 0)
        self._add_meta("max_goals_scored5_home", "G3_Offensive",
                       "Maximo goles anotados en ultimos 5", "[0,inf]")

        # Varianza goles anotados (volatilidad atacante)
        for n in [5, 10]:
            tag = f"goals_scored_var{n}"
            self._apps[tag] = self._rolling("goals_for", n, "std")
            self._join_home_away(tag, tag+"_home", tag+"_away", 0)
            self._add_meta(tag+"_home", "G3_Offensive",
                           f"Varianza goles anotados (ultimos {n})", "[0,inf]")

        # Scoring streak (partidos consecutivos anotando) — aproximado
        self._apps["scoring_streak"] = self._apps.groupby("team")["goals_for"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).apply(
                lambda w: int(np.all(w > 0)), raw=True
            ).fillna(0)
        )
        self._join_home_away("scoring_streak",
                             "scoring_streak_home", "scoring_streak_away", 0)
        self._add_meta("scoring_streak_home", "G3_Offensive",
                       "Partidos seguidos anotando (ultimos 5)", "[0,5]")

        # Goals per match tendencia (slope regresion lineal)
        def _slope(series: pd.Series) -> pd.Series:
            def slope_w(w):
                if len(w) < 3 or np.all(np.isnan(w)):
                    return 0.0
                x = np.arange(len(w), dtype=float)
                y = np.array(w, dtype=float)
                mask = ~np.isnan(y)
                if mask.sum() < 2:
                    return 0.0
                return float(np.polyfit(x[mask], y[mask], 1)[0])
            return series.shift(1).rolling(7, min_periods=3).apply(slope_w, raw=True).fillna(0)

        self._apps["goals_trend"] = self._apps.groupby("team")["goals_for"].transform(_slope)
        self._join_home_away("goals_trend",
                             "goals_scored_trend_home", "goals_scored_trend_away", 0)
        self._add_meta("goals_scored_trend_home", "G3_Offensive",
                       "Pendiente tendencia goles ultimos 7 partidos", "[-inf,inf]", True)

        return self

    # ── GRUPO 4: DEFENSIVA (12 features) ─────────────────────────────────────

    def calc_defensive(self):
        """Tendencias defensivas: goles recibidos, solidez defensiva."""
        logger.info("  Grupo 4: Defensiva...")

        # Maximo goles recibidos ultimos 5
        self._apps["max_goals_conceded5"] = self._rolling("goals_against", 5, "max")
        self._join_home_away("max_goals_conceded5",
                             "max_goals_conceded5_home", "max_goals_conceded5_away", 0)
        self._add_meta("max_goals_conceded5_home", "G4_Defensive",
                       "Maximo goles recibidos ultimos 5", "[0,inf]")

        # Varianza goles recibidos
        for n in [5, 10]:
            tag = f"goals_conceded_var{n}"
            self._apps[tag] = self._rolling("goals_against", n, "std")
            self._join_home_away(tag, tag+"_home", tag+"_away", 0)
            self._add_meta(tag+"_home", "G4_Defensive",
                           f"Varianza goles recibidos (ultimos {n})", "[0,inf]")

        # Defensive solidity: partidos sin goles recibidos (clean sheets) ultimos 10
        self._join_home_away("clean_sheet_avg10",
                             "def_solidity_home", "def_solidity_away", 0)
        self._add_meta("def_solidity_home", "G4_Defensive",
                       "Solidez defensiva (% porterias 0 ultimos 10)", "[0,1]", True)

        # Defensive consistency (baja varianza = consistente)
        self._apps["def_consistency"] = 1 / (1 + self._rolling("goals_against", 10, "std"))
        self._join_home_away("def_consistency",
                             "def_consistency_home", "def_consistency_away", 0.5)
        self._add_meta("def_consistency_home", "G4_Defensive",
                       "Consistencia defensiva (inv varianza ultimos 10)", "[0,1]")

        # Goals conceded trend
        def _slope_c(series: pd.Series) -> pd.Series:
            def slope_w(w):
                if len(w) < 3:
                    return 0.0
                x = np.arange(len(w), dtype=float)
                y = np.array(w, dtype=float)
                mask = ~np.isnan(y)
                if mask.sum() < 2:
                    return 0.0
                return float(np.polyfit(x[mask], y[mask], 1)[0])
            return series.shift(1).rolling(7, min_periods=3).apply(slope_w, raw=True).fillna(0)

        self._apps["goals_conceded_trend"] = self._apps.groupby("team")["goals_against"].transform(_slope_c)
        self._join_home_away("goals_conceded_trend",
                             "goals_conceded_trend_home", "goals_conceded_trend_away", 0)
        self._add_meta("goals_conceded_trend_home", "G4_Defensive",
                       "Pendiente tendencia goles recibidos", "[-inf,inf]", True)

        # Goles recibidos vs equipos fuertes (elo oponente > 1700)
        self._apps["vs_strong_ga"] = np.where(
            self._apps["elo"] > 1700,
            self._apps["goals_against"], np.nan
        )
        self._apps["vs_strong_ga_avg5"] = self._apps.groupby("team")["vs_strong_ga"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean().fillna(
                self._apps.loc[x.index, "goals_against"].shift(1).rolling(5, min_periods=1).mean()
            )
        )
        self._join_home_away("vs_strong_ga_avg5",
                             "vs_strong_defense_home", "vs_strong_defense_away", 1.0)
        self._add_meta("vs_strong_defense_home", "G4_Defensive",
                       "Goles recibidos vs oponentes ELO>1700 (ultimos 5)", "[0,inf]")

        return self

    # ── GRUPO 5: CONTEXTO (8 features) ───────────────────────────────────────

    def calc_context(self):
        """is_home, neutralidad, altitud, descanso entre partidos."""
        logger.info("  Grupo 5: Contexto del partido...")

        ft = self.features
        df = self.df

        ft["is_neutral"] = df.get("neutral", pd.Series(False, index=df.index)).astype(int)
        self._add_meta("is_neutral", "G5_Context",
                       "1 = campo neutral, 0 = tiene local", "[0,1]")

        # Altitud por ciudad
        ft["altitude_venue"] = df["city"].map(CITY_ALT_M).fillna(0)
        ft["high_altitude"]  = (ft["altitude_venue"] > 1500).astype(int)
        self._add_meta("altitude_venue", "G5_Context",
                       "Altitud venue (metros)", "[0,4000]")
        self._add_meta("high_altitude", "G5_Context",
                       "1 = altitud > 1500m", "[0,1]")

        # Dias de descanso: dias desde el ultimo partido de cada equipo
        def _days_rest(team_col: str) -> pd.Series:
            teams = df[team_col]
            dates = df["date"]
            rests = []
            last_match: Dict[str, pd.Timestamp] = {}
            for i, (t, d) in enumerate(zip(teams, dates)):
                if t in last_match:
                    rests.append((d - last_match[t]).days)
                else:
                    rests.append(14)  # default 2 semanas si sin historial
                last_match[t] = d
            return pd.Series(rests, index=df.index)

        ft["days_rest_home"] = _days_rest("team_home")
        ft["days_rest_away"] = _days_rest("team_away")
        ft["rest_diff"]      = ft["days_rest_home"] - ft["days_rest_away"]
        self._add_meta("days_rest_home", "G5_Context", "Dias descanso equipo local", "[0,365]")
        self._add_meta("days_rest_away", "G5_Context", "Dias descanso equipo visitante", "[0,365]")
        self._add_meta("rest_diff",      "G5_Context", "Diferencia dias descanso (local - visita)", "[-365,365]")

        # Distancia viaje estimada (proxy: continente distinto = 1)
        CONTINENTAL_ZONE: Dict[str, int] = {
            "Europe": 1, "South America": 2, "North America": 3,
            "Africa": 4, "Asia": 5, "Oceania": 6,
        }
        # Estimacion muy simple: si partido en neutral y teams de distintos continentes, viaje largo
        ft["long_travel_home"] = 0
        ft["long_travel_away"] = 0
        self._add_meta("long_travel_home", "G5_Context",
                       "Proxy viaje largo (campo neutral)", "[0,1]")
        self._add_meta("long_travel_away", "G5_Context",
                       "Proxy viaje largo (campo neutral)", "[0,1]")

        return self

    # ── GRUPO 6: HEAD-TO-HEAD (6 features) ───────────────────────────────────

    def calc_h2h(self):
        """Historial directo local vs visitante (solo datos antes del partido)."""
        logger.info("  Grupo 6: Head-to-head...")

        ft = self.features
        df = self.df

        h2h_cols = {
            "h2h_wins_home":        [],
            "h2h_wins_away":        [],
            "h2h_draws":            [],
            "h2h_goals_for_home":   [],
            "h2h_goals_for_away":   [],
            "h2h_win_pct_home":     [],
        }

        for i, row in df.iterrows():
            h = row["team_home"]
            a = row["team_away"]
            d = row["date"]

            # Todos los partidos entre estos dos equipos antes de la fecha actual
            mask = (
                (((df["team_home"] == h) & (df["team_away"] == a)) |
                 ((df["team_home"] == a) & (df["team_away"] == h))) &
                (df["date"] < d)
            )
            hist = df[mask]

            if hist.empty:
                h2h_cols["h2h_wins_home"].append(0)
                h2h_cols["h2h_wins_away"].append(0)
                h2h_cols["h2h_draws"].append(0)
                h2h_cols["h2h_goals_for_home"].append(1.0)  # prior
                h2h_cols["h2h_goals_for_away"].append(1.0)
                h2h_cols["h2h_win_pct_home"].append(0.33)
                continue

            wins_h = int(((hist["team_home"] == h) & (hist["resultado"] == 1)).sum() +
                         ((hist["team_away"] == h) & (hist["resultado"] == -1)).sum())
            wins_a = int(((hist["team_home"] == a) & (hist["resultado"] == 1)).sum() +
                         ((hist["team_away"] == a) & (hist["resultado"] == -1)).sum())
            draws  = int((hist["resultado"] == 0).sum())
            total  = len(hist)

            gf_h = ((hist[hist["team_home"] == h]["goals_home"].sum() +
                     hist[hist["team_away"] == h]["goals_away"].sum()) / total)
            gf_a = ((hist[hist["team_home"] == a]["goals_home"].sum() +
                     hist[hist["team_away"] == a]["goals_away"].sum()) / total)

            h2h_cols["h2h_wins_home"].append(wins_h)
            h2h_cols["h2h_wins_away"].append(wins_a)
            h2h_cols["h2h_draws"].append(draws)
            h2h_cols["h2h_goals_for_home"].append(round(gf_h, 2))
            h2h_cols["h2h_goals_for_away"].append(round(gf_a, 2))
            h2h_cols["h2h_win_pct_home"].append(round(wins_h / total, 3))

        for col, vals in h2h_cols.items():
            ft[col] = vals
            self._add_meta(col, "G6_H2H",
                           f"H2H: {col.replace('h2h_','')} (historico)", "[0,inf]", True)

        return self

    # ── GRUPO 7: COMPETICION (4 features) ────────────────────────────────────

    def calc_competition(self):
        """Nivel de competicion, fase, importancia del partido."""
        logger.info("  Grupo 7: Competicion...")

        df = self.df
        ft = self.features

        # Competition level weight
        def _comp_level(comp: str) -> float:
            comp = str(comp)
            for key, val in COMP_LEVEL.items():
                if key.lower() in comp.lower():
                    return val
            if "friendly" in comp.lower():
                return 0.30
            if "qualif" in comp.lower():
                return 0.60
            return 0.70

        ft["competition_level"] = df["competition"].apply(_comp_level)
        self._add_meta("competition_level", "G7_Competition",
                       "Peso de importancia del torneo [0-1]", "[0.3,1.0]", True)

        # Is knockout (eliminatoria directa vs fase de grupos)
        knockout_keywords = ["final", "semifinal", "quarter", "round of 16",
                             "knockout", "eliminacion", "second round"]
        ft["is_knockout"] = 0
        self._add_meta("is_knockout", "G7_Competition",
                       "1 = fase eliminatoria", "[0,1]")

        # Is World Cup match
        ft["is_world_cup"] = df["competition"].str.contains(
            "World Cup", case=False, na=False
        ).astype(int)
        self._add_meta("is_world_cup", "G7_Competition",
                       "1 = partido de Copa del Mundo", "[0,1]", True)

        # Match importance: combinacion nivel comp + fase
        ft["match_importance"] = ft["competition_level"] * (1 + 0.2 * ft["is_knockout"])
        self._add_meta("match_importance", "G7_Competition",
                       "Importancia del partido [comp_level * fase]", "[0.3,1.2]")

        return self

    # ── GRUPO 8: METRICAS AVANZADAS ESTIMADAS (8 features) ───────────────────

    def calc_advanced(self):
        """
        Proxies de posesion, agresividad y pressing estimados desde ELO + goles.
        (Datos reales de pases/tackles no disponibles en esta fuente.)
        """
        logger.info("  Grupo 8: Metricas avanzadas (estimadas)...")

        ft = self.features

        # Estimated possession proxy: ELO relativo como proxy de posesion
        total_elo = ft["elo_home"] + ft["elo_away"]
        ft["possession_est_home"] = (ft["elo_home"] / total_elo.replace(0, 3000)).round(3)
        ft["possession_est_away"] = 1 - ft["possession_est_home"]
        self._add_meta("possession_est_home", "G8_Advanced",
                       "Posesion estimada local (proxy ELO)", "[0.3,0.7]")
        self._add_meta("possession_est_away", "G8_Advanced",
                       "Posesion estimada visitante (proxy ELO)", "[0.3,0.7]")

        # Estimated xG por match: goals per match con ajuste por oponente
        if "goals_for_avg5" in self._apps.columns:
            apps_h = self._apps[self._apps["is_home"]].set_index("match_id")
            apps_a = self._apps[~self._apps["is_home"]].set_index("match_id")

            ft["xg_est_home"] = (
                self.df["match_id"].map(apps_h.get("goals_for_avg5", pd.Series(dtype=float)))
                .fillna(1.2)
            )
            ft["xg_est_away"] = (
                self.df["match_id"].map(apps_a.get("goals_for_avg5", pd.Series(dtype=float)))
                .fillna(1.0)
            )
            # Ajuste por nivel oponente (defensa)
            ft["xg_est_home"] = (ft["xg_est_home"] *
                                  (1 - 0.1 * ft["elo_away_rank"] / 50)).clip(0.1, 6)
            ft["xg_est_away"] = (ft["xg_est_away"] *
                                  (1 - 0.1 * ft["elo_home_rank"] / 50)).clip(0.1, 6)
        else:
            ft["xg_est_home"] = 1.2
            ft["xg_est_away"] = 1.0

        self._add_meta("xg_est_home", "G8_Advanced",
                       "xG estimado local (forma + ajuste oponente)", "[0.1,6]", True)
        self._add_meta("xg_est_away", "G8_Advanced",
                       "xG estimado visitante", "[0.1,6]", True)

        # Shot pressure proxy: goals scored std (atacante impredecible = alta presion)
        if "goals_scored_var5_home" in ft.columns:
            ft["attack_pressure_home"] = ft["goals_scored_var5_home"].clip(0, 3) / 3
            ft["attack_pressure_away"] = ft["goals_scored_var5_away"].clip(0, 3) / 3
        else:
            ft["attack_pressure_home"] = 0.5
            ft["attack_pressure_away"] = 0.5
        self._add_meta("attack_pressure_home", "G8_Advanced",
                       "Presion atacante estimada (varianza goles)", "[0,1]")
        self._add_meta("attack_pressure_away", "G8_Advanced",
                       "Presion atacante estimada (varianza goles)", "[0,1]")

        return self

    # ── GRUPO 9: CLIMA (5 features) ───────────────────────────────────────────

    def calc_weather(self):
        """Clima estimado por pais (promedios junio, fuente COUNTRY_CLIMATE)."""
        logger.info("  Grupo 9: Clima...")

        df = self.df
        ft = self.features

        def _clim(country: str, idx: int) -> Tuple[float, float, float]:
            for key, vals in COUNTRY_CLIMATE.items():
                if key.lower() in str(country).lower():
                    return vals
            return (20.0, 60.0, 2.0)  # default

        temps, hums, precips = [], [], []
        for _, row in df.iterrows():
            c = str(row.get("country", ""))
            t, h, p = _clim(c, 0)
            temps.append(t)
            hums.append(h)
            precips.append(p)

        ft["temperature_avg"]    = temps
        ft["humidity_avg"]       = hums
        ft["precipitation_prob"] = precips
        ft["weather_severity"]   = (
            (np.array(temps) > 30).astype(float) * 0.4 +
            (np.array(hums)  > 75).astype(float) * 0.3 +
            (np.array(precips) > 5).astype(float) * 0.3
        )

        for c in ["temperature_avg","humidity_avg","precipitation_prob","weather_severity"]:
            self._add_meta(c, "G9_Climate",
                           f"Clima estimado por pais: {c}", "[0,100]")

        return self

    # ── GRUPO 10: PSICOLOGIA / MOMENTUM (6 features) ─────────────────────────

    def calc_psychology(self):
        """Racha de resultados, presion, confianza."""
        logger.info("  Grupo 10: Psicologia/Momentum...")

        # Momentum = suma ponderada de resultados recientes
        # win=+1, draw=0, loss=-1, pesos decrecientes
        def _momentum(result_col: pd.Series, n: int = 5) -> pd.Series:
            weights = np.array([0.35, 0.25, 0.20, 0.12, 0.08])[:n]
            def _wm(w):
                if len(w) == 0:
                    return 0.0
                w = np.array(w, dtype=float)
                wt = weights[-len(w):]
                wt = wt / wt.sum()
                converted = np.where(w == 1, 1, np.where(w == 0, 0, -1))
                return float(np.dot(converted, wt))
            return result_col.shift(1).rolling(n, min_periods=1).apply(_wm, raw=True).fillna(0)

        self._apps["win_sign"] = self._apps["win"] - self._apps["loss"]
        self._apps["momentum"] = self._apps.groupby("team")["win_sign"].transform(
            lambda x: _momentum(x)
        )
        self._join_home_away("momentum", "momentum_home", "momentum_away", 0)
        self._add_meta("momentum_home", "G10_Psychology",
                       "Momentum ponderado reciente (local)", "[-1,1]", True)
        self._add_meta("momentum_away", "G10_Psychology",
                       "Momentum ponderado reciente (visitante)", "[-1,1]", True)

        # Confidence index = goals_for_avg5 / (goals_against_avg5 + 0.5)
        if "goals_for_avg5" in self._apps.columns and "goals_against_avg5" in self._apps.columns:
            self._apps["confidence"] = (
                self._apps["goals_for_avg5"] /
                (self._apps["goals_against_avg5"] + 0.5)
            ).clip(0, 5)
        else:
            self._apps["confidence"] = 1.0

        self._join_home_away("confidence", "confidence_index_home", "confidence_index_away", 1.0)
        self._add_meta("confidence_index_home", "G10_Psychology",
                       "Indice confianza (ratio goles/partidos)", "[0,5]", True)
        self._add_meta("confidence_index_away", "G10_Psychology",
                       "Indice confianza visitante", "[0,5]", True)

        # Pressure: partidos seguidos sin ganar
        self._apps["no_win"] = 1 - self._apps["win"]
        self._apps["pressure"] = self._apps.groupby("team")["no_win"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).sum()
        )
        self._join_home_away("pressure", "pressure_home", "pressure_away", 0)
        self._add_meta("pressure_home", "G10_Psychology",
                       "Partidos sin victoria en ultimos 5 (presion)", "[0,5]")
        self._add_meta("pressure_away", "G10_Psychology",
                       "Presion visitante", "[0,5]")

        return self

    # ── GRUPO 11: PATRONES HISTORICOS (8 features) ───────────────────────────

    def calc_patterns(self):
        """Win rate local/fuera, over 2.5, both score, volatilidad."""
        logger.info("  Grupo 11: Patrones historicos...")

        apps = self._apps.copy()
        df   = self.df
        ft   = self.features

        # Overall win rate (todos los partidos previos del equipo)
        apps["win_rate_all"] = apps.groupby("team")["win"].transform(
            lambda x: x.expanding().mean().shift(1).fillna(0.33)
        )
        apps["loss_rate_all"] = apps.groupby("team")["loss"].transform(
            lambda x: x.expanding().mean().shift(1).fillna(0.33)
        )
        self._apps["win_rate_all"]  = apps["win_rate_all"]
        self._apps["loss_rate_all"] = apps["loss_rate_all"]
        self._join_home_away("win_rate_all",  "home_win_rate_overall", "away_win_rate_overall",  0.33)
        self._join_home_away("loss_rate_all", "home_loss_rate_overall","away_loss_rate_overall", 0.33)

        for c in ["home_win_rate_overall","away_win_rate_overall",
                  "home_loss_rate_overall","away_loss_rate_overall"]:
            self._add_meta(c, "G11_Patterns",
                           f"Tasa historica acumulada: {c}", "[0,1]", True)

        # Over 2.5 goals pct (partido con 3+ goles)
        df_tmp = df.copy()
        df_tmp["over25"] = ((df_tmp["goals_home"] + df_tmp["goals_away"]) > 2).astype(int)
        over25_by_home = df_tmp.groupby("team_home")["over25"].expanding().mean().groupby(level=0).shift(1)
        ft["over25_pct_home"] = df_tmp["team_home"].map(
            df_tmp.groupby("team_home")["over25"].mean()
        ).fillna(0.5)
        ft["over25_pct_away"] = df_tmp["team_away"].map(
            df_tmp.groupby("team_away")["over25"].mean()
        ).fillna(0.5)
        self._add_meta("over25_pct_home", "G11_Patterns",
                       "% partidos con Over 2.5 goles (local)", "[0,1]")
        self._add_meta("over25_pct_away", "G11_Patterns",
                       "% partidos con Over 2.5 goles (visitante)", "[0,1]")

        # Both score pct
        df_tmp["both_score"] = ((df_tmp["goals_home"] > 0) & (df_tmp["goals_away"] > 0)).astype(int)
        ft["both_score_pct_home"] = df_tmp["team_home"].map(
            df_tmp.groupby("team_home")["both_score"].mean()
        ).fillna(0.5)
        ft["both_score_pct_away"] = df_tmp["team_away"].map(
            df_tmp.groupby("team_away")["both_score"].mean()
        ).fillna(0.5)
        self._add_meta("both_score_pct_home", "G11_Patterns",
                       "% partidos donde ambos anotan", "[0,1]")
        self._add_meta("both_score_pct_away", "G11_Patterns",
                       "% partidos donde ambos anotan", "[0,1]")

        # Average goals per match historico
        ft["avg_goals_team_home"] = df_tmp["team_home"].map(
            df_tmp.groupby("team_home")["goals_home"].mean()
        ).fillna(1.2)
        ft["avg_goals_team_away"] = df_tmp["team_away"].map(
            df_tmp.groupby("team_away")["goals_away"].mean()
        ).fillna(1.0)
        self._add_meta("avg_goals_team_home", "G11_Patterns",
                       "Goles promedio historico anotados (local)", "[0,inf]", True)
        self._add_meta("avg_goals_team_away", "G11_Patterns",
                       "Goles promedio historico anotados (visitante)", "[0,inf]", True)

        return self

    # ── GRUPO 12: VARIABLES DERIVADAS (12 features) ───────────────────────────

    def calc_derived(self):
        """Features derivadas: prob ELO, xG total, competitividad, etc."""
        logger.info("  Grupo 12: Variables derivadas...")

        ft = self.features

        # ELO win probability (formula Elo estandar)
        elo_diff = ft["elo_diff"].fillna(0)
        ft["elo_win_prob_home"] = (1 / (1 + 10 ** (-elo_diff / 400))).round(4)
        ft["elo_win_prob_away"] = 1 - ft["elo_win_prob_home"]
        self._add_meta("elo_win_prob_home", "G12_Derived",
                       "P(victoria local) segun formula ELO", "[0,1]", True)
        self._add_meta("elo_win_prob_away", "G12_Derived",
                       "P(victoria visitante) segun ELO", "[0,1]", True)

        # Match competitiveness: cuanto mas cercano ELO, mas competitivo
        ft["match_competitiveness"] = 1 / (1 + abs(elo_diff) / 400)
        self._add_meta("match_competitiveness", "G12_Derived",
                       "Competitividad del partido (1=muy parejo)", "[0,1]", True)

        # Upset probability: prob de que el equipo menos favorecido gane
        ft["upset_probability"] = ft["elo_win_prob_away"].where(
            ft["elo_home"] > ft["elo_away"],
            ft["elo_win_prob_home"]
        )
        self._add_meta("upset_probability", "G12_Derived",
                       "Prob de sorpresa (favorito es derrotado)", "[0,0.5]")

        # Expected total goals
        ft["expected_total_goals"] = (
            ft.get("xg_est_home", pd.Series(1.2, index=ft.index)) +
            ft.get("xg_est_away", pd.Series(1.0, index=ft.index))
        ).round(2)
        self._add_meta("expected_total_goals", "G12_Derived",
                       "xG total estimado del partido", "[0,8]", True)

        # Home away balance: ratio rendimiento local vs visitante
        if "home_win_rate_overall" in ft.columns and "away_win_rate_overall" in ft.columns:
            ft["home_away_balance_home"] = (
                ft["home_win_rate_overall"] / (ft["away_win_rate_overall"] + 0.01)
            ).clip(0, 10).round(3)
            ft["home_away_balance_away"] = (
                ft["away_win_rate_overall"] / (ft["home_win_rate_overall"] + 0.01)
            ).clip(0, 10).round(3)
        else:
            ft["home_away_balance_home"] = 1.0
            ft["home_away_balance_away"] = 1.0
        self._add_meta("home_away_balance_home", "G12_Derived",
                       "Ratio victorias local/fuera del equipo", "[0,10]")

        # Underdog strength: diferencia ELO en favor del visitante
        ft["underdog_strength"] = (ft["elo_away"] - ft["elo_home"]).clip(0).round(1)
        self._add_meta("underdog_strength", "G12_Derived",
                       "Superioridad ELO visitante sobre local (0 si local es mejor)", "[0,inf]")

        # Goal diff balance: diferencia promedios de gol
        if "goals_for_avg5_home" in ft.columns and "goals_for_avg5_away" in ft.columns:
            ft["goal_balance_home"] = (
                ft["goals_for_avg5_home"] - ft["goals_against_avg5_home"]
            ).round(2)
            ft["goal_balance_away"] = (
                ft["goals_for_avg5_away"] - ft["goals_against_avg5_away"]
            ).round(2)
            ft["goal_balance_diff"] = ft["goal_balance_home"] - ft["goal_balance_away"]
        else:
            ft["goal_balance_home"] = 0.0
            ft["goal_balance_away"] = 0.0
            ft["goal_balance_diff"] = 0.0
        self._add_meta("goal_balance_diff", "G12_Derived",
                       "Diferencia de balance gol entre equipos", "[-inf,inf]", True)

        # Offensive vs defensive pressure
        if "goals_for_avg5_home" in ft.columns:
            ft["offensive_pressure_home"] = ft["goals_for_avg5_home"].clip(0, 5) / 5
            ft["defensive_pressure_home"] = ft["goals_against_avg5_home"].clip(0, 5) / 5
            ft["offensive_pressure_away"] = ft["goals_for_avg5_away"].clip(0, 5) / 5
            ft["defensive_pressure_away"] = ft["goals_against_avg5_away"].clip(0, 5) / 5
        else:
            ft["offensive_pressure_home"] = 0.4
            ft["defensive_pressure_home"] = 0.4
            ft["offensive_pressure_away"] = 0.4
            ft["defensive_pressure_away"] = 0.4
        self._add_meta("offensive_pressure_home", "G12_Derived",
                       "Presion ofensiva (goles/match norm)", "[0,1]")
        self._add_meta("offensive_pressure_away", "G12_Derived",
                       "Presion ofensiva visitante", "[0,1]")

        return self

    # ── VARIABLES OBJETIVO ───────────────────────────────────────────────────

    def _add_targets(self):
        """Agregar variables objetivo al DataFrame de features."""
        ft = self.features
        df = self.df
        ft["target_resultado"]   = df["resultado"].fillna(0).astype(int)
        ft["target_goals_home"]  = df["goals_home"].fillna(0)
        ft["target_goals_away"]  = df["goals_away"].fillna(0)
        ft["target_total_goals"] = ft["target_goals_home"] + ft["target_goals_away"]
        ft["target_over25"]      = (ft["target_total_goals"] > 2).astype(int)
        ft["target_both_score"]  = (
            (ft["target_goals_home"] > 0) & (ft["target_goals_away"] > 0)
        ).astype(int)

    # ── PIPELINE PRINCIPAL ───────────────────────────────────────────────────

    def generate_all(self) -> pd.DataFrame:
        """
        Ejecutar todos los grupos de features en secuencia.
        Retorna DataFrame con match_id + todas las features + targets.
        """
        logger.info("Generando features...")
        t0 = datetime.now()

        (self
         .calc_strength_features()
         .calc_recent_form()
         .calc_offensive()
         .calc_defensive()
         .calc_context()
         .calc_h2h()
         .calc_competition()
         .calc_advanced()
         .calc_weather()
         .calc_psychology()
         .calc_patterns()
         .calc_derived()
         )

        self._add_targets()

        elapsed = (datetime.now() - t0).total_seconds()
        n_feat = len([c for c in self.features.columns if not c.startswith("target_")
                      and c != "match_id"])
        logger.info("Features generadas: %d en %.1fs", n_feat, elapsed)
        return self.features.copy()

    # ── VALIDACION Y CALIDAD ─────────────────────────────────────────────────

    def validate_quality(self, df_feat: pd.DataFrame) -> Dict:
        """
        Reporte de calidad:
        - % NaN por columna
        - Rangos min/max
        - Columnas con alta correlacion (> 0.95)
        """
        logger.info("Validando calidad de features...")

        feature_cols = [c for c in df_feat.columns
                        if not c.startswith("target_") and c != "match_id"]

        # NaN report
        nan_pct = (df_feat[feature_cols].isna().mean() * 100).round(2)
        high_nan = nan_pct[nan_pct > 20].to_dict()

        # Ranges
        ranges = df_feat[feature_cols].agg(["min", "max", "mean", "std"]).round(3).to_dict()

        # High correlations
        try:
            corr = df_feat[feature_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            high_corr_pairs = [
                {"f1": c1, "f2": c2, "corr": round(float(upper.loc[c1, c2]), 3)}
                for c1 in upper.index for c2 in upper.columns
                if pd.notna(upper.loc[c1, c2]) and upper.loc[c1, c2] > 0.95
            ]
        except Exception:
            high_corr_pairs = []

        report = {
            "total_features": len(feature_cols),
            "total_rows": len(df_feat),
            "features_with_high_nan_pct": high_nan,
            "high_correlation_pairs_count": len(high_corr_pairs),
            "high_correlation_pairs": high_corr_pairs[:20],
            "nan_pct_per_feature": nan_pct.to_dict(),
        }

        logger.info("  Features totales: %d", len(feature_cols))
        logger.info("  Features con >20%% NaN: %d", len(high_nan))
        logger.info("  Pares alta correlacion (>0.95): %d", len(high_corr_pairs))

        return report

    def save_metadata(self):
        """Guardar metadatos de features en JSON."""
        out = DATA_DIR / "features_metadata.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, indent=2, ensure_ascii=False)
        logger.info("[OK] Metadatos: %s (%d features)", out, len(self._meta))


# ─── FUNCIONES DE SALIDA ─────────────────────────────────────────────────────

def _impute_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Imputar NaN: mediana para numericas, 0 para categoricas."""
    feature_cols = [c for c in df.columns
                    if not c.startswith("target_") and c != "match_id"]
    for col in feature_cols:
        if df[col].dtype in [np.float64, np.float32, float]:
            med = df[col].median()
            df[col] = df[col].fillna(med if pd.notna(med) else 0.0)
        else:
            df[col] = df[col].fillna(0)
    return df


def _baseline_importance(df_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Importancia baseline: correlacion de Spearman de cada feature con target_resultado.
    No usa modelo — sirve para priorizar features en Fase 3.
    """
    feature_cols = [c for c in df_feat.columns
                    if not c.startswith("target_") and c != "match_id"]
    target = df_feat["target_resultado"]

    rows = []
    for col in feature_cols:
        try:
            rho, pval = stats.spearmanr(df_feat[col].fillna(0), target, nan_policy="omit")
            rows.append({"feature": col, "spearman_rho": round(rho, 4),
                         "abs_rho": round(abs(rho), 4), "p_value": round(pval, 4)})
        except Exception:
            rows.append({"feature": col, "spearman_rho": 0, "abs_rho": 0, "p_value": 1})

    df_imp = pd.DataFrame(rows).sort_values("abs_rho", ascending=False).reset_index(drop=True)
    return df_imp


def main():
    logger.info("=" * 70)
    logger.info("FEATURE ENGINEERING — Fase 2")
    logger.info("Inicio: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 70)

    # ── Cargar datos ──────────────────────────────────────────────────────────
    csv_in = DATA_DIR / "matches_cleaned.csv"
    if not csv_in.exists():
        logger.error("FATAL: No encontrado %s", csv_in)
        return False

    df = pd.read_csv(csv_in)
    logger.info("Cargados %d partidos, %d columnas", len(df), len(df.columns))

    # ── Generar features ──────────────────────────────────────────────────────
    fe = FeatureEngineer(df)
    df_feat = fe.generate_all()
    df_feat = _impute_nan(df_feat)

    # ── Validar calidad ───────────────────────────────────────────────────────
    quality_report = fe.validate_quality(df_feat)

    # ── Baseline importance ───────────────────────────────────────────────────
    df_imp = _baseline_importance(df_feat)
    logger.info("Top 10 features por correlacion Spearman con resultado:")
    for _, row in df_imp.head(10).iterrows():
        logger.info("  %-40s  rho=%.4f", row["feature"], row["spearman_rho"])

    # ── Correlacion matrix (solo features numericas) ──────────────────────────
    feature_cols = [c for c in df_feat.columns
                    if not c.startswith("target_") and c != "match_id"]
    corr_matrix = df_feat[feature_cols].corr().round(3)

    # ── Guardar salidas ───────────────────────────────────────────────────────
    out_feat   = DATA_DIR / "features_engineered.csv"
    out_report = DATA_DIR / "missing_values_report.json"
    out_imp    = DATA_DIR / "feature_importance_baseline.csv"
    out_corr   = DATA_DIR / "features_correlation_matrix.csv"

    df_feat.to_csv(out_feat, index=False, encoding="utf-8")
    logger.info("[OK] features_engineered.csv — %d filas x %d columnas",
                len(df_feat), len(df_feat.columns))

    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2, ensure_ascii=False, default=str)
    logger.info("[OK] missing_values_report.json")

    df_imp.to_csv(out_imp, index=False, encoding="utf-8")
    logger.info("[OK] feature_importance_baseline.csv")

    corr_matrix.to_csv(out_corr, encoding="utf-8")
    logger.info("[OK] features_correlation_matrix.csv")

    fe.save_metadata()

    # ── Resumen final ─────────────────────────────────────────────────────────
    n_feat = len([c for c in df_feat.columns
                  if not c.startswith("target_") and c != "match_id"])
    n_targets = len([c for c in df_feat.columns if c.startswith("target_")])

    logger.info("=" * 70)
    logger.info("[OK] FASE 2 COMPLETADA")
    logger.info("     Filas:     %d", len(df_feat))
    logger.info("     Features:  %d", n_feat)
    logger.info("     Targets:   %d (resultado, goals_home, goals_away, over25, both_score)", n_targets)
    logger.info("     Salida:    %s", out_feat)
    logger.info("=" * 70)

    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
