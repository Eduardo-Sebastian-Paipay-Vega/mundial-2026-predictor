#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PIPELINE FINAL - Fase 1 Completa
Modelo Predictivo Mundial 2026

Fuentes:
  1. Football-Data.org (historico 2017-2025)
  2. ELO Ratings (CSV de GitHub)
  3. FIFA Rankings (API oficial)

Salida: data/matches_cleaned.csv (50+ filas con datos historicos)
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from dotenv import dotenv_values

warnings.filterwarnings("ignore")

# ===============================================================
# LOGGING (sin caracteres especiales para Windows)
# ===============================================================

def setup_logging(log_dir="data"):
    """Logging a archivo + consola (Windows-safe)"""
    Path(log_dir).mkdir(exist_ok=True)
    log_file = Path(log_dir) / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ===============================================================
# 1. FOOTBALL-DATA.ORG (HISTORICO)
# ===============================================================

def descarga_partidos_historicos(start_year=2017, end_year=2025):
    """
    Descarga partidos internacionales historicos desde martj42/international_results.
    Filtra a torneos mayores: WC, Euro, Copa America, AFCON, Gold Cup, Nations League.
    Retorna: (df_matches, df_results_full)
      df_matches    = partidos filtrados en rango y torneo
      df_results_full = CSV completo (para calcular ELO desde inicio)
    """
    logger.info("=== PARTIDOS HISTORICOS (international_results) ===")

    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

    try:
        logger.info("  Descargando results.csv...")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        from io import StringIO
        df_full = pd.read_csv(StringIO(resp.text))
        logger.info(f"  Total historial: {len(df_full)} partidos")

        df_full["date"] = pd.to_datetime(df_full["date"], errors="coerce")
        df_full = df_full.dropna(subset=["date", "home_team", "away_team",
                                          "home_score", "away_score"])

        # Torneos mayores (filtro por keywords en tournament)
        major_keywords = [
            "FIFA World Cup", "UEFA Euro", "Copa America",
            "Africa Cup", "AFCON", "Gold Cup", "Asian Cup",
            "UEFA Nations League", "CONCACAF Nations League",
            "World Cup qualification", "UEFA Euro qualification",
        ]
        pattern = "|".join(major_keywords)
        mask_tourn = df_full["tournament"].str.contains(pattern, case=False, na=False)

        # Filtro por año
        mask_year  = (df_full["date"].dt.year >= start_year) & \
                     (df_full["date"].dt.year <= end_year)

        df_matches = df_full[mask_tourn & mask_year].copy()
        df_matches = df_matches.reset_index(drop=True)
        df_matches["match_id"] = "mr_" + df_matches.index.astype(str)
        df_matches = df_matches.rename(columns={
            "home_team": "team_home",
            "away_team": "team_away",
            "home_score": "goals_home",
            "away_score": "goals_away",
            "tournament": "competition",
        })
        df_matches["date"] = df_matches["date"].dt.strftime("%Y-%m-%d")
        df_matches["goals_home"] = pd.to_numeric(df_matches["goals_home"], errors="coerce")
        df_matches["goals_away"] = pd.to_numeric(df_matches["goals_away"], errors="coerce")
        df_matches["resultado"] = df_matches.apply(
            lambda r: 1 if r["goals_home"] > r["goals_away"] else
                      -1 if r["goals_home"] < r["goals_away"] else 0,
            axis=1
        )

        logger.info(f"  Partidos en rango {start_year}-{end_year} (torneos mayores): {len(df_matches)}")
        return df_matches, df_full

    except Exception as e:
        logger.warning(f"  ERROR: {str(e)[:100]}")
        return pd.DataFrame(), pd.DataFrame()

# ===============================================================
# 2. ELO RATINGS (CSV GitHub)
# ===============================================================

def calcula_elo(df_full: pd.DataFrame) -> dict:
    """
    Calcula ELO rolling desde el historial completo.
    Retorna dict {team: elo_rating} con rating al final del historial.
    """
    logger.info("=== RANKING ELO ===")

    K_DEFAULT = 30
    HOME_ADV   = 100
    ELO_INIT   = 1500

    elo_ratings = {}

    df = df_full.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").dropna(subset=["home_team", "away_team", "home_score", "away_score"])

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        elo_h = elo_ratings.get(home, ELO_INIT)
        elo_a = elo_ratings.get(away, ELO_INIT)

        neutral = bool(row.get("neutral", False))
        h_adj = elo_h if neutral else elo_h + HOME_ADV

        expected_h = 1 / (1 + 10 ** ((elo_a - h_adj) / 400))

        try:
            gh = int(float(row["home_score"]))
            ga = int(float(row["away_score"]))
        except (ValueError, TypeError):
            continue
        actual_h = 1 if gh > ga else 0.5 if gh == ga else 0

        tname = str(row.get("tournament", "")).lower()
        if "world cup" in tname and "qualif" not in tname:
            K = 60
        elif any(x in tname for x in ["euro", "copa america", "africa cup", "asian cup", "gold cup"]):
            K = 50
        elif any(x in tname for x in ["qualifier", "qualification", "nations league"]):
            K = 40
        elif "friendly" in tname:
            K = 20
        else:
            K = K_DEFAULT

        elo_ratings[home] = elo_h + K * (actual_h - expected_h)
        elo_ratings[away] = elo_a + K * ((1 - actual_h) - (1 - expected_h))

    logger.info(f"  Equipos ELO calculados: {len(elo_ratings)}")
    return elo_ratings


def enriquece_elo_por_fecha(df_matches: pd.DataFrame, df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna a cada partido el ELO pre-partido de ambos equipos
    (calculado desde todos los partidos anteriores a esa fecha).
    """
    logger.info("  Calculando ELO pre-partido por fecha...")

    K_DEFAULT = 30
    HOME_ADV   = 100
    ELO_INIT   = 1500

    df_full2 = df_full.copy()
    df_full2["date"] = pd.to_datetime(df_full2["date"], errors="coerce")
    df_full2 = df_full2.sort_values("date").dropna(
        subset=["home_team", "away_team", "home_score", "away_score"]
    )

    match_dates = set(pd.to_datetime(df_matches["date"]).dt.date)
    elo_ratings: dict = {}
    elo_snapshots: dict = {}  # date -> {team: elo}

    for _, row in df_full2.iterrows():
        d = row["date"].date()
        if d in match_dates:
            # Guardar snapshot ANTES de este partido
            if d not in elo_snapshots:
                elo_snapshots[d] = dict(elo_ratings)

        home = row["home_team"]
        away = row["away_team"]
        elo_h = elo_ratings.get(home, ELO_INIT)
        elo_a = elo_ratings.get(away, ELO_INIT)
        neutral = bool(row.get("neutral", False))
        h_adj = elo_h if neutral else elo_h + HOME_ADV
        expected_h = 1 / (1 + 10 ** ((elo_a - h_adj) / 400))
        try:
            gh = int(float(row["home_score"]))
            ga = int(float(row["away_score"]))
        except (ValueError, TypeError):
            continue
        actual_h = 1 if gh > ga else 0.5 if gh == ga else 0
        tname = str(row.get("tournament", "")).lower()
        if "world cup" in tname and "qualif" not in tname:
            K = 60
        elif any(x in tname for x in ["euro", "copa america", "africa cup", "asian cup", "gold cup"]):
            K = 50
        elif any(x in tname for x in ["qualifier", "qualification", "nations league"]):
            K = 40
        elif "friendly" in tname:
            K = 20
        else:
            K = K_DEFAULT
        elo_ratings[home] = elo_h + K * (actual_h - expected_h)
        elo_ratings[away] = elo_a + K * ((1 - actual_h) - (1 - expected_h))

    # Asignar ELO a cada match usando el snapshot del dia del partido
    elo_home_list = []
    elo_away_list = []
    for _, row in df_matches.iterrows():
        d = pd.to_datetime(row["date"]).date()
        snap = elo_snapshots.get(d, elo_ratings)
        elo_home_list.append(round(snap.get(row["team_home"], ELO_INIT), 1))
        elo_away_list.append(round(snap.get(row["team_away"], ELO_INIT), 1))

    df_out = df_matches.copy()
    df_out["elo_home"] = elo_home_list
    df_out["elo_away"] = elo_away_list
    df_out["diff_elo"] = df_out["elo_home"] - df_out["elo_away"]
    return df_out

# ===============================================================
# 3. FIFA RANKINGS (API publica)
# ===============================================================

def descarga_fifa_api():
    """
    Descarga FIFA rankings desde api.fifa.com (endpoint no oficial).
    Retorna DataFrame vacio si falla (no critico).
    """
    logger.info("=== RANKING FIFA ===")

    try:
        date_id = "id_20240620"  # Rankings junio 2024
        url = f"https://api.fifa.com/api/v1/rankings/FIFA?locale=en&dateId={date_id}&count=250"
        headers = {"User-Agent": "Mozilla/5.0"}

        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        data = resp.json()
        rankings = data.get("Rankings", [])

        rows = []
        for entry in rankings:
            name_obj = entry.get("TeamName", {})
            name = name_obj.get("Description", "") if isinstance(name_obj, dict) else str(name_obj)
            rows.append({
                "team_fifa": name,
                "fifa_rank": entry.get("Rank"),
                "fifa_points": entry.get("TotalPoints"),
            })

        df = pd.DataFrame(rows)
        logger.info(f"  Equipos FIFA: {len(df)}")
        return df

    except Exception as e:
        logger.warning(f"  FIFA API no disponible: {str(e)[:80]}")
        return pd.DataFrame()

# ===============================================================
# CONSOLIDACION
# ===============================================================

def consolida_datos(df_matches, df_full, df_fifa):
    """Consolidar partidos + ELO por fecha + FIFA"""
    logger.info("=== CONSOLIDACION ===")

    if df_matches.empty:
        logger.error("FATAL: Sin partidos historicos")
        return pd.DataFrame()

    # ELO pre-partido por fecha
    df = enriquece_elo_por_fecha(df_matches, df_full)
    logger.info("  ELO enriquecido (pre-partido por fecha)")

    if not df_fifa.empty:
        df_fh = df_fifa.rename(columns={"team_fifa": "team_home", "fifa_rank": "fifa_rank_home"})
        df = df.merge(df_fh[["team_home", "fifa_rank_home"]], on="team_home", how="left")
        df_fa = df_fifa.rename(columns={"team_fifa": "team_away", "fifa_rank": "fifa_rank_away"})
        df = df.merge(df_fa[["team_away", "fifa_rank_away"]], on="team_away", how="left")
        logger.info("  FIFA enriquecido")

    df = df.drop_duplicates(subset=["match_id"])
    logger.info(f"  Total matches consolidados: {len(df)}")
    return df

def valida_calidad(df):
    """Validar datos"""
    logger.info("=== VALIDACION ===")

    goles_pct = (df['goals_home'].notna().sum() / len(df) * 100) if len(df) > 0 else 0

    report = {
        "total_filas": len(df),
        "total_columnas": len(df.columns),
        "columnas": list(df.columns),
        "fecha_rango": f"{df['date'].min()} a {df['date'].max()}" if 'date' in df.columns else "N/A",
        "goles_completos_pct": round(goles_pct, 1),
        "nulos_por_columna": df.isnull().sum().to_dict(),
    }

    logger.info(f"  Filas: {report['total_filas']}")
    logger.info(f"  Columnas: {report['total_columnas']}")
    logger.info(f"  Rango fechas: {report['fecha_rango']}")
    logger.info(f"  Goles completados: {goles_pct:.1f}%")

    return report

# ===============================================================
# MAIN
# ===============================================================

def main():
    """Ejecutar pipeline"""

    logger.info("=" * 70)
    logger.info("PIPELINE FASE 1 - FINAL")
    logger.info(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    env = dotenv_values(".env")
    api_key_fd = env.get("FOOTBALL_DATA_API_KEY", "")
    start_year = int(env.get("START_YEAR", "2017"))
    end_year   = int(env.get("END_YEAR", "2025"))

    logger.info(f"Periodo: {start_year}-{end_year}")
    logger.info(f"API keys: FD={bool(api_key_fd)}")

    df_matches, df_full = descarga_partidos_historicos(start_year, end_year)
    df_fifa             = descarga_fifa_api()

    df = consolida_datos(df_matches, df_full, df_fifa)

    if df.empty:
        logger.error("FATAL: DataFrame consolidado vacio")
        return False

    report = valida_calidad(df)

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    csv_file = data_dir / "matches_cleaned.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8')
    logger.info(f"[OK] Guardado: {csv_file}")

    report_file = data_dir / "data_quality_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"[OK] Reporte: {report_file}")

    logger.info("=" * 70)
    logger.info(f"[OK] PIPELINE COMPLETADO - {len(df)} matches")
    logger.info("=" * 70)

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
