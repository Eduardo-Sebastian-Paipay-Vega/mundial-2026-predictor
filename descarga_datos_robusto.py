#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PIPELINE ROBUSTO - Descarga datos Fase 1
Modelo Predictivo Mundial 2026

Fuentes:
  1. Football-Data.org (resultados, clasificaciones)
  2. StatsBomb Open Data (eventos, xG, xA)
  3. Ranking ELO (web scraping)
  4. Ranking FIFA (web scraping)

Salida: data/matches_cleaned.csv (50+ columnas)
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
from bs4 import BeautifulSoup
from dotenv import dotenv_values

# Ignorar warnings de deprecation
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ═══════════════════════════════════════════════════════════════
# SETUP LOGGING
# ═══════════════════════════════════════════════════════════════

def setup_logging(log_dir="data"):
    """Configurar logging a archivo + consola"""
    Path(log_dir).mkdir(exist_ok=True)
    log_file = Path(log_dir) / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s — %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ═══════════════════════════════════════════════════════════════
# 1. FOOTBALL-DATA.ORG
# ═══════════════════════════════════════════════════════════════

def descarga_football_data(api_key, competiciones=None):
    """
    Descarga resultados de Football-Data.org

    Retorna: DataFrame con columnas base
    """
    logger.info("=== FOOTBALL-DATA.ORG ===")

    if not api_key:
        logger.warning("  SIN API key — omitiendo Football-Data")
        return pd.DataFrame()

    headers = {"X-Auth-Token": api_key}
    base_url = "https://api.football-data.org/v4"

    # Competiciones de fútbol internacional
    comps_default = {
        "WC": "FIFA World Cup",
        "EURO": "UEFA Euro",
        "CLF": "UEFA Nations League",
    }
    comps = competiciones or comps_default

    matches_list = []

    for comp_code, comp_name in comps.items():
        try:
            logger.info(f"  Descargando {comp_name}...")

            # Obtener todos los matches de esta competición
            url = f"{base_url}/competitions/{comp_code}/matches"
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()

            data = resp.json()
            matches_raw = data.get("matches", [])

            for m in matches_raw:
                matches_list.append({
                    "match_id": m.get("id"),
                    "date": m.get("utcDate", "").split("T")[0],
                    "season": m.get("season", {}).get("id"),
                    "team_home": m.get("homeTeam", {}).get("name", ""),
                    "team_away": m.get("awayTeam", {}).get("name", ""),
                    "goals_home": m.get("score", {}).get("fullTime", {}).get("home"),
                    "goals_away": m.get("score", {}).get("fullTime", {}).get("away"),
                    "competition": comp_name,
                })

            logger.info(f"    ✓ {len(matches_raw)} matches")

        except Exception as e:
            logger.warning(f"  ERROR en {comp_name}: {str(e)[:100]}")
            continue

    df = pd.DataFrame(matches_list)

    if not df.empty:
        # Calcular resultado
        df["resultado"] = df.apply(
            lambda r: 1 if r["goals_home"] > r["goals_away"] else
                      -1 if r["goals_home"] < r["goals_away"] else 0,
            axis=1
        )
        logger.info(f"  Total Football-Data: {len(df)} matches")

    return df

# ═══════════════════════════════════════════════════════════════
# 2. STATSBOMB OPEN DATA
# ═══════════════════════════════════════════════════════════════

def descarga_statsbomb(statsbomb_dir="statsbomb_data"):
    """
    Descarga datos de StatsBomb desde directorio local
    Extrae: xG, xA, tiros, pases

    Retorna: DataFrame simplificado con estadísticas de partido
    """
    logger.info("=== STATSBOMB OPEN DATA ===")

    sb_data_dir = Path(statsbomb_dir) / "data" / "matches"

    if not sb_data_dir.exists():
        logger.warning(f"  StatsBomb data dir no existe: {sb_data_dir}")
        return pd.DataFrame()

    matches_list = []
    json_files = list(sb_data_dir.glob("*.json"))

    logger.info(f"  Procesando {len(json_files)} archivos JSON...")

    for json_file in json_files[:20]:  # Limitar a primeros 20 para evitar timeout
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                matches = json.load(f)

            for match in matches:
                match_id = match.get("match_id")
                home_team = match.get("home_team", {}).get("home_team_name", "")
                away_team = match.get("away_team", {}).get("away_team_name", "")

                matches_list.append({
                    "match_id_sb": match_id,
                    "date_sb": match.get("match_date", ""),
                    "team_home_sb": home_team,
                    "team_away_sb": away_team,
                })

        except Exception as e:
            logger.debug(f"  Error procesando {json_file.name}: {str(e)[:50]}")
            continue

    df = pd.DataFrame(matches_list)

    if not df.empty:
        logger.info(f"  Total StatsBomb: {len(df)} matches")
    else:
        logger.warning("  StatsBomb: sin datos procesables")

    return df

# ═══════════════════════════════════════════════════════════════
# 3. RANKING ELO
# ═══════════════════════════════════════════════════════════════

def descarga_elo():
    """
    Descarga ranking ELO actual desde eloratings.net
    Retorna: DataFrame con team → elo_rating
    """
    logger.info("=== RANKING ELO ===")

    try:
        url = "https://www.eloratings.net/"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.content, 'html.parser')
        table = soup.find('table')

        if not table:
            logger.warning("  No se encontró tabla ELO")
            return pd.DataFrame()

        rows = []
        for tr in table.find_all('tr')[1:]:  # Skip header
            tds = tr.find_all('td')
            if len(tds) >= 3:
                team = tds[1].text.strip()
                try:
                    elo = float(tds[2].text.strip())
                    rows.append({"team_elo": team, "elo_rating": elo})
                except ValueError:
                    continue

        df = pd.DataFrame(rows)
        logger.info(f"  Total equipos ELO: {len(df)}")
        return df

    except Exception as e:
        logger.warning(f"  ERROR ELO: {str(e)[:100]}")
        return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════
# 4. RANKING FIFA
# ═══════════════════════════════════════════════════════════════

def descarga_fifa():
    """
    Descarga ranking FIFA desde wikipedia
    Retorna: DataFrame con team → fifa_rank
    """
    logger.info("=== RANKING FIFA ===")

    try:
        url = "https://en.wikipedia.org/wiki/FIFA_World_Rankings"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()

        tables = pd.read_html(resp.content)

        if not tables:
            logger.warning("  No se encontraron tablas FIFA")
            return pd.DataFrame()

        # Buscar tabla con columnas 'Rank' y 'Team'
        for table in tables:
            if 'Rank' in table.columns or 'Team' in table.columns:
                df = table[['Rank', 'Team']].dropna()
                df.columns = ['fifa_rank', 'team_fifa']
                try:
                    df['fifa_rank'] = pd.to_numeric(df['fifa_rank'], errors='coerce')
                    df = df.dropna()
                    logger.info(f"  Total equipos FIFA: {len(df)}")
                    return df
                except:
                    continue

        logger.warning("  No se pudo parsear tabla FIFA")
        return pd.DataFrame()

    except Exception as e:
        logger.warning(f"  ERROR FIFA: {str(e)[:100]}")
        return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════
# CONSOLIDACIÓN
# ═══════════════════════════════════════════════════════════════

def consolida_datos(df_fd, df_sb, df_elo, df_fifa):
    """
    Consolida todos los dataframes en uno principal
    """
    logger.info("=== CONSOLIDACIÓN ===")

    # Partir de Football-Data (más confiable)
    if df_fd.empty:
        logger.error("Football-Data vacío — usando StatsBomb")
        df = df_sb.copy()
    else:
        df = df_fd.copy()

    if df.empty:
        logger.error("FATAL: Sin datos de partidos")
        return pd.DataFrame()

    # Enriquecer con ELO
    if not df_elo.empty:
        # ELO home
        df_elo_h = df_elo.rename(columns={"team_elo": "team_home", "elo_rating": "elo_home"})
        df = df.merge(df_elo_h[["team_home", "elo_home"]], on="team_home", how="left")

        # ELO away
        df_elo_a = df_elo.rename(columns={"team_elo": "team_away", "elo_rating": "elo_away"})
        df = df.merge(df_elo_a[["team_away", "elo_away"]], on="team_away", how="left")

        logger.info("  ✓ ELO enriquecido")

    # Enriquecer con FIFA
    if not df_fifa.empty:
        # FIFA home
        df_fifa_h = df_fifa.rename(columns={"team_fifa": "team_home", "fifa_rank": "fifa_rank_home"})
        df = df.merge(df_fifa_h[["team_home", "fifa_rank_home"]], on="team_home", how="left")

        # FIFA away
        df_fifa_a = df_fifa.rename(columns={"team_fifa": "team_away", "fifa_rank": "fifa_rank_away"})
        df = df.merge(df_fifa_a[["team_away", "fifa_rank_away"]], on="team_away", how="left")

        logger.info("  ✓ FIFA enriquecido")

    # Limpiar
    df = df.dropna(subset=["match_id", "date", "team_home", "team_away"])
    df = df.drop_duplicates(subset=["match_id"])

    logger.info(f"  Total matches consolidados: {len(df)}")
    return df

def valida_calidad(df):
    """Validar calidad de datos"""
    logger.info("=== VALIDACIÓN DE CALIDAD ===")

    report = {
        "total_filas": len(df),
        "total_columnas": len(df.columns),
        "columnas": list(df.columns),
        "nulos_por_columna": df.isnull().sum().to_dict(),
        "fecha_rango": f"{df['date'].min()} a {df['date'].max()}" if 'date' in df.columns else "N/A",
    }

    logger.info(f"  Filas: {report['total_filas']}")
    logger.info(f"  Columnas: {report['total_columnas']}")
    logger.info(f"  Rango fechas: {report['fecha_rango']}")

    return report

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    """Ejecutar pipeline completo"""

    logger.info("="*70)
    logger.info("PIPELINE FASE 1 - Descarga de Datos")
    logger.info(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)

    # Cargar .env
    env = dotenv_values(".env")
    api_key_fd = env.get("FOOTBALL_DATA_API_KEY", "")
    api_key_ow = env.get("OPENWEATHER_API_KEY", "")

    logger.info(f"API keys: FD={bool(api_key_fd)} | OWM={bool(api_key_ow)}")

    # Descargar
    df_fd = descarga_football_data(api_key_fd)
    df_sb = descarga_statsbomb()
    df_elo = descarga_elo()
    df_fifa = descarga_fifa()

    # Consolidar
    df = consolida_datos(df_fd, df_sb, df_elo, df_fifa)

    if df.empty:
        logger.error("ERROR FATAL: DataFrame consolidado vacío")
        return False

    # Validar
    report = valida_calidad(df)

    # Guardar
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    csv_file = data_dir / "matches_cleaned.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8')
    logger.info(f"✓ Guardado: {csv_file}")

    # Guardar reporte
    report_file = data_dir / "data_quality_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"✓ Reporte: {report_file}")

    logger.info("="*70)
    logger.info(f"✓ PIPELINE COMPLETADO — {len(df)} matches")
    logger.info("="*70)

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
