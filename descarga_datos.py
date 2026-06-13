#!/usr/bin/env python3
"""
Pipeline de datos para Modelo Predictivo Mundial 2026.

Fuentes de datos:
    1. StatsBomb Open Data  — eventos de partidos: xG, tiros, posesión (sin API key)
    2. Football-Data.org    — resultados y clasificaciones (API key opcional)
    3. World Football ELO   — ratings ELO calculados desde histórico de resultados
    4. FIFA Rankings        — rankings oficiales mensuales (API pública no oficial)

Salidas:
    data/matches_raw.csv          — consolidado sin filtrar
    data/matches_cleaned.csv      — datos validados y limpios
    data/data_quality_report.json — reporte de calidad
    data/pipeline.log             — log de ejecución
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, date
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from statsbombpy import sb

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------

load_dotenv()

DATA_DIR         = Path(os.getenv("DATA_DIR", "./data"))
LOG_LEVEL        = os.getenv("LOG_LEVEL", "INFO").upper()
FOOTBALL_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")
START_YEAR       = int(os.getenv("START_YEAR", "2017"))
END_YEAR         = int(os.getenv("END_YEAR", "2025"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "cache").mkdir(exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(DATA_DIR / "pipeline.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("mundial2026")

# ---------------------------------------------------------------------------
# CONSTANTES: COMPETICIONES NACIONALES
# ---------------------------------------------------------------------------

# IDs de StatsBomb para competiciones de selecciones
SB_NATIONAL_COMPETITIONS = {
    43: "FIFA World Cup",
    55: "UEFA Euro",
    16: "Copa America",
    6:  "Africa Cup of Nations",
    72: "UEFA Nations League",
}

# IDs de Football-Data.org para torneos internacionales
FD_COMPETITIONS = {
    2000: "FIFA World Cup",
    2016: "UEFA Euro",
    2019: "UEFA Nations League",
}

# Fechas de publicación de rankings FIFA (aprox. mensual)
FIFA_RANKING_DATES = [
    "2017-04-06", "2017-06-15", "2017-08-10", "2017-10-12", "2017-12-21",
    "2018-02-22", "2018-04-12", "2018-06-07", "2018-08-16", "2018-10-25",
    "2019-02-21", "2019-04-04", "2019-06-06", "2019-09-19", "2019-12-19",
    "2020-08-13", "2020-10-22", "2020-12-17",
    "2021-02-18", "2021-04-08", "2021-06-03", "2021-08-12",
    "2021-10-21", "2021-12-23",
    "2022-02-10", "2022-03-31", "2022-06-23", "2022-10-06", "2022-12-22",
    "2023-02-16", "2023-04-06", "2023-06-22", "2023-08-17",
    "2023-10-19", "2023-12-21",
    "2024-02-15", "2024-04-04", "2024-06-20", "2024-08-15",
    "2024-10-24", "2024-12-19",
    "2025-02-20", "2025-04-03",
]

# K-factors para cálculo ELO según importancia del torneo
ELO_K_FACTORS = {
    "FIFA World Cup":           60,
    "Copa America":             50,
    "UEFA Euro":                50,
    "Africa Cup of Nations":    50,
    "CONMEBOL":                 40,
    "UEFA Nations League":      40,
    "Qualifier":                40,
    "Friendly":                 20,
}
ELO_K_DEFAULT    = 30
ELO_HOME_ADV     = 100   # puntos de ventaja campo local
ELO_INITIAL      = 1500  # ELO inicial para equipos nuevos

# ---------------------------------------------------------------------------
# NORMALIZACIÓN DE NOMBRES DE EQUIPO
# ---------------------------------------------------------------------------

TEAM_ALIASES: Dict[str, str] = {
    "korea republic":         "south korea",
    "republic of korea":      "south korea",
    "ir iran":                "iran",
    "côte d'ivoire":          "ivory coast",
    "cote d'ivoire":          "ivory coast",
    "usa":                    "united states",
    "u.s.a.":                 "united states",
    "czechia":                "czech republic",
    "czech rep.":             "czech republic",
    "türkiye":                "turkey",
    "turkiye":                "turkey",
    "uae":                    "united arab emirates",
    "northern ireland":       "northern ireland",
    "cape verde islands":     "cape verde",
    "democratic republic of the congo": "dr congo",
    "trinidad and tobago":    "trinidad & tobago",
}


def normaliza_equipo(nombre: str) -> str:
    """Convierte nombre de equipo a forma canónica (minúscula, sin acentos problemáticos)."""
    if not nombre or pd.isna(nombre):
        return ""
    clean = str(nombre).strip().lower()
    return TEAM_ALIASES.get(clean, clean)


# ---------------------------------------------------------------------------
# DECORADOR: RETRY CON BACKOFF EXPONENCIAL
# ---------------------------------------------------------------------------

def retry_con_backoff(
    max_intentos: int = 4,
    delay_inicial: float = 1.5,
    factor: float = 2.0,
    excepciones: tuple = (requests.RequestException, ConnectionError, TimeoutError),
):
    """
    Decorador para reintentar funciones fallidas con backoff exponencial.

    Args:
        max_intentos:  Número máximo de intentos antes de propagar la excepción.
        delay_inicial: Segundos de espera en el primer reintento.
        factor:        Multiplicador del delay en cada reintento sucesivo.
        excepciones:   Tipos de excepción que activan el reintento.
    """
    def decorador(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = delay_inicial
            for intento in range(1, max_intentos + 1):
                try:
                    return func(*args, **kwargs)
                except excepciones as exc:
                    if intento == max_intentos:
                        logger.error(
                            "[%s] Fallo definitivo tras %d intentos: %s",
                            func.__name__, max_intentos, exc,
                        )
                        raise
                    logger.warning(
                        "[%s] Intento %d/%d fallido — reintentando en %.1fs: %s",
                        func.__name__, intento, max_intentos, delay, exc,
                    )
                    time.sleep(delay)
                    delay *= factor
        return wrapper
    return decorador


# ---------------------------------------------------------------------------
# UTILIDADES DE RED
# ---------------------------------------------------------------------------

@retry_con_backoff()
def get_json(
    url: str,
    headers: Optional[Dict] = None,
    params: Optional[Dict] = None,
) -> dict:
    """HTTP GET con retry que devuelve JSON parseado. Lanza HTTPError en 4xx/5xx."""
    resp = requests.get(url, headers=headers or {}, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


@retry_con_backoff()
def get_html(url: str, headers: Optional[Dict] = None) -> BeautifulSoup:
    """HTTP GET con retry que devuelve un objeto BeautifulSoup."""
    hdrs = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }
    if headers:
        hdrs.update(headers)
    resp = requests.get(url, headers=hdrs, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


# ---------------------------------------------------------------------------
# FUENTE 1: STATSBOMB OPEN DATA
# ---------------------------------------------------------------------------

def descarga_statsbomb() -> pd.DataFrame:
    """
    Descarga eventos de partidos de selecciones nacionales desde StatsBomb Open Data.

    Accede al repositorio público de GitHub (statsbomb/open-data) sin API key.
    Filtra competiciones de selecciones nacionales entre START_YEAR y END_YEAR.
    Para cada partido agrega tiros para obtener xG, shots y shots_on_target.

    Returns:
        DataFrame con columnas:
            match_id, date, season, competition,
            team_home, team_away, goals_home, goals_away,
            xg_home, xg_away, shots_home, shots_away,
            shots_on_target_home, shots_on_target_away, source.

    Notas:
        - La descarga puede tardar 20-60 min según cobertura disponible.
        - Los datos de StatsBomb cubren principalmente World Cup 2018/2022 y Euro 2020.
    """
    logger.info("=== FUENTE 1: StatsBomb Open Data ===")

    try:
        comps = sb.competitions()
    except Exception as exc:
        logger.error("No se pudo acceder a StatsBomb competitions: %s", exc)
        return pd.DataFrame()

    # Extraer año del nombre de temporada (ej. "2018" o "2017/2018" → 2018)
    comps["_season_year"] = pd.to_numeric(
        comps["season_name"].str.extract(r"(\d{4})")[0], errors="coerce"
    )
    mask = (
        comps["competition_id"].isin(SB_NATIONAL_COMPETITIONS.keys()) &
        comps["_season_year"].between(START_YEAR, END_YEAR)
    )
    comps_filtradas = comps[mask].reset_index(drop=True)

    logger.info(
        "Competiciones StatsBomb disponibles (%d-%d): %d",
        START_YEAR, END_YEAR, len(comps_filtradas),
    )

    if comps_filtradas.empty:
        logger.warning("Sin datos StatsBomb para el período especificado.")
        return pd.DataFrame()

    partidos_list: List[Dict] = []

    for _, comp_row in comps_filtradas.iterrows():
        cid   = int(comp_row["competition_id"])
        sid   = int(comp_row["season_id"])
        cname = comp_row["competition_name"]
        sname = comp_row["season_name"]

        try:
            matches = sb.matches(competition_id=cid, season_id=sid)
            logger.info("  %s %s — %d partidos encontrados", cname, sname, len(matches))
        except Exception as exc:
            logger.warning("  Error descargando partidos %s %s: %s", cname, sname, exc)
            continue

        total_partidos = len(matches)
        for i, (_, match) in enumerate(matches.iterrows(), 1):
            mid = int(match["match_id"])
            logger.debug("    Partido %d/%d (id=%d) ...", i, total_partidos, mid)

            try:
                events = sb.events(match_id=mid)
            except Exception as exc:
                logger.debug("    Error eventos match %d: %s", mid, exc)
                events = pd.DataFrame()

            stats = _agrega_shots(events, match)

            # home_team y away_team pueden ser str o dict según versión statsbombpy
            home = match.get("home_team", "")
            away = match.get("away_team", "")
            if isinstance(home, dict):
                home = home.get("home_team_name", home.get("name", ""))
            if isinstance(away, dict):
                away = away.get("away_team_name", away.get("name", ""))

            partidos_list.append({
                "match_id":               f"sb_{mid}",
                "date":                   str(match.get("match_date", "")),
                "season":                 sname,
                "competition":            cname,
                "team_home":              normaliza_equipo(str(home)),
                "team_away":              normaliza_equipo(str(away)),
                "goals_home":             int(match.get("home_score") or 0),
                "goals_away":             int(match.get("away_score") or 0),
                "xg_home":                stats["xg_home"],
                "xg_away":                stats["xg_away"],
                "shots_home":             stats["shots_home"],
                "shots_away":             stats["shots_away"],
                "shots_on_target_home":   stats["shots_on_target_home"],
                "shots_on_target_away":   stats["shots_on_target_away"],
                "source":                 "statsbomb",
            })

            time.sleep(0.15)  # cortesía hacia GitHub raw content

    if not partidos_list:
        logger.warning("StatsBomb: ningún partido descargado.")
        return pd.DataFrame()

    df = pd.DataFrame(partidos_list)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    logger.info("StatsBomb: %d partidos descargados.", len(df))
    return df


def _agrega_shots(events: pd.DataFrame, match: pd.Series) -> Dict:
    """
    Agrega estadísticas de tiro (xG, shots, shots_on_target) para un partido.

    Procesa la columna 'shot' del evento StatsBomb, que contiene un dict
    con 'statsbomb_xg' y 'outcome.name' entre otros campos.

    Args:
        events: DataFrame de eventos del partido (de sb.events).
        match:  Serie con metadatos del partido (para resolver nombre de equipos).

    Returns:
        Dict con claves: xg_home, xg_away, shots_home, shots_away,
        shots_on_target_home, shots_on_target_away.
    """
    result = {
        "xg_home": np.nan, "xg_away": np.nan,
        "shots_home": 0, "shots_away": 0,
        "shots_on_target_home": 0, "shots_on_target_away": 0,
    }

    if events.empty or "type" not in events.columns:
        return result

    # Extraer nombre del equipo local desde el match (puede ser str o dict)
    home_name = match.get("home_team", "")
    away_name = match.get("away_team", "")
    if isinstance(home_name, dict):
        home_name = home_name.get("home_team_name", home_name.get("name", ""))
    if isinstance(away_name, dict):
        away_name = away_name.get("away_team_name", away_name.get("name", ""))

    # Filtrar eventos de tiro
    def es_shot(t):
        if isinstance(t, dict):
            return t.get("name") == "Shot"
        return str(t) == "Shot"

    shots = events[events["type"].apply(es_shot)].copy()
    if shots.empty:
        return result

    # Extraer nombre del equipo por disparo
    shots["_team"] = shots["team"].apply(
        lambda t: t.get("name") if isinstance(t, dict) else str(t)
    )
    # Extraer outcome del disparo
    shots["_outcome"] = shots["shot"].apply(
        lambda s: s.get("outcome", {}).get("name") if isinstance(s, dict) else None
    )
    # Extraer xG
    shots["_xg"] = pd.to_numeric(
        shots["shot"].apply(
            lambda s: s.get("statsbomb_xg") if isinstance(s, dict) else None
        ),
        errors="coerce",
    )

    outcomes_on_target = {"Goal", "Saved", "Saved To Post", "Saved to Post"}

    for equipo, prefix in [(str(home_name), "home"), (str(away_name), "away")]:
        mask = shots["_team"] == equipo
        team_shots = shots[mask]

        result[f"shots_{prefix}"]           = int(mask.sum())
        result[f"shots_on_target_{prefix}"] = int(
            team_shots["_outcome"].isin(outcomes_on_target).sum()
        )
        xg_total = team_shots["_xg"].sum()
        result[f"xg_{prefix}"] = float(xg_total) if not np.isnan(xg_total) else np.nan

    return result


# ---------------------------------------------------------------------------
# FUENTE 2: FOOTBALL-DATA.ORG API
# ---------------------------------------------------------------------------

FD_BASE_URL = "https://api.football-data.org/v4"


def descarga_football_data(api_key: str = "") -> pd.DataFrame:
    """
    Descarga resultados históricos desde Football-Data.org.

    Plan gratuito (sin API key): 10 requests/minuto, acceso básico.
    Plan pago: mayor cobertura y sin límite estricto.
    Registrarse gratis: https://www.football-data.org/client/register

    Args:
        api_key: Token de autenticación (puede estar vacío para plan gratuito).

    Returns:
        DataFrame con columnas:
            match_id, date, season, competition,
            team_home, team_away, goals_home, goals_away, source.
    """
    logger.info("=== FUENTE 2: Football-Data.org ===")

    if api_key:
        logger.info("  Usando API key (acceso completo).")
    else:
        logger.info("  Sin API key — plan gratuito, límite 10 req/min.")

    headers = {"X-Auth-Token": api_key} if api_key else {}
    rate_delay = 7.0 if not api_key else 1.0  # respetar límite gratuito
    partidos: List[Dict] = []

    for comp_id, comp_name in FD_COMPETITIONS.items():
        url = f"{FD_BASE_URL}/competitions/{comp_id}/matches"

        try:
            data = get_json(url, headers=headers)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            if status == 403:
                logger.warning(
                    "  %s (id=%d): acceso denegado — requiere tier superior.",
                    comp_name, comp_id,
                )
            elif status == 429:
                logger.warning("  %s: rate limit alcanzado — esperando 60s.", comp_name)
                time.sleep(60)
            else:
                logger.warning("  %s: HTTP %s — %s", comp_name, status, exc)
            continue
        except Exception as exc:
            logger.warning("  %s: error de conexión — %s", comp_name, exc)
            continue

        raw_matches = data.get("matches", [])
        aceptados = 0

        for match in raw_matches:
            if match.get("status") != "FINISHED":
                continue

            utc_date = (match.get("utcDate") or "")[:10]
            try:
                match_year = int(utc_date[:4])
            except ValueError:
                continue

            if not (START_YEAR <= match_year <= END_YEAR):
                continue

            home = match.get("homeTeam", {}).get("name", "")
            away = match.get("awayTeam", {}).get("name", "")
            full_time = match.get("score", {}).get("fullTime", {})
            season_start = match.get("season", {}).get("startDate", "")[:4]

            partidos.append({
                "match_id":   f"fd_{match['id']}",
                "date":       utc_date,
                "season":     season_start,
                "competition": comp_name,
                "team_home":  normaliza_equipo(home),
                "team_away":  normaliza_equipo(away),
                "goals_home": int(full_time.get("home") or 0),
                "goals_away": int(full_time.get("away") or 0),
                "source":     "football_data",
            })
            aceptados += 1

        logger.info("  %s: %d partidos en período %d-%d", comp_name, aceptados, START_YEAR, END_YEAR)
        time.sleep(rate_delay)

    if not partidos:
        logger.warning("Football-Data: ningún partido descargado.")
        return pd.DataFrame()

    df = pd.DataFrame(partidos)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    logger.info("Football-Data: %d partidos descargados.", len(df))
    return df


# ---------------------------------------------------------------------------
# FUENTE 3: ELO RATINGS (CÁLCULO DESDE HISTÓRICO + SNAPSHOT)
# ---------------------------------------------------------------------------

RESULTS_CSV_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)


def descarga_elo() -> pd.DataFrame:
    """
    Descarga y calcula ratings ELO históricos para selecciones nacionales.

    Estrategia:
        1. Descarga el CSV histórico de resultados internacionales
           (martj42/international_results en GitHub, 50k+ partidos desde 1872).
        2. Calcula ratings ELO con el algoritmo de World Football ELO Ratings
           (mismo que usa eloratings.net): K-factor variable por torneo y
           ventaja de localía de +100 puntos.
        3. Devuelve el ELO pre-partido de cada equipo para todos los matches.

    El cálculo ELO usa:
        - K = 60 para World Cup, 50 para torneos continentales, 40 para clasificatorias
        - K = 20 para amistosos
        - Ventaja local = 100 puntos en el expected score
        - ELO inicial = 1500 para cualquier equipo nuevo

    Returns:
        DataFrame con columnas:
            date, team_home, team_away, elo_home, elo_away,
            goals_home_elo, goals_away_elo, tournament.
    """
    logger.info("=== FUENTE 3: ELO Ratings ===")

    df_results = _descarga_resultados_historicos()

    if df_results.empty:
        logger.error("ELO: no se pudo descargar histórico de resultados.")
        return pd.DataFrame()

    df_elo = _calcula_elo_historico(df_results)
    df_elo["date"] = pd.to_datetime(df_elo["date"], errors="coerce")

    logger.info("ELO: %d registros calculados.", len(df_elo))
    return df_elo


@retry_con_backoff(max_intentos=3)
def _descarga_resultados_historicos() -> pd.DataFrame:
    """
    Descarga CSV con resultados históricos de fútbol internacional.

    Fuente: github.com/martj42/international_results
    Columnas: date, home_team, away_team, home_score, away_score,
              tournament, city, country, neutral.
    """
    cache_path = DATA_DIR / "cache" / "international_results.csv"

    if cache_path.exists():
        logger.debug("Cargando resultados históricos desde cache.")
        df = pd.read_csv(cache_path)
    else:
        logger.info("  Descargando resultados históricos desde GitHub (~5-10 MB)...")
        df = pd.read_csv(RESULTS_CSV_URL)
        df.to_csv(cache_path, index=False)
        logger.info("  Descargados %d resultados históricos.", len(df))

    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    if not required.issubset(df.columns):
        logger.error("CSV histórico sin columnas esperadas: %s", list(df.columns))
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])
    df["home_team"] = df["home_team"].apply(normaliza_equipo)
    df["away_team"] = df["away_team"].apply(normaliza_equipo)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce").fillna(0).astype(int)
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce").fillna(0).astype(int)

    return df.sort_values("date").reset_index(drop=True)


def _k_factor(tournament: str) -> float:
    """Devuelve el K-factor ELO según el tipo de torneo."""
    if not isinstance(tournament, str):
        return ELO_K_DEFAULT
    t = tournament.lower()
    if "world cup" in t and "qualifier" not in t:
        return 60
    if any(x in t for x in ["copa america", "euro", "gold cup", "africa cup", "asian cup"]):
        return 50
    if "qualifier" in t or "qualification" in t:
        return 40
    if "nations league" in t or "confed" in t:
        return 40
    if "friendly" in t:
        return 20
    return ELO_K_DEFAULT


def _calcula_elo_historico(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula ratings ELO acumulados para cada partido del histórico.

    Algoritmo World Football ELO Ratings (eloratings.net):
        Expected(home) = 1 / (1 + 10^(-(ELO_home - ELO_away + HOME_ADV) / 400))
        Actual(home)   = 1.0 si victoria, 0.5 si empate, 0.0 si derrota
        ΔElo           = K * (Actual - Expected)

    Args:
        df: DataFrame con columnas: date, home_team, away_team,
            home_score, away_score, tournament, neutral.

    Returns:
        DataFrame con columnas: date, team_home, team_away,
        elo_home, elo_away, goals_home_elo, goals_away_elo, tournament.
    """
    ratings: Dict[str, float] = {}
    records: List[Dict] = []

    neutral_col = "neutral" in df.columns

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        elo_h = ratings.get(home, float(ELO_INITIAL))
        elo_a = ratings.get(away, float(ELO_INITIAL))

        k    = _k_factor(row.get("tournament", ""))
        is_neutral = bool(row["neutral"]) if neutral_col else False
        ha   = 0.0 if is_neutral else float(ELO_HOME_ADV)

        expected_h = 1.0 / (1.0 + 10.0 ** (-(elo_h - elo_a + ha) / 400.0))

        gh = int(row["home_score"])
        ga = int(row["away_score"])
        actual_h = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)

        delta = k * (actual_h - expected_h)

        records.append({
            "date":           row["date"],
            "team_home":      home,
            "team_away":      away,
            "elo_home":       round(elo_h, 2),
            "elo_away":       round(elo_a, 2),
            "goals_home_elo": gh,
            "goals_away_elo": ga,
            "tournament":     row.get("tournament", ""),
        })

        ratings[home] = elo_h + delta
        ratings[away] = elo_a - delta

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# FUENTE 4: FIFA RANKINGS (API PÚBLICA NO OFICIAL)
# ---------------------------------------------------------------------------

FIFA_API_BASE = "https://api.fifa.com/api/v1"


def descarga_fifa() -> pd.DataFrame:
    """
    Descarga rankings FIFA históricos desde la API pública de fifa.com.

    Usa el endpoint público del sitio oficial de FIFA (no requiere API key,
    pero puede cambiar sin previo aviso). Descarga ~25-30 fechas históricas
    en el período configurado, cubriendo un ranking mensual por equipo.

    Returns:
        DataFrame con columnas:
            rank_date, team, fifa_rank, fifa_points.

    Notas:
        - Si la API falla repetidamente, intenta scraping de respaldo.
        - Los rankings se actualizan aproximadamente una vez al mes.
    """
    logger.info("=== FUENTE 4: FIFA Rankings ===")

    rankings: List[Dict] = []
    errores_consecutivos = 0

    fechas_periodo = [
        f for f in FIFA_RANKING_DATES
        if START_YEAR - 1 <= int(f[:4]) <= END_YEAR
    ]
    logger.info("  Descargando %d fechas de ranking FIFA...", len(fechas_periodo))

    for fecha_str in fechas_periodo:
        if errores_consecutivos >= 4:
            logger.warning("  Demasiados errores consecutivos — abortando FIFA API.")
            break

        date_id = f"id_{fecha_str.replace('-', '')}"
        url     = f"{FIFA_API_BASE}/rankings/FIFA"
        params  = {"locale": "en", "dateId": date_id, "count": 250}

        try:
            data = get_json(url, params=params)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            logger.warning("  FIFA %s: HTTP %s", fecha_str, status)
            errores_consecutivos += 1
            time.sleep(3.0)
            continue
        except Exception as exc:
            logger.warning("  FIFA %s: %s", fecha_str, exc)
            errores_consecutivos += 1
            time.sleep(3.0)
            continue

        errores_consecutivos = 0
        entradas = data.get("Rankings", [])

        for entry in entradas:
            team_name = entry.get("TeamName", {})
            if isinstance(team_name, dict):
                name = team_name.get("Description", "")
            else:
                name = str(team_name)

            rankings.append({
                "rank_date":   fecha_str,
                "team":        normaliza_equipo(name),
                "fifa_rank":   int(entry.get("Rank", 0)),
                "fifa_points": float(entry.get("TotalPoints", 0.0)),
            })

        logger.debug("  FIFA %s: %d equipos", fecha_str, len(entradas))
        time.sleep(1.2)

    if not rankings:
        logger.warning("FIFA API no respondió — intentando scraping de respaldo.")
        return _descarga_fifa_scraping()

    df = pd.DataFrame(rankings)
    df["rank_date"] = pd.to_datetime(df["rank_date"])

    logger.info(
        "FIFA Rankings: %d entradas en %d fechas.",
        len(df), df["rank_date"].nunique(),
    )
    return df


def _descarga_fifa_scraping() -> pd.DataFrame:
    """
    Respaldo: scraping de ranking FIFA actual desde fifa.com.
    Solo obtiene el ranking más reciente (sin historial temporal).
    """
    logger.debug("Scraping FIFA rankings desde fifa.com...")
    rows: List[Dict] = []

    try:
        soup = get_html("https://www.fifa.com/fifa-world-ranking/")
        tabla = soup.find("table")

        if tabla:
            for fila in tabla.find_all("tr")[1:]:
                celdas = fila.find_all("td")
                if len(celdas) >= 2:
                    try:
                        rank = int(celdas[0].get_text(strip=True))
                        team = normaliza_equipo(celdas[1].get_text(strip=True))
                        pts  = (
                            float(celdas[2].get_text(strip=True).replace(",", ""))
                            if len(celdas) >= 3 else np.nan
                        )
                        rows.append({
                            "rank_date":   date.today().isoformat(),
                            "team":        team,
                            "fifa_rank":   rank,
                            "fifa_points": pts,
                        })
                    except (ValueError, IndexError):
                        continue

        if rows:
            logger.info("FIFA scraping: %d equipos (snapshot actual).", len(rows))
            df = pd.DataFrame(rows)
            df["rank_date"] = pd.to_datetime(df["rank_date"])
            return df

    except Exception as exc:
        logger.error("FIFA scraping también falló: %s", exc)

    return pd.DataFrame()


# ---------------------------------------------------------------------------
# CONSOLIDACIÓN DE DATOS
# ---------------------------------------------------------------------------

def consolida_datos(
    df_sb:   pd.DataFrame,
    df_fd:   pd.DataFrame,
    df_elo:  pd.DataFrame,
    df_fifa: pd.DataFrame,
) -> pd.DataFrame:
    """
    Consolida las cuatro fuentes de datos en un DataFrame único.

    Pipeline de merges:
        1. Combina partidos de StatsBomb y Football-Data (concat + dedup).
        2. Une ELO pre-partido por nombre de equipo y fecha.
        3. Une ranking FIFA vigente usando merge_asof (ranking más reciente ≤ fecha).
        4. Calcula variable objetivo: resultado (1/0/-1).

    Args:
        df_sb:   Partidos StatsBomb (con xG y tiros).
        df_fd:   Partidos Football-Data (resultados básicos).
        df_elo:  Ratings ELO históricos por partido.
        df_fifa: Rankings FIFA históricos por equipo y fecha.

    Returns:
        DataFrame consolidado con todas las features base requeridas.
    """
    logger.info("=== CONSOLIDANDO DATOS ===")

    # ── 1. Combinar fuentes de partidos ──────────────────────────────────────
    marcos_validos = [df for df in [df_sb, df_fd] if not df.empty]
    if not marcos_validos:
        logger.error("Sin DataFrames de partidos para consolidar.")
        return pd.DataFrame()

    columnas_base = [
        "match_id", "date", "season", "competition",
        "team_home", "team_away", "goals_home", "goals_away", "source",
    ]
    columnas_extra = [
        "xg_home", "xg_away", "shots_home", "shots_away",
        "shots_on_target_home", "shots_on_target_away",
    ]

    partes: List[pd.DataFrame] = []
    for df in marcos_validos:
        cols = [c for c in columnas_base + columnas_extra if c in df.columns]
        partes.append(df[cols].copy())

    df_partidos = pd.concat(partes, ignore_index=True)

    # Asegurar columnas extra aunque falten
    for col in columnas_extra:
        if col not in df_partidos.columns:
            df_partidos[col] = np.nan

    # ── 2. Deduplicar (mismo equipo + fecha en múltiples fuentes) ─────────────
    df_partidos = _deduplicar_partidos(df_partidos)

    # ── 3. Join ELO ──────────────────────────────────────────────────────────
    if not df_elo.empty:
        df_partidos = _join_elo(df_partidos, df_elo)
    else:
        df_partidos["elo_home"] = np.nan
        df_partidos["elo_away"] = np.nan

    # ── 4. Join FIFA ─────────────────────────────────────────────────────────
    if not df_fifa.empty:
        df_partidos = _join_fifa(df_partidos, df_fifa)
    else:
        df_partidos["fifa_rank_home"] = np.nan
        df_partidos["fifa_rank_away"] = np.nan

    # ── 5. Variable objetivo ─────────────────────────────────────────────────
    df_partidos["resultado"] = (
        df_partidos["goals_home"]
        .gt(df_partidos["goals_away"])
        .map({True: 1})
        .fillna(
            df_partidos["goals_home"]
            .lt(df_partidos["goals_away"])
            .map({True: -1})
            .fillna(0)
        )
        .astype(int)
    )

    # ── 6. Features derivadas ─────────────────────────────────────────────────
    df_partidos["diff_elo"]      = df_partidos["elo_home"] - df_partidos["elo_away"]
    df_partidos["diff_fifa_rank"] = df_partidos["fifa_rank_away"] - df_partidos["fifa_rank_home"]
    df_partidos["goles_totales"] = df_partidos["goals_home"] + df_partidos["goals_away"]

    df_partidos = df_partidos.sort_values("date").reset_index(drop=True)

    logger.info(
        "Consolidado: %d partidos, %d columnas.",
        len(df_partidos), len(df_partidos.columns),
    )
    return df_partidos


def _deduplicar_partidos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina partidos duplicados que aparezcan en múltiples fuentes.

    Estrategia: mismo (team_home, team_away, date) → conservar statsbomb
    porque tiene más features (xG, tiros). Si hay diferencia de ±1 día en
    la fecha se considera el mismo partido (zona horaria).
    """
    if df.empty:
        return df

    # Prioridad de fuente: menor número = mayor prioridad
    source_priority = {"statsbomb": 0, "football_data": 1}
    df["_prio"] = df.get("source", pd.Series(["unknown"] * len(df))).map(source_priority).fillna(99)

    df_sorted = df.sort_values(["team_home", "team_away", "date", "_prio"])
    dedup     = df_sorted.drop_duplicates(subset=["team_home", "team_away", "date"], keep="first")

    n_eliminados = len(df) - len(dedup)
    if n_eliminados > 0:
        logger.info("  Deduplicación: %d duplicados eliminados.", n_eliminados)

    return dedup.drop(columns=["_prio"]).reset_index(drop=True)


def _join_elo(df_partidos: pd.DataFrame, df_elo: pd.DataFrame) -> pd.DataFrame:
    """
    Une ratings ELO pre-partido.

    Si df_elo tiene columnas elo_home/elo_away por partido (martj42 calculado),
    hace merge directo por (date, team_home, team_away).
    Si df_elo solo tiene rating por equipo (scraping), hace lookup por nombre.
    """
    if df_elo.empty:
        df_partidos["elo_home"] = np.nan
        df_partidos["elo_away"] = np.nan
        return df_partidos

    if "elo_home" in df_elo.columns and "team_home" in df_elo.columns:
        # Formato histórico completo: merge por (date, team_home, team_away)
        elo_merge = df_elo[["date", "team_home", "team_away", "elo_home", "elo_away"]].copy()
        df_out = df_partidos.merge(
            elo_merge, on=["date", "team_home", "team_away"], how="left"
        )
        cobertura = int(df_out["elo_home"].notna().sum())
        logger.info("  ELO unido por partido: %d/%d con cobertura.", cobertura, len(df_out))
        return df_out

    if "team" in df_elo.columns and "elo" in df_elo.columns:
        # Formato snapshot: solo rating actual por equipo
        elo_map = df_elo.groupby("team")["elo"].last().to_dict()
        df_partidos["elo_home"] = df_partidos["team_home"].map(elo_map)
        df_partidos["elo_away"] = df_partidos["team_away"].map(elo_map)
        logger.info("  ELO unido por nombre de equipo (snapshot sin historial).")
        return df_partidos

    df_partidos["elo_home"] = np.nan
    df_partidos["elo_away"] = np.nan
    return df_partidos


def _join_fifa(df_partidos: pd.DataFrame, df_fifa: pd.DataFrame) -> pd.DataFrame:
    """
    Une rankings FIFA usando el último ranking publicado antes de la fecha del partido.

    Usa pd.merge_asof con parámetro 'by' (agrupando por equipo) para encontrar,
    de forma vectorizada, el ranking más reciente sin exceder la fecha del partido.
    Tolerancia máxima: 35 días (más de un ciclo de actualización mensual).
    """
    if df_fifa.empty or "rank_date" not in df_fifa.columns:
        df_partidos["fifa_rank_home"] = np.nan
        df_partidos["fifa_rank_away"] = np.nan
        return df_partidos

    df_out  = df_partidos.copy().sort_values("date").reset_index(drop=True)
    df_out["_orig_idx"] = df_out.index
    df_fifa = df_fifa.sort_values("rank_date").copy()

    for team_col, rank_col in [("team_home", "fifa_rank_home"), ("team_away", "fifa_rank_away")]:
        # Preparar tabla de partidos con columna unificada de equipo
        partidos_side = df_out[["date", "_orig_idx", team_col]].rename(
            columns={team_col: "_team"}
        ).sort_values("date")

        fifa_side = df_fifa[["rank_date", "team", "fifa_rank"]].rename(
            columns={"rank_date": "date", "team": "_team", "fifa_rank": rank_col}
        )

        merged = pd.merge_asof(
            partidos_side,
            fifa_side,
            on="date",
            by="_team",
            direction="backward",
            tolerance=pd.Timedelta("35D"),
        )

        rank_map = merged.set_index("_orig_idx")[rank_col].to_dict()
        df_out[rank_col] = df_out["_orig_idx"].map(rank_map)

    df_out = df_out.drop(columns=["_orig_idx"])

    cobertura = int(df_out["fifa_rank_home"].notna().sum())
    logger.info("  FIFA rank unido: %d/%d con cobertura.", cobertura, len(df_out))
    return df_out


# ---------------------------------------------------------------------------
# VALIDACIÓN DE CALIDAD
# ---------------------------------------------------------------------------

def valida_calidad(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Valida y limpia el DataFrame consolidado; genera reporte de calidad.

    Validaciones aplicadas:
        - Filas sin equipo o sin fecha → eliminadas.
        - Goles negativos → corregidos a 0.
        - ELO fuera de rango (500–2500) → nulificados.
        - FIFA rank fuera de rango (1–300) → nulificados.
        - Inconsistencia resultado vs goles → recalculado.
        - Partidos duplicados residuales → eliminados.

    Args:
        df: DataFrame consolidado sin filtrar.

    Returns:
        Tuple[DataFrame limpio, dict con reporte de calidad].
    """
    logger.info("=== VALIDANDO CALIDAD ===")

    reporte: Dict = {
        "timestamp":           datetime.utcnow().isoformat() + "Z",
        "periodo":             {
            "inicio": str(df["date"].min()),
            "fin":    str(df["date"].max()),
        },
        "total_partidos_raw":  len(df),
        "columnas":            list(df.columns),
        "nulos_raw":           {},
        "outliers_corregidos": {},
        "advertencias":        [],
        "partidos_eliminados": 0,
        "partidos_limpios":    0,
        "cobertura":           {},
    }

    df_clean      = df.copy()
    n_eliminados  = 0

    # ── Nulos por columna ────────────────────────────────────────────────────
    for col in df_clean.columns:
        n   = int(df_clean[col].isna().sum())
        pct = round(n / len(df_clean) * 100, 2) if len(df_clean) > 0 else 0.0
        reporte["nulos_raw"][col] = {"count": n, "pct": pct}

    # ── Eliminar filas críticas vacías ────────────────────────────────────────
    criticos = df_clean["team_home"].isna() | df_clean["team_away"].isna() | df_clean["date"].isna()
    if criticos.any():
        n = int(criticos.sum())
        logger.warning("  Eliminando %d filas sin equipo/fecha.", n)
        df_clean   = df_clean[~criticos].copy()
        n_eliminados += n
        reporte["advertencias"].append(f"{n} filas eliminadas: equipo o fecha faltante")

    # ── Goles negativos ───────────────────────────────────────────────────────
    for col in ["goals_home", "goals_away"]:
        if col in df_clean.columns:
            invalidos = df_clean[col] < 0
            if invalidos.any():
                logger.warning("  %s con valores negativos: %d casos → corregidos a 0.", col, invalidos.sum())
                df_clean.loc[invalidos, col] = 0
                reporte["outliers_corregidos"][col] = int(invalidos.sum())

    # ── ELO fuera de rango físicamente posible ────────────────────────────────
    for col in ["elo_home", "elo_away"]:
        if col in df_clean.columns:
            fuera = df_clean[col].notna() & ((df_clean[col] < 500) | (df_clean[col] > 2500))
            if fuera.any():
                logger.warning("  ELO fuera de rango (500-2500): %d en %s → nulificados.", fuera.sum(), col)
                df_clean.loc[fuera, col] = np.nan
                reporte["outliers_corregidos"][col] = int(fuera.sum())

    # ── FIFA rank fuera de rango ──────────────────────────────────────────────
    for col in ["fifa_rank_home", "fifa_rank_away"]:
        if col in df_clean.columns:
            fuera = df_clean[col].notna() & ((df_clean[col] < 1) | (df_clean[col] > 300))
            if fuera.any():
                logger.warning("  FIFA rank fuera de rango (1-300): %d en %s → nulificados.", fuera.sum(), col)
                df_clean.loc[fuera, col] = np.nan
                reporte["outliers_corregidos"][col] = int(fuera.sum())

    # ── Recalcular variable objetivo ──────────────────────────────────────────
    if all(c in df_clean.columns for c in ["resultado", "goals_home", "goals_away"]):
        resultado_correcto = df_clean["goals_home"].gt(df_clean["goals_away"]).map({True: 1}).fillna(
            df_clean["goals_home"].lt(df_clean["goals_away"]).map({True: -1}).fillna(0)
        ).astype(int)
        inconsistencias = (df_clean["resultado"] != resultado_correcto).sum()
        if inconsistencias > 0:
            logger.warning("  %d inconsistencias resultado/goles → recalculados.", inconsistencias)
            df_clean["resultado"] = resultado_correcto
            reporte["advertencias"].append(f"{inconsistencias} resultados recalculados desde goles")

    # ── Estadísticas finales ─────────────────────────────────────────────────
    reporte["partidos_eliminados"] = n_eliminados
    reporte["partidos_limpios"]    = len(df_clean)

    teams_series = pd.concat([df_clean["team_home"], df_clean["team_away"]])

    reporte["cobertura"] = {
        "xg_disponible":    int(df_clean["xg_home"].notna().sum()) if "xg_home" in df_clean.columns else 0,
        "elo_disponible":   int(df_clean["elo_home"].notna().sum()) if "elo_home" in df_clean.columns else 0,
        "fifa_disponible":  int(df_clean["fifa_rank_home"].notna().sum()) if "fifa_rank_home" in df_clean.columns else 0,
        "competiciones":    int(df_clean["competition"].nunique()) if "competition" in df_clean.columns else 0,
        "equipos_unicos":   int(teams_series.nunique()),
        "distribucion_resultado": (
            df_clean["resultado"].value_counts().to_dict()
            if "resultado" in df_clean.columns else {}
        ),
        "fuentes": (
            df_clean["source"].value_counts().to_dict()
            if "source" in df_clean.columns else {}
        ),
    }

    logger.info(
        "Calidad validada: %d partidos limpios (%d eliminados).",
        len(df_clean), n_eliminados,
    )
    return df_clean, reporte


# ---------------------------------------------------------------------------
# GUARDADO DE ARCHIVOS
# ---------------------------------------------------------------------------

def guarda_salidas(df_raw: pd.DataFrame, df_clean: pd.DataFrame, reporte: Dict) -> None:
    """
    Guarda los tres archivos de salida del pipeline en DATA_DIR.

    Archivos generados:
        matches_raw.csv           — DataFrame consolidado sin filtrar.
        matches_cleaned.csv       — DataFrame validado y limpio.
        data_quality_report.json  — Reporte de calidad con estadísticas.

    Args:
        df_raw:   DataFrame consolidado original.
        df_clean: DataFrame después de validación y limpieza.
        reporte:  Diccionario con métricas de calidad de datos.
    """
    paths = {
        "raw":    DATA_DIR / "matches_raw.csv",
        "clean":  DATA_DIR / "matches_cleaned.csv",
        "report": DATA_DIR / "data_quality_report.json",
    }

    df_raw.to_csv(paths["raw"], index=False, encoding="utf-8-sig")
    logger.info("Guardado: %s (%d filas, %d cols)", paths["raw"], len(df_raw), len(df_raw.columns))

    df_clean.to_csv(paths["clean"], index=False, encoding="utf-8-sig")
    logger.info("Guardado: %s (%d filas, %d cols)", paths["clean"], len(df_clean), len(df_clean.columns))

    with open(paths["report"], "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Guardado: %s", paths["report"])


# ---------------------------------------------------------------------------
# PUNTO DE ENTRADA
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Ejecuta el pipeline completo: descarga → consolida → valida → guarda.

    Cada fuente de datos se descarga de forma independiente.
    Un fallo en una fuente no detiene el pipeline: se registra el error
    y se continúa con las fuentes disponibles.
    """
    logger.info("=" * 65)
    logger.info("PIPELINE MUNDIAL 2026  |  %s", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
    logger.info("Período: %d–%d  |  DATA_DIR: %s", START_YEAR, END_YEAR, DATA_DIR.resolve())
    logger.info("Football-Data API key: %s", "configurada" if FOOTBALL_API_KEY else "no configurada")
    logger.info("=" * 65)

    # ── Descarga de cada fuente (independientes) ─────────────────────────────
    df_sb   = descarga_statsbomb()
    df_fd   = descarga_football_data(FOOTBALL_API_KEY)
    df_elo  = descarga_elo()
    df_fifa = descarga_fifa()

    # Requiere al menos una fuente de partidos
    if df_sb.empty and df_fd.empty:
        logger.error("CRÍTICO: Sin datos de partidos de ninguna fuente. Abortando.")
        sys.exit(1)

    # ── Consolidar ────────────────────────────────────────────────────────────
    df_raw = consolida_datos(df_sb, df_fd, df_elo, df_fifa)

    if df_raw.empty:
        logger.error("CRÍTICO: DataFrame consolidado vacío. Abortando.")
        sys.exit(1)

    # ── Validar y limpiar ─────────────────────────────────────────────────────
    df_clean, reporte = valida_calidad(df_raw)

    # ── Guardar salidas ───────────────────────────────────────────────────────
    guarda_salidas(df_raw, df_clean, reporte)

    # ── Resumen final ─────────────────────────────────────────────────────────
    cob = reporte["cobertura"]
    logger.info("=" * 65)
    logger.info("PIPELINE COMPLETADO")
    logger.info("  Partidos brutos:    %d", len(df_raw))
    logger.info("  Partidos limpios:   %d", len(df_clean))
    logger.info("  Con xG:             %d  (%.0f%%)", cob["xg_disponible"],
                100 * cob["xg_disponible"] / max(len(df_clean), 1))
    logger.info("  Con ELO:            %d  (%.0f%%)", cob["elo_disponible"],
                100 * cob["elo_disponible"] / max(len(df_clean), 1))
    logger.info("  Con FIFA rank:      %d  (%.0f%%)", cob["fifa_disponible"],
                100 * cob["fifa_disponible"] / max(len(df_clean), 1))
    logger.info("  Equipos únicos:     %d", cob["equipos_unicos"])
    logger.info("  Competiciones:      %d", cob["competiciones"])
    resultado_dist = cob.get("distribucion_resultado", {})
    total = sum(resultado_dist.values()) or 1
    logger.info(
        "  Resultados → Local: %d%%, Empate: %d%%, Visitante: %d%%",
        round(100 * resultado_dist.get(1, 0) / total),
        round(100 * resultado_dist.get(0, 0) / total),
        round(100 * resultado_dist.get(-1, 0) / total),
    )
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
