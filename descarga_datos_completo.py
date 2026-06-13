#!/usr/bin/env python3
"""
Pipeline de datos COMPLETO para Modelo Predictivo Mundial 2026.

Expansión de descarga_datos.py con nuevas fuentes:
    - Clima: Open-Meteo (primario, sin key), OpenWeatherMap, NOAA CDO
    - Estadios WC2026: 16 venues con altitud, distancia, capacidad
    - Lesiones: Transfermarkt scraping con cloudscraper
    - Jugadores: valores de mercado desde Transfermarkt
    - StatsBomb: estadísticas de pases y stats por jugador (xG, xA)

Salidas:
    data/matches_full.csv
    data/matches_cleaned.csv
    data/stadiums_data.csv
    data/players_key.csv
    data/injuries_current.csv
    data/climate_data.csv
    data/data_quality_report.json
    data/pipeline.log
"""

import os
import sys
import json
import math
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

try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------

load_dotenv()

DATA_DIR          = Path(os.getenv("DATA_DIR", "./data"))
LOG_LEVEL         = os.getenv("LOG_LEVEL", "INFO").upper()
FOOTBALL_API_KEY  = os.getenv("FOOTBALL_DATA_API_KEY", "")
OPENWEATHER_KEY   = os.getenv("OPENWEATHER_API_KEY", "")
NOAA_TOKEN        = os.getenv("NOAA_TOKEN", "")
START_YEAR        = int(os.getenv("START_YEAR", "2017"))
END_YEAR          = int(os.getenv("END_YEAR", "2025"))
SCRAPE_DELAY      = float(os.getenv("SCRAPE_DELAY", "1.5"))

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
logger = logging.getLogger("mundial2026_completo")

# ---------------------------------------------------------------------------
# CONSTANTES: COMPETICIONES
# ---------------------------------------------------------------------------

SB_NATIONAL_COMPETITIONS = {
    43: "FIFA World Cup",
    55: "UEFA Euro",
    16: "Copa America",
    6:  "Africa Cup of Nations",
    72: "UEFA Nations League",
}

FD_COMPETITIONS = {
    2000: "FIFA World Cup",
    2016: "UEFA Euro",
    2019: "UEFA Nations League",
}

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

ELO_K_FACTORS = {
    "FIFA World Cup":        60,
    "Copa America":          50,
    "UEFA Euro":             50,
    "Africa Cup of Nations": 50,
    "CONMEBOL":              40,
    "UEFA Nations League":   40,
    "Qualifier":             40,
    "Friendly":              20,
}
ELO_K_DEFAULT = 30
ELO_HOME_ADV  = 100
ELO_INITIAL   = 1500

RESULTS_CSV_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
FIFA_API_BASE   = "https://api.fifa.com/api/v1"
FD_BASE_URL     = "https://api.football-data.org/v4"

# ---------------------------------------------------------------------------
# VENUES WC2026 — 16 estadios confirmados
# ---------------------------------------------------------------------------

WC2026_VENUES: Dict[str, Dict] = {
    "metlife": {
        "name": "MetLife Stadium", "city": "East Rutherford", "country": "USA",
        "lat": 40.8135, "lon": -74.0745, "altitude_m": 2, "capacity": 82500,
        "slug": "new-york-red-bulls", "tm_id": 399,
    },
    "sofi": {
        "name": "SoFi Stadium", "city": "Inglewood", "country": "USA",
        "lat": 33.9534, "lon": -118.3390, "altitude_m": 27, "capacity": 70240,
        "slug": "la-galaxy", "tm_id": 1146,
    },
    "att": {
        "name": "AT&T Stadium", "city": "Arlington", "country": "USA",
        "lat": 32.7480, "lon": -97.0929, "altitude_m": 183, "capacity": 80000,
        "slug": "fc-dallas", "tm_id": 776,
    },
    "levis": {
        "name": "Levi's Stadium", "city": "Santa Clara", "country": "USA",
        "lat": 37.4033, "lon": -121.9694, "altitude_m": 14, "capacity": 68500,
        "slug": "san-jose-earthquakes", "tm_id": 777,
    },
    "arrowhead": {
        "name": "Arrowhead Stadium", "city": "Kansas City", "country": "USA",
        "lat": 39.0489, "lon": -94.4839, "altitude_m": 270, "capacity": 76416,
        "slug": "sporting-kansas-city", "tm_id": 1147,
    },
    "gillette": {
        "name": "Gillette Stadium", "city": "Foxborough", "country": "USA",
        "lat": 42.0909, "lon": -71.2643, "altitude_m": 9, "capacity": 65878,
        "slug": "new-england-revolution", "tm_id": 1148,
    },
    "lincoln_financial": {
        "name": "Lincoln Financial Field", "city": "Philadelphia", "country": "USA",
        "lat": 39.9008, "lon": -75.1675, "altitude_m": 10, "capacity": 69328,
        "slug": "philadelphia-union", "tm_id": 1149,
    },
    "hard_rock": {
        "name": "Hard Rock Stadium", "city": "Miami Gardens", "country": "USA",
        "lat": 25.9580, "lon": -80.2389, "altitude_m": 2, "capacity": 65326,
        "slug": "inter-miami-cf", "tm_id": 46859,
    },
    "nrg": {
        "name": "NRG Stadium", "city": "Houston", "country": "USA",
        "lat": 29.6847, "lon": -95.4107, "altitude_m": 13, "capacity": 72220,
        "slug": "houston-dynamo", "tm_id": 1150,
    },
    "lumen_field": {
        "name": "Lumen Field", "city": "Seattle", "country": "USA",
        "lat": 47.5952, "lon": -122.3316, "altitude_m": 2, "capacity": 69000,
        "slug": "seattle-sounders-fc", "tm_id": 1151,
    },
    "mercedes_benz": {
        "name": "Mercedes-Benz Stadium", "city": "Atlanta", "country": "USA",
        "lat": 33.7554, "lon": -84.4010, "altitude_m": 288, "capacity": 71000,
        "slug": "atlanta-united-fc", "tm_id": 36286,
    },
    "bmo_field": {
        "name": "BMO Field", "city": "Toronto", "country": "Canada",
        "lat": 43.6333, "lon": -79.4189, "altitude_m": 76, "capacity": 30000,
        "slug": "toronto-fc", "tm_id": 6386,
    },
    "bc_place": {
        "name": "BC Place", "city": "Vancouver", "country": "Canada",
        "lat": 49.2767, "lon": -123.1118, "altitude_m": 0, "capacity": 54500,
        "slug": "vancouver-whitecaps-fc", "tm_id": 3614,
    },
    "commonwealth": {
        "name": "Commonwealth Stadium", "city": "Edmonton", "country": "Canada",
        "lat": 53.4642, "lon": -113.4980, "altitude_m": 671, "capacity": 56302,
        "slug": "fc-edmonton", "tm_id": 14112,
    },
    "azteca": {
        "name": "Estadio Azteca", "city": "Mexico City", "country": "Mexico",
        "lat": 19.3029, "lon": -99.1505, "altitude_m": 2224, "capacity": 87523,
        "slug": "club-america", "tm_id": 724,
    },
    "bbva": {
        "name": "Estadio BBVA", "city": "Monterrey", "country": "Mexico",
        "lat": 25.6694, "lon": -100.3102, "altitude_m": 550, "capacity": 53500,
        "slug": "cf-monterrey", "tm_id": 3002,
    },
}

# Transfermarkt team IDs para scraping de lesiones y jugadores.
# IDs verificados con el patrón de URL: transfermarkt.com/{slug}/kader/verein/{id}
# El rango 3375-3450 corresponde a selecciones nacionales europeas/globales en TM.
TM_TEAM_IDS: Dict[str, int] = {
    "argentina":     3437,
    "france":        3377,
    "brazil":        3439,
    "england":       3381,
    "spain":         3376,
    "germany":       3378,
    "portugal":      3380,
    "netherlands":   3379,
    "belgium":       3382,
    "italy":         3375,
    "croatia":       3384,
    "morocco":       3411,
    "mexico":        3397,
    "united states": 3505,
    "canada":        3600,
    "japan":         3404,
    "south korea":   3402,
    "uruguay":       3389,
    "colombia":      3390,
    "senegal":       3412,
    "australia":     3450,
    "denmark":       3385,
    "poland":        3387,
    "switzerland":   3383,
    "ecuador":       3388,
}

TM_TEAM_SLUGS: Dict[str, str] = {
    "argentina":     "argentinien",
    "france":        "frankreich",
    "brazil":        "brasilien",
    "england":       "england",
    "spain":         "spanien",
    "germany":       "deutschland",
    "portugal":      "portugal",
    "netherlands":   "niederlande",
    "belgium":       "belgien",
    "italy":         "italien",
    "croatia":       "kroatien",
    "morocco":       "marokko",
    "mexico":        "mexiko",
    "united states": "vereinigte-staaten",
    "canada":        "kanada",
    "japan":         "japan",
    "south korea":   "südkorea",
    "uruguay":       "uruguay",
    "colombia":      "kolumbien",
    "senegal":       "senegal",
}

# Coordenadas de ciudades base para equipos nacionales (para distancia de viaje)
TEAM_HOME_CITIES: Dict[str, Tuple[float, float]] = {
    "argentina":     (-34.6037, -58.3816),
    "france":        (48.8566, 2.3522),
    "brazil":        (-15.7797, -47.9297),
    "england":       (51.5074, -0.1278),
    "spain":         (40.4168, -3.7038),
    "germany":       (52.5200, 13.4050),
    "portugal":      (38.7169, -9.1399),
    "netherlands":   (52.3676, 4.9041),
    "belgium":       (50.8503, 4.3517),
    "italy":         (41.9028, 12.4964),
    "croatia":       (45.8150, 15.9819),
    "morocco":       (33.9716, -6.8498),
    "mexico":        (19.4326, -99.1332),
    "united states": (38.9072, -77.0369),
    "canada":        (45.4215, -75.6972),
    "japan":         (35.6762, 139.6503),
    "south korea":   (37.5665, 126.9780),
    "uruguay":       (-34.9011, -56.1645),
    "colombia":      (4.7110, -74.0721),
    "senegal":       (14.6928, -17.4467),
}

# ---------------------------------------------------------------------------
# NORMALIZACIÓN DE NOMBRES DE EQUIPO
# ---------------------------------------------------------------------------

TEAM_ALIASES: Dict[str, str] = {
    "korea republic":                    "south korea",
    "republic of korea":                 "south korea",
    "ir iran":                           "iran",
    "côte d'ivoire":                     "ivory coast",
    "cote d'ivoire":                     "ivory coast",
    "usa":                               "united states",
    "u.s.a.":                            "united states",
    "czechia":                           "czech republic",
    "czech rep.":                        "czech republic",
    "türkiye":                           "turkey",
    "turkiye":                           "turkey",
    "uae":                               "united arab emirates",
    "cape verde islands":                "cape verde",
    "democratic republic of the congo":  "dr congo",
    "trinidad and tobago":               "trinidad & tobago",
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
    """HTTP GET con retry que devuelve JSON parseado."""
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


def _get_cloudscraper() -> Optional[object]:
    """Devuelve instancia de cloudscraper o None si no está disponible."""
    if not CLOUDSCRAPER_AVAILABLE:
        logger.warning("cloudscraper no instalado — pip install cloudscraper")
        return None
    return cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )


def calcula_distancia_haversine(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Calcula la distancia ortodrómica entre dos puntos usando la fórmula haversine.

    Implementación matemática pura (sin API). Precisión ±0.5%, suficiente para
    modelar el impacto del viaje sobre el rendimiento.

    Args:
        lat1, lon1: Coordenadas del punto de origen (grados decimales).
        lat2, lon2: Coordenadas del punto de destino.

    Returns:
        Distancia en kilómetros (float).

    Example:
        >>> calcula_distancia_haversine(-34.60, -58.38, 33.75, -84.40)  # BUE→ATL
        9288.3
    """
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return round(2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 1)


# Alias privado para compatibilidad interna
_haversine = calcula_distancia_haversine


# ---------------------------------------------------------------------------
# GRUPO A — PARTIDOS
# ---------------------------------------------------------------------------

# ── Football-Data.org ────────────────────────────────────────────────────────

def descarga_football_data(api_key: str = "") -> pd.DataFrame:
    """
    Descarga resultados históricos desde Football-Data.org.

    Competiciones: FIFA World Cup (2000), UEFA Euro (2016), Nations League (2019).
    Plan gratuito: 10 req/min. Con key: acceso completo.

    Args:
        api_key: Token de autenticación X-Auth-Token.

    Returns:
        DataFrame con columnas: match_id, date, season, competition,
        team_home, team_away, goals_home, goals_away, source.
    """
    logger.info("=== FUENTE: Football-Data.org ===")

    if api_key:
        logger.info("  API key configurada.")
    else:
        logger.info("  Sin API key — plan gratuito (10 req/min).")

    headers    = {"X-Auth-Token": api_key} if api_key else {}
    rate_delay = 7.0 if not api_key else 1.0
    partidos: List[Dict] = []

    for comp_id, comp_name in FD_COMPETITIONS.items():
        url = f"{FD_BASE_URL}/competitions/{comp_id}/matches"
        try:
            data = get_json(url, headers=headers)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            if status == 403:
                logger.warning("  %s (id=%d): acceso denegado.", comp_name, comp_id)
            elif status == 429:
                logger.warning("  Rate limit — esperando 60s.")
                time.sleep(60)
            else:
                logger.warning("  %s: HTTP %s", comp_name, status)
            continue
        except Exception as exc:
            logger.warning("  %s: error — %s", comp_name, exc)
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

            home      = match.get("homeTeam", {}).get("name", "")
            away      = match.get("awayTeam", {}).get("name", "")
            full_time = match.get("score", {}).get("fullTime", {})
            season_s  = match.get("season", {}).get("startDate", "")[:4]

            partidos.append({
                "match_id":    f"fd_{match['id']}",
                "date":        utc_date,
                "season":      season_s,
                "competition": comp_name,
                "team_home":   normaliza_equipo(home),
                "team_away":   normaliza_equipo(away),
                "goals_home":  int(full_time.get("home") or 0),
                "goals_away":  int(full_time.get("away") or 0),
                "source":      "football_data",
            })
            aceptados += 1

        logger.info("  %s: %d partidos (%d-%d)", comp_name, aceptados, START_YEAR, END_YEAR)
        time.sleep(rate_delay)

    if not partidos:
        logger.warning("Football-Data: ningún partido descargado.")
        return pd.DataFrame()

    df = pd.DataFrame(partidos)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    logger.info("Football-Data: %d partidos.", len(df))
    return df


# ── StatsBomb ────────────────────────────────────────────────────────────────

def descarga_statsbomb() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Descarga eventos de partidos de selecciones desde StatsBomb Open Data.

    Extrae xG, tiros, estadísticas de pase por partido y stats por jugador.

    Returns:
        Tuple (df_matches, df_players) donde:
        - df_matches: match_id, date, season, competition, team_home, team_away,
          goals_home/away, xg_home/away, shots_home/away, shots_on_target_home/away,
          passes_home/away, pass_accuracy_home/away, source.
        - df_players: match_id, team, player_name, xg, xa, shots,
          goals_last5, minutes.
    """
    logger.info("=== FUENTE: StatsBomb Open Data ===")

    try:
        comps = sb.competitions()
    except Exception as exc:
        logger.error("No se pudo acceder a StatsBomb: %s", exc)
        return pd.DataFrame(), pd.DataFrame()

    comps["_season_year"] = pd.to_numeric(
        comps["season_name"].str.extract(r"(\d{4})")[0], errors="coerce"
    )
    mask = (
        comps["competition_id"].isin(SB_NATIONAL_COMPETITIONS.keys()) &
        comps["_season_year"].between(START_YEAR, END_YEAR)
    )
    comps_filtradas = comps[mask].reset_index(drop=True)

    logger.info(
        "StatsBomb competiciones (%d-%d): %d",
        START_YEAR, END_YEAR, len(comps_filtradas),
    )

    if comps_filtradas.empty:
        return pd.DataFrame(), pd.DataFrame()

    partidos_list: List[Dict] = []
    players_list:  List[Dict] = []

    for _, comp_row in comps_filtradas.iterrows():
        cid   = int(comp_row["competition_id"])
        sid   = int(comp_row["season_id"])
        cname = comp_row["competition_name"]
        sname = comp_row["season_name"]

        try:
            matches = sb.matches(competition_id=cid, season_id=sid)
            logger.info("  %s %s — %d partidos", cname, sname, len(matches))
        except Exception as exc:
            logger.warning("  Error partidos %s %s: %s", cname, sname, exc)
            continue

        for i, (_, match) in enumerate(matches.iterrows(), 1):
            mid = int(match["match_id"])
            logger.debug("    Partido %d/%d (id=%d)", i, len(matches), mid)

            try:
                events = sb.events(match_id=mid)
            except Exception as exc:
                logger.debug("    Error eventos %d: %s", mid, exc)
                events = pd.DataFrame()

            home = match.get("home_team", "")
            away = match.get("away_team", "")
            if isinstance(home, dict):
                home = home.get("home_team_name", home.get("name", ""))
            if isinstance(away, dict):
                away = away.get("away_team_name", away.get("name", ""))

            shot_stats = _agrega_shots(events, match)
            pass_stats = _agrega_passes(events, str(home), str(away))

            match_id_str = f"sb_{mid}"
            partidos_list.append({
                "match_id":                match_id_str,
                "date":                    str(match.get("match_date", "")),
                "season":                  sname,
                "competition":             cname,
                "team_home":               normaliza_equipo(str(home)),
                "team_away":               normaliza_equipo(str(away)),
                "goals_home":              int(match.get("home_score") or 0),
                "goals_away":              int(match.get("away_score") or 0),
                "xg_home":                 shot_stats["xg_home"],
                "xg_away":                 shot_stats["xg_away"],
                "shots_home":              shot_stats["shots_home"],
                "shots_away":              shot_stats["shots_away"],
                "shots_on_target_home":    shot_stats["shots_on_target_home"],
                "shots_on_target_away":    shot_stats["shots_on_target_away"],
                "passes_home":             pass_stats["passes_home"],
                "passes_away":             pass_stats["passes_away"],
                "pass_accuracy_home":      pass_stats["pass_accuracy_home"],
                "pass_accuracy_away":      pass_stats["pass_accuracy_away"],
                "source":                  "statsbomb",
            })

            player_rows = _agrega_player_stats(events, mid, str(home), str(away))
            players_list.extend(player_rows)

            time.sleep(0.15)

    if not partidos_list:
        logger.warning("StatsBomb: ningún partido.")
        return pd.DataFrame(), pd.DataFrame()

    df_matches = pd.DataFrame(partidos_list)
    df_matches["date"] = pd.to_datetime(df_matches["date"], errors="coerce")

    df_players = pd.DataFrame(players_list) if players_list else pd.DataFrame(
        columns=["match_id", "team", "player_name", "xg", "xa", "shots", "goals_last5", "minutes"]
    )

    logger.info(
        "StatsBomb: %d partidos, %d registros de jugadores.",
        len(df_matches), len(df_players),
    )
    return df_matches, df_players


def _agrega_shots(events: pd.DataFrame, match: pd.Series) -> Dict:
    """Agrega xG, tiros y tiros a puerta por equipo para un partido."""
    result = {
        "xg_home": np.nan, "xg_away": np.nan,
        "shots_home": 0, "shots_away": 0,
        "shots_on_target_home": 0, "shots_on_target_away": 0,
    }
    if events.empty or "type" not in events.columns:
        return result

    home_name = match.get("home_team", "")
    away_name = match.get("away_team", "")
    if isinstance(home_name, dict):
        home_name = home_name.get("home_team_name", home_name.get("name", ""))
    if isinstance(away_name, dict):
        away_name = away_name.get("away_team_name", away_name.get("name", ""))

    def es_shot(t):
        if isinstance(t, dict):
            return t.get("name") == "Shot"
        return str(t) == "Shot"

    shots = events[events["type"].apply(es_shot)].copy()
    if shots.empty:
        return result

    shots["_team"] = shots["team"].apply(
        lambda t: t.get("name") if isinstance(t, dict) else str(t)
    )
    shots["_outcome"] = shots["shot"].apply(
        lambda s: s.get("outcome", {}).get("name") if isinstance(s, dict) else None
    )
    shots["_xg"] = pd.to_numeric(
        shots["shot"].apply(
            lambda s: s.get("statsbomb_xg") if isinstance(s, dict) else None
        ),
        errors="coerce",
    )

    outcomes_on_target = {"Goal", "Saved", "Saved To Post", "Saved to Post"}

    for equipo, prefix in [(str(home_name), "home"), (str(away_name), "away")]:
        mask       = shots["_team"] == equipo
        team_shots = shots[mask]
        result[f"shots_{prefix}"]           = int(mask.sum())
        result[f"shots_on_target_{prefix}"] = int(
            team_shots["_outcome"].isin(outcomes_on_target).sum()
        )
        xg_total = team_shots["_xg"].sum()
        result[f"xg_{prefix}"] = float(xg_total) if not np.isnan(xg_total) else np.nan

    return result


def _agrega_passes(events: pd.DataFrame, home_name: str, away_name: str) -> Dict:
    """Agrega estadísticas de pases (total y precisión) por equipo."""
    result = {
        "passes_home": 0, "passes_away": 0,
        "pass_accuracy_home": np.nan, "pass_accuracy_away": np.nan,
    }
    if events.empty or "type" not in events.columns:
        return result

    def es_pass(t):
        if isinstance(t, dict):
            return t.get("name") == "Pass"
        return str(t) == "Pass"

    passes = events[events["type"].apply(es_pass)].copy()
    if passes.empty:
        return result

    passes["_team"] = passes["team"].apply(
        lambda t: t.get("name") if isinstance(t, dict) else str(t)
    )
    passes["_outcome"] = passes["pass"].apply(
        lambda p: p.get("outcome", {}).get("name") if isinstance(p, dict) else None
    )

    for equipo, prefix in [(home_name, "home"), (away_name, "away")]:
        team_passes = passes[passes["_team"] == equipo]
        total = len(team_passes)
        result[f"passes_{prefix}"] = total
        if total > 0:
            incomplete = team_passes["_outcome"].notna().sum()
            result[f"pass_accuracy_{prefix}"] = round(
                (total - incomplete) / total * 100, 2
            )

    return result


def _agrega_player_stats(
    events: pd.DataFrame,
    match_id: int,
    home_name: str,
    away_name: str,
) -> List[Dict]:
    """
    Extrae estadísticas por jugador: xG, xA, tiros, minutos jugados.

    goals_last5 se calcula externamente durante la consolidación;
    aquí se fija en NaN como placeholder.
    """
    rows: List[Dict] = []
    if events.empty or "type" not in events.columns:
        return rows

    match_id_str = f"sb_{match_id}"

    def get_name(t):
        if isinstance(t, dict):
            return t.get("name") if isinstance(t, dict) else str(t)
        return str(t)

    def get_team(t):
        if isinstance(t, dict):
            return t.get("name", "")
        return str(t)

    # xG por jugador desde shots
    def es_shot(t):
        return get_name(t) == "Shot"

    shot_events = events[events["type"].apply(es_shot)].copy()
    player_xg: Dict[Tuple[str, str], float] = {}
    player_shots: Dict[Tuple[str, str], int] = {}

    for _, ev in shot_events.iterrows():
        pname = str(ev.get("player", "") or "")
        if isinstance(ev.get("player"), dict):
            pname = ev["player"].get("name", "")
        team  = get_team(ev.get("team", ""))
        key   = (team, pname)
        xg_val = np.nan
        if isinstance(ev.get("shot"), dict):
            xg_val = float(ev["shot"].get("statsbomb_xg") or 0)
        player_xg[key]    = player_xg.get(key, 0.0) + (xg_val if not np.isnan(xg_val) else 0.0)
        player_shots[key] = player_shots.get(key, 0) + 1

    # xA por jugador desde passes que llevan a gol
    def es_pass(t):
        return get_name(t) == "Pass"

    pass_events = events[events["type"].apply(es_pass)].copy()
    player_xa: Dict[Tuple[str, str], float] = {}

    for _, ev in pass_events.iterrows():
        pdata = ev.get("pass", {})
        if not isinstance(pdata, dict):
            continue
        goal_assist = pdata.get("goal_assist") or False
        xg_assist   = float(pdata.get("shot_assist_xg") or 0.0)
        if goal_assist or xg_assist > 0:
            pname = ""
            if isinstance(ev.get("player"), dict):
                pname = ev["player"].get("name", "")
            else:
                pname = str(ev.get("player", "") or "")
            team  = get_team(ev.get("team", ""))
            key   = (team, pname)
            player_xa[key] = player_xa.get(key, 0.0) + xg_assist

    # Minutos jugados desde lineup / player off events
    player_minutes: Dict[Tuple[str, str], float] = {}

    def es_player_off(t):
        return get_name(t) in ("Substitution", "Half Start", "Half End")

    # Aproximación: cualquier jugador con evento = jugó ~90 min en ausencia de datos exactos
    all_players: Dict[Tuple[str, str], bool] = {}
    if "player" in events.columns:
        for _, ev in events.iterrows():
            pname = ""
            if isinstance(ev.get("player"), dict):
                pname = ev["player"].get("name", "")
            else:
                pname = str(ev.get("player", "") or "")
            if not pname or pname == "nan":
                continue
            team = get_team(ev.get("team", ""))
            all_players[(team, pname)] = True

    for key in all_players:
        player_minutes[key] = 90.0  # placeholder

    # Combinar en filas
    all_keys = set(list(player_xg.keys()) + list(player_xa.keys()) + list(player_minutes.keys()))
    for (team, pname) in all_keys:
        if not pname:
            continue
        rows.append({
            "match_id":    match_id_str,
            "team":        normaliza_equipo(team),
            "player_name": pname,
            "xg":          round(player_xg.get((team, pname), 0.0), 4),
            "xa":          round(player_xa.get((team, pname), 0.0), 4),
            "shots":       player_shots.get((team, pname), 0),
            "goals_last5": np.nan,
            "minutes":     player_minutes.get((team, pname), 90.0),
        })

    return rows


# ---------------------------------------------------------------------------
# GRUPO B — RANKINGS
# ---------------------------------------------------------------------------

# ── ELO ─────────────────────────────────────────────────────────────────────

def descarga_elo() -> pd.DataFrame:
    """
    Calcula ratings ELO históricos usando el CSV de martj42/international_results.

    Returns:
        DataFrame con columnas: date, team_home, team_away,
        elo_home, elo_away, goals_home_elo, goals_away_elo, tournament.
    """
    logger.info("=== FUENTE: ELO Ratings ===")

    df_results = _descarga_resultados_historicos()
    if df_results.empty:
        logger.error("ELO: no se pudo descargar histórico.")
        return pd.DataFrame()

    df_elo = _calcula_elo_historico(df_results)
    df_elo["date"] = pd.to_datetime(df_elo["date"], errors="coerce")
    logger.info("ELO: %d registros calculados.", len(df_elo))
    return df_elo


@retry_con_backoff(max_intentos=3)
def _descarga_resultados_historicos() -> pd.DataFrame:
    """Descarga CSV histórico de resultados internacionales (martj42)."""
    cache_path = DATA_DIR / "cache" / "international_results.csv"

    if cache_path.exists():
        logger.debug("Cargando resultados desde cache.")
        df = pd.read_csv(cache_path)
    else:
        logger.info("  Descargando resultados históricos (~5-10 MB)...")
        df = pd.read_csv(RESULTS_CSV_URL)
        df.to_csv(cache_path, index=False)
        logger.info("  Descargados %d resultados.", len(df))

    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    if not required.issubset(df.columns):
        logger.error("CSV sin columnas esperadas: %s", list(df.columns))
        return pd.DataFrame()

    df["date"]       = pd.to_datetime(df["date"], errors="coerce")
    df               = df.dropna(subset=["date", "home_team", "away_team"])
    df["home_team"]  = df["home_team"].apply(normaliza_equipo)
    df["away_team"]  = df["away_team"].apply(normaliza_equipo)
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
    Calcula ratings ELO acumulados con el algoritmo de World Football ELO Ratings.

    Expected(home) = 1 / (1 + 10^(-(ELO_h - ELO_a + HOME_ADV) / 400))
    Delta = K * (Actual - Expected)
    """
    ratings: Dict[str, float] = {}
    records: List[Dict] = []
    neutral_col = "neutral" in df.columns

    for _, row in df.iterrows():
        home  = row["home_team"]
        away  = row["away_team"]
        elo_h = ratings.get(home, float(ELO_INITIAL))
        elo_a = ratings.get(away, float(ELO_INITIAL))
        k     = _k_factor(row.get("tournament", ""))
        is_neutral = bool(row["neutral"]) if neutral_col else False
        ha    = 0.0 if is_neutral else float(ELO_HOME_ADV)
        expected_h = 1.0 / (1.0 + 10.0 ** (-(elo_h - elo_a + ha) / 400.0))
        gh    = int(row["home_score"])
        ga    = int(row["away_score"])
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


# ── FIFA Rankings ────────────────────────────────────────────────────────────

def descarga_fifa() -> pd.DataFrame:
    """
    Descarga rankings FIFA históricos desde la API pública de fifa.com.

    Returns:
        DataFrame con columnas: rank_date, team, fifa_rank, fifa_points.
    """
    logger.info("=== FUENTE: FIFA Rankings ===")

    rankings: List[Dict] = []
    errores_consecutivos = 0

    fechas_periodo = [
        f for f in FIFA_RANKING_DATES
        if START_YEAR - 1 <= int(f[:4]) <= END_YEAR
    ]
    logger.info("  Descargando %d fechas FIFA...", len(fechas_periodo))

    for fecha_str in fechas_periodo:
        if errores_consecutivos >= 4:
            logger.warning("  Demasiados errores FIFA — abortando.")
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
        for entry in data.get("Rankings", []):
            team_name = entry.get("TeamName", {})
            name = team_name.get("Description", "") if isinstance(team_name, dict) else str(team_name)
            rankings.append({
                "rank_date":   fecha_str,
                "team":        normaliza_equipo(name),
                "fifa_rank":   int(entry.get("Rank", 0)),
                "fifa_points": float(entry.get("TotalPoints", 0.0)),
            })

        logger.debug("  FIFA %s: %d equipos", fecha_str, len(data.get("Rankings", [])))
        time.sleep(1.2)

    if not rankings:
        logger.warning("FIFA API sin respuesta — intentando scraping.")
        return _descarga_fifa_scraping()

    df = pd.DataFrame(rankings)
    df["rank_date"] = pd.to_datetime(df["rank_date"])
    logger.info("FIFA: %d entradas en %d fechas.", len(df), df["rank_date"].nunique())
    return df


def _descarga_fifa_scraping() -> pd.DataFrame:
    """Respaldo: scraping del ranking FIFA actual desde fifa.com."""
    logger.debug("Scraping FIFA desde fifa.com...")
    rows: List[Dict] = []
    try:
        soup  = get_html("https://www.fifa.com/fifa-world-ranking/")
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
            logger.info("FIFA scraping: %d equipos.", len(rows))
            df = pd.DataFrame(rows)
            df["rank_date"] = pd.to_datetime(df["rank_date"])
            return df
    except Exception as exc:
        logger.error("FIFA scraping fallido: %s", exc)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# GRUPO C — CLIMA
# ---------------------------------------------------------------------------

OPENMETEO_ARCHIVE_URL   = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
OPENWEATHER_URL         = "https://api.openweathermap.org/data/2.5/weather"
NOAA_BASE_URL           = "https://www.ncei.noaa.gov/cdo-web/api/v2/"

# Años a promediar para climatología Jun-Jul
CLIMA_YEARS = list(range(2019, 2024))  # 5 años: 2019-2023


def descarga_clima_openmeteo() -> pd.DataFrame:
    """
    Descarga climatología histórica Jun-Jul para los 16 venues WC2026.

    Fuente: Open-Meteo Archive API (gratuita, sin key).
    Para cada venue: promedia los últimos 5 años de datos diarios de junio-julio.
    Incluye temperatura, precipitación, viento, humedad y código de clima.
    También obtiene altitud desde el endpoint de elevación de Open-Meteo.

    Returns:
        DataFrame con columnas: venue_id, name, city, country, lat, lon,
        altitude_m, temp_avg_c, temp_max_avg_c, temp_min_avg_c,
        humidity_pct, wind_kmh, precip_mm, rain_days, weathercode_mode.
    """
    logger.info("=== FUENTE: Open-Meteo Climate (primario, sin key) ===")

    cache_path = DATA_DIR / "cache" / "climate_openmeteo.csv"
    if cache_path.exists():
        logger.info("  Cargando clima desde cache.")
        return pd.read_csv(cache_path)

    rows: List[Dict] = []

    for venue_id, venue in WC2026_VENUES.items():
        lat = venue["lat"]
        lon = venue["lon"]
        logger.info("  Procesando clima: %s (%s)", venue["name"], venue["city"])

        # Obtener altitud desde Open-Meteo si no está en el dict o es 0
        altitude = venue.get("altitude_m", 0)
        try:
            elev_data = get_json(
                OPENMETEO_ELEVATION_URL,
                params={"latitude": lat, "longitude": lon},
            )
            elevation_list = elev_data.get("elevation", [altitude])
            if elevation_list:
                altitude = float(elevation_list[0])
            time.sleep(0.5)
        except Exception as exc:
            logger.debug("  Altitud Open-Meteo %s: %s", venue_id, exc)

        # Acumular datos diarios de Jun-Jul para cada año
        daily_records: Dict[str, List[float]] = {
            "temp_max": [], "temp_min": [], "precip": [],
            "wind": [], "humidity": [], "weathercode": [],
        }

        for year in CLIMA_YEARS:
            start_date = f"{year}-06-01"
            end_date   = f"{year}-07-31"

            # Llamada 1: variables diarias básicas
            params_daily = {
                "latitude":   lat,
                "longitude":  lon,
                "start_date": start_date,
                "end_date":   end_date,
                "daily":      "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,weathercode",
                "timezone":   "UTC",
            }
            try:
                data = get_json(OPENMETEO_ARCHIVE_URL, params=params_daily)
                d = data.get("daily", {})
                daily_records["temp_max"].extend(
                    [v for v in (d.get("temperature_2m_max") or []) if v is not None]
                )
                daily_records["temp_min"].extend(
                    [v for v in (d.get("temperature_2m_min") or []) if v is not None]
                )
                daily_records["precip"].extend(
                    [v for v in (d.get("precipitation_sum") or []) if v is not None]
                )
                daily_records["wind"].extend(
                    [v for v in (d.get("windspeed_10m_max") or []) if v is not None]
                )
                daily_records["weathercode"].extend(
                    [v for v in (d.get("weathercode") or []) if v is not None]
                )
                time.sleep(0.5)
            except Exception as exc:
                logger.debug("  Open-Meteo daily %s %d: %s", venue_id, year, exc)

            # Llamada 2: humedad horaria → agregar a diaria
            params_hourly = {
                "latitude":   lat,
                "longitude":  lon,
                "start_date": start_date,
                "end_date":   end_date,
                "hourly":     "relativehumidity_2m",
                "timezone":   "UTC",
            }
            try:
                data_h = get_json(OPENMETEO_ARCHIVE_URL, params=params_hourly)
                h = data_h.get("hourly", {})
                hum_vals = [v for v in (h.get("relativehumidity_2m") or []) if v is not None]
                if hum_vals:
                    times_h = h.get("time", [])
                    # Agrupar 24 horas → media diaria
                    for day_idx in range(len(hum_vals) // 24):
                        chunk = hum_vals[day_idx * 24: (day_idx + 1) * 24]
                        if chunk:
                            daily_records["humidity"].append(float(np.mean(chunk)))
                time.sleep(0.5)
            except Exception as exc:
                logger.debug("  Open-Meteo hourly %s %d: %s", venue_id, year, exc)

        # Calcular promedios
        def safe_mean(lst):
            return round(float(np.mean(lst)), 2) if lst else np.nan

        def safe_mode(lst):
            if not lst:
                return np.nan
            vals, counts = np.unique(lst, return_counts=True)
            return int(vals[np.argmax(counts)])

        temp_max_vals = daily_records["temp_max"]
        temp_min_vals = daily_records["temp_min"]
        temp_avg_vals = [(h + l) / 2 for h, l in zip(temp_max_vals, temp_min_vals)]
        rain_days = int(sum(1 for p in daily_records["precip"] if p > 1.0))

        rows.append({
            "venue_id":         venue_id,
            "name":             venue["name"],
            "city":             venue["city"],
            "country":          venue["country"],
            "lat":              lat,
            "lon":              lon,
            "altitude_m":       round(altitude, 1),
            "temp_avg_c":       safe_mean(temp_avg_vals),
            "temp_max_avg_c":   safe_mean(temp_max_vals),
            "temp_min_avg_c":   safe_mean(temp_min_vals),
            "humidity_pct":     safe_mean(daily_records["humidity"]),
            "wind_kmh":         safe_mean(daily_records["wind"]),
            "precip_mm":        safe_mean(daily_records["precip"]),
            "rain_days":        rain_days,
            "weathercode_mode": safe_mode(daily_records["weathercode"]),
            "years_averaged":   len(CLIMA_YEARS),
            "source":           "open-meteo",
        })

        logger.debug("  %s: temp_avg=%.1f°C, hum=%.0f%%", venue["name"],
                     rows[-1]["temp_avg_c"] or 0, rows[-1]["humidity_pct"] or 0)

    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    logger.info("Open-Meteo: %d venues procesados.", len(df))
    return df


def descarga_clima_openweather(api_key: str = "") -> pd.DataFrame:
    """
    Obtiene condiciones climáticas actuales para los 16 venues WC2026.

    Usa la API de OpenWeatherMap (requiere key gratuita).
    Útil para planificación de partidos próximos.

    Args:
        api_key: API key de OpenWeatherMap. Si está vacía, retorna DataFrame vacío.

    Returns:
        DataFrame con columnas: venue_id, temp_c, humidity_pct, wind_kmh,
        weather_main, weather_description, timestamp_utc.
    """
    logger.info("=== FUENTE: OpenWeatherMap (condiciones actuales) ===")

    if not api_key:
        logger.info("  OPENWEATHER_API_KEY no configurada — omitiendo.")
        return pd.DataFrame()

    rows: List[Dict] = []

    for venue_id, venue in WC2026_VENUES.items():
        params = {
            "lat":   venue["lat"],
            "lon":   venue["lon"],
            "appid": api_key,
            "units": "metric",
        }
        try:
            data = get_json(OPENWEATHER_URL, params=params)
            main    = data.get("main", {})
            wind    = data.get("wind", {})
            weather = data.get("weather", [{}])[0]

            rows.append({
                "venue_id":            venue_id,
                "name":                venue["name"],
                "city":                venue["city"],
                "temp_c":              float(main.get("temp", np.nan)),
                "humidity_pct":        float(main.get("humidity", np.nan)),
                "wind_kmh":            round(float(wind.get("speed", 0)) * 3.6, 2),
                "weather_main":        weather.get("main", ""),
                "weather_description": weather.get("description", ""),
                "timestamp_utc":       datetime.utcnow().isoformat() + "Z",
                "source":              "openweathermap",
            })
            logger.debug("  %s: %.1f°C, %s", venue["name"], rows[-1]["temp_c"],
                         rows[-1]["weather_main"])
            time.sleep(0.5)

        except Exception as exc:
            logger.warning("  OpenWeather %s: %s", venue_id, exc)

    df = pd.DataFrame(rows)
    logger.info("OpenWeatherMap: %d venues.", len(df))
    return df


def descarga_clima_noaa() -> pd.DataFrame:
    """
    Descarga normales climáticas mensuales desde NOAA CDO API.

    Requiere token gratuito de https://www.ncdc.noaa.gov/cdo-web/token.
    Busca la estación GHCND más cercana a cada venue y obtiene normales de Jun-Jul.
    Retorna DataFrame vacío con log informativo si el token no está configurado.

    Returns:
        DataFrame con columnas: venue_id, station_id, station_name,
        month, temp_normal_c, precip_normal_mm, source.
    """
    logger.info("=== FUENTE: NOAA CDO API ===")

    if not NOAA_TOKEN:
        logger.info("  NOAA_TOKEN no configurado — omitiendo. "
                    "Obtén uno gratis en https://www.ncdc.noaa.gov/cdo-web/token")
        return pd.DataFrame()

    headers = {"token": NOAA_TOKEN}
    rows: List[Dict] = []

    for venue_id, venue in WC2026_VENUES.items():
        lat = venue["lat"]
        lon = venue["lon"]

        # Buscar estación GHCND más cercana
        station_id   = None
        station_name = ""
        try:
            params_station = {
                "datasetid":  "NORMAL_MLY",
                "datatypeid": "MLY-TMAX-NORMAL",
                "extent":     f"{lat - 0.5},{lon - 0.5},{lat + 0.5},{lon + 0.5}",
                "limit":      5,
            }
            station_data = get_json(
                NOAA_BASE_URL + "stations",
                headers=headers,
                params=params_station,
            )
            results = station_data.get("results", [])
            if results:
                station_id   = results[0]["id"]
                station_name = results[0].get("name", "")
            time.sleep(0.5)
        except Exception as exc:
            logger.debug("  NOAA station %s: %s", venue_id, exc)
            continue

        if not station_id:
            continue

        # Obtener normales mensuales para Jun (month=6) y Jul (month=7)
        for month in [6, 7]:
            try:
                params_data = {
                    "datasetid":  "NORMAL_MLY",
                    "datatypeid": "MLY-TMAX-NORMAL,MLY-PRCP-NORMAL",
                    "stationid":  station_id,
                    "startdate":  f"2010-{month:02d}-01",
                    "enddate":    f"2010-{month:02d}-01",
                    "limit":      10,
                }
                climate_data = get_json(
                    NOAA_BASE_URL + "data",
                    headers=headers,
                    params=params_data,
                )
                temp_c   = np.nan
                precip   = np.nan
                for item in climate_data.get("results", []):
                    dtype = item.get("datatype", "")
                    val   = item.get("value", np.nan)
                    if dtype == "MLY-TMAX-NORMAL":
                        temp_c = round((float(val) - 32) * 5 / 9, 2)  # °F → °C
                    elif dtype == "MLY-PRCP-NORMAL":
                        precip = round(float(val) * 0.254, 2)  # 0.01in → mm

                rows.append({
                    "venue_id":          venue_id,
                    "station_id":        station_id,
                    "station_name":      station_name,
                    "month":             month,
                    "temp_normal_c":     temp_c,
                    "precip_normal_mm":  precip,
                    "source":            "noaa_cdo",
                })
                time.sleep(0.5)

            except Exception as exc:
                logger.debug("  NOAA data %s m%d: %s", venue_id, month, exc)

    df = pd.DataFrame(rows)
    logger.info("NOAA CDO: %d registros para %d venues.", len(df), df["venue_id"].nunique() if not df.empty else 0)
    return df


# ---------------------------------------------------------------------------
# GRUPO D — ESTADIOS
# ---------------------------------------------------------------------------

def descarga_altitud_openmeteo(lat: float, lon: float) -> float:
    """
    Consulta la altitud de un punto geográfico usando Open-Meteo Elevation API.

    API completamente gratuita, sin autenticación requerida.
    Endpoint: https://api.open-meteo.com/v1/elevation

    Args:
        lat: Latitud en grados decimales (-90 a 90).
        lon: Longitud en grados decimales (-180 a 180).

    Returns:
        Altitud en metros sobre el nivel del mar (float).
        Devuelve 0.0 si la API no responde.

    Example:
        >>> descarga_altitud_openmeteo(19.3029, -99.1505)  # Estadio Azteca
        2224.0
    """
    try:
        data = get_json(
            OPENMETEO_ELEVATION_URL,
            params={"latitude": lat, "longitude": lon},
        )
        elev_list = data.get("elevation", [])
        if elev_list:
            return round(float(elev_list[0]), 1)
    except Exception as exc:
        logger.debug("  Open-Meteo Elevation (%.4f, %.4f): %s", lat, lon, exc)
    return 0.0


def descarga_altitudes_venues() -> pd.DataFrame:
    """
    Obtiene altitud para los 16 venues WC2026 usando Open-Meteo (sin API key).

    Para cada venue: primero intenta Open-Meteo Elevation API; si falla usa
    el valor pre-calculado del diccionario WC2026_VENUES como fallback.

    Returns:
        DataFrame con columnas: venue_id, name, city, country,
        latitude, longitude, altitude_m, source.
    """
    logger.info("=== FUENTE: Altitud venues (Open-Meteo) ===")
    rows: List[Dict] = []

    for venue_id, venue in WC2026_VENUES.items():
        lat      = venue["lat"]
        lon      = venue["lon"]
        altitude = float(venue.get("altitude_m", 0))
        source   = "wc2026_dict"

        api_alt = descarga_altitud_openmeteo(lat, lon)
        if api_alt > 0.0:
            altitude = api_alt
            source   = "open-meteo_elevation"
        time.sleep(0.4)

        rows.append({
            "venue_id":   venue_id,
            "name":       venue["name"],
            "city":       venue["city"],
            "country":    venue["country"],
            "latitude":   lat,
            "longitude":  lon,
            "altitude_m": altitude,
            "source":     source,
        })
        logger.debug("  %s: %.0fm (%s)", venue["name"], altitude, source)

    df = pd.DataFrame(rows)
    logger.info("Altitud: %d venues (Open-Meteo sin key).", len(df))
    return df


def descarga_distancias_venues() -> pd.DataFrame:
    """
    Calcula distancias de viaje desde las ciudades base de cada equipo a cada venue WC2026.

    Usa exclusivamente la fórmula haversine (implementación matemática pura, sin API).
    Las coordenadas de ciudades base están en TEAM_HOME_CITIES.

    Returns:
        DataFrame con columnas: team, venue_id, venue_name, distance_km, source.
        'source' siempre es 'haversine' (sin dependencia externa).

    Nota:
        La distancia haversine es la distancia en línea recta por la superficie terrestre
        (great-circle distance). Para vuelos intercontinentales es una excelente
        aproximación del tiempo de viaje real (correlación >0.95 con tiempo de vuelo).
    """
    logger.info("=== FUENTE: Distancias de viaje (Haversine) ===")
    rows: List[Dict] = []

    for team, (team_lat, team_lon) in TEAM_HOME_CITIES.items():
        for venue_id, venue in WC2026_VENUES.items():
            dist_km = calcula_distancia_haversine(
                team_lat, team_lon, venue["lat"], venue["lon"]
            )
            rows.append({
                "team":        team,
                "venue_id":    venue_id,
                "venue_name":  venue["name"],
                "distance_km": dist_km,
                "source":      "haversine",
            })

    df = pd.DataFrame(rows)
    logger.info(
        "Distancias: %d pares equipo-venue calculados (haversine, sin API).", len(df)
    )
    return df


def scrape_transfermarkt_estadios() -> pd.DataFrame:
    """
    Scraping de información de estadios desde Transfermarkt.

    Usa cloudscraper para manejar la protección Cloudflare.
    Parsea: stadium_name, capacity, grass_type, city.
    Fallback a WC2026_VENUES dict si el scraping falla.

    Returns:
        DataFrame con columnas: venue_id, name, city, country, capacity,
        grass_type, lat, lon, altitude_m, source.
    """
    logger.info("=== FUENTE: Transfermarkt estadios ===")

    scraper = _get_cloudscraper()
    rows: List[Dict] = []

    for venue_id, venue in WC2026_VENUES.items():
        slug   = venue.get("slug", "")
        tm_id  = venue.get("tm_id", 0)
        scraped = False

        if scraper and slug and tm_id:
            url = f"https://www.transfermarkt.com/{slug}/stadion/verein/{tm_id}"
            try:
                time.sleep(SCRAPE_DELAY)
                resp = scraper.get(url, timeout=30)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "lxml")

                stadium_name = ""
                capacity     = venue["capacity"]
                grass_type   = "unknown"

                # Buscar nombre del estadio
                h1 = soup.find("h1")
                if h1:
                    stadium_name = h1.get_text(strip=True)

                # Buscar tabla de datos del estadio
                for row in soup.find_all("tr"):
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        label = cells[0].get_text(strip=True).lower()
                        val   = cells[1].get_text(strip=True)
                        if "capacity" in label or "fassungsverm" in label:
                            try:
                                capacity = int(val.replace(".", "").replace(",", "").split()[0])
                            except ValueError:
                                pass
                        if "grass" in label or "surface" in label or "untergrund" in label:
                            grass_type = val.lower()

                rows.append({
                    "venue_id":   venue_id,
                    "name":       stadium_name or venue["name"],
                    "city":       venue["city"],
                    "country":    venue["country"],
                    "capacity":   capacity,
                    "grass_type": grass_type,
                    "lat":        venue["lat"],
                    "lon":        venue["lon"],
                    "altitude_m": venue["altitude_m"],
                    "source":     "transfermarkt",
                })
                scraped = True
                logger.debug("  TM estadio %s: OK", venue["name"])

            except Exception as exc:
                logger.warning("  TM estadio %s: %s — usando dict", venue["name"], exc)

        if not scraped:
            rows.append({
                "venue_id":   venue_id,
                "name":       venue["name"],
                "city":       venue["city"],
                "country":    venue["country"],
                "capacity":   venue["capacity"],
                "grass_type": "natural",
                "lat":        venue["lat"],
                "lon":        venue["lon"],
                "altitude_m": venue["altitude_m"],
                "source":     "wc2026_dict",
            })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "stadiums_data.csv", index=False, encoding="utf-8-sig")
    logger.info("Estadios: %d venues.", len(df))
    return df


# ---------------------------------------------------------------------------
# GRUPO E — LESIONES
# ---------------------------------------------------------------------------

def scrape_transfermarkt_lesiones() -> pd.DataFrame:
    """
    Scraping de lesiones actuales de equipos nacionales desde Transfermarkt.

    Parsea: player_name, injury_type, injury_date, return_date, days_absent.
    Calcula: injured_count, days_until_return (avg), critical_injury flag.
    Jugador "estrella": si aparece en top 3 por valor de mercado del squad.

    Returns:
        DataFrame con columnas: team, player_name, injury_type,
        injury_date, return_date, days_absent, days_until_return,
        critical_injury, injured_count.
    """
    logger.info("=== FUENTE: Transfermarkt lesiones ===")

    scraper = _get_cloudscraper()
    if not scraper:
        logger.warning("  cloudscraper no disponible — omitiendo lesiones.")
        return pd.DataFrame()

    all_rows: List[Dict] = []

    for team, tm_id in TM_TEAM_IDS.items():
        slug = TM_TEAM_SLUGS.get(team, team.replace(" ", "-"))
        url  = f"https://www.transfermarkt.com/{slug}/verletzungen/verein/{tm_id}"

        try:
            time.sleep(SCRAPE_DELAY)
            resp = scraper.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            # Obtener top 3 jugadores por valor para determinar estrella
            top_players = _get_top_players_by_value(scraper, slug, tm_id)

            team_rows: List[Dict] = []
            tabla = soup.find("table", class_=lambda c: c and "items" in c)
            if not tabla:
                logger.debug("  Sin tabla lesiones para %s", team)
                continue

            for row in tabla.find_all("tr")[1:]:
                cells = row.find_all("td")
                if len(cells) < 5:
                    continue
                try:
                    player_name  = cells[0].get_text(strip=True)
                    injury_type  = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                    injury_date  = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                    return_date  = cells[3].get_text(strip=True) if len(cells) > 3 else ""
                    days_absent_str = cells[4].get_text(strip=True) if len(cells) > 4 else "0"

                    days_absent = 0
                    for part in days_absent_str.split():
                        try:
                            days_absent = int(part)
                            break
                        except ValueError:
                            pass

                    days_until_return = _calc_days_until_return(return_date)
                    is_critical = int(player_name in top_players)

                    team_rows.append({
                        "team":               normaliza_equipo(team),
                        "player_name":        player_name,
                        "injury_type":        injury_type,
                        "injury_date":        injury_date,
                        "return_date":        return_date,
                        "days_absent":        days_absent,
                        "days_until_return":  days_until_return,
                        "critical_injury":    is_critical,
                    })
                except Exception:
                    continue

            injured_count = len(team_rows)
            for r in team_rows:
                r["injured_count"] = injured_count

            all_rows.extend(team_rows)
            logger.info("  %s: %d lesiones", team, injured_count)

        except Exception as exc:
            logger.warning("  TM lesiones %s bloqueado/error: %s", team, exc)

    if not all_rows:
        logger.warning("Transfermarkt lesiones: sin datos.")
        df = pd.DataFrame(columns=[
            "team", "player_name", "injury_type", "injury_date",
            "return_date", "days_absent", "days_until_return",
            "critical_injury", "injured_count",
        ])
    else:
        df = pd.DataFrame(all_rows)

    df.to_csv(DATA_DIR / "injuries_current.csv", index=False, encoding="utf-8-sig")
    logger.info("Lesiones: %d registros, %d equipos.", len(df),
                df["team"].nunique() if not df.empty else 0)
    return df


def _calc_days_until_return(return_date_str: str) -> float:
    """Calcula días hasta la fecha de retorno desde hoy. Retorna NaN si no parseable."""
    if not return_date_str or return_date_str in ("-", "Unknown", ""):
        return np.nan
    for fmt in ("%b %d, %Y", "%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y"):
        try:
            rd = datetime.strptime(return_date_str.strip(), fmt).date()
            return max(0, (rd - date.today()).days)
        except ValueError:
            continue
    return np.nan


def _get_top_players_by_value(scraper, slug: str, tm_id: int) -> List[str]:
    """Obtiene los 3 jugadores de mayor valor de mercado del equipo en TM."""
    url = f"https://www.transfermarkt.com/{slug}/kader/verein/{tm_id}"
    try:
        time.sleep(SCRAPE_DELAY)
        resp = scraper.get(url, timeout=30)
        resp.raise_for_status()
        soup  = BeautifulSoup(resp.text, "lxml")
        tabla = soup.find("table", class_=lambda c: c and "items" in c)
        if not tabla:
            return []

        players_vals: List[Tuple[str, float]] = []
        for row in tabla.find_all("tr")[1:]:
            cells = row.find_all("td")
            if len(cells) < 6:
                continue
            try:
                name = cells[0].get_text(strip=True)
                val_str = cells[-1].get_text(strip=True)
                val  = _parse_market_value(val_str)
                if name and val > 0:
                    players_vals.append((name, val))
            except Exception:
                continue

        players_vals.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in players_vals[:3]]

    except Exception:
        return []


def _parse_market_value(val_str: str) -> float:
    """Parsea string de valor de mercado TM (ej '€45.00m', '€500k') a float EUR."""
    val_str = val_str.strip().replace("€", "").replace(",", ".")
    multiplier = 1.0
    if "m" in val_str.lower():
        val_str    = val_str.lower().replace("m", "")
        multiplier = 1_000_000
    elif "k" in val_str.lower():
        val_str    = val_str.lower().replace("k", "")
        multiplier = 1_000
    try:
        return float(val_str) * multiplier
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# GRUPO F — JUGADORES
# ---------------------------------------------------------------------------

def scrape_transfermarkt_jugadores() -> pd.DataFrame:
    """
    Scraping de plantillas y valores de mercado desde Transfermarkt.

    Parsea: player_name, position, age, market_value_eur, nationality.
    Agrega por equipo: total_market_value_eur, squad_size, avg_age.
    Extrae top 3 jugadores por valor para cada equipo.

    Returns:
        DataFrame con columnas: team, player_name, position, age,
        market_value_eur, nationality, is_top3, total_squad_value_eur,
        squad_size, avg_age.
    """
    logger.info("=== FUENTE: Transfermarkt jugadores ===")

    scraper = _get_cloudscraper()
    if not scraper:
        logger.warning("  cloudscraper no disponible — omitiendo jugadores.")
        return pd.DataFrame()

    all_rows: List[Dict] = []

    for team, tm_id in TM_TEAM_IDS.items():
        slug = TM_TEAM_SLUGS.get(team, team.replace(" ", "-"))
        url  = f"https://www.transfermarkt.com/{slug}/kader/verein/{tm_id}"

        try:
            time.sleep(SCRAPE_DELAY)
            resp = scraper.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            tabla = soup.find("table", class_=lambda c: c and "items" in c)
            if not tabla:
                logger.debug("  Sin tabla para %s", team)
                continue

            team_players: List[Dict] = []
            for row in tabla.find_all("tr")[1:]:
                cells = row.find_all("td")
                if len(cells) < 4:
                    continue
                try:
                    player_name  = ""
                    position     = ""
                    age          = np.nan
                    market_val   = 0.0
                    nationality  = ""

                    # Nombre: puede estar en <a> dentro de la primera celda relevante
                    name_cell = row.find("td", class_=lambda c: c and "hauptlink" in (c or ""))
                    if name_cell:
                        player_name = name_cell.get_text(strip=True)

                    if not player_name and len(cells) > 0:
                        player_name = cells[0].get_text(strip=True)

                    # Posición
                    if len(cells) > 1:
                        position = cells[1].get_text(strip=True)

                    # Edad
                    for cell in cells:
                        txt = cell.get_text(strip=True)
                        if txt.isdigit() and 15 <= int(txt) <= 45:
                            age = int(txt)
                            break

                    # Nacionalidad (bandera alt)
                    flag = row.find("img", class_=lambda c: c and "flagge" in (c or ""))
                    if flag:
                        nationality = flag.get("title", flag.get("alt", ""))

                    # Valor de mercado (última celda)
                    market_val = _parse_market_value(cells[-1].get_text(strip=True))

                    if player_name:
                        team_players.append({
                            "team":              normaliza_equipo(team),
                            "player_name":       player_name,
                            "position":          position,
                            "age":               age,
                            "market_value_eur":  market_val,
                            "nationality":       nationality,
                        })
                except Exception:
                    continue

            if not team_players:
                continue

            # Calcular métricas agregadas
            total_value = sum(p["market_value_eur"] for p in team_players)
            squad_size  = len(team_players)
            ages        = [p["age"] for p in team_players if not (isinstance(p["age"], float) and np.isnan(p["age"]))]
            avg_age     = round(float(np.mean(ages)), 1) if ages else np.nan

            # Marcar top 3
            sorted_players = sorted(team_players, key=lambda x: x["market_value_eur"], reverse=True)
            top3_names     = {p["player_name"] for p in sorted_players[:3]}

            for p in team_players:
                p["is_top3"]              = int(p["player_name"] in top3_names)
                p["total_squad_value_eur"] = total_value
                p["squad_size"]            = squad_size
                p["avg_age"]               = avg_age

            all_rows.extend(team_players)
            logger.info("  %s: %d jugadores, valor total €%.1fM",
                        team, squad_size, total_value / 1e6)

        except Exception as exc:
            logger.warning("  TM jugadores %s bloqueado/error: %s", team, exc)

    if not all_rows:
        logger.warning("Transfermarkt jugadores: sin datos.")
        df = pd.DataFrame(columns=[
            "team", "player_name", "position", "age", "market_value_eur",
            "nationality", "is_top3", "total_squad_value_eur", "squad_size", "avg_age",
        ])
    else:
        df = pd.DataFrame(all_rows)

    df.to_csv(DATA_DIR / "players_key.csv", index=False, encoding="utf-8-sig")
    logger.info("Jugadores: %d registros, %d equipos.", len(df),
                df["team"].nunique() if not df.empty else 0)
    return df


# ---------------------------------------------------------------------------
# CONSOLIDACIÓN MAESTRA
# ---------------------------------------------------------------------------

def consolida_datos_maestro(
    df_matches:  pd.DataFrame,
    df_players:  pd.DataFrame,
    df_elo:      pd.DataFrame,
    df_fifa:     pd.DataFrame,
    df_climate:  pd.DataFrame,
    df_stadiums: pd.DataFrame,
    df_injuries: pd.DataFrame,
    df_squad:    pd.DataFrame,
) -> pd.DataFrame:
    """
    Consolida todas las fuentes en un DataFrame único para el modelo predictivo.

    Pipeline de joins:
        1. Matches base (StatsBomb preferido, FD para cobertura adicional, dedup).
        2. ELO pre-partido por (date, team_home, team_away) exacto.
        3. FIFA rank por merge_asof (último ranking ≤ fecha del partido, tolerancia 35D).
        4. Clima por venue directo; si no hay venue → promedio de temporada del país.
        5. Lesiones as-of: snapshot más reciente ≤ fecha del partido por equipo.
        6. Valor de plantilla más reciente por equipo.
        7. Features derivadas: diff_elo, diff_fifa_rank, goles_totales, resultado.

    Args:
        df_matches:  Partidos de StatsBomb (con xG y pases).
        df_players:  Stats por jugador de StatsBomb.
        df_elo:      ELO histórico calculado.
        df_fifa:     Rankings FIFA históricos.
        df_climate:  Climatología Jun-Jul por venue.
        df_stadiums: Datos de estadios WC2026.
        df_injuries: Estado de lesiones por equipo.
        df_squad:    Valores de plantilla por equipo.

    Returns:
        DataFrame consolidado con todas las features.
    """
    logger.info("=== CONSOLIDANDO DATOS MAESTRO ===")

    if df_matches.empty:
        logger.error("Sin DataFrame de partidos para consolidar.")
        return pd.DataFrame()

    df = df_matches.copy()

    # Asegurar columnas de pase si no existen
    for col in ["passes_home", "passes_away", "pass_accuracy_home", "pass_accuracy_away"]:
        if col not in df.columns:
            df[col] = np.nan

    # ── 1. Deduplicar ────────────────────────────────────────────────────────
    df = _deduplicar_partidos(df)

    # ── 2. Join ELO ──────────────────────────────────────────────────────────
    if not df_elo.empty:
        df = _join_elo(df, df_elo)
    else:
        df["elo_home"] = np.nan
        df["elo_away"] = np.nan

    # ── 3. Join FIFA ─────────────────────────────────────────────────────────
    if not df_fifa.empty:
        df = _join_fifa(df, df_fifa)
    else:
        df["fifa_rank_home"] = np.nan
        df["fifa_rank_away"] = np.nan

    # ── 4. Join Clima ─────────────────────────────────────────────────────────
    if not df_climate.empty and not df_stadiums.empty:
        df = _join_climate(df, df_climate, df_stadiums)
    else:
        for col in ["temp_avg_c", "humidity_pct", "wind_kmh", "precip_mm", "altitude_m"]:
            df[col] = np.nan

    # ── 5. Join Lesiones ─────────────────────────────────────────────────────
    if not df_injuries.empty:
        df = _join_injuries(df, df_injuries)
    else:
        df["injured_count_home"] = np.nan
        df["injured_count_away"] = np.nan
        df["critical_injury_home"] = 0
        df["critical_injury_away"] = 0

    # ── 6. Join Squad Values ─────────────────────────────────────────────────
    if not df_squad.empty:
        df = _join_squad(df, df_squad)
    else:
        df["squad_value_home_eur"] = np.nan
        df["squad_value_away_eur"] = np.nan

    # ── 7. Variables objetivo y derivadas ────────────────────────────────────
    df["resultado"] = (
        df["goals_home"].gt(df["goals_away"]).map({True: 1})
        .fillna(
            df["goals_home"].lt(df["goals_away"]).map({True: -1}).fillna(0)
        )
        .astype(int)
    )

    df["diff_elo"]       = df["elo_home"] - df["elo_away"]
    df["diff_fifa_rank"] = df["fifa_rank_away"] - df["fifa_rank_home"]
    df["goles_totales"]  = df["goals_home"] + df["goals_away"]
    df["diff_squad_val"] = (
        df["squad_value_home_eur"].fillna(0) - df["squad_value_away_eur"].fillna(0)
    )

    df = df.sort_values("date").reset_index(drop=True)

    logger.info("Consolidado maestro: %d partidos, %d columnas.", len(df), len(df.columns))
    return df


def _deduplicar_partidos(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados entre fuentes; prioriza statsbomb sobre football_data."""
    if df.empty:
        return df
    source_priority = {"statsbomb": 0, "football_data": 1}
    df["_prio"] = df.get("source", pd.Series(["unknown"] * len(df))).map(source_priority).fillna(99)
    df_sorted   = df.sort_values(["team_home", "team_away", "date", "_prio"])
    dedup       = df_sorted.drop_duplicates(subset=["team_home", "team_away", "date"], keep="first")
    n_eliminados = len(df) - len(dedup)
    if n_eliminados > 0:
        logger.info("  Deduplicación: %d duplicados eliminados.", n_eliminados)
    return dedup.drop(columns=["_prio"]).reset_index(drop=True)


def _join_elo(df_partidos: pd.DataFrame, df_elo: pd.DataFrame) -> pd.DataFrame:
    """Une ratings ELO pre-partido por (date, team_home, team_away)."""
    if df_elo.empty:
        df_partidos["elo_home"] = np.nan
        df_partidos["elo_away"] = np.nan
        return df_partidos

    if "elo_home" in df_elo.columns and "team_home" in df_elo.columns:
        elo_merge = df_elo[["date", "team_home", "team_away", "elo_home", "elo_away"]].copy()
        df_out    = df_partidos.merge(elo_merge, on=["date", "team_home", "team_away"], how="left")
        logger.info("  ELO unido: %d/%d con cobertura.", int(df_out["elo_home"].notna().sum()), len(df_out))
        return df_out

    df_partidos["elo_home"] = np.nan
    df_partidos["elo_away"] = np.nan
    return df_partidos


def _join_fifa(df_partidos: pd.DataFrame, df_fifa: pd.DataFrame) -> pd.DataFrame:
    """Une FIFA rank más reciente ≤ fecha del partido; tolerancia 35 días."""
    if df_fifa.empty or "rank_date" not in df_fifa.columns:
        df_partidos["fifa_rank_home"] = np.nan
        df_partidos["fifa_rank_away"] = np.nan
        return df_partidos

    df_out  = df_partidos.copy().sort_values("date").reset_index(drop=True)
    df_out["_orig_idx"] = df_out.index
    df_fifa = df_fifa.sort_values("rank_date").copy()

    for team_col, rank_col in [("team_home", "fifa_rank_home"), ("team_away", "fifa_rank_away")]:
        partidos_side = df_out[["date", "_orig_idx", team_col]].rename(
            columns={team_col: "_team"}
        ).sort_values("date")
        fifa_side = df_fifa[["rank_date", "team", "fifa_rank"]].rename(
            columns={"rank_date": "date", "team": "_team", "fifa_rank": rank_col}
        )
        merged = pd.merge_asof(
            partidos_side, fifa_side, on="date", by="_team",
            direction="backward", tolerance=pd.Timedelta("35D"),
        )
        rank_map        = merged.set_index("_orig_idx")[rank_col].to_dict()
        df_out[rank_col] = df_out["_orig_idx"].map(rank_map)

    df_out = df_out.drop(columns=["_orig_idx"])
    logger.info("  FIFA rank: %d/%d con cobertura.", int(df_out["fifa_rank_home"].notna().sum()), len(df_out))
    return df_out


def _join_climate(
    df_partidos: pd.DataFrame,
    df_climate:  pd.DataFrame,
    df_stadiums: pd.DataFrame,
) -> pd.DataFrame:
    """
    Une datos climáticos por venue.

    Si el partido tiene venue_id → lookup directo.
    Si no → usa promedio del país del equipo local (fallback por country).
    """
    climate_cols = ["temp_avg_c", "humidity_pct", "wind_kmh", "precip_mm",
                    "rain_days", "altitude_m"]
    for col in climate_cols:
        df_partidos[col] = np.nan

    if df_climate.empty:
        return df_partidos

    # Crear mapa venue_id → clima
    climate_map = df_climate.set_index("venue_id")[climate_cols].to_dict("index")

    # Mapa country → promedio de clima (fallback)
    country_climate: Dict[str, Dict] = {}
    if "country" in df_climate.columns:
        for country, grp in df_climate.groupby("country"):
            country_climate[country.lower()] = {
                col: grp[col].mean() for col in climate_cols if col in grp.columns
            }

    # Si los partidos tienen venue info
    venue_col = "venue_id" if "venue_id" in df_partidos.columns else None

    for idx, row in df_partidos.iterrows():
        matched = False
        if venue_col and pd.notna(row.get(venue_col)):
            vid = row[venue_col]
            if vid in climate_map:
                for col in climate_cols:
                    df_partidos.at[idx, col] = climate_map[vid].get(col, np.nan)
                matched = True

        if not matched:
            # Fallback: si la competicion es WC2026 y hay venue en los venues
            comp = str(row.get("competition", "")).lower()
            if "world cup" in comp and "2026" in comp:
                # USA como sede principal
                if "usa" in country_climate:
                    for col in climate_cols:
                        df_partidos.at[idx, col] = country_climate["usa"].get(col, np.nan)

    logger.info("  Clima unido: %d/%d con cobertura.",
                int(df_partidos["temp_avg_c"].notna().sum()), len(df_partidos))
    return df_partidos


def _join_injuries(df_partidos: pd.DataFrame, df_injuries: pd.DataFrame) -> pd.DataFrame:
    """
    Une estado de lesiones as-of (snapshot más reciente ≤ fecha del partido).

    Dado que las lesiones son un snapshot único (scrapeado hoy),
    se une directamente por nombre de equipo y se aplica a todos los partidos.
    """
    for col in ["injured_count_home", "injured_count_away",
                "critical_injury_home", "critical_injury_away",
                "avg_days_return_home", "avg_days_return_away"]:
        df_partidos[col] = np.nan if "days" in col or "count" in col else 0

    if df_injuries.empty:
        return df_partidos

    # Agregar por equipo
    inj_agg = df_injuries.groupby("team").agg(
        injured_count=("player_name", "count"),
        critical_injury=("critical_injury", "max"),
        avg_days_return=("days_until_return", "mean"),
    ).reset_index()

    inj_map = inj_agg.set_index("team").to_dict("index")

    for idx, row in df_partidos.iterrows():
        home = row.get("team_home", "")
        away = row.get("team_away", "")
        if home in inj_map:
            df_partidos.at[idx, "injured_count_home"]   = inj_map[home]["injured_count"]
            df_partidos.at[idx, "critical_injury_home"] = inj_map[home]["critical_injury"]
            df_partidos.at[idx, "avg_days_return_home"] = inj_map[home]["avg_days_return"]
        if away in inj_map:
            df_partidos.at[idx, "injured_count_away"]   = inj_map[away]["injured_count"]
            df_partidos.at[idx, "critical_injury_away"] = inj_map[away]["critical_injury"]
            df_partidos.at[idx, "avg_days_return_away"] = inj_map[away]["avg_days_return"]

    logger.info("  Lesiones unidas: %d equipos.",
                int((df_partidos["injured_count_home"].notna()).sum()))
    return df_partidos


def _join_squad(df_partidos: pd.DataFrame, df_squad: pd.DataFrame) -> pd.DataFrame:
    """Une valor total de plantilla por equipo."""
    df_partidos["squad_value_home_eur"] = np.nan
    df_partidos["squad_value_away_eur"] = np.nan

    if df_squad.empty:
        return df_partidos

    squad_map = (
        df_squad.groupby("team")["total_squad_value_eur"].first().to_dict()
        if "total_squad_value_eur" in df_squad.columns
        else {}
    )

    df_partidos["squad_value_home_eur"] = df_partidos["team_home"].map(squad_map)
    df_partidos["squad_value_away_eur"] = df_partidos["team_away"].map(squad_map)

    logger.info("  Squad values unidos: %d/%d home con cobertura.",
                int(df_partidos["squad_value_home_eur"].notna().sum()), len(df_partidos))
    return df_partidos


# ---------------------------------------------------------------------------
# VALIDACIÓN DE CALIDAD
# ---------------------------------------------------------------------------

def valida_calidad(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Valida y limpia el DataFrame consolidado; genera reporte de calidad.

    Validaciones base (de descarga_datos.py) + nuevas:
        - Rango de temperatura: -30 a 60°C
        - Humedad: 0 a 100%
        - Viento: 0 a 200 km/h
        - Squad value: > 0
        - Lesiones: 0 a 30 jugadores

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

    df_clean     = df.copy()
    n_eliminados = 0

    # ── Nulos por columna ─────────────────────────────────────────────────────
    for col in df_clean.columns:
        n   = int(df_clean[col].isna().sum())
        pct = round(n / len(df_clean) * 100, 2) if len(df_clean) > 0 else 0.0
        reporte["nulos_raw"][col] = {"count": n, "pct": pct}

    # ── Eliminar filas críticas vacías ────────────────────────────────────────
    criticos = (
        df_clean["team_home"].isna() |
        df_clean["team_away"].isna() |
        df_clean["date"].isna()
    )
    if criticos.any():
        n = int(criticos.sum())
        logger.warning("  Eliminando %d filas sin equipo/fecha.", n)
        df_clean     = df_clean[~criticos].copy()
        n_eliminados += n
        reporte["advertencias"].append(f"{n} filas eliminadas: equipo o fecha faltante")

    # ── Goles negativos ───────────────────────────────────────────────────────
    for col in ["goals_home", "goals_away"]:
        if col in df_clean.columns:
            invalidos = df_clean[col] < 0
            if invalidos.any():
                df_clean.loc[invalidos, col] = 0
                reporte["outliers_corregidos"][col] = int(invalidos.sum())

    # ── ELO fuera de rango ────────────────────────────────────────────────────
    for col in ["elo_home", "elo_away"]:
        if col in df_clean.columns:
            fuera = df_clean[col].notna() & ((df_clean[col] < 500) | (df_clean[col] > 2500))
            if fuera.any():
                df_clean.loc[fuera, col] = np.nan
                reporte["outliers_corregidos"][col] = int(fuera.sum())

    # ── FIFA rank fuera de rango ──────────────────────────────────────────────
    for col in ["fifa_rank_home", "fifa_rank_away"]:
        if col in df_clean.columns:
            fuera = df_clean[col].notna() & ((df_clean[col] < 1) | (df_clean[col] > 300))
            if fuera.any():
                df_clean.loc[fuera, col] = np.nan
                reporte["outliers_corregidos"][col] = int(fuera.sum())

    # ── Clima: rangos físicos ─────────────────────────────────────────────────
    clima_ranges = {
        "temp_avg_c":   (-30, 60),
        "temp_max_avg_c": (-30, 60),
        "temp_min_avg_c": (-30, 60),
        "humidity_pct": (0, 100),
        "wind_kmh":     (0, 200),
        "precip_mm":    (0, 1000),
    }
    for col, (lo, hi) in clima_ranges.items():
        if col in df_clean.columns:
            fuera = df_clean[col].notna() & ((df_clean[col] < lo) | (df_clean[col] > hi))
            if fuera.any():
                logger.warning("  %s fuera de rango (%s-%s): %d casos → nulificados.", col, lo, hi, fuera.sum())
                df_clean.loc[fuera, col] = np.nan
                reporte["outliers_corregidos"][col] = int(fuera.sum())

    # ── Squad value ≤ 0 → nulificar ──────────────────────────────────────────
    for col in ["squad_value_home_eur", "squad_value_away_eur"]:
        if col in df_clean.columns:
            invalidos = df_clean[col].notna() & (df_clean[col] <= 0)
            if invalidos.any():
                df_clean.loc[invalidos, col] = np.nan
                reporte["outliers_corregidos"][col] = int(invalidos.sum())

    # ── Lesiones fuera de rango 0-30 ─────────────────────────────────────────
    for col in ["injured_count_home", "injured_count_away"]:
        if col in df_clean.columns:
            fuera = df_clean[col].notna() & ((df_clean[col] < 0) | (df_clean[col] > 30))
            if fuera.any():
                df_clean.loc[fuera, col] = np.nan
                reporte["outliers_corregidos"][col] = int(fuera.sum())

    # ── Recalcular variable objetivo ──────────────────────────────────────────
    if all(c in df_clean.columns for c in ["resultado", "goals_home", "goals_away"]):
        resultado_correcto = (
            df_clean["goals_home"].gt(df_clean["goals_away"]).map({True: 1})
            .fillna(df_clean["goals_home"].lt(df_clean["goals_away"]).map({True: -1}).fillna(0))
            .astype(int)
        )
        inconsistencias = (df_clean["resultado"] != resultado_correcto).sum()
        if inconsistencias > 0:
            logger.warning("  %d inconsistencias resultado/goles → recalculados.", inconsistencias)
            df_clean["resultado"] = resultado_correcto
            reporte["advertencias"].append(f"{inconsistencias} resultados recalculados")

    # ── Estadísticas finales ──────────────────────────────────────────────────
    reporte["partidos_eliminados"] = n_eliminados
    reporte["partidos_limpios"]    = len(df_clean)

    teams_series = pd.concat([df_clean["team_home"], df_clean["team_away"]])

    def _cobertura(col):
        return int(df_clean[col].notna().sum()) if col in df_clean.columns else 0

    reporte["cobertura"] = {
        "xg_disponible":         _cobertura("xg_home"),
        "elo_disponible":        _cobertura("elo_home"),
        "fifa_disponible":       _cobertura("fifa_rank_home"),
        "clima_disponible":      _cobertura("temp_avg_c"),
        "squad_val_disponible":  _cobertura("squad_value_home_eur"),
        "lesiones_disponible":   _cobertura("injured_count_home"),
        "pases_disponible":      _cobertura("passes_home"),
        "competiciones":         int(df_clean["competition"].nunique()) if "competition" in df_clean.columns else 0,
        "equipos_unicos":        int(teams_series.nunique()),
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
# GUARDADO DE SALIDAS
# ---------------------------------------------------------------------------

def guarda_salidas(
    df_full:      pd.DataFrame,
    df_clean:     pd.DataFrame,
    df_climate:   pd.DataFrame,
    df_stadiums:  pd.DataFrame,
    reporte:      Dict,
) -> None:
    """Guarda todos los archivos de salida del pipeline."""
    files = {
        "matches_full":          (DATA_DIR / "matches_full.csv",    df_full),
        "matches_cleaned":       (DATA_DIR / "matches_cleaned.csv", df_clean),
        "climate_data":          (DATA_DIR / "climate_data.csv",    df_climate),
        "stadiums_data":         (DATA_DIR / "stadiums_data.csv",   df_stadiums),
    }
    for name, (path, df) in files.items():
        if not df.empty:
            df.to_csv(path, index=False, encoding="utf-8-sig")
            logger.info("Guardado: %s (%d filas, %d cols)", path, len(df), len(df.columns))
        else:
            logger.info("Omitido (vacío): %s", path)

    report_path = DATA_DIR / "data_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Guardado: %s", report_path)


# ---------------------------------------------------------------------------
# PUNTO DE ENTRADA
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Ejecuta el pipeline completo de 7 grupos de fuentes.

    Un fallo en cualquier fuente individual no detiene el pipeline.
    Requiere al menos una fuente de partidos (StatsBomb o Football-Data).
    """
    logger.info("=" * 70)
    logger.info(
        "PIPELINE COMPLETO MUNDIAL 2026  |  %s",
        datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    )
    logger.info("Período: %d–%d  |  DATA_DIR: %s", START_YEAR, END_YEAR, DATA_DIR.resolve())
    logger.info("API keys configuradas: FD=%s | OWM=%s | NOAA=%s",
                "sí" if FOOTBALL_API_KEY else "no",
                "sí" if OPENWEATHER_KEY else "no",
                "sí" if NOAA_TOKEN else "no")
    logger.info("=" * 70)

    # ── GRUPO A: Partidos ────────────────────────────────────────────────────
    df_sb, df_players = descarga_statsbomb()
    df_fd             = descarga_football_data(FOOTBALL_API_KEY)

    if df_sb.empty and df_fd.empty:
        logger.error("CRÍTICO: Sin datos de partidos de ninguna fuente. Abortando.")
        sys.exit(1)

    # Combinar fuentes de partidos
    marcos_validos = [d for d in [df_sb, df_fd] if not d.empty]
    columnas_base  = [
        "match_id", "date", "season", "competition",
        "team_home", "team_away", "goals_home", "goals_away", "source",
        "xg_home", "xg_away", "shots_home", "shots_away",
        "shots_on_target_home", "shots_on_target_away",
        "passes_home", "passes_away", "pass_accuracy_home", "pass_accuracy_away",
    ]
    partes = []
    for d in marcos_validos:
        cols = [c for c in columnas_base if c in d.columns]
        partes.append(d[cols].copy())
    df_matches_combined = pd.concat(partes, ignore_index=True)

    # ── GRUPO B: Rankings ────────────────────────────────────────────────────
    df_elo  = descarga_elo()
    df_fifa = descarga_fifa()

    # ── GRUPO C: Clima ───────────────────────────────────────────────────────
    df_climate_om  = descarga_clima_openmeteo()
    df_climate_ow  = descarga_clima_openweather(OPENWEATHER_KEY)
    df_climate_noaa = descarga_clima_noaa()

    # Consolidar fuentes climáticas (Open-Meteo es primaria)
    df_climate = df_climate_om if not df_climate_om.empty else df_climate_ow

    # ── GRUPO D: Estadios ────────────────────────────────────────────────────
    df_stadiums   = scrape_transfermarkt_estadios()
    df_altitudes  = descarga_altitudes_venues()       # Open-Meteo, sin key
    df_distancias = descarga_distancias_venues()      # Haversine, sin API

    # Merge altitud en stadiums
    if not df_altitudes.empty and not df_stadiums.empty:
        alt_map = df_altitudes.set_index("venue_id")["altitude_m"].to_dict()
        df_stadiums["altitude_m"] = df_stadiums.get("venue_id", pd.Series()).map(alt_map)

    # ── GRUPO E: Lesiones ────────────────────────────────────────────────────
    df_injuries = scrape_transfermarkt_lesiones()

    # ── GRUPO F: Jugadores ───────────────────────────────────────────────────
    df_squad = scrape_transfermarkt_jugadores()

    # ── CONSOLIDACIÓN ────────────────────────────────────────────────────────
    df_full = consolida_datos_maestro(
        df_matches  = df_matches_combined,
        df_players  = df_players,
        df_elo      = df_elo,
        df_fifa     = df_fifa,
        df_climate  = df_climate,
        df_stadiums = df_stadiums,
        df_injuries = df_injuries,
        df_squad    = df_squad,
    )

    if df_full.empty:
        logger.error("CRÍTICO: DataFrame consolidado vacío. Abortando.")
        sys.exit(1)

    # ── VALIDACIÓN ───────────────────────────────────────────────────────────
    df_clean, reporte = valida_calidad(df_full)

    # Guardar distancias separado
    if not df_distancias.empty:
        df_distancias.to_csv(DATA_DIR / "travel_distances.csv", index=False, encoding="utf-8-sig")
        logger.info("Guardado: travel_distances.csv (%d filas)", len(df_distancias))

    # Guardar OpenWeather por separado si existe
    if not df_climate_ow.empty:
        df_climate_ow.to_csv(DATA_DIR / "climate_current.csv", index=False, encoding="utf-8-sig")

    # ── GUARDAR SALIDAS ───────────────────────────────────────────────────────
    guarda_salidas(df_full, df_clean, df_climate, df_stadiums, reporte)

    # ── RESUMEN ───────────────────────────────────────────────────────────────
    cob = reporte["cobertura"]
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETADO")
    logger.info("  Partidos brutos:       %d", len(df_full))
    logger.info("  Partidos limpios:      %d", len(df_clean))
    logger.info("  Con xG:                %d (%.0f%%)", cob["xg_disponible"],
                100 * cob["xg_disponible"] / max(len(df_clean), 1))
    logger.info("  Con ELO:               %d (%.0f%%)", cob["elo_disponible"],
                100 * cob["elo_disponible"] / max(len(df_clean), 1))
    logger.info("  Con FIFA rank:         %d (%.0f%%)", cob["fifa_disponible"],
                100 * cob["fifa_disponible"] / max(len(df_clean), 1))
    logger.info("  Con clima:             %d (%.0f%%)", cob["clima_disponible"],
                100 * cob["clima_disponible"] / max(len(df_clean), 1))
    logger.info("  Con squad values:      %d (%.0f%%)", cob["squad_val_disponible"],
                100 * cob["squad_val_disponible"] / max(len(df_clean), 1))
    logger.info("  Con lesiones:          %d (%.0f%%)", cob["lesiones_disponible"],
                100 * cob["lesiones_disponible"] / max(len(df_clean), 1))
    logger.info("  Con pases:             %d (%.0f%%)", cob["pases_disponible"],
                100 * cob["pases_disponible"] / max(len(df_clean), 1))
    logger.info("  Equipos únicos:        %d", cob["equipos_unicos"])
    logger.info("  Competiciones:         %d", cob["competiciones"])
    rdist = cob.get("distribucion_resultado", {})
    total = sum(rdist.values()) or 1
    logger.info(
        "  Resultados → Local: %d%%, Empate: %d%%, Visitante: %d%%",
        round(100 * rdist.get(1, 0) / total),
        round(100 * rdist.get(0, 0) / total),
        round(100 * rdist.get(-1, 0) / total),
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
