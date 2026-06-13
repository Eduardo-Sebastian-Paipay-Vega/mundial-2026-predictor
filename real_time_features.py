"""
real_time_features.py — Fase 6.2: Features en tiempo real
Lesiones (Transfermarkt), Clima (OpenWeatherMap), Tecnico (Wikipedia)
"""
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import requests

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("realtime")

ROOT = Path(__file__).parent

# Cargar API key de .env
def _load_env():
    env_path = ROOT / ".env"
    env = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env

_ENV = _load_env()
OWM_API_KEY = _ENV.get("OPENWEATHER_API_KEY", "")

# Mapping equipo → ID Transfermarkt (equipos WC2026 más comunes)
TEAM_IDS: Dict[str, int] = {
    "Argentina":      17480,
    "Brazil":         3468,
    "France":         3377,
    "Germany":        3376,
    "Spain":          3375,
    "England":        3,
    "Portugal":       3928,
    "Netherlands":    3379,
    "Italy":          3380,
    "Belgium":        3382,
    "Croatia":        3099,
    "Morocco":        3375,
    "Senegal":        3384,
    "Uruguay":        3386,
    "Mexico":         3386,
    "United States":  3411,
    "Japan":          3414,
    "South Korea":    3430,
    "Australia":      3412,
    "Canada":         3388,
    "Ecuador":        3387,
    "Peru":           3388,
    "Chile":          3390,
    "Colombia":       3391,
    "Venezuela":      3392,
    "Bolivia":        3393,
    "Paraguay":       3394,
    "Costa Rica":     3395,
    "Panama":         3396,
    "Honduras":       3397,
    "Jamaica":        3398,
    "El Salvador":    3399,
    "Saudi Arabia":   3400,
    "Iran":           3401,
    "Qatar":          3402,
    "Cameroon":       3403,
    "Ghana":          3404,
    "Tunisia":        3405,
    "Nigeria":        3406,
    "Egypt":          3407,
    "Algeria":        3408,
    "South Africa":   3409,
    "Serbia":         3410,
    "Switzerland":    3413,
    "Denmark":        3415,
    "Sweden":         3416,
    "Norway":         3417,
    "Poland":         3418,
    "Ukraine":        3419,
    "Austria":        3420,
    "Turkey":         3421,
    "Czech Republic": 3422,
    "Romania":        3423,
    "Hungary":        3424,
    "Slovakia":       3425,
    "Greece":         3426,
    "Scotland":       3427,
    "Wales":          3428,
    "Ireland":        3429,
    "New Zealand":    3431,
    "Ivory Coast":    3432,
    "Mali":           3433,
    "Guinea":         3434,
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

# ── LESIONES ─────────────────────────────────────────────────────────────────

def get_injuries_transfermarkt(team_name: str) -> Dict:
    """
    Obtiene lesiones actuales desde Transfermarkt.
    Retorna: injured_count, injury_impact (0-1), critical_injury (0/1).
    """
    team_id = TEAM_IDS.get(team_name)
    if team_id is None:
        logger.warning("  Team ID no encontrado para %s", team_name)
        return {"injured_count": 0, "injury_impact": 0.0, "critical_injury": 0}

    slug = team_name.lower().replace(" ", "-")
    url  = f"https://www.transfermarkt.com/{slug}/verletzungen/verein/{team_id}"

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()

        try:
            from bs4 import BeautifulSoup
            soup  = BeautifulSoup(resp.content, "html.parser")
            table = soup.find("table", {"class": "items"})
        except ImportError:
            logger.warning("  beautifulsoup4 no instalado; usando fallback")
            return _injury_fallback()

        if not table:
            return {"injured_count": 0, "injury_impact": 0.0, "critical_injury": 0}

        KEY_POSITIONS = {"CB", "ST", "CF", "CAM", "CM", "GK"}
        injured = []

        for row in table.find_all("tr")[1:]:
            tds = row.find_all("td")
            if len(tds) >= 3:
                try:
                    position = tds[1].get_text(strip=True)
                    injured.append({"position": position,
                                    "is_key": position in KEY_POSITIONS})
                except Exception:
                    continue

        n   = len(injured)
        crit = int(any(i["is_key"] for i in injured))
        impact = round(min(n * 0.04, 0.25), 3)

        logger.info("  %s: %d lesionados, critico=%d", team_name, n, crit)
        return {"injured_count": n, "injury_impact": impact, "critical_injury": crit}

    except requests.RequestException as e:
        logger.warning("  ERROR Transfermarkt %s: %s", team_name, str(e)[:60])
        return _injury_fallback()

def _injury_fallback() -> Dict:
    return {"injured_count": 0, "injury_impact": 0.0, "critical_injury": 0}

# ── CLIMA ─────────────────────────────────────────────────────────────────────

def get_weather_openweathermap(city: str, api_key: Optional[str] = None) -> Dict:
    """
    Obtiene datos climaticos desde OpenWeatherMap.
    Retorna: temperature, humidity, wind_speed_kmh, precipitation_prob, difficulty_score.
    """
    key = api_key or OWM_API_KEY
    if not key:
        logger.warning("  OPENWEATHER_API_KEY no configurada — usando defaults")
        return _weather_defaults()

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        resp = requests.get(url, params={"q": city, "appid": key, "units": "metric"},
                            timeout=10)
        resp.raise_for_status()
        data = resp.json()

        temp     = float(data["main"].get("temp", 20))
        humidity = float(data["main"].get("humidity", 50))
        wind_ms  = float(data["wind"].get("speed", 0))
        wind_kmh = round(wind_ms * 3.6, 1)
        clouds   = float(data.get("clouds", {}).get("all", 0))
        precip   = round(clouds / 100.0, 3)

        # Difficulty: desviacion de condiciones ideales
        temp_factor = abs(temp - 20) / 20.0
        wind_factor = min(wind_kmh / 50.0, 1.0)
        difficulty  = round(min((temp_factor + wind_factor + precip) / 3.0, 1.0), 3)

        logger.info("  %s: %.1f°C viento=%.1f km/h lluvia=%.0f%% dif=%.2f",
                    city, temp, wind_kmh, precip * 100, difficulty)

        return {
            "temperature":        round(temp, 1),
            "humidity":           round(humidity, 1),
            "wind_speed_kmh":     wind_kmh,
            "precipitation_prob": precip,
            "difficulty_score":   difficulty,
        }

    except requests.RequestException as e:
        logger.warning("  ERROR OpenWeatherMap %s: %s", city, str(e)[:60])
        return _weather_defaults()

def _weather_defaults() -> Dict:
    return {
        "temperature":        22.0,
        "humidity":           55.0,
        "wind_speed_kmh":     10.0,
        "precipitation_prob": 0.1,
        "difficulty_score":   0.2,
    }

# ── TECNICO ───────────────────────────────────────────────────────────────────

_MANAGER_CACHE: Dict[str, Dict] = {}

def get_manager_info(team_name: str) -> Dict:
    """
    Obtiene info del tecnico desde Wikipedia API.
    Retorna: manager, tenure_years.
    """
    if team_name in _MANAGER_CACHE:
        return _MANAGER_CACHE[team_name]

    try:
        url    = "https://en.wikipedia.org/w/api.php"
        params = {
            "action":  "query",
            "titles":  f"{team_name} national football team",
            "prop":    "revisions",
            "rvprop":  "content",
            "format":  "json",
            "rvsection": "0",
        }
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        content = ""
        for page in pages.values():
            revs = page.get("revisions", [])
            if revs:
                content = revs[0].get("*", "")

        # Buscar manager y año de inicio
        manager = "Unknown"
        tenure  = 1.0

        m = re.search(r"\|\s*manager\s*=\s*\[\[([^\]|]+)", content, re.IGNORECASE)
        if m:
            manager = m.group(1).strip()

        m_year = re.search(r"\|\s*managerstart\s*=\s*(\d{4})", content, re.IGNORECASE)
        if m_year:
            tenure = round(datetime.now().year - int(m_year.group(1)) + 0.5, 1)

        result = {"manager": manager, "tenure_years": tenure}
        _MANAGER_CACHE[team_name] = result
        logger.info("  %s: manager=%s tenure=%.1f y", team_name, manager, tenure)
        return result

    except Exception as e:
        logger.warning("  ERROR Wikipedia %s: %s", team_name, str(e)[:60])
        result = {"manager": "Unknown", "tenure_years": 1.0}
        _MANAGER_CACHE[team_name] = result
        return result

# ── FUNCION PRINCIPAL ─────────────────────────────────────────────────────────

def enrich_match(team_home: str, team_away: str,
                  date: str, city: str,
                  api_key_ow: Optional[str] = None) -> Dict:
    """
    Enriquece un partido con datos en tiempo real.

    Retorna dict con 12 features nuevas listas para agregar al vector de prediccion:
      - injured_players_count_home/away
      - injury_impact_score_home/away
      - critical_position_injured_home/away
      - temperature_match, precipitation_probability, wind_speed_kmh, weather_difficulty_score
      - manager_tenure_years_home/away
    """
    logger.info("Enriching: %s vs %s @ %s (%s)", team_home, team_away, city, date)

    t0 = time.time()

    inj_h = get_injuries_transfermarkt(team_home)
    inj_a = get_injuries_transfermarkt(team_away)
    weather = get_weather_openweathermap(city, api_key_ow)
    mgr_h = get_manager_info(team_home)
    mgr_a = get_manager_info(team_away)

    features = {
        "injured_players_count_home":     float(inj_h["injured_count"]),
        "injured_players_count_away":     float(inj_a["injured_count"]),
        "injury_impact_score_home":       inj_h["injury_impact"],
        "injury_impact_score_away":       inj_a["injury_impact"],
        "critical_position_injured_home": float(inj_h["critical_injury"]),
        "critical_position_injured_away": float(inj_a["critical_injury"]),
        "temperature_match":              weather["temperature"],
        "precipitation_probability":      weather["precipitation_prob"],
        "wind_speed_kmh":                 weather["wind_speed_kmh"],
        "weather_difficulty_score":       weather["difficulty_score"],
        "manager_tenure_years_home":      mgr_h["tenure_years"],
        "manager_tenure_years_away":      mgr_a["tenure_years"],
    }

    elapsed = round(time.time() - t0, 2)
    logger.info("Enriquecimiento completado en %.2f s", elapsed)
    logger.info("Features: %s", json.dumps(features, indent=2))
    return features


# ── REENTRENAMIENTO CON NUEVAS FEATURES ──────────────────────────────────────

def update_features_csv(new_features: Dict, match_id: str):
    """
    Agrega features en tiempo real a features_engineered.csv para un match_id dado.
    """
    feat_path = ROOT / "data" / "features_engineered.csv"
    if not feat_path.exists():
        logger.error("features_engineered.csv no encontrado")
        return

    df = pd.read_csv(feat_path) if "pandas" in sys.modules else None
    if df is None:
        import pandas as pd
        df = pd.read_csv(feat_path)

    mask = df["match_id"].astype(str) == str(match_id)
    if not mask.any():
        logger.warning("match_id %s no encontrado", match_id)
        return

    for col, val in new_features.items():
        df.loc[mask, col] = val

    df.to_csv(feat_path, index=False)
    logger.info("features_engineered.csv actualizado para match_id=%s", match_id)


if __name__ == "__main__":
    import sys

    result = enrich_match(
        team_home="Argentina",
        team_away="France",
        date="2026-06-14",
        city="East Rutherford",
    )
    print("\n12 Features de tiempo real:")
    for k, v in result.items():
        print(f"  {k:45s}: {v}")
