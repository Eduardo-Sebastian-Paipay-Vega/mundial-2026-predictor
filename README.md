# Pipeline de Datos Completo — Modelo Predictivo Mundial 2026

Script `descarga_datos_completo.py`: expansión de `descarga_datos.py` con
climatología por venue, datos de estadios, lesiones y valores de plantilla.

---

## Quick Start

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar variables de entorno
copy .env.example .env
# Editar .env — la key de Football-Data ya está incluida

# 3. Ejecutar el pipeline completo
python descarga_datos_completo.py
```

---

## Fuentes de datos

| Fuente | Qué descarga | Tiempo est. | Tamaño est. | Key requerida |
|---|---|---|---|---|
| StatsBomb Open Data | Partidos con xG, tiros, pases, stats por jugador | 30–90 min | 50–200 MB caché | No |
| Football-Data.org | Resultados WC/Euro/NL (2000, 2016, 2019) | 5–15 min | < 1 MB | No (opcional mejora) |
| ELO (martj42) | Ratings ELO calculados desde 50k+ resultados históricos | 3–8 min | ~10 MB caché | No |
| FIFA Rankings | Rankings mensuales históricos 2017–2025 | 5–10 min | < 2 MB | No |
| Open-Meteo Archive | Climatología Jun-Jul 5 años para 16 venues | 10–20 min | < 1 MB | No |
| OpenWeatherMap | Condiciones actuales en los 16 venues | 1–2 min | < 0.1 MB | Opcional (gratis) |
| NOAA CDO | Normales climáticas mensuales oficiales | 5–10 min | < 1 MB | Opcional (gratis) |
| Google Elevation | Altitud exacta por venue | 1 min | < 0.1 MB | Opcional (gratis tier) |
| Google Distance Matrix | Distancia de viaje equipo→venue | 2–5 min | < 0.1 MB | Opcional (gratis tier) |
| Transfermarkt (estadios) | Capacidad, tipo de césped | 5–15 min | < 0.1 MB | No (scraping) |
| Transfermarkt (lesiones) | Lesiones actuales por equipo | 10–20 min | < 0.5 MB | No (scraping) |
| Transfermarkt (jugadores) | Valores de mercado, top 3 por equipo | 10–20 min | < 1 MB | No (scraping) |

---

## Cómo obtener cada API key

### Football-Data.org (ya configurada)
La key `5f7f1fe50ad44cb6a5b77bec90e5987e` ya está en `.env.example`.
Proporciona acceso a FIFA World Cup (id=2000), UEFA Euro (id=2016) y UEFA Nations League (id=2019).
- Plan gratuito: 10 requests/minuto, cobertura completa de torneos principales.
- Registro: https://www.football-data.org/client/register

### OpenWeatherMap (opcional — gratis)
1. Crear cuenta en https://openweathermap.org/
2. Ir a perfil → "API Keys"
3. Copiar la key por defecto o generar una nueva
4. Pegar en `.env` como `OPENWEATHER_API_KEY=tu_key`
- Límite gratuito: 1,000 llamadas/día, suficiente para los 16 venues.

### Google Cloud Platform (opcional — gratis hasta límite mensual)
1. Ir a https://console.cloud.google.com/
2. Crear proyecto nuevo
3. Habilitar "Elevation API" y "Distance Matrix API" en "APIs & Services"
4. Ir a "Credentials" → "Create Credentials" → "API key"
5. Pegar en `.env` como `GOOGLE_MAPS_API_KEY=tu_key`
- Crédito gratuito mensual: $200 USD, cubre miles de llamadas.
- Sin key: el script usa Open-Meteo para altitud y haversine para distancias.

### NOAA CDO Token (opcional — completamente gratis)
1. Ir a https://www.ncdc.noaa.gov/cdo-web/token
2. Ingresar tu email
3. El token llega al instante al correo
4. Pegar en `.env` como `NOAA_TOKEN=tu_token`
- Límite: 1,000 requests/día, sin costo.
- Sin token: se omite NOAA y se usa Open-Meteo como fuente climática primaria.

---

## Diccionario de columnas — `matches_cleaned.csv`

| Columna | Tipo | Fuente | Descripción |
|---|---|---|---|
| `match_id` | str | StatsBomb/FD | ID único (`sb_XXXX` o `fd_XXXX`) |
| `date` | datetime | StatsBomb/FD | Fecha del partido |
| `season` | str | StatsBomb/FD | Temporada (ej. "2022") |
| `competition` | str | StatsBomb/FD | Nombre del torneo |
| `team_home` | str | StatsBomb/FD | Equipo local (normalizado, minúsculas) |
| `team_away` | str | StatsBomb/FD | Equipo visitante |
| `goals_home` | int | StatsBomb/FD | Goles del equipo local |
| `goals_away` | int | StatsBomb/FD | Goles del equipo visitante |
| `resultado` | int | calculado | 1=local gana, 0=empate, -1=visitante gana |
| `goles_totales` | int | calculado | `goals_home + goals_away` |
| `xg_home` | float | StatsBomb | Expected Goals equipo local |
| `xg_away` | float | StatsBomb | Expected Goals equipo visitante |
| `shots_home` | int | StatsBomb | Total tiros equipo local |
| `shots_away` | int | StatsBomb | Total tiros equipo visitante |
| `shots_on_target_home` | int | StatsBomb | Tiros a puerta equipo local |
| `shots_on_target_away` | int | StatsBomb | Tiros a puerta equipo visitante |
| `passes_home` | int | StatsBomb | Total pases equipo local |
| `passes_away` | int | StatsBomb | Total pases equipo visitante |
| `pass_accuracy_home` | float | StatsBomb | Precisión de pase local (%) |
| `pass_accuracy_away` | float | StatsBomb | Precisión de pase visitante (%) |
| `elo_home` | float | calculado (martj42) | Rating ELO pre-partido equipo local |
| `elo_away` | float | calculado (martj42) | Rating ELO pre-partido equipo visitante |
| `diff_elo` | float | calculado | `elo_home - elo_away` |
| `fifa_rank_home` | float | FIFA API | Ranking FIFA local (más reciente ≤ partido) |
| `fifa_rank_away` | float | FIFA API | Ranking FIFA visitante |
| `diff_fifa_rank` | float | calculado | `fifa_rank_away - fifa_rank_home` (positivo = local mejor clasificado) |
| `temp_avg_c` | float | Open-Meteo | Temperatura promedio Jun-Jul venue (°C) |
| `temp_max_avg_c` | float | Open-Meteo | Temperatura máxima promedio (°C) |
| `temp_min_avg_c` | float | Open-Meteo | Temperatura mínima promedio (°C) |
| `humidity_pct` | float | Open-Meteo | Humedad relativa promedio (%) |
| `wind_kmh` | float | Open-Meteo | Velocidad máxima de viento promedio (km/h) |
| `precip_mm` | float | Open-Meteo | Precipitación promedio diaria (mm) |
| `rain_days` | int | Open-Meteo | Días con lluvia > 1mm en período Jun-Jul |
| `altitude_m` | float | Open-Meteo/Google | Altitud del venue sobre el nivel del mar (m) |
| `injured_count_home` | float | Transfermarkt | Jugadores lesionados equipo local |
| `injured_count_away` | float | Transfermarkt | Jugadores lesionados equipo visitante |
| `critical_injury_home` | int | Transfermarkt | 1 si hay lesión crítica (estrella) en local |
| `critical_injury_away` | int | Transfermarkt | 1 si hay lesión crítica (estrella) en visitante |
| `avg_days_return_home` | float | Transfermarkt | Días promedio hasta retorno lesionados local |
| `avg_days_return_away` | float | Transfermarkt | Días promedio hasta retorno lesionados visitante |
| `squad_value_home_eur` | float | Transfermarkt | Valor total plantilla local (EUR) |
| `squad_value_away_eur` | float | Transfermarkt | Valor total plantilla visitante (EUR) |
| `diff_squad_val` | float | calculado | `squad_value_home - squad_value_away` (EUR) |
| `source` | str | meta | Fuente primaria del partido |

---

## Criticidad de fuentes

### CRITICAS — el pipeline falla sin ellas
| Fuente | Por qué es crítica |
|---|---|
| StatsBomb o Football-Data | Sin partidos no hay pipeline |
| ELO (martj42 CSV) | Feature más predictiva; descarga pública sin key |

### OPCIONALES — mejoran cobertura significativamente
| Fuente | Impacto si falta |
|---|---|
| Football-Data API key | Sin key funciona a 10 req/min; con key sin throttle |
| FIFA Rankings | ~20% de partidos sin ranking → columna NaN |
| Open-Meteo | Clima disponible sin key; sin él columnas clima = NaN |

### ENHANCEMENT — añaden valor diferencial
| Fuente | Qué aporta |
|---|---|
| Transfermarkt lesiones | Feature de contexto pre-partido única |
| Transfermarkt jugadores | Proxy de calidad de plantilla por valor de mercado |
| OpenWeatherMap | Condiciones el día del partido (tiempo real) |
| Google Elevation/Distance | Altitud exacta y penalización por viaje largo |
| NOAA CDO | Normales climáticas oficiales como validación |

---

## Flujo de datos

```
                    ┌─────────────────────────────────────────────┐
                    │           FUENTES DE DATOS                  │
                    └─────────────────────────────────────────────┘

  GRUPO A              GRUPO B          GRUPO C            GRUPO D/E/F
  Partidos             Rankings         Clima              Contexto
  ────────             ────────         ─────              ───────────
  StatsBomb ──┐        ELO ────┐        Open-Meteo ──┐    TM Estadios ──┐
  Football-D ─┤        FIFA ───┤        OpenWeather ─┤    TM Lesiones ──┤
              │                │        NOAA ─────── ┤    TM Jugadores ─┤
              ▼                │                     │    Google Alt. ──┤
       ┌─────────────┐         │                     │    Google Dist. ─┘
       │ df_matches  │         │                     │         │
       │ df_players  │         │                     │         │
       └──────┬──────┘         │                     │         │
              │                │                     │         │
              ▼                ▼                     ▼         ▼
       ┌──────────────────────────────────────────────────────────┐
       │              consolida_datos_maestro()                    │
       │                                                           │
       │  1. dedup (statsbomb > football_data)                     │
       │  2. join ELO por (date, team_home, team_away)             │
       │  3. join FIFA por merge_asof (tolerancia 35D)             │
       │  4. join clima por venue_id → country fallback            │
       │  5. join lesiones por equipo (snapshot actual)            │
       │  6. join squad values por equipo                          │
       │  7. calcular diff_elo, diff_fifa_rank, resultado          │
       └──────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ valida_calidad() │
                         │ rangos físicos   │
                         │ nulos críticos   │
                         │ recalc objetivo  │
                         └────────┬────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
     matches_full.csv   matches_cleaned.csv   data_quality_report.json
     stadiums_data.csv  climate_data.csv      players_key.csv
     injuries_current.csv                     pipeline.log
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'cloudscraper'`
```bash
pip install cloudscraper==1.2.71
```
Sin cloudscraper, los grupos D/E/F (Transfermarkt) se omiten con advertencia.

### Transfermarkt devuelve HTTP 403 o bloqueo Cloudflare
Aumentar `SCRAPE_DELAY` en `.env` a `2.5` o `3.0`.
Si el bloqueo persiste, el script registra la advertencia y continúa con datos de `WC2026_VENUES`.

### StatsBomb descarga 0 partidos
Los datos gratuitos cubren torneos específicos. Verificar que `START_YEAR`/`END_YEAR` incluyan 2018, 2021 o 2022.

### Football-Data devuelve HTTP 403
El plan gratuito sin key tiene acceso limitado. La key del `.env.example` debe cubrir los torneos configurados.

### Open-Meteo tarda mucho
El pipeline hace 2 llamadas por año por venue (16 venues × 5 años × 2 = 160 llamadas).
Con `sleep(0.5)` entre llamadas, el total es ~80 segundos mínimo.
El resultado se guarda en `data/cache/climate_openmeteo.csv` para evitar re-descargas.

### FIFA Rankings retorna error de conexión
La API `api.fifa.com` es no oficial y puede cambiar. El script tiene scraping de respaldo automático.
Si ambos fallan, el pipeline continúa con `fifa_rank_home/away = NaN`.

### El pipeline tarda más de 3 horas
StatsBomb descarga eventos partido por partido. Es el cuello de botella principal.
Para pruebas rápidas, reducir el período: `START_YEAR=2022 END_YEAR=2022` en `.env`.

### ELO da 1500 para muchos equipos
El ELO inicial es 1500 para equipos sin historial. Equipos con pocos partidos en el período configurado convergen lentamente. El cache de `international_results.csv` incluye partidos desde 1872, por lo que todos los equipos con historia relevante tendrán ELO calculado.

### `data/cache/` ocupa mucho espacio
El archivo `international_results.csv` pesa ~10 MB. Los eventos StatsBomb no se cachean (son temporales en RAM). Para limpiar: `rm -rf data/cache/` y volver a ejecutar.
