"""
04_about.py — Metodologia y metricas del modelo
"""
import streamlit as st

st.set_page_config(page_title="Acerca del modelo", page_icon="ℹ️", layout="wide")
st.title("ℹ️ Metodologia del Modelo")

st.markdown("""
## Pipeline de Prediccion

Este sistema combina **machine learning supervisado** con **simulacion Monte Carlo** para
estimar probabilidades de resultado en partidos del Mundial 2026.

---

### Datos de Entrenamiento

| Fuente | Descripcion | Cobertura |
|--------|-------------|-----------|
| martj42/international-football-results | Resultados internacionales historicos | 2017–2025 |
| ELO propio | Calculado desde 1872 sobre 49,477 partidos | Todos |
| OpenWeather API | Temperatura y humedad (features climaticos) | Cuando disponible |

- **4,401 partidos** tras filtrar a torneos mayores (WC, Euro, Copa America, AFCON, etc.)
- **14 columnas base**: date, teams, goals, competition, ELO, resultado

---

### Feature Engineering (Fase 2)

**141 features** organizados en 12 grupos, todos con proteccion anti-leakage (`shift(1)` antes de rolling):

| Grupo | Descripcion | Features |
|-------|-------------|----------|
| G1 | Fuerza ELO | 12 |
| G2 | Forma reciente (3/5/10 partidos) | 24+ |
| G3 | Ofensiva | 12 |
| G4 | Defensiva | 12 |
| G5 | Contexto (local, neutral, etc.) | 8 |
| G6 | Head-to-head | 6 |
| G7 | Competicion | 4 |
| G8 | Stats avanzadas estimadas | 8 |
| G9 | Clima | 5 |
| G10 | Psicologia (momentum, racha) | 6 |
| G11 | Patrones (remontadas, etc.) | 8 |
| G12 | Derivados (ratios) | 12 |

**Top features por correlacion Spearman:**
- `elo_ratio` (0.60)
- `elo_win_prob_home` (0.60)
- `elo_diff` (0.60)
- `goal_balance_diff` (0.42)

---

### Modelos CatBoost (Fase 3)

**Division temporal** (no aleatoria):

| Split | Periodo | Partidos |
|-------|---------|----------|
| Train | 2017–2023 | 3,133 |
| Valid | 2024 | 640 |
| Test  | 2025+ | 628 |

| Modelo | Tipo | Metrica | Valor |
|--------|------|---------|-------|
| `model_resultado` | Clasificacion 3 clases | Accuracy | **63.7%** |
| `model_resultado` | — | F1 macro | **58.8%** |
| `model_goals_home` | Regresion Poisson | MAE | **0.99** |
| `model_goals_away` | Regresion Poisson | MAE | **0.85** |
| `model_over25` | Clasificacion binaria | AUC | **0.726** |

**Hiperparametros**: `iterations=500`, `learning_rate=0.05`, `depth=6`, early stopping 50 rondas.

---

### Simulacion Monte Carlo (Fase 4)

Para cada partido se realizan **100,000 simulaciones** con distribucion de Poisson:

```
goals_home ~ Poisson(lambda_home)
goals_away ~ Poisson(lambda_away)
```

Donde `lambda_home` y `lambda_away` son las predicciones del modelo de regresion.

Esto permite estimar:
- Probabilidad exacta de cada marcador (ej. 1-0 = 14.2%)
- Over/Under 2.5 goles
- Ambos equipos marcan
- Porteria a cero
- Distribucion completa de goles

---

### Limitaciones y Disclaimer

> **Este modelo es un sistema experimental para fines academicos/investigacion.**
> Las predicciones NO constituyen consejo de apuestas.

- El modelo fue entrenado sobre partidos 2017–2025; equipos sin historial reciente tendran menor precision
- No considera lesiones, sanciones ni alineaciones en tiempo real
- Los clasificadores de futbol tienen un techo natural de ~65% accuracy (alta varianza del deporte)
- Accuracy de referencia naive (siempre predecir local gana): ~46%
""")

st.divider()

c1, c2, c3 = st.columns(3)
c1.metric("Partidos entrenamiento", "4,401")
c2.metric("Features seleccionadas", "118 (de 141)")
c3.metric("Accuracy test resultado", "63.7%")

c4, c5, c6 = st.columns(3)
c4.metric("AUC Over 2.5",     "0.726")
c5.metric("MAE goles local",  "0.99")
c6.metric("MAE goles visita", "0.85")

st.caption("Modelo entrenado con Python 3.13 · CatBoost 1.2.10 · scikit-learn 1.9.0")
