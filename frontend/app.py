"""
app.py — Pagina principal del dashboard Mundial 2026
"""
import os
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Mundial 2026 Predictor",
    page_icon="⚽",
    layout="wide",
)

# ── URL DE LA API (configurable para produccion) ──────────────────────────────
# Orden de prioridad:
#   1. st.secrets["api"]["url"]  (Streamlit Cloud dashboard / .streamlit/secrets.toml local)
#   2. Variable de entorno API_URL
#   3. Fallback a localhost para desarrollo
try:
    API = st.secrets["api"]["url"]
except Exception:
    API = os.environ.get("API_URL", "http://localhost:8000")

# Guardar en session_state para que las paginas puedan reutilizarla
st.session_state["API_URL"] = API

# ── HEADER ────────────────────────────────────────────────────────────────────

st.title("⚽ Mundial 2026 — Sistema de Prediccion")
st.caption("CatBoost + Monte Carlo Poisson | 4,401 partidos historicos | 141 features")

# ── SIDEBAR — URL activa (util para debugging) ────────────────────────────────
with st.sidebar:
    st.caption(f"**API:** `{API}`")

# ── HEALTH CHECK ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def get_health():
    try:
        r = requests.get(f"{API}/health", timeout=3)
        return r.json()
    except Exception:
        return None

health = get_health()

if health:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Estado API", "En linea" if health["status"] == "ok" else "Error")
    col2.metric("Partidos historicos", f"{health['historical_matches']:,}")
    col3.metric("Features ML", health["n_features"])
    col4.metric("Version", health["version"])
else:
    st.error("API no disponible — inicia el backend: `uvicorn backend.main:app --reload --port 8000`")
    st.stop()

st.divider()

# ── ULTIMAS PREDICCIONES ─────────────────────────────────────────────────────

@st.cache_data(ttl=120)
def get_latest_predictions(n=10):
    try:
        r = requests.get(f"{API}/predictions/latest?limit={n}", timeout=5)
        return r.json().get("predictions", [])
    except Exception:
        return []

preds = get_latest_predictions(10)

col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("Ultimas Predicciones")
    if preds:
        df_preds = pd.DataFrame(preds)
        df_preds["Partido"] = df_preds["team_home"] + " vs " + df_preds["team_away"]
        df_preds["Local %"]  = (df_preds["p_home"] * 100).round(1)
        df_preds["Empate %"] = (df_preds["p_draw"] * 100).round(1)
        df_preds["Visita %"] = (df_preds["p_away"] * 100).round(1)
        df_preds["xG L"]    = df_preds["lambda_home"].round(2)
        df_preds["xG V"]    = df_preds["lambda_away"].round(2)

        st.dataframe(
            df_preds[["date", "Partido", "Local %", "Empate %", "Visita %",
                       "xG L", "xG V", "favorite"]],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Sin predicciones guardadas aun. Ejecuta `predict_mundial.py` para generarlas.")

with col_right:
    st.subheader("Probabilidad Local por Partido")
    if preds:
        df_chart = pd.DataFrame(preds)
        df_chart["Partido"] = df_chart["team_home"] + " vs " + df_chart["team_away"]
        fig = px.bar(
            df_chart,
            x="p_home",
            y="Partido",
            orientation="h",
            color="p_home",
            color_continuous_scale="Blues",
            range_x=[0, 1],
            labels={"p_home": "P(Local)", "Partido": ""},
        )
        fig.update_layout(
            height=380,
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.empty()

st.divider()

# ── PREDICCION RAPIDA ─────────────────────────────────────────────────────────

st.subheader("Prediccion Rapida")

@st.cache_data(ttl=3600)
def get_teams():
    try:
        r = requests.get(f"{API}/teams", timeout=5)
        return r.json().get("teams", [])
    except Exception:
        return []

teams = get_teams()

c1, c2, c3, c4 = st.columns([1.5, 1.5, 1, 1])
team_home = c1.selectbox("Equipo local", teams, index=teams.index("Argentina") if "Argentina" in teams else 0, key="qh")
team_away = c2.selectbox("Equipo visitante", teams, index=teams.index("France") if "France" in teams else 1, key="qa")
match_date = c3.date_input("Fecha", value=pd.Timestamp("2026-06-14"), key="qd")

if c4.button("Predecir", type="primary", use_container_width=True):
    if team_home == team_away:
        st.error("Los equipos deben ser diferentes")
    else:
        with st.spinner("Calculando..."):
            try:
                resp = requests.post(f"{API}/predict", json={
                    "team_home": team_home,
                    "team_away": team_away,
                    "date":      str(match_date),
                    "neutral":   True,
                    "n_sim":     50_000,
                }, timeout=30)
                data = resp.json()

                col_a, col_b, col_c = st.columns(3)
                ph   = data["p_home"] * 100
                pd_  = data["p_draw"] * 100
                pa   = data["p_away"] * 100
                col_a.metric(f"{team_home}", f"{ph:.1f}%")
                col_b.metric("Empate",       f"{pd_:.1f}%")
                col_c.metric(f"{team_away}", f"{pa:.1f}%")

                col_x, col_y, col_z = st.columns(3)
                col_x.metric("xG Local",     f"{data['lambda_home']:.2f}")
                col_y.metric("Over 2.5",     f"{data['mc_p_over25']*100:.1f}%")
                col_z.metric("xG Visitante", f"{data['lambda_away']:.2f}")

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=ph,
                    title={"text": f"P({team_home} gana) %"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": "#1f77b4"},
                        "steps": [
                            {"range": [0, 33], "color": "#ff7f0e"},
                            {"range": [33, 50], "color": "#ffbb78"},
                            {"range": [50, 100], "color": "#aec7e8"},
                        ],
                    },
                    number={"suffix": "%"},
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
