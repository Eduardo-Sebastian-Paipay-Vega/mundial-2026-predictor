"""
01_predicciones.py — Prediccion detallada con distribucion MC
"""
import os
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# URL de la API: primero session_state (seteado por app.py), luego secrets, luego env, luego localhost
API = st.session_state.get("API_URL") or (
    st.secrets.get("api", {}).get("url") if hasattr(st, "secrets") else None
) or os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Predicciones", page_icon="📊", layout="wide")
st.title("📊 Prediccion Detallada")

# ── EQUIPOS ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_teams():
    try:
        r = requests.get(f"{API}/teams", timeout=5)
        return r.json().get("teams", [])
    except Exception:
        return ["Argentina", "France", "Brazil", "Germany", "Spain", "England",
                "Portugal", "Netherlands", "Morocco", "Japan", "Mexico", "USA"]

teams = get_teams()

# ── FORMULARIO ────────────────────────────────────────────────────────────────

with st.form("prediction_form"):
    c1, c2 = st.columns(2)
    team_home = c1.selectbox("Equipo Local", teams,
                             index=teams.index("Argentina") if "Argentina" in teams else 0)
    team_away = c2.selectbox("Equipo Visitante", teams,
                             index=teams.index("France") if "France" in teams else 1)

    c3, c4, c5 = st.columns(3)
    match_date = c3.date_input("Fecha", value=pd.Timestamp("2026-06-14"))
    city       = c4.text_input("Ciudad", value="East Rutherford")
    n_sim      = c5.select_slider("Simulaciones MC",
                                  options=[10_000, 50_000, 100_000, 200_000],
                                  value=100_000)

    submitted = st.form_submit_button("Calcular Prediccion", type="primary",
                                      use_container_width=True)

if not submitted:
    st.stop()

# ── PREDICCION ────────────────────────────────────────────────────────────────

if team_home == team_away:
    st.error("Los equipos deben ser diferentes")
    st.stop()

with st.spinner(f"Simulando {n_sim:,} partidos..."):
    try:
        resp = requests.post(f"{API}/predict", json={
            "team_home": team_home,
            "team_away": team_away,
            "date":      str(match_date),
            "city":      city,
            "neutral":   True,
            "n_sim":     n_sim,
        }, timeout=60)
        if resp.status_code != 200:
            st.error(f"API error {resp.status_code}: {resp.text}")
            st.stop()
        d = resp.json()
    except Exception as e:
        st.error(f"No se pudo conectar con la API: {e}")
        st.stop()

# ── RESULTADO ─────────────────────────────────────────────────────────────────

st.divider()
st.subheader(f"{team_home} vs {team_away}")

# Probabilidades principales
col_a, col_b, col_c = st.columns(3)
col_a.metric(team_home,  f"{d['mc_p_home']*100:.1f}%",
             delta=f"ML: {d['p_home']*100:.1f}%")
col_b.metric("Empate",   f"{d['mc_p_draw']*100:.1f}%",
             delta=f"ML: {d['p_draw']*100:.1f}%")
col_c.metric(team_away,  f"{d['mc_p_away']*100:.1f}%",
             delta=f"ML: {d['p_away']*100:.1f}%")

# ELO y xG
col_d, col_e, col_f, col_g = st.columns(4)
col_d.metric("ELO Local",     f"{d['elo_home']:.0f}")
col_e.metric("xG Local",      f"{d['lambda_home']:.2f}")
col_f.metric("xG Visitante",  f"{d['lambda_away']:.2f}")
col_g.metric("ELO Visitante", f"{d['elo_away']:.0f}")

col_h, col_i, col_j = st.columns(3)
col_h.metric("Over 2.5",      f"{d['mc_p_over25']*100:.1f}%")
col_i.metric("Ambos Marcan",  f"{d['mc_p_both_score']*100:.1f}%")
col_j.metric("Confianza",     d["confidence"])

st.divider()

# ── MARCADORES MAS PROBABLES ──────────────────────────────────────────────────

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Top 10 Marcadores Probables")
    if d.get("top_scores"):
        df_sc = pd.DataFrame(d["top_scores"]).head(10)
        df_sc["Probabilidad %"] = (df_sc["probability"] * 100).round(2)
        fig_sc = px.bar(
            df_sc,
            x="score", y="Probabilidad %",
            color="Probabilidad %",
            color_continuous_scale="Blues",
            labels={"score": "Marcador", "Probabilidad %": "%"},
        )
        fig_sc.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                             coloraxis_showscale=False)
        st.plotly_chart(fig_sc, use_container_width=True)

with col_right:
    st.subheader("Probabilidades Comparadas")
    categories = ["Local", "Empate", "Visitante"]
    ml_vals    = [d["p_home"], d["p_draw"], d["p_away"]]
    mc_vals    = [d["mc_p_home"], d["mc_p_draw"], d["mc_p_away"]]

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name="CatBoost ML", x=categories,
        y=[v * 100 for v in ml_vals],
        marker_color="#1f77b4",
    ))
    fig_comp.add_trace(go.Bar(
        name="Monte Carlo", x=categories,
        y=[v * 100 for v in mc_vals],
        marker_color="#ff7f0e",
    ))
    fig_comp.update_layout(
        barmode="group", height=320,
        yaxis_title="%",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# ── RESUMEN TEXTUAL ───────────────────────────────────────────────────────────

st.divider()
fav   = d["favorite"]
ph    = d["mc_p_home"] * 100
pd_   = d["mc_p_draw"] * 100
pa    = d["mc_p_away"] * 100
lh, la = d["lambda_home"], d["lambda_away"]
st.info(
    f"**Favorito: {fav}** | "
    f"{team_home} {ph:.1f}% — Empate {pd_:.1f}% — {team_away} {pa:.1f}% | "
    f"Goles esperados: {lh:.2f} – {la:.2f} | "
    f"Over 2.5: {d['mc_p_over25']*100:.1f}%"
)
