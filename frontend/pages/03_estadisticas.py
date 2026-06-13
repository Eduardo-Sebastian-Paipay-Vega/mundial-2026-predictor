"""
03_estadisticas.py — Estadisticas de equipos y ELO
"""
import os
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# URL de la API: primero session_state (seteado por app.py), luego secrets, luego env, luego localhost
API = st.session_state.get("API_URL") or (
    st.secrets.get("api", {}).get("url") if hasattr(st, "secrets") else None
) or os.environ.get("API_URL", "http://localhost:8000")
DATA_DIR = Path(__file__).parent.parent.parent / "data"

st.set_page_config(page_title="Estadisticas", page_icon="📈", layout="wide")
st.title("📈 Estadisticas de Equipos")


@st.cache_data(ttl=3600)
def get_teams():
    try:
        r = requests.get(f"{API}/teams", timeout=5)
        return r.json().get("teams", [])
    except Exception:
        return []


@st.cache_data(ttl=300)
def get_stats(team: str):
    try:
        r = requests.get(f"{API}/stats/{team}", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600)
def load_history():
    path = DATA_DIR / "matches_cleaned.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


teams    = get_teams()
df_hist  = load_history()

if not teams:
    st.error("API no disponible")
    st.stop()

# ── SELECTOR ─────────────────────────────────────────────────────────────────

col_a, col_b = st.columns([1, 1])
default_a = teams.index("Argentina") if "Argentina" in teams else 0
default_b = teams.index("Brazil") if "Brazil" in teams else 1

team_a = col_a.selectbox("Equipo A", teams, index=default_a, key="ea")
team_b = col_b.selectbox("Equipo B (comparar)", teams, index=default_b, key="eb")

# ── STATS CARD ────────────────────────────────────────────────────────────────

def render_stats_card(team: str, col):
    stats = get_stats(team)
    if stats is None:
        col.error(f"No se encontraron datos para {team}")
        return

    col.subheader(f"{team}")
    m1, m2, m3 = col.columns(3)
    m1.metric("ELO",         f"{stats['elo_rating']:.0f}")
    m2.metric("Partidos",    stats["total_matches"])
    m3.metric("Forma",       stats["recent_form"])

    m4, m5, m6 = col.columns(3)
    m4.metric("Victorias",   f"{stats['win_rate']*100:.1f}%")
    m5.metric("Empates",     f"{stats['draw_rate']*100:.1f}%")
    m6.metric("Derrotas",    f"{stats['loss_rate']*100:.1f}%")

    m7, m8, m9 = col.columns(3)
    m7.metric("xGF",         stats["avg_goals_for"])
    m8.metric("xGA",         stats["avg_goals_against"])
    m9.metric("Over 2.5",    f"{stats['over25_rate']*100:.1f}%")

    # Donut
    labels = ["Victorias", "Empates", "Derrotas"]
    values = [stats["win_rate"], stats["draw_rate"], stats["loss_rate"]]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55, marker_colors=["#1f77b4", "#aec7e8", "#ff7f0e"],
        textinfo="label+percent",
    ))
    fig.update_layout(height=230, margin=dict(l=0, r=0, t=10, b=0),
                      showlegend=False)
    col.plotly_chart(fig, use_container_width=True)


col_left, col_right = st.columns(2)
render_stats_card(team_a, col_left)
render_stats_card(team_b, col_right)

st.divider()

# ── ELO TIMELINE ─────────────────────────────────────────────────────────────

if not df_hist.empty:
    st.subheader("Evolucion ELO")

    def elo_series(team: str):
        rows = []
        for _, r in df_hist.sort_values("date").iterrows():
            if r["team_home"] == team and pd.notna(r.get("elo_home")):
                rows.append({"date": r["date"], "elo": r["elo_home"]})
            elif r["team_away"] == team and pd.notna(r.get("elo_away")):
                rows.append({"date": r["date"], "elo": r["elo_away"]})
        return pd.DataFrame(rows)

    s_a = elo_series(team_a)
    s_b = elo_series(team_b)

    fig_elo = go.Figure()
    if not s_a.empty:
        fig_elo.add_trace(go.Scatter(
            x=s_a["date"], y=s_a["elo"], mode="lines",
            name=team_a, line=dict(color="#1f77b4", width=2),
        ))
    if not s_b.empty:
        fig_elo.add_trace(go.Scatter(
            x=s_b["date"], y=s_b["elo"], mode="lines",
            name=team_b, line=dict(color="#ff7f0e", width=2),
        ))
    fig_elo.update_layout(
        height=350, xaxis_title="Fecha", yaxis_title="ELO Rating",
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_elo, use_container_width=True)

    st.divider()

    # Distribucion de goles
    st.subheader("Distribucion de Goles Marcados (ultimos 30 partidos)")
    c_left, c_right = st.columns(2)

    def goal_dist(team: str, col):
        mask = (df_hist["team_home"] == team) | (df_hist["team_away"] == team)
        recent = df_hist[mask].sort_values("date").tail(30)
        gf = []
        for _, r in recent.iterrows():
            g = r["goals_home"] if r["team_home"] == team else r["goals_away"]
            if pd.notna(g):
                gf.append(int(g))
        if not gf:
            return
        s = pd.Series(gf).value_counts().sort_index().reset_index()
        s.columns = ["Goles", "Partidos"]
        fig = px.bar(s, x="Goles", y="Partidos", color="Goles",
                     color_continuous_scale="Blues", title=team)
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0),
                          coloraxis_showscale=False)
        col.plotly_chart(fig, use_container_width=True)

    goal_dist(team_a, c_left)
    goal_dist(team_b, c_right)
