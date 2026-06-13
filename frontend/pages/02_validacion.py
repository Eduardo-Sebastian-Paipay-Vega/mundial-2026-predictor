"""
02_validacion.py — Validacion de predicciones vs resultados reales
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).parent.parent.parent / "data"

st.set_page_config(page_title="Validacion", page_icon="✅", layout="wide")
st.title("✅ Validacion del Modelo")
st.caption("Comparacion de predicciones contra resultados reales")


@st.cache_data(ttl=300)
def load_validation_data():
    """Carga predicciones del CSV y las une con resultados reales."""
    pred_path = DATA_DIR / "predictions_detailed.csv"
    hist_path = DATA_DIR / "matches_cleaned.csv"

    if not pred_path.exists():
        return None, "No se encontro predictions_detailed.csv. Ejecuta predict_mundial.py primero."

    df_pred = pd.read_csv(pred_path)
    df_pred["date"] = pd.to_datetime(df_pred["date"])

    if not hist_path.exists():
        return df_pred, None

    df_hist = pd.read_csv(hist_path)
    df_hist["date"] = pd.to_datetime(df_hist["date"])

    # Unir por equipo local + visitante + fecha
    df = df_pred.merge(
        df_hist[["date", "team_home", "team_away", "goals_home", "goals_away", "resultado"]],
        on=["date", "team_home", "team_away"],
        how="left",
    )

    # Prediccion del modelo
    df["pred_resultado"] = df.apply(
        lambda r: 1 if r["p_home"] >= r["p_away"] and r["p_home"] >= r["p_draw"]
                  else (-1 if r["p_away"] > r["p_home"] and r["p_away"] > r["p_draw"]
                        else 0),
        axis=1,
    )

    # Resultado real (solo partidos ya jugados)
    df["resultado_real"] = pd.to_numeric(df.get("resultado_x",
                                                 df.get("resultado", None)),
                                         errors="coerce")
    df["correct"] = (df["pred_resultado"] == df["resultado_real"]).astype(float)
    return df, None


df, error = load_validation_data()

if error:
    st.warning(error)
    st.stop()

if df is None:
    st.error("No se pudieron cargar los datos")
    st.stop()

# Solo partidos con resultado real
df_played = df[df["resultado_real"].notna()].copy()
n_total   = len(df)
n_played  = len(df_played)

# ── METRICAS GLOBALES ─────────────────────────────────────────────────────────

if n_played > 0:
    accuracy = df_played["correct"].mean()
    n_correct = int(df_played["correct"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Partidos con prediccion", n_total)
    c2.metric("Jugados",                 n_played)
    c3.metric("Correctos",               n_correct)
    c4.metric("Accuracy",                f"{accuracy*100:.1f}%")

    st.divider()

    # Breakdown por tipo de resultado
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Accuracy por Resultado Real")
        breakdown = df_played.groupby("resultado_real")["correct"].agg(["mean", "count"]).reset_index()
        breakdown["resultado_real"] = breakdown["resultado_real"].map(
            {1: "Victoria Local", 0: "Empate", -1: "Victoria Visitante"}
        )
        breakdown.columns = ["Resultado", "Accuracy", "N"]
        breakdown["Accuracy %"] = (breakdown["Accuracy"] * 100).round(1)

        fig_br = px.bar(
            breakdown, x="Resultado", y="Accuracy %",
            color="Accuracy %", color_continuous_scale="Blues",
            text="Accuracy %",
        )
        fig_br.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_br.update_layout(height=350, coloraxis_showscale=False,
                             margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_br, use_container_width=True)

    with col_right:
        st.subheader("Calibracion: P(Local) vs Resultado")
        df_cal = df_played.copy()
        df_cal["bin"] = pd.cut(df_cal["p_home"], bins=10, labels=False)
        cal = df_cal.groupby("bin").agg(
            mean_pred=("p_home", "mean"),
            mean_real=("correct", "mean"),
            count=("correct", "count"),
        ).reset_index()

        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(
            x=cal["mean_pred"], y=cal["mean_real"],
            mode="markers+lines",
            marker=dict(size=cal["count"] / cal["count"].max() * 30 + 5,
                        color="#1f77b4"),
            name="Calibracion real",
        ))
        fig_cal.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines", line=dict(dash="dash", color="gray"),
            name="Calibracion perfecta",
        ))
        fig_cal.update_layout(
            height=350, xaxis_title="P predicha", yaxis_title="Fraccion correcta",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_cal, use_container_width=True)

    st.divider()

    # Tabla detallada
    st.subheader("Detalle por Partido")
    df_show = df_played[["date", "team_home", "team_away",
                          "p_home", "p_draw", "p_away",
                          "resultado_real", "pred_resultado", "correct"]].copy()
    df_show["date"] = df_show["date"].dt.strftime("%Y-%m-%d")
    df_show["p_home"] = (df_show["p_home"] * 100).round(1)
    df_show["p_draw"] = (df_show["p_draw"] * 100).round(1)
    df_show["p_away"] = (df_show["p_away"] * 100).round(1)
    df_show["resultado_real"] = df_show["resultado_real"].map(
        {1: "L gana", 0: "Empate", -1: "V gana"}
    )
    df_show["pred_resultado"] = df_show["pred_resultado"].map(
        {1: "L gana", 0: "Empate", -1: "V gana"}
    )
    df_show["correcto"] = df_show["correct"].map({1.0: "Si", 0.0: "No"})
    df_show.columns = ["Fecha", "Local", "Visitante", "P(L)%", "P(E)%", "P(V)%",
                       "Real", "Prediccion", "_", "Correcto"]
    st.dataframe(df_show.drop(columns=["_"]), hide_index=True,
                 use_container_width=True)
else:
    st.info(f"{n_total} predicciones cargadas, pero ninguna tiene resultado real todavia.")
    if n_total > 0:
        st.dataframe(df[["date", "team_home", "team_away",
                          "p_home", "p_draw", "p_away"]].head(20),
                     hide_index=True, use_container_width=True)
