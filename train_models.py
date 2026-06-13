#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_models.py
===============
Fase 3 - Entrenamiento CatBoost
Modelo Predictivo Mundial 2026

Entrada:  data/features_engineered.csv  (4,401 x 148 col)
Salida:   models/model_resultado.pkl    (clasificacion 3 clases)
          models/model_goals_home.pkl   (regresion)
          models/model_goals_away.pkl   (regresion)
          models/model_over25.pkl       (clasificacion binaria)
          data/model_evaluation.json   (metricas por modelo)
          data/feature_importance_catboost.csv
"""

import json
import logging
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error, r2_score,
)

warnings.filterwarnings("ignore")

# ─── CONFIG ─────────────────────────────────────────────────────────────────

DATA_DIR   = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(DATA_DIR / "log_training.txt", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("training")

RANDOM_STATE = 42

# Split temporal
TRAIN_END = "2023-12-31"
VALID_END = "2024-12-31"
# Test: 2025-01-01 en adelante

# Columnas que nunca son features
NON_FEATURE_COLS = {
    "match_id", "date", "team_home", "team_away",
    "competition", "city", "country", "neutral",
    "target_resultado", "target_goals_home", "target_goals_away",
    "target_total_goals", "target_over25", "target_both_score",
    # columnas del CSV base que pudieran haberse arrastrado
    "goals_home", "goals_away", "resultado",
    "elo_home", "elo_away",   # ya estan como features con nombre distinto
}


# ─── ETAPA 1: SELECCION DE FEATURES ─────────────────────────────────────────

def selecciona_features(df: pd.DataFrame, corr_threshold: float = 0.95) -> List[str]:
    """
    Elimina multicolinealidad: de cada par con corr > threshold,
    retiene el de mayor correlacion absoluta con target_resultado.
    Retorna lista de features seleccionadas.
    """
    logger.info("Seleccion de features (umbral corr=%.2f)...", corr_threshold)

    # Candidatos: todas las columnas que no son non-feature
    candidates = [c for c in df.columns if c not in NON_FEATURE_COLS]

    # Cargar importancia Spearman (Fase 2)
    imp_path = DATA_DIR / "feature_importance_baseline.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path).set_index("feature")["abs_rho"]
    else:
        # Calcular on-the-fly
        from scipy import stats
        target = df["target_resultado"].fillna(0)
        imp_vals = {}
        for c in candidates:
            try:
                rho, _ = stats.spearmanr(df[c].fillna(0), target, nan_policy="omit")
                imp_vals[c] = abs(rho)
            except Exception:
                imp_vals[c] = 0.0
        imp = pd.Series(imp_vals)

    # Cargar matriz de correlacion
    corr_path = DATA_DIR / "features_correlation_matrix.csv"
    if corr_path.exists():
        corr_matrix = pd.read_csv(corr_path, index_col=0)
        # Mantener solo candidatos
        common = [c for c in candidates if c in corr_matrix.columns]
        corr_matrix = corr_matrix.loc[common, common].abs()
    else:
        logger.warning("No se encontro correlation matrix — usando todas las features")
        return candidates

    # Algoritmo greedy: eliminar la de menor importancia de cada par correlacionado
    to_drop = set()
    checked_cols = [c for c in candidates if c in corr_matrix.columns]

    for i, col_a in enumerate(checked_cols):
        if col_a in to_drop:
            continue
        for col_b in checked_cols[i+1:]:
            if col_b in to_drop:
                continue
            try:
                corr_val = float(corr_matrix.loc[col_a, col_b])
            except (KeyError, ValueError):
                continue
            if corr_val > corr_threshold:
                imp_a = float(imp.get(col_a, 0))
                imp_b = float(imp.get(col_b, 0))
                # Eliminar el de menor importancia
                to_drop.add(col_b if imp_a >= imp_b else col_a)

    selected = [c for c in candidates if c not in to_drop and c in df.columns]
    logger.info("  Candidatos: %d  |  Eliminados: %d  |  Seleccionados: %d",
                len(candidates), len(to_drop), len(selected))
    return selected


# ─── ETAPA 2: SPLIT TEMPORAL ─────────────────────────────────────────────────

def split_temporal(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split por fecha (no aleatorio — evita data leakage temporal).
    Retorna (train, valid, test).
    """
    logger.info("Split temporal...")

    # Necesitamos la columna 'date' — leerla desde CSV base si no esta
    if "date" not in df.columns:
        base = pd.read_csv(DATA_DIR / "matches_cleaned.csv", usecols=["match_id", "date"])
        df = df.merge(base, on="match_id", how="left")

    df["date"] = pd.to_datetime(df["date"])

    train = df[df["date"] <= TRAIN_END].copy()
    valid = df[(df["date"] > TRAIN_END) & (df["date"] <= VALID_END)].copy()
    test  = df[df["date"] > VALID_END].copy()

    logger.info("  Train: %d partidos (hasta %s)", len(train), TRAIN_END)
    logger.info("  Valid: %d partidos (%s - %s)", len(valid), TRAIN_END[:4]+"-01-01", VALID_END)
    logger.info("  Test:  %d partidos (2025+)", len(test))
    return train, valid, test


# ─── ETAPA 3: MODELOS ────────────────────────────────────────────────────────

def _xy(df: pd.DataFrame, features: List[str], target_col: str):
    """Extraer X, y desde un split."""
    X = df[features].fillna(0).astype(np.float32)
    y = df[target_col].fillna(0)
    return X, y


def _encode_resultado(y: pd.Series) -> pd.Series:
    """Mapear -1→0, 0→1, 1→2 para CatBoost classifier."""
    return y.map({-1: 0, 0: 1, 1: 2}).fillna(1).astype(int)


def _decode_resultado(y: np.ndarray) -> np.ndarray:
    """Invertir: 0→-1, 1→0, 2→1."""
    mapping = {0: -1, 1: 0, 2: 1}
    return np.vectorize(mapping.get)(y)


# ── MODELO 1: CLASIFICACION RESULTADO (3 clases) ─────────────────────────────

def train_model_resultado(
    train, valid, test, features: List[str]
) -> Tuple[CatBoostClassifier, Dict]:
    """
    CatBoostClassifier multiclase: predice 1=local, 0=empate, -1=visitante.
    Clases mapeadas: -1->0, 0->1, 1->2 internamente.
    """
    logger.info("Entrenando Modelo 1: Clasificacion Resultado...")

    X_tr, y_tr = _xy(train, features, "target_resultado")
    X_va, y_va = _xy(valid, features, "target_resultado")
    X_te, y_te = _xy(test,  features, "target_resultado")

    y_tr_enc = _encode_resultado(y_tr)
    y_va_enc = _encode_resultado(y_va)
    y_te_enc = _encode_resultado(y_te)

    # Distribucion de clases
    dist = y_tr_enc.value_counts().sort_index()
    logger.info("  Distribucion train — victoria_visita:%d empate:%d victoria_local:%d",
                dist.get(0, 0), dist.get(1, 0), dist.get(2, 0))

    # Pesos para compensar desbalance (mas empates = clase difícil)
    n = len(y_tr_enc)
    weights = {cls: round(n / (3 * cnt), 3) for cls, cnt in dist.items()}
    class_weights = [weights.get(i, 1.0) for i in range(3)]
    logger.info("  class_weights: %s", class_weights)

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=RANDOM_STATE,
        eval_metric="Accuracy",
        loss_function="MultiClass",
        class_weights=class_weights,
        verbose=100,
        use_best_model=True,
        early_stopping_rounds=50,
    )

    model.fit(
        X_tr, y_tr_enc,
        eval_set=(X_va, y_va_enc),
        verbose=100,
    )

    # Evaluacion
    y_pred_va = model.predict(X_va).flatten()
    y_pred_te = model.predict(X_te).flatten()

    metrics = {
        "valid_accuracy":  round(accuracy_score(y_va_enc, y_pred_va), 4),
        "test_accuracy":   round(accuracy_score(y_te_enc, y_pred_te), 4),
        "valid_f1_macro":  round(f1_score(y_va_enc, y_pred_va, average="macro"), 4),
        "test_f1_macro":   round(f1_score(y_te_enc, y_pred_te, average="macro"), 4),
        "best_iteration":  model.best_iteration_,
        "class_mapping":   {"-1 (vis gana)": 0, "0 (empate)": 1, "1 (local gana)": 2},
        "confusion_matrix_test": confusion_matrix(y_te_enc, y_pred_te).tolist(),
    }

    logger.info("  Valid accuracy:  %.4f", metrics["valid_accuracy"])
    logger.info("  Test  accuracy:  %.4f", metrics["test_accuracy"])
    logger.info("  Test  F1 macro:  %.4f", metrics["test_f1_macro"])

    return model, metrics


# ── MODELO 2: REGRESION GOALS HOME ───────────────────────────────────────────

def train_model_goals(
    train, valid, test, features: List[str], target: str, label: str
) -> Tuple[CatBoostRegressor, Dict]:
    """
    CatBoostRegressor para goles (distribucion Poisson-like).
    loss_function=Poisson captura la naturaleza de conteo.
    """
    logger.info("Entrenando Modelo: %s (%s)...", label, target)

    X_tr, y_tr = _xy(train, features, target)
    X_va, y_va = _xy(valid, features, target)
    X_te, y_te = _xy(test,  features, target)

    # Clip a rango razonable (0-8 goles)
    y_tr = y_tr.clip(0, 8)
    y_va = y_va.clip(0, 8)
    y_te = y_te.clip(0, 8)

    logger.info("  Media goles train: %.3f  |  std: %.3f",
                float(y_tr.mean()), float(y_tr.std()))

    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=RANDOM_STATE,
        loss_function="Poisson",
        eval_metric="RMSE",
        verbose=100,
        use_best_model=True,
        early_stopping_rounds=50,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        verbose=100,
    )

    y_pred_va = model.predict(X_va).clip(0, 8)
    y_pred_te = model.predict(X_te).clip(0, 8)

    metrics = {
        "valid_mae":      round(mean_absolute_error(y_va, y_pred_va), 4),
        "test_mae":       round(mean_absolute_error(y_te, y_pred_te), 4),
        "valid_rmse":     round(float(np.sqrt(mean_squared_error(y_va, y_pred_va))), 4),
        "test_rmse":      round(float(np.sqrt(mean_squared_error(y_te, y_pred_te))), 4),
        "test_r2":        round(r2_score(y_te, y_pred_te), 4),
        "best_iteration": model.best_iteration_,
        "train_mean_y":   round(float(y_tr.mean()), 3),
    }

    logger.info("  Test MAE:  %.4f  (promedio %s = %.2f)",
                metrics["test_mae"], target.replace("target_",""), metrics["train_mean_y"])
    logger.info("  Test RMSE: %.4f", metrics["test_rmse"])
    logger.info("  Test R2:   %.4f", metrics["test_r2"])

    return model, metrics


# ── MODELO 4: CLASIFICACION OVER 2.5 ─────────────────────────────────────────

def train_model_over25(
    train, valid, test, features: List[str]
) -> Tuple[CatBoostClassifier, Dict]:
    """
    CatBoostClassifier binario: predice si el partido tiene Over 2.5 goles.
    """
    logger.info("Entrenando Modelo 4: Clasificacion Over 2.5...")

    X_tr, y_tr = _xy(train, features, "target_over25")
    X_va, y_va = _xy(valid, features, "target_over25")
    X_te, y_te = _xy(test,  features, "target_over25")

    y_tr, y_va, y_te = y_tr.astype(int), y_va.astype(int), y_te.astype(int)

    dist = y_tr.value_counts()
    logger.info("  Over25 dist train — Over:%d Under:%d",
                dist.get(1, 0), dist.get(0, 0))

    model = CatBoostClassifier(
        iterations=400,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=RANDOM_STATE,
        eval_metric="AUC",
        loss_function="Logloss",
        verbose=100,
        use_best_model=True,
        early_stopping_rounds=40,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        verbose=100,
    )

    from sklearn.metrics import roc_auc_score
    y_pred_prob_va = model.predict_proba(X_va)[:, 1]
    y_pred_prob_te = model.predict_proba(X_te)[:, 1]
    y_pred_te      = (y_pred_prob_te > 0.5).astype(int)

    metrics = {
        "valid_auc":      round(roc_auc_score(y_va, y_pred_prob_va), 4),
        "test_auc":       round(roc_auc_score(y_te, y_pred_prob_te), 4),
        "test_accuracy":  round(accuracy_score(y_te, y_pred_te), 4),
        "test_f1":        round(f1_score(y_te, y_pred_te), 4),
        "best_iteration": model.best_iteration_,
    }

    logger.info("  Test AUC:      %.4f", metrics["test_auc"])
    logger.info("  Test Accuracy: %.4f", metrics["test_accuracy"])

    return model, metrics


# ─── IMPORTANCIA DE FEATURES ─────────────────────────────────────────────────

def save_feature_importance(
    model_resultado: CatBoostClassifier,
    model_gh: CatBoostRegressor,
    model_ga: CatBoostRegressor,
    features: List[str],
):
    """Guardar importancia de features de los 3 modelos en CSV."""
    imp_resultado = model_resultado.get_feature_importance()
    imp_gh        = model_gh.get_feature_importance()
    imp_ga        = model_ga.get_feature_importance()

    df_imp = pd.DataFrame({
        "feature":             features,
        "importance_resultado": imp_resultado,
        "importance_goals_h":  imp_gh,
        "importance_goals_a":  imp_ga,
    })
    df_imp["importance_avg"] = df_imp[
        ["importance_resultado","importance_goals_h","importance_goals_a"]
    ].mean(axis=1)
    df_imp = df_imp.sort_values("importance_avg", ascending=False).reset_index(drop=True)

    out = DATA_DIR / "feature_importance_catboost.csv"
    df_imp.to_csv(out, index=False, encoding="utf-8")
    logger.info("[OK] feature_importance_catboost.csv")

    logger.info("Top 15 features (importancia promedio CatBoost):")
    for _, row in df_imp.head(15).iterrows():
        logger.info("  %-40s  %.2f", row["feature"], row["importance_avg"])

    return df_imp


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("ENTRENAMIENTO CATBOOST - Fase 3")
    logger.info("Inicio: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 70)

    # 1. Cargar features
    feat_path = DATA_DIR / "features_engineered.csv"
    if not feat_path.exists():
        logger.error("FATAL: No encontrado %s", feat_path)
        return False

    df = pd.read_csv(feat_path)
    logger.info("Cargado: %d filas x %d columnas", len(df), len(df.columns))

    # Necesitamos la fecha para el split temporal
    base_path = DATA_DIR / "matches_cleaned.csv"
    if base_path.exists() and "date" not in df.columns:
        base = pd.read_csv(base_path, usecols=["match_id", "date"])
        df = df.merge(base, on="match_id", how="left")
        logger.info("Fecha agregada desde matches_cleaned.csv")

    # 2. Seleccion de features
    features = selecciona_features(df, corr_threshold=0.95)
    logger.info("Features seleccionadas: %d", len(features))

    # 3. Split temporal
    train, valid, test = split_temporal(df)

    if len(train) < 100 or len(valid) < 20 or len(test) < 20:
        logger.error("FATAL: Split demasiado pequeño — verificar fechas en CSV")
        return False

    # 4. Entrenar modelos
    all_metrics = {}

    # Modelo 1: Resultado (1/0/-1)
    m_resultado, met_resultado = train_model_resultado(train, valid, test, features)
    all_metrics["modelo_resultado"] = met_resultado

    # Modelo 2: Goles local
    m_gh, met_gh = train_model_goals(
        train, valid, test, features,
        target="target_goals_home", label="Goals Home"
    )
    all_metrics["modelo_goals_home"] = met_gh

    # Modelo 3: Goles visitante
    m_ga, met_ga = train_model_goals(
        train, valid, test, features,
        target="target_goals_away", label="Goals Away"
    )
    all_metrics["modelo_goals_away"] = met_ga

    # Modelo 4: Over 2.5
    m_over25, met_over25 = train_model_over25(train, valid, test, features)
    all_metrics["modelo_over25"] = met_over25

    # 5. Importancia de features
    df_imp = save_feature_importance(m_resultado, m_gh, m_ga, features)

    # 6. Guardar modelos .pkl
    models_to_save = {
        "model_resultado.pkl":  m_resultado,
        "model_goals_home.pkl": m_gh,
        "model_goals_away.pkl": m_ga,
        "model_over25.pkl":     m_over25,
    }
    for fname, model in models_to_save.items():
        path = MODELS_DIR / fname
        with open(path, "wb") as f:
            pickle.dump(model, f)
        size_kb = path.stat().st_size // 1024
        logger.info("[OK] %s (%d KB)", path, size_kb)

    # Guardar lista de features seleccionadas (necesaria en Fase 4)
    features_path = MODELS_DIR / "selected_features.json"
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump({"features": features, "n_features": len(features)}, f, indent=2)
    logger.info("[OK] selected_features.json (%d features)", len(features))

    # 7. Guardar metricas
    eval_path = DATA_DIR / "model_evaluation.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False, default=str)
    logger.info("[OK] model_evaluation.json")

    # 8. Resumen final
    logger.info("=" * 70)
    logger.info("[OK] FASE 3 COMPLETADA")
    logger.info("")
    logger.info("RESUMEN DE METRICAS:")
    logger.info("  Resultado (clasificacion 3 clases):")
    logger.info("    Valid accuracy: %.4f", met_resultado["valid_accuracy"])
    logger.info("    Test  accuracy: %.4f", met_resultado["test_accuracy"])
    logger.info("    Test  F1-macro: %.4f", met_resultado["test_f1_macro"])
    logger.info("  Goals Home (regresion Poisson):")
    logger.info("    Test MAE:  %.4f  RMSE: %.4f  R2: %.4f",
                met_gh["test_mae"], met_gh["test_rmse"], met_gh["test_r2"])
    logger.info("  Goals Away (regresion Poisson):")
    logger.info("    Test MAE:  %.4f  RMSE: %.4f  R2: %.4f",
                met_ga["test_mae"], met_ga["test_rmse"], met_ga["test_r2"])
    logger.info("  Over 2.5 (clasificacion binaria):")
    logger.info("    Test AUC:  %.4f  Accuracy: %.4f",
                met_over25["test_auc"], met_over25["test_accuracy"])
    logger.info("")
    logger.info("Modelos guardados en: %s/", MODELS_DIR)
    logger.info("=" * 70)

    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
