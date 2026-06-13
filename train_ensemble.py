"""
train_ensemble.py — Fase 6.1: Ensemble CatBoost + XGBoost + LightGBM
Stacking meta-learner para resultado, goles y over2.5
"""
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, classification_report,
                              mean_absolute_error, mean_squared_error, r2_score,
                              roc_auc_score)
from xgboost import XGBClassifier, XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/ensemble.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("ensemble")

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
LOG_DIR   = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

TRAIN_END = "2023-12-31"
VALID_END = "2024-12-31"

# ── CARGA DE DATOS ────────────────────────────────────────────────────────────

def load_data():
    logger.info("Cargando datos...")
    df_feat  = pd.read_csv(DATA_DIR / "features_engineered.csv")
    df_hist  = pd.read_csv(DATA_DIR / "matches_cleaned.csv")
    df_hist["date"] = pd.to_datetime(df_hist["date"])

    with open(MODEL_DIR / "selected_features.json", encoding="utf-8") as f:
        feats = json.load(f)["features"]

    # Target encoding (igual que train_models.py)
    def encode(s):
        return s.map({-1: 0, 0: 1, 1: 2})

    df = df_feat.merge(df_hist[["match_id", "date", "resultado",
                                  "goals_home", "goals_away"]], on="match_id")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    train = df[df["date"] <= TRAIN_END]
    valid = df[(df["date"] > TRAIN_END) & (df["date"] <= VALID_END)]
    test  = df[df["date"] > VALID_END]

    avail = [f for f in feats if f in df.columns]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        for mf in missing:
            df[mf] = 0.0
        avail = feats

    X_train = train[avail].fillna(0)
    X_valid = valid[avail].fillna(0)
    X_test  = test[avail].fillna(0)

    y_res_train = encode(train["resultado"])
    y_res_valid = encode(valid["resultado"])
    y_res_test  = encode(test["resultado"])

    y_gh_train = train["goals_home"].fillna(0)
    y_gh_valid = valid["goals_home"].fillna(0)
    y_gh_test  = test["goals_home"].fillna(0)

    y_ga_train = train["goals_away"].fillna(0)
    y_ga_valid = valid["goals_away"].fillna(0)
    y_ga_test  = test["goals_away"].fillna(0)

    y_o25_train = ((train["goals_home"] + train["goals_away"]) >= 3).astype(int)
    y_o25_valid = ((valid["goals_home"] + valid["goals_away"]) >= 3).astype(int)
    y_o25_test  = ((test["goals_home"]  + test["goals_away"])  >= 3).astype(int)

    logger.info("Train: %d | Valid: %d | Test: %d | Features: %d",
                len(train), len(valid), len(test), len(avail))
    return {
        "X_train": X_train, "X_valid": X_valid, "X_test": X_test,
        "y_res": (y_res_train, y_res_valid, y_res_test),
        "y_gh":  (y_gh_train,  y_gh_valid,  y_gh_test),
        "y_ga":  (y_ga_train,  y_ga_valid,  y_ga_test),
        "y_o25": (y_o25_train, y_o25_valid, y_o25_test),
    }

# ── MODELOS BASE ──────────────────────────────────────────────────────────────

def train_xgb_classifier(X_tr, y_tr, X_vl, y_vl):
    logger.info("  Entrenando XGBoost Clasificador...")
    m = XGBClassifier(
        learning_rate=0.1, max_depth=7, n_estimators=400,
        random_state=42, eval_metric="mlogloss",
        tree_method="hist", verbosity=0,
        early_stopping_rounds=50,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
    return m

def train_lgb_classifier(X_tr, y_tr, X_vl, y_vl):
    logger.info("  Entrenando LightGBM Clasificador...")
    m = LGBMClassifier(
        learning_rate=0.05, num_leaves=50, n_estimators=500,
        random_state=42, objective="multiclass", num_class=3,
        verbose=-1,
    )
    m.fit(X_tr, y_tr,
          eval_set=[(X_vl, y_vl)],
          callbacks=[])
    return m

def train_xgb_regressor(X_tr, y_tr, X_vl, y_vl, label=""):
    logger.info("  Entrenando XGBoost Regresor %s...", label)
    m = XGBRegressor(
        learning_rate=0.1, max_depth=6, n_estimators=400,
        random_state=42, objective="count:poisson",
        tree_method="hist", verbosity=0,
        early_stopping_rounds=50,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
    return m

def train_lgb_regressor(X_tr, y_tr, X_vl, y_vl, label=""):
    logger.info("  Entrenando LightGBM Regresor %s...", label)
    m = LGBMRegressor(
        learning_rate=0.05, num_leaves=50, n_estimators=500,
        random_state=42, objective="poisson", verbose=-1,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)])
    return m

# ── STACKING ──────────────────────────────────────────────────────────────────

def build_meta_features_clf(models, X):
    parts = [m.predict_proba(X) for m in models]
    return np.hstack(parts)

def build_meta_features_reg(models, X):
    parts = [m.predict(X).reshape(-1, 1) for m in models]
    return np.hstack(parts)

def train_stacking_clf(base_models, X_tr, y_tr, X_vl, y_vl, X_te, y_te):
    logger.info("  Stacking clasificador...")
    X_meta_tr = build_meta_features_clf(base_models, X_tr)
    X_meta_vl = build_meta_features_clf(base_models, X_vl)
    X_meta_te = build_meta_features_clf(base_models, X_te)

    meta = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    meta.fit(X_meta_tr, y_tr)

    acc_vl = accuracy_score(y_vl, meta.predict(X_meta_vl))
    acc_te = accuracy_score(y_te, meta.predict(X_meta_te))
    logger.info("    Ensemble → Valid: %.4f | Test: %.4f", acc_vl, acc_te)
    return meta, (acc_vl, acc_te)

def train_stacking_reg(base_models, X_tr, y_tr, X_vl, y_vl, X_te, y_te):
    logger.info("  Stacking regresor...")
    X_meta_tr = build_meta_features_reg(base_models, X_tr)
    X_meta_vl = build_meta_features_reg(base_models, X_vl)
    X_meta_te = build_meta_features_reg(base_models, X_te)

    meta = Ridge(alpha=1.0)
    meta.fit(X_meta_tr, y_tr)

    y_pred_te = meta.predict(X_meta_te)
    mae = mean_absolute_error(y_te, y_pred_te)
    r2  = r2_score(y_te, y_pred_te)
    logger.info("    Ensemble → MAE: %.4f | R2: %.4f", mae, r2)
    return meta, (mae, r2)

# ── PIPELINE PRINCIPAL ────────────────────────────────────────────────────────

def train_resultado(d):
    logger.info("=" * 60)
    logger.info("ENSEMBLE: RESULTADO (Clasificacion 3 clases)")
    logger.info("=" * 60)

    X_tr, X_vl, X_te = d["X_train"], d["X_valid"], d["X_test"]
    y_tr, y_vl, y_te = d["y_res"]

    # Cargar CatBoost existente
    catb = pickle.load(open(MODEL_DIR / "model_resultado.pkl", "rb"))
    acc_catb = accuracy_score(y_te, catb.predict(X_te))
    logger.info("CatBoost (base) Test: %.4f", acc_catb)

    xgb = train_xgb_classifier(X_tr, y_tr, X_vl, y_vl)
    acc_xgb = accuracy_score(y_te, xgb.predict(X_te))
    logger.info("XGBoost Test: %.4f", acc_xgb)

    lgb = train_lgb_classifier(X_tr, y_tr, X_vl, y_vl)
    acc_lgb = accuracy_score(y_te, lgb.predict(X_te))
    logger.info("LightGBM Test: %.4f", acc_lgb)

    base_models = [catb, xgb, lgb]
    meta, (acc_vl, acc_te) = train_stacking_clf(
        base_models, X_tr, y_tr, X_vl, y_vl, X_te, y_te
    )

    improvement = acc_te - acc_catb
    logger.info("Mejora Ensemble: %+.4f (%.2f pp)", improvement, improvement * 100)

    pickle.dump(xgb,  open(MODEL_DIR / "model_xgboost.pkl",  "wb"))
    pickle.dump(lgb,  open(MODEL_DIR / "model_lightgbm.pkl", "wb"))
    pickle.dump(meta, open(MODEL_DIR / "ensemble_resultado.pkl", "wb"))

    return {
        "catboost_test": acc_catb, "xgboost_test": acc_xgb,
        "lightgbm_test": acc_lgb, "ensemble_test": acc_te,
        "improvement": improvement,
    }

def train_goals(d, target="home"):
    logger.info("=" * 60)
    logger.info("ENSEMBLE: GOALS_%s (Regresion Poisson)", target.upper())
    logger.info("=" * 60)

    X_tr, X_vl, X_te = d["X_train"], d["X_valid"], d["X_test"]
    key = "y_gh" if target == "home" else "y_ga"
    y_tr, y_vl, y_te = d[key]

    catb_key = f"model_goals_{target}.pkl"
    catb = pickle.load(open(MODEL_DIR / catb_key, "rb"))
    mae_catb = mean_absolute_error(y_te, catb.predict(X_te))
    r2_catb  = r2_score(y_te, catb.predict(X_te))
    logger.info("CatBoost (base) MAE: %.4f | R2: %.4f", mae_catb, r2_catb)

    xgb = train_xgb_regressor(X_tr, y_tr, X_vl, y_vl, label=target)
    lgb = train_lgb_regressor(X_tr, y_tr, X_vl, y_vl, label=target)

    mae_xgb = mean_absolute_error(y_te, xgb.predict(X_te))
    mae_lgb = mean_absolute_error(y_te, lgb.predict(X_te))
    logger.info("XGBoost MAE: %.4f | LightGBM MAE: %.4f", mae_xgb, mae_lgb)

    base_models = [catb, xgb, lgb]
    meta, (mae_en, r2_en) = train_stacking_reg(
        base_models, X_tr, y_tr, X_vl, y_vl, X_te, y_te
    )

    pickle.dump(meta, open(MODEL_DIR / f"ensemble_goals_{target}.pkl", "wb"))
    return {"catboost_mae": mae_catb, "ensemble_mae": mae_en,
            "catboost_r2": r2_catb, "ensemble_r2": r2_en}

def train_over25(d):
    logger.info("=" * 60)
    logger.info("ENSEMBLE: OVER 2.5 (Clasificacion Binaria)")
    logger.info("=" * 60)

    X_tr, X_vl, X_te = d["X_train"], d["X_valid"], d["X_test"]
    y_tr, y_vl, y_te = d["y_o25"]

    catb = pickle.load(open(MODEL_DIR / "model_over25.pkl", "rb"))
    auc_catb = roc_auc_score(y_te, catb.predict_proba(X_te)[:, 1])
    acc_catb  = accuracy_score(y_te, catb.predict(X_te))
    logger.info("CatBoost AUC: %.4f | Acc: %.4f", auc_catb, acc_catb)

    xgb = XGBClassifier(
        learning_rate=0.1, max_depth=6, n_estimators=400,
        random_state=42, eval_metric="logloss",
        tree_method="hist", verbosity=0, early_stopping_rounds=50,
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)

    lgb = LGBMClassifier(
        learning_rate=0.05, num_leaves=50, n_estimators=500,
        random_state=42, objective="binary", verbose=-1,
    )
    lgb.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)])

    auc_xgb = roc_auc_score(y_te, xgb.predict_proba(X_te)[:, 1])
    auc_lgb = roc_auc_score(y_te, lgb.predict_proba(X_te)[:, 1])
    logger.info("XGBoost AUC: %.4f | LightGBM AUC: %.4f", auc_xgb, auc_lgb)

    base_models = [catb, xgb, lgb]
    meta, (acc_vl, acc_te) = train_stacking_clf(
        base_models, X_tr, y_tr, X_vl, y_vl, X_te, y_te
    )

    auc_en = roc_auc_score(y_te,
        meta.predict_proba(build_meta_features_clf(base_models, X_te))[:, 1])
    logger.info("Ensemble AUC: %.4f | Acc: %.4f", auc_en, acc_te)

    pickle.dump(meta, open(MODEL_DIR / "ensemble_over25.pkl", "wb"))
    return {"catboost_auc": auc_catb, "ensemble_auc": auc_en,
            "catboost_acc": acc_catb, "ensemble_acc": acc_te}

def main():
    t0 = time.time()
    logger.info("FASE 6.1: ENSEMBLE TRAINING")
    logger.info("XGBoost %s | LightGBM %s", __import__("xgboost").__version__,
                __import__("lightgbm").__version__)

    d = load_data()

    res   = train_resultado(d)
    gh    = train_goals(d, "home")
    ga    = train_goals(d, "away")
    o25   = train_over25(d)

    config = {
        "type": "stacking",
        "base_models": ["catboost", "xgboost", "lightgbm"],
        "meta_learner_clf": "logistic_regression",
        "meta_learner_reg": "ridge",
        "resultado": res,
        "goals_home": gh,
        "goals_away": ga,
        "over25": o25,
        "elapsed_s": round(time.time() - t0, 1),
    }

    with open(MODEL_DIR / "ensemble_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logger.info("=" * 60)
    logger.info("RESUMEN FINAL")
    logger.info("=" * 60)
    logger.info("Resultado: CatBoost %.4f -> Ensemble %.4f (%+.2f pp)",
                res["catboost_test"], res["ensemble_test"],
                res["improvement"] * 100)
    logger.info("Goles H:  CatBoost MAE %.4f -> Ensemble MAE %.4f",
                gh["catboost_mae"], gh["ensemble_mae"])
    logger.info("Goles A:  CatBoost MAE %.4f -> Ensemble MAE %.4f",
                ga["catboost_mae"], ga["ensemble_mae"])
    logger.info("Over2.5:  CatBoost AUC %.4f -> Ensemble AUC %.4f",
                o25["catboost_auc"], o25["ensemble_auc"])
    logger.info("Tiempo total: %.1f s", time.time() - t0)
    logger.info("Modelos guardados en models/")


if __name__ == "__main__":
    main()
