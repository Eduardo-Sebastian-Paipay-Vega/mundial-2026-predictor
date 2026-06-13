"""
optimize_hyperparameters.py — Fase 6.3: AutoML con Optuna
Busqueda de hiperparametros optimos para CatBoost, XGBoost y LightGBM
"""
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/optuna.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("optuna_search")

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
LOG_DIR   = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

TRAIN_END = "2023-12-31"
VALID_END = "2024-12-31"
N_TRIALS  = 50

# ── CARGA DE DATOS ────────────────────────────────────────────────────────────

def load_data():
    df_feat = pd.read_csv(DATA_DIR / "features_engineered.csv")
    df_hist = pd.read_csv(DATA_DIR / "matches_cleaned.csv")
    df_hist["date"] = pd.to_datetime(df_hist["date"])

    with open(MODEL_DIR / "selected_features.json", encoding="utf-8") as f:
        feats = json.load(f)["features"]

    df = df_feat.merge(
        df_hist[["match_id", "date", "resultado"]], on="match_id"
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    def encode(s):
        return s.map({-1: 0, 0: 1, 1: 2})

    avail = [f for f in feats if f in df.columns]
    missing = set(feats) - set(avail)
    for mf in missing:
        df[mf] = 0.0

    train = df[df["date"] <= TRAIN_END]
    valid = df[(df["date"] > TRAIN_END) & (df["date"] <= VALID_END)]
    test  = df[df["date"] > VALID_END]

    X_tr = train[feats].fillna(0)
    X_vl = valid[feats].fillna(0)
    X_te = test[feats].fillna(0)

    y_tr = encode(train["resultado"])
    y_vl = encode(valid["resultado"])
    y_te = encode(test["resultado"])

    logger.info("Datos: Train=%d Valid=%d Test=%d Features=%d",
                len(train), len(valid), len(test), len(feats))
    return X_tr, X_vl, X_te, y_tr, y_vl, y_te

# ── OBJECTIVES ────────────────────────────────────────────────────────────────

def make_objective_catboost(X_tr, y_tr, X_vl, y_vl):
    def objective(trial):
        # bootstrap_type=Bernoulli es el unico que soporta subsample
        params = {
            "learning_rate":    trial.suggest_float("lr", 0.01, 0.3, log=True),
            "depth":            trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg":      trial.suggest_float("l2", 1.0, 10.0),
            "iterations":       trial.suggest_int("iters", 200, 600),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "random_strength":  trial.suggest_float("rand_str", 0.1, 5.0),
            "bootstrap_type":   "Bernoulli",
        }
        model = CatBoostClassifier(
            **params,
            random_seed=42,
            loss_function="MultiClass",
            eval_metric="Accuracy",
            use_best_model=True,
            early_stopping_rounds=30,
            verbose=0,
        )
        model.fit(X_tr, y_tr,
                  eval_set=[(X_vl, y_vl)],
                  verbose=False)
        return accuracy_score(y_vl, model.predict(X_vl))
    return objective

def make_objective_xgboost(X_tr, y_tr, X_vl, y_vl):
    def objective(trial):
        params = {
            "learning_rate":    trial.suggest_float("lr", 0.01, 0.3, log=True),
            "max_depth":        trial.suggest_int("depth", 4, 10),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample", 0.5, 1.0),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_int("min_child", 1, 7),
            "reg_alpha":        trial.suggest_float("alpha", 0.0, 5.0),
            "reg_lambda":       trial.suggest_float("lambda", 0.5, 5.0),
        }
        model = XGBClassifier(
            **params,
            n_estimators=400,
            random_state=42,
            eval_metric="mlogloss",
            tree_method="hist",
            verbosity=0,
            early_stopping_rounds=30,
        )
        model.fit(X_tr, y_tr,
                  eval_set=[(X_vl, y_vl)],
                  verbose=False)
        return accuracy_score(y_vl, model.predict(X_vl))
    return objective

def make_objective_lightgbm(X_tr, y_tr, X_vl, y_vl):
    def objective(trial):
        params = {
            "learning_rate":    trial.suggest_float("lr", 0.01, 0.2, log=True),
            "num_leaves":       trial.suggest_int("leaves", 20, 100),
            "min_data_in_leaf": trial.suggest_int("min_data", 5, 50),
            "lambda_l1":        trial.suggest_float("l1", 0.0, 10.0),
            "lambda_l2":        trial.suggest_float("l2", 0.0, 10.0),
            "feature_fraction": trial.suggest_float("feat_frac", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bag_frac", 0.5, 1.0),
            "bagging_freq":     trial.suggest_int("bag_freq", 1, 7),
        }
        model = LGBMClassifier(
            **params,
            n_estimators=500,
            random_state=42,
            objective="multiclass",
            num_class=3,
            verbose=-1,
        )
        model.fit(X_tr, y_tr,
                  eval_set=[(X_vl, y_vl)],
                  callbacks=[])
        return accuracy_score(y_vl, model.predict(X_vl))
    return objective

# ── OPTIMIZACION ──────────────────────────────────────────────────────────────

def optimize_model(name: str, objective_fn, n_trials: int = N_TRIALS):
    logger.info("[%s] Iniciando %d trials Optuna...", name, n_trials)
    t0 = time.time()

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=name,
    )

    study.optimize(objective_fn, n_trials=n_trials,
                   show_progress_bar=True, gc_after_trial=True)

    elapsed = round(time.time() - t0, 1)
    best    = study.best_params
    acc     = study.best_value

    logger.info("[%s] Best Accuracy: %.4f | Elapsed: %.1f s", name, acc, elapsed)
    logger.info("[%s] Best Params: %s", name, json.dumps(best, indent=2))
    return {"best_params": best, "best_accuracy": acc, "n_trials": n_trials,
            "elapsed_s": elapsed}

# ── REENTRENAMIENTO ───────────────────────────────────────────────────────────

def _remap_catboost(p: dict) -> dict:
    """Traduce nombres de trial Optuna a parametros reales de CatBoost."""
    return {
        "learning_rate":    p.get("lr", 0.05),
        "depth":            p.get("depth", 6),
        "l2_leaf_reg":      p.get("l2", 3.0),
        "iterations":       p.get("iters", 400),
        "subsample":        p.get("subsample", 0.8),
        "random_strength":  p.get("rand_str", 1.0),
    }

def _remap_xgboost(p: dict) -> dict:
    return {
        "learning_rate":    p.get("lr", 0.1),
        "max_depth":        p.get("depth", 7),
        "subsample":        p.get("subsample", 0.8),
        "colsample_bytree": p.get("colsample", 0.8),
        "gamma":            p.get("gamma", 0.0),
        "min_child_weight": p.get("min_child", 3),
        "reg_alpha":        p.get("alpha", 0.0),
        "reg_lambda":       p.get("lambda", 1.0),
    }

def _remap_lightgbm(p: dict) -> dict:
    return {
        "learning_rate":    p.get("lr", 0.05),
        "num_leaves":       p.get("leaves", 50),
        "min_data_in_leaf": p.get("min_data", 20),
        "lambda_l1":        p.get("l1", 0.0),
        "lambda_l2":        p.get("l2", 0.0),
        "feature_fraction": p.get("feat_frac", 0.8),
        "bagging_fraction": p.get("bag_frac", 0.8),
        "bagging_freq":     p.get("bag_freq", 5),
    }

def retrain_best(best_params: dict, X_tr, y_tr, X_vl, y_vl, X_te, y_te):
    import pickle
    logger.info("Reentrenando con hiperparametros optimos...")
    results = {}

    # CatBoost
    raw_catb = best_params.get("catboost", {}).get("best_params", {})
    if raw_catb:
        params = _remap_catboost(raw_catb)
        model = CatBoostClassifier(
            **params,
            bootstrap_type="Bernoulli",
            random_seed=42,
            loss_function="MultiClass",
            eval_metric="Accuracy",
            use_best_model=True,
            early_stopping_rounds=50,
            verbose=0,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
        acc = accuracy_score(y_te, model.predict(X_te))
        logger.info("CatBoost optimizado Test: %.4f", acc)
        results["catboost_opt_acc"] = acc
        pickle.dump(model, open(MODEL_DIR / "model_resultado_opt.pkl", "wb"))

    # XGBoost
    raw_xgb = best_params.get("xgboost", {}).get("best_params", {})
    if raw_xgb:
        params = _remap_xgboost(raw_xgb)
        model = XGBClassifier(
            **params,
            n_estimators=400,
            random_state=42,
            eval_metric="mlogloss",
            tree_method="hist",
            verbosity=0,
            early_stopping_rounds=50,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
        acc = accuracy_score(y_te, model.predict(X_te))
        logger.info("XGBoost optimizado Test: %.4f", acc)
        results["xgboost_opt_acc"] = acc
        pickle.dump(model, open(MODEL_DIR / "model_xgboost_opt.pkl", "wb"))

    # LightGBM
    raw_lgb = best_params.get("lightgbm", {}).get("best_params", {})
    if raw_lgb:
        params = _remap_lightgbm(raw_lgb)
        model = LGBMClassifier(
            **params,
            n_estimators=500,
            random_state=42,
            objective="multiclass",
            num_class=3,
            verbose=-1,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], callbacks=[])
        acc = accuracy_score(y_te, model.predict(X_te))
        logger.info("LightGBM optimizado Test: %.4f", acc)
        results["lightgbm_opt_acc"] = acc
        pickle.dump(model, open(MODEL_DIR / "model_lightgbm_opt.pkl", "wb"))

    return results

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("FASE 6.3: AUTOML — Optuna Hyperparameter Search")
    logger.info("Trials por modelo: %d", N_TRIALS)

    X_tr, X_vl, X_te, y_tr, y_vl, y_te = load_data()

    best_params = {}

    # CatBoost
    logger.info("=" * 60)
    logger.info("[1/3] CATBOOST")
    logger.info("=" * 60)
    res = optimize_model("catboost",
                         make_objective_catboost(X_tr, y_tr, X_vl, y_vl))
    best_params["catboost"] = res

    # XGBoost
    logger.info("=" * 60)
    logger.info("[2/3] XGBOOST")
    logger.info("=" * 60)
    res = optimize_model("xgboost",
                         make_objective_xgboost(X_tr, y_tr, X_vl, y_vl))
    best_params["xgboost"] = res

    # LightGBM
    logger.info("=" * 60)
    logger.info("[3/3] LIGHTGBM")
    logger.info("=" * 60)
    res = optimize_model("lightgbm",
                         make_objective_lightgbm(X_tr, y_tr, X_vl, y_vl))
    best_params["lightgbm"] = res

    # Guardar hiperparametros
    with open(MODEL_DIR / "best_hyperparams.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    logger.info("best_hyperparams.json guardado")

    # Reentrenar con mejores params
    retrain_results = retrain_best(best_params, X_tr, y_tr, X_vl, y_vl, X_te, y_te)

    # Resumen
    logger.info("=" * 60)
    logger.info("RESUMEN")
    logger.info("=" * 60)
    for model_name, res in best_params.items():
        logger.info("%s: Best Valid Accuracy = %.4f (%d trials)",
                    model_name, res["best_accuracy"], res["n_trials"])

    if retrain_results:
        logger.info("Test accuracies con params optimos: %s", retrain_results)

    return best_params


if __name__ == "__main__":
    main()
