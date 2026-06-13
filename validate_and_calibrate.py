"""
validate_and_calibrate.py — Fase 6.4: Validacion temporal + calibracion
Time Series CV (5 folds) + calibracion sigmoid de probabilidades
"""
import json
import logging
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (accuracy_score, brier_score_loss,
                              classification_report, roc_auc_score)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/validation.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("validate")

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
PLOT_DIR  = ROOT / "plots"
LOG_DIR   = ROOT / "logs"
PLOT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

N_SPLITS  = 5
TRAIN_END = "2023-12-31"
VALID_END = "2024-12-31"

# ── CARGA ─────────────────────────────────────────────────────────────────────

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
    df = df.sort_values("date").reset_index(drop=True)

    for mf in feats:
        if mf not in df.columns:
            df[mf] = 0.0

    def encode(s):
        return s.map({-1: 0, 0: 1, 1: 2})

    X = df[feats].fillna(0)
    y = encode(df["resultado"])
    dates = df["date"]

    return X, y, dates, feats

# ── TIME SERIES CV ────────────────────────────────────────────────────────────

def time_series_cv(X, y, dates, n_splits: int = N_SPLITS):
    """
    Time Series Cross-Validation con ventanas deslizantes anuales.
    Garantiza que train < valid en tiempo (sin data leakage).
    """
    logger.info("=" * 60)
    logger.info("TIME SERIES CV — %d folds", n_splits)
    logger.info("=" * 60)

    min_date = dates.min()
    max_date = dates.max()
    total_years = (max_date - min_date).days / 365.25

    results = []

    for fold in range(1, n_splits + 1):
        # Cada fold: train crece, test es el siguiente año
        train_end_frac = (fold - 1) / n_splits + 1.0 / n_splits
        test_end_frac  = fold / n_splits

        train_end_date = min_date + pd.Timedelta(
            days=int(total_years * train_end_frac * 365.25 * 0.8)
        )
        test_start_date = train_end_date
        test_end_date   = min_date + pd.Timedelta(
            days=int(total_years * test_end_frac * 365.25)
        )

        train_mask = dates <= train_end_date
        test_mask  = (dates > test_start_date) & (dates <= test_end_date)

        if train_mask.sum() < 500 or test_mask.sum() < 50:
            logger.warning("Fold %d: datos insuficientes, omitiendo", fold)
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask],  y[test_mask]

        # Usar CatBoost (modelo principal)
        model = CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            loss_function="MultiClass", eval_metric="Accuracy",
            random_seed=42, verbose=0, early_stopping_rounds=30,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

        acc = accuracy_score(y_te, model.predict(X_te))

        row = {
            "fold":         fold,
            "train_start":  str(dates[train_mask].min())[:10],
            "train_end":    str(dates[train_mask].max())[:10],
            "test_start":   str(dates[test_mask].min())[:10],
            "test_end":     str(dates[test_mask].max())[:10],
            "n_train":      int(train_mask.sum()),
            "n_test":       int(test_mask.sum()),
            "accuracy":     round(acc, 4),
        }
        results.append(row)

        logger.info(
            "Fold %d: Train %s-%s (%d) → Test %s-%s (%d) | Acc: %.4f",
            fold,
            row["train_start"], row["train_end"], row["n_train"],
            row["test_start"],  row["test_end"],  row["n_test"],
            acc,
        )

    df_res = pd.DataFrame(results)
    mean_acc = df_res["accuracy"].mean()
    std_acc  = df_res["accuracy"].std()

    logger.info("-" * 60)
    logger.info("CV Accuracy: %.4f ± %.4f", mean_acc, std_acc)
    logger.info("Min: %.4f | Max: %.4f", df_res["accuracy"].min(), df_res["accuracy"].max())

    df_res.to_csv(DATA_DIR / "cv_results.csv", index=False)
    return df_res, mean_acc, std_acc

# ── CALIBRACION ───────────────────────────────────────────────────────────────

class _PlattCalibrator:
    """
    Calibrador Platt (sigmoid) multiclase para modelos ya entrenados.
    Para cada clase entrena un LogisticRegression 1D sobre la probabilidad raw.
    """
    def __init__(self):
        self._calibrators = {}
        self.classes_ = np.array([0, 1, 2])

    def fit(self, proba_raw: np.ndarray, y: np.ndarray):
        from sklearn.linear_model import LogisticRegression
        n_classes = proba_raw.shape[1]
        for i in range(n_classes):
            y_bin = (y == i).astype(int)
            p = proba_raw[:, i].reshape(-1, 1)
            lr = LogisticRegression(max_iter=1000, C=1.0)
            lr.fit(p, y_bin)
            self._calibrators[i] = lr
        return self

    def predict_proba_from_raw(self, proba_raw: np.ndarray) -> np.ndarray:
        n_classes = proba_raw.shape[1]
        cal = np.zeros_like(proba_raw)
        for i in range(n_classes):
            p = proba_raw[:, i].reshape(-1, 1)
            cal[:, i] = self._calibrators[i].predict_proba(p)[:, 1]
        # Renormalizar para que sumen 1
        row_sums = cal.sum(axis=1, keepdims=True)
        return cal / np.maximum(row_sums, 1e-9)

    def predict(self, proba_raw: np.ndarray) -> np.ndarray:
        return self.predict_proba_from_raw(proba_raw).argmax(axis=1)


def calibrate_model(model, X_vl, y_vl):
    """
    Calibra las probabilidades del modelo usando Platt scaling (sigmoid) multiclase.
    """
    logger.info("Calibrando probabilidades (Platt scaling)...")
    proba_raw = model.predict_proba(X_vl)
    calibrator = _PlattCalibrator()
    calibrator.fit(proba_raw, np.array(y_vl))
    logger.info("Calibrador entrenado OK")
    return calibrator

def plot_calibration_curves(model, calibrator, X_vl, y_vl):
    """Genera curvas de calibracion: sin calibrar vs calibrado."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    class_names = ["Local Win", "Draw", "Away Win"]
    colors_unc  = ["#d62728", "#ff7f0e", "#1f77b4"]
    colors_cal  = ["#2ca02c", "#9467bd", "#8c564b"]

    probs_unc = model.predict_proba(X_vl)
    probs_cal = calibrator.predict_proba_from_raw(probs_unc)

    for i, (ax, name) in enumerate(zip(axes, class_names)):
        y_bin = (y_vl == i).astype(int)

        frac_unc, mean_unc = calibration_curve(y_bin, probs_unc[:, i], n_bins=8)
        frac_cal, mean_cal = calibration_curve(y_bin, probs_cal[:, i], n_bins=8)

        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfectamente calibrado")
        ax.plot(mean_unc, frac_unc, "o-", color=colors_unc[i],
                label="Sin calibrar", lw=2)
        ax.plot(mean_cal, frac_cal, "s-", color=colors_cal[i],
                label="Calibrado (sigmoid)", lw=2)

        bs_unc = brier_score_loss(y_bin, probs_unc[:, i])
        bs_cal = brier_score_loss(y_bin, probs_cal[:, i])

        ax.set_title(f"{name}\nBrier: {bs_unc:.4f} -> {bs_cal:.4f}")
        ax.set_xlabel("Probabilidad predicha")
        ax.set_ylabel("Fraccion observada")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Curvas de Calibracion: Antes vs Despues (Sigmoid)", fontsize=13)
    plt.tight_layout()

    path = PLOT_DIR / "calibration_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Curva de calibracion guardada: %s", path)

# ── EVALUACION DETALLADA ──────────────────────────────────────────────────────

def detailed_evaluation(y_true, model, calibrator, X_te):
    logger.info("=" * 60)
    logger.info("EVALUACION DETALLADA")
    logger.info("=" * 60)

    proba_raw = model.predict_proba(X_te)
    proba_cal = calibrator.predict_proba_from_raw(proba_raw)

    y_pred_raw = np.array(model.predict(X_te), dtype=int).flatten()
    y_pred_cal = calibrator.predict(proba_raw)

    acc_raw = accuracy_score(y_true, y_pred_raw)
    acc_cal = accuracy_score(y_true, y_pred_cal)

    logger.info("Accuracy sin calibrar: %.4f", acc_raw)
    logger.info("Accuracy calibrado:    %.4f", acc_cal)

    logger.info("\nClassification Report (calibrado):")
    logger.info("\n%s", classification_report(y_true, y_pred_cal,
                                               target_names=["Local", "Empate", "Visitante"]))

    auc_scores = {}
    for i, name in enumerate(["Local", "Empate", "Visitante"]):
        y_bin = (y_true == i).astype(int)
        auc_r = roc_auc_score(y_bin, proba_raw[:, i])
        auc_c = roc_auc_score(y_bin, proba_cal[:, i])
        auc_scores[name] = {"uncalibrated": round(auc_r, 4),
                             "calibrated":   round(auc_c, 4)}
        logger.info("AUC %s: %.4f -> %.4f (calibrado)", name, auc_r, auc_c)

    brier_raw = sum(
        brier_score_loss((y_true == i).astype(int), proba_raw[:, i])
        for i in range(3)
    ) / 3
    brier_cal = sum(
        brier_score_loss((y_true == i).astype(int), proba_cal[:, i])
        for i in range(3)
    ) / 3
    logger.info("Brier Score (avg): %.4f -> %.4f", brier_raw, brier_cal)

    return {
        "accuracy_uncalibrated": round(acc_raw, 4),
        "accuracy_calibrated":   round(acc_cal, 4),
        "auc_by_class":          auc_scores,
        "brier_uncalibrated":    round(brier_raw, 4),
        "brier_calibrated":      round(brier_cal, 4),
    }

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("FASE 6.4: VALIDACION TEMPORAL + CALIBRACION")

    X, y, dates, feats = load_data()

    # 1. Time Series CV
    df_cv, mean_acc, std_acc = time_series_cv(X, y, dates, N_SPLITS)

    # 2. Cargar modelo final (entrenado en train+valid)
    model_path = MODEL_DIR / "model_resultado.pkl"
    model = pickle.load(open(model_path, "rb"))
    logger.info("Modelo cargado: %s", model_path.name)

    # Splits para calibracion
    train_mask = dates <= TRAIN_END
    valid_mask = (dates > TRAIN_END) & (dates <= VALID_END)
    test_mask  = dates > VALID_END

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_vl, y_vl = X[valid_mask], y[valid_mask]
    X_te, y_te = X[test_mask],  y[test_mask]

    # 3. Calibrar
    calibrator = calibrate_model(model, X_vl, y_vl)

    # 4. Curvas de calibracion
    plot_calibration_curves(model, calibrator, X_vl, y_vl)

    # 5. Evaluacion detallada
    eval_res = detailed_evaluation(y_te, model, calibrator, X_te)

    # 6. Guardar calibrador
    pickle.dump(calibrator, open(MODEL_DIR / "calibrator.pkl", "wb"))
    logger.info("calibrator.pkl guardado")

    # 7. Reporte final
    report = {
        "cv_folds":     N_SPLITS,
        "cv_results":   df_cv.to_dict("records"),
        "cv_mean_accuracy": round(mean_acc, 4),
        "cv_std_accuracy":  round(std_acc, 4),
        "final_evaluation": eval_res,
        "calibration_method": "sigmoid",
    }

    with open(MODEL_DIR / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 60)
    logger.info("RESUMEN FINAL")
    logger.info("=" * 60)
    logger.info("CV Accuracy:    %.4f ± %.4f (target: 70%%+)", mean_acc, std_acc)
    logger.info("Test Accuracy sin calibrar: %.4f", eval_res["accuracy_uncalibrated"])
    logger.info("Test Accuracy calibrado:    %.4f", eval_res["accuracy_calibrated"])
    logger.info("Brier Score:    %.4f -> %.4f (menor es mejor)",
                eval_res["brier_uncalibrated"], eval_res["brier_calibrated"])
    logger.info("validation_report.json guardado")

    return report


if __name__ == "__main__":
    main()
