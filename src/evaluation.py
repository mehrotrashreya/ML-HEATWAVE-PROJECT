"""Regression + heatwave-category classification metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, median_absolute_error,
                             max_error, accuracy_score, precision_recall_fscore_support, confusion_matrix)
from src.heatwave_rules import classify_heatwave, CATEGORY_NAMES

CLASS_LABELS = [0, 1, 2]


def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mse = mean_squared_error(y_true, y_pred)
    return {"n": len(y_true), "MAE (°C)": mean_absolute_error(y_true, y_pred), "MSE (°C²)": mse,
            "RMSE (°C)": float(np.sqrt(mse)), "R²": r2_score(y_true, y_pred),
            "MedAE (°C)": median_absolute_error(y_true, y_pred), "Max error (°C)": max_error(y_true, y_pred),
            "Bias (°C)": float(np.mean(y_pred - y_true))}


def regression_by_horizon(y_true, y_pred, horizons):
    y_true, y_pred, horizons = map(np.asarray, (y_true, y_pred, horizons))
    rows = {f"h={int(h)}": regression_metrics(y_true[horizons == h], y_pred[horizons == h]) for h in sorted(set(horizons))}
    rows["all"] = regression_metrics(y_true, y_pred)
    return pd.DataFrame(rows).T


def to_heatwave_codes(temps):
    return np.array([classify_heatwave(float(t))[0] for t in temps])


def classification_metrics(y_true_temp, y_pred_temp):
    # Both temperatures -> heatwave codes (0/1/2), then accuracy / precision / recall / F1 / support / confusion matrix.
    t, p = to_heatwave_codes(y_true_temp), to_heatwave_codes(y_pred_temp)
    prec, rec, f1, sup_ = precision_recall_fscore_support(t, p, labels=CLASS_LABELS, zero_division=0)
    per_class = pd.DataFrame({"precision": prec, "recall": rec, "f1": f1, "support": sup_},
                             index=[f"{c} - {CATEGORY_NAMES[c]}" for c in CLASS_LABELS])
    has = sup_ > 0
    summary = {"accuracy": accuracy_score(t, p),
               "macro precision": float(prec[has].mean()) if has.any() else np.nan,
               "macro recall": float(rec[has].mean()) if has.any() else np.nan,
               "macro F1": float(f1[has].mean()) if has.any() else np.nan,
               "weighted F1": float(np.average(f1, weights=sup_)) if sup_.sum() else np.nan,
               "total support": int(sup_.sum())}
    cm = pd.DataFrame(confusion_matrix(t, p, labels=CLASS_LABELS),
                      index=[f"actual {c}" for c in CLASS_LABELS], columns=[f"pred {c}" for c in CLASS_LABELS])
    return summary, per_class, cm

