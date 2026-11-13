from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..config import ROOT
from .prepare_features import prepare_features

logger = logging.getLogger(__name__)


def train(input_path: Path, model_path: Path, metrics_path: Path, device: str = "auto") -> dict:
    observations = pd.read_csv(input_path)
    X, y, target_dates = prepare_features(observations)
    dates = pd.to_datetime(target_dates)
    unique_dates = np.sort(dates.unique())
    if len(unique_dates) < 2:
        raise ValueError("At least two target dates are needed for chronological train/test evaluation")
    split_index = max(1, min(len(unique_dates) - 1, int(len(unique_dates) * 0.8)))
    cutoff = pd.Timestamp(unique_dates[split_index])
    train_mask = (dates < cutoff).to_numpy()
    test_mask = (dates >= cutoff).to_numpy()
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Chronological split produced an empty train or test set")

    rf_evaluation = RandomForestRegressor(
        n_estimators=100, max_depth=12, min_samples_leaf=2,
        random_state=42, n_jobs=1,
    )
    rf_evaluation.fit(X.iloc[train_mask], y.iloc[train_mask])
    rf_prediction = np.maximum(0, rf_evaluation.predict(X.iloc[test_mask]))

    gpu_enabled = device != "cpu"
    if device == "auto":
        try:
            # Ask XGBoost to allocate on CUDA; this fails clearly on a CPU-only install.
            xgb_evaluation = xgb.XGBRegressor(
                n_estimators=500, max_depth=7, learning_rate=0.05,
                min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=5.0, objective="reg:squarederror", eval_metric="rmse",
                tree_method="hist", device="cuda", n_jobs=1, random_state=42,
            )
            xgb_evaluation.fit(X.iloc[train_mask].astype("float32"),
                               y.iloc[train_mask].astype("float32"), verbose=False)
        except Exception as exc:
            logger.warning("XGBoost CUDA training unavailable; falling back to CPU: %s", exc)
            gpu_enabled = False
    elif device == "cuda":
        xgb_evaluation = xgb.XGBRegressor(
            n_estimators=500, max_depth=7, learning_rate=0.05,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=5.0, objective="reg:squarederror", eval_metric="rmse",
            tree_method="hist", device=device, n_jobs=1, random_state=42,
        )
        xgb_evaluation.fit(X.iloc[train_mask].astype("float32"),
                           y.iloc[train_mask].astype("float32"), verbose=False)
    else:
        xgb_evaluation = xgb.XGBRegressor(
            n_estimators=500, max_depth=7, learning_rate=0.05,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=5.0, objective="reg:squarederror", eval_metric="rmse",
            tree_method="hist", device="cpu", n_jobs=-1, random_state=42,
        )
        xgb_evaluation.fit(X.iloc[train_mask].astype("float32"),
                           y.iloc[train_mask].astype("float32"), verbose=False)

    if not gpu_enabled and device == "auto":
        xgb_evaluation = xgb.XGBRegressor(
            n_estimators=500, max_depth=7, learning_rate=0.05,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=5.0, objective="reg:squarederror", eval_metric="rmse",
            tree_method="hist", device="cpu", n_jobs=-1, random_state=42,
        )
        xgb_evaluation.fit(X.iloc[train_mask].astype("float32"),
                           y.iloc[train_mask].astype("float32"), verbose=False)
    # Evaluation rows are already in host memory; CPU inference avoids a
    # needless round-trip warning on this small output batch.
    xgb_evaluation.set_params(device="cpu")
    xgb_prediction = np.maximum(0, xgb_evaluation.predict(X.iloc[test_mask].astype("float32")))
    gpu_device = "cuda" if gpu_enabled else "cpu"
    baseline = np.maximum(0, X.iloc[test_mask]["lag_1"].to_numpy())
    actual = y.iloc[test_mask].to_numpy()

    def score(values):
        return {
            "mae_mm": float(mean_absolute_error(actual, values)),
            "rmse_mm": float(np.sqrt(mean_squared_error(actual, values))),
            "r2": float(r2_score(actual, values)) if len(actual) > 1 else None,
        }

    baseline_metrics = score(baseline)
    rf_metrics = score(rf_prediction)
    xgb_metrics = score(xgb_prediction)
    selected_name, selected_eval, model_metrics = min(
        (("random_forest", rf_evaluation, rf_metrics),
         ("xgboost", xgb_evaluation, xgb_metrics)),
        key=lambda candidate: (candidate[2]["mae_mm"], candidate[2]["rmse_mm"]),
    )
    if selected_name == "xgboost":
        model_version = "xgboost-next-day-v1"
        importance_values = selected_eval.feature_importances_
    else:
        model_version = "rf-next-day-v2-multiyear"
        importance_values = selected_eval.feature_importances_
    feature_importance = sorted(
        ({"feature": name, "importance": float(value)}
         for name, value in zip(X.columns, importance_values)),
        key=lambda item: item["importance"], reverse=True,
    )
    test_actual = pd.Series(actual)
    wet_days = int((test_actual >= 0.1).sum())
    heavy_days = int((test_actual >= 64.5).sum())
    metrics = {
        "source_file": str(input_path),
        "target": "next_calendar_day_rainfall_mm",
        "feature_names": list(X.columns),
        "model_version": model_version,
        "selected_model": selected_name,
        "training_device": gpu_device if selected_name == "xgboost" else "cpu",
        "rows_total": int(len(X)),
        "rows_train": int(train_mask.sum()),
        "rows_test": int(test_mask.sum()),
        "training_target_start": str(dates[train_mask].min().date()),
        "training_target_end": str(dates[train_mask].max().date()),
        "test_target_start": str(dates[test_mask].min().date()),
        "test_target_end": str(dates[test_mask].max().date()),
        "baseline_previous_day": baseline_metrics,
        "random_forest": rf_metrics,
        "xgboost": xgb_metrics,
        "selected_model_metrics": model_metrics,
        "skill_vs_previous_day_percent": {
            "mae": float(100 * (baseline_metrics["mae_mm"] - model_metrics["mae_mm"]) / baseline_metrics["mae_mm"]),
            "rmse": float(100 * (baseline_metrics["rmse_mm"] - model_metrics["rmse_mm"]) / baseline_metrics["rmse_mm"]),
        },
        "test_rainfall_profile": {
            "mean_mm": float(test_actual.mean()),
            "wet_grid_days_at_least_0_1mm": wet_days,
            "wet_grid_day_percent": float(100 * wet_days / len(test_actual)),
            "heavy_grid_days_at_least_64_5mm": heavy_days,
        },
        "top_feature_importance": feature_importance[:10],
        "model_selection_metric": "Lowest holdout MAE; consult RMSE too because heavy-rain outliers matter.",
        "model_note": f"{selected_name} next-day grid-cell estimate using rainfall history; XGBoost candidate training device was {gpu_device}. Not an official forecast. Scores are from a chronological holdout; compare against persistence before use.",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    # Refit the deployable model on all available history only after the holdout
    # evaluation is complete, so the test period never influences its score.
    if selected_name == "xgboost":
        deployment_model = xgb.XGBRegressor(
            n_estimators=500, max_depth=7, learning_rate=0.05,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=5.0, objective="reg:squarederror", eval_metric="rmse",
            tree_method="hist", device=gpu_device, n_jobs=1, random_state=42,
        )
        deployment_model.fit(X.astype("float32"), y.astype("float32"), verbose=False)
    else:
        deployment_model = RandomForestRegressor(
            n_estimators=100, max_depth=12, min_samples_leaf=2,
            random_state=42, n_jobs=1,
        )
        deployment_model.fit(X, y)
    joblib.dump({"model": deployment_model, "features": list(X.columns),
                 "model_version": metrics["model_version"]}, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train next-day rainfall model from verified observations CSV")
    parser.add_argument("input", type=Path, help="CSV with observation_date, rainfall_mm, and location columns")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "rainfall_predictive_project_model.joblib")
    parser.add_argument("--metrics", type=Path, default=ROOT / "models" / "rainfall_predictive_project_metrics.json")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                        help="XGBoost training device; auto tries CUDA then uses CPU if unavailable")
    args = parser.parse_args()
    result = train(args.input, args.model, args.metrics, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
