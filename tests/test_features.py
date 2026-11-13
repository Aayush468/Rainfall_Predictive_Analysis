import pandas as pd
import pytest

from src.ml.prepare_features import latest_features, prepare_features
from src.ml.train import train


def test_features_use_past_data_and_next_day_target():
    frame = pd.DataFrame({
        "grid_latitude": [10.0] * 20,
        "grid_longitude": [75.0] * 20,
        "observation_date": pd.date_range("2024-01-01", periods=20),
        "rainfall_mm": list(range(20)),
    })
    X, y, target_dates = prepare_features(frame)
    assert X.iloc[0].lag_1 == 13
    assert X.iloc[0].lag_2 == 12
    assert X.iloc[0].rolling_mean_7 == 10
    assert y.iloc[0] == 14
    assert target_dates.iloc[0] == pd.Timestamp("2024-01-15")


def test_features_do_not_bridge_missing_calendar_days():
    dates = pd.date_range("2024-01-01", periods=21).delete(5)
    frame = pd.DataFrame({"observation_date": dates, "rainfall_mm": range(20)})
    X, _, _ = prepare_features(frame)
    assert len(X) == 1


def test_negative_rainfall_rejected():
    frame = pd.DataFrame({"observation_date": pd.date_range("2024-01-01", periods=20), "rainfall_mm": [-1] + [0] * 19})
    with pytest.raises(ValueError, match="non-negative"):
        prepare_features(frame)


def test_training_writes_chronological_metrics_and_model(tmp_path):
    history = pd.DataFrame({
        "grid_latitude": [10.0] * 90,
        "grid_longitude": [75.0] * 90,
        "observation_date": pd.date_range("2022-01-01", periods=90),
        "rainfall_mm": [float((day % 9) * 2) for day in range(90)],
    })
    source = tmp_path / "observations.csv"
    model = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    history.to_csv(source, index=False)
    metrics = train(source, model, metrics_path)
    assert model.exists() and metrics_path.exists()
    assert metrics["test_target_start"] > metrics["training_target_end"]
    assert set(metrics["random_forest"]) == {"mae_mm", "rmse_mm", "r2"}
    assert "skill_vs_previous_day_percent" in metrics
    assert metrics["model_version"] in {"rf-next-day-v2-multiyear", "xgboost-next-day-v1"}
    assert metrics["top_feature_importance"]


def test_latest_features_require_consecutive_history():
    frame = pd.DataFrame({
        "grid_latitude": [10.0] * 20,
        "grid_longitude": [75.0] * 20,
        "observation_date": pd.date_range("2024-01-01", periods=20),
        "rainfall_mm": list(range(20)),
    })
    latest = latest_features(frame)
    assert latest.iloc[0].prediction_date == pd.Timestamp("2024-01-21")
    assert latest.iloc[0].lag_1 == 19
    assert latest.iloc[0].lag_14 == 6
