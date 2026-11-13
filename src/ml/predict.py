from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.dialects.postgresql import insert

from ..config import ROOT
from ..database.models import GridMLPrediction
from ..database.session import make_engine
from .prepare_features import latest_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict next-day rainfall for locations with 14 recent daily observations")
    parser.add_argument("history", type=Path, help="Verified rainfall CSV (same location columns used for training)")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "rainfall_predictive_project_model.joblib")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "rainfall_predictive_project_predictions.csv")
    parser.add_argument("--store-db", action="store_true", help="Store grid-cell predictions in PostgreSQL")
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    frame = pd.read_csv(args.history)
    features = latest_features(frame)
    if features.empty:
        raise SystemExit("No location has 14 consecutive daily observations")
    feature_names = bundle["features"]
    missing = [name for name in feature_names if name not in features]
    if missing:
        raise SystemExit(f"History does not provide the model feature columns: {missing}")
    model = bundle["model"]
    # Inference usually covers few locations; run GPU-trained XGBoost bundles
    # on CPU here unless the caller explicitly opts into device transfer.
    if model.__class__.__module__.startswith("xgboost"):
        model.set_params(device="cpu")
    values = np.maximum(0, model.predict(features[feature_names]))
    identity = [column for column in ("district_id", "grid_latitude", "grid_longitude", "location_id") if column in features]
    output = features[identity + ["prediction_date"]].copy()
    output["predicted_rainfall_mm"] = values
    output["prediction_type"] = "ML estimate; not an official IMD forecast"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    if args.store_db:
        if not {"grid_latitude", "grid_longitude"}.issubset(output.columns):
            raise SystemExit("Database persistence currently requires grid_latitude/grid_longitude model locations")
        records = [{
            "grid_latitude": row.grid_latitude,
            "grid_longitude": row.grid_longitude,
            "prediction_date": pd.Timestamp(row.prediction_date).date(),
            "predicted_rainfall_mm": float(row.predicted_rainfall_mm),
            "model_version": bundle["model_version"],
        } for row in output.itertuples(index=False)]
        engine = make_engine()
        with engine.begin() as connection:
            connection.execute(insert(GridMLPrediction).on_conflict_do_nothing(
                constraint="uq_grid_prediction_date_model"), records)
    print(f"Wrote {len(output)} ML estimates to {args.output}")


if __name__ == "__main__":
    main()
