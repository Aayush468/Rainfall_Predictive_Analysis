from __future__ import annotations

import pandas as pd

IDENTITY_COLUMNS = ("district_id", "grid_latitude", "grid_longitude", "location_id")


def prepare_features(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create next-calendar-day targets with past-only rainfall features.

    Input must be one observation per location and day. Rows with gaps are not
    treated as adjacent days: reindexing each series prevents lag leakage over
    missing calendar dates.
    """
    required = {"observation_date", "rainfall_mm"}
    if not required.issubset(observations.columns):
        raise ValueError(f"Input requires columns {sorted(required)}")
    frame = observations.copy()
    frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce").dt.normalize()
    frame["rainfall_mm"] = pd.to_numeric(frame["rainfall_mm"], errors="coerce")
    frame = frame.dropna(subset=["observation_date", "rainfall_mm"])
    if (frame["rainfall_mm"] < 0).any():
        raise ValueError("Rainfall observations must be non-negative")
    location_cols = [column for column in IDENTITY_COLUMNS if column in frame.columns]
    if not location_cols:
        frame["location_id"] = "India"
        location_cols = ["location_id"]
    if frame.duplicated(location_cols + ["observation_date"]).any():
        raise ValueError("Input has duplicate location/date rows; aggregate explicitly before training")
    frame = frame.sort_values(location_cols + ["observation_date"])
    groups = frame.groupby(location_cols, sort=False, dropna=False)
    feature_parts, target_parts, date_parts = [], [], []
    for _, group in groups:
        group = group.set_index("observation_date").sort_index()
        daily = group["rainfall_mm"].asfreq("D")
        features = pd.DataFrame(index=daily.index)
        location = group.iloc[0]
        for column in ("grid_latitude", "grid_longitude", "district_id"):
            if column in location_cols:
                features[column] = pd.to_numeric(location[column], errors="coerce")
        # This row is observation date t; target is rainfall at t+1.
        for lag in (1, 2, 3, 7, 14):
            features[f"lag_{lag}"] = daily.shift(lag - 1)
        features["rolling_mean_7"] = daily.rolling(7, min_periods=7).mean()
        features["rolling_mean_14"] = daily.rolling(14, min_periods=14).mean()
        target_dates = features.index + pd.Timedelta(days=1)
        features["month"] = target_dates.month
        features["day_of_year"] = target_dates.day_of_year
        target = daily.shift(-1).rename("target_rainfall_mm")
        valid = features.notna().all(axis=1) & target.notna()
        if valid.any():
            feature_parts.append(features.loc[valid])
            target_parts.append(target.loc[valid])
            date_parts.append(pd.Series(target_dates[valid], index=features.index[valid], name="target_date"))
    if not feature_parts:
        raise ValueError("Not enough consecutive daily observations to build lag/rolling features")
    X = pd.concat(feature_parts, ignore_index=True).fillna(0)
    y = pd.concat(target_parts, ignore_index=True)
    target_dates = pd.concat(date_parts, ignore_index=True)
    return X, y, target_dates


def latest_features(observations: pd.DataFrame) -> pd.DataFrame:
    """Build one next-day feature row per location from its latest 14 days."""
    required = {"observation_date", "rainfall_mm"}
    if not required.issubset(observations.columns):
        raise ValueError(f"Input requires columns {sorted(required)}")
    frame = observations.copy()
    frame["observation_date"] = pd.to_datetime(frame.observation_date, errors="coerce").dt.normalize()
    frame["rainfall_mm"] = pd.to_numeric(frame.rainfall_mm, errors="coerce")
    frame = frame.dropna(subset=["observation_date", "rainfall_mm"])
    if (frame.rainfall_mm < 0).any():
        raise ValueError("Rainfall observations must be non-negative")
    location_cols = [column for column in IDENTITY_COLUMNS if column in frame.columns]
    if not location_cols:
        frame["location_id"] = "India"
        location_cols = ["location_id"]
    if frame.duplicated(location_cols + ["observation_date"]).any():
        raise ValueError("Input has duplicate location/date rows")
    result = []
    for _, group in frame.groupby(location_cols, sort=False, dropna=False):
        group = group.sort_values("observation_date")
        history = group.set_index("observation_date").rainfall_mm.asfreq("D").dropna()
        if len(history) < 14 or (history.index[-1] - history.index[-14]).days != 13:
            continue
        last = history.iloc[-14:]
        target_date = history.index[-1] + pd.Timedelta(days=1)
        row = {f"lag_{lag}": float(last.iloc[-lag]) for lag in (1, 2, 3, 7, 14)}
        row["rolling_mean_7"] = float(last.iloc[-7:].mean())
        row["rolling_mean_14"] = float(last.mean())
        row["month"] = int(target_date.month)
        row["day_of_year"] = int(target_date.day_of_year)
        location = group.iloc[0]
        for column in ("grid_latitude", "grid_longitude", "district_id"):
            if column in location_cols:
                row[column] = pd.to_numeric(location[column], errors="coerce")
        for column in location_cols:
            row[column] = location[column]
        row["prediction_date"] = target_date
        result.append(row)
    return pd.DataFrame(result)
