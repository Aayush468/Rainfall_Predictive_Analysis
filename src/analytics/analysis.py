from __future__ import annotations

import pandas as pd
import plotly.express as px


def imd_daily_rain_category(amount_mm: float) -> str:
    """Return the IMD daily rainfall amount band (amount thresholds only)."""
    if pd.isna(amount_mm) or amount_mm < 0.1:
        return "No measurable rain"
    if amount_mm <= 2.4:
        return "Very light"
    if amount_mm <= 15.5:
        return "Light"
    if amount_mm <= 64.4:
        return "Moderate"
    if amount_mm <= 115.5:
        return "Heavy"
    if amount_mm <= 204.4:
        return "Very heavy"
    return "Extremely heavy"


def rainfall_figures(observations: pd.DataFrame) -> dict[str, object]:
    """Create descriptive figures from observed district rainfall only."""
    if observations.empty:
        return {}
    frame = observations.copy()
    frame["observation_date"] = pd.to_datetime(frame["observation_date"])
    frame["month"] = frame["observation_date"].dt.month
    frame["season"] = frame["month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Pre-monsoon", 4: "Pre-monsoon", 5: "Pre-monsoon",
        6: "Southwest monsoon", 7: "Southwest monsoon", 8: "Southwest monsoon", 9: "Southwest monsoon",
        10: "Post-monsoon", 11: "Post-monsoon",
    })
    figures = {}
    if "state_name" in frame and frame["state_name"].notna().any():
        by_state = frame.groupby("state_name", as_index=False).actual_rainfall_mm.mean().sort_values("actual_rainfall_mm", ascending=False)
        figures["state_average"] = px.bar(by_state, x="state_name", y="actual_rainfall_mm", title="Mean Daily Rainfall by State (Available Records)", labels={"state_name": "State", "actual_rainfall_mm": "Rainfall (mm)"})
    if "district_name" in frame:
        by_district = frame.groupby("district_name", as_index=False).actual_rainfall_mm.mean().sort_values("actual_rainfall_mm", ascending=False)
        figures["district_highest"] = px.bar(by_district.head(15), x="district_name", y="actual_rainfall_mm", title="Highest Mean Daily Rainfall by District", labels={"district_name": "District", "actual_rainfall_mm": "Rainfall (mm)"})
        figures["district_lowest"] = px.bar(by_district.tail(15).sort_values("actual_rainfall_mm"), x="district_name", y="actual_rainfall_mm", title="Lowest Mean Daily Rainfall by District", labels={"district_name": "District", "actual_rainfall_mm": "Rainfall (mm)"})
    monthly = frame.groupby("month", as_index=False).actual_rainfall_mm.mean()
    figures["monthly"] = px.bar(monthly, x="month", y="actual_rainfall_mm", title="Mean Daily Rainfall by Calendar Month", labels={"month": "Month", "actual_rainfall_mm": "Rainfall (mm)"})
    seasonal = frame.groupby("season", as_index=False).actual_rainfall_mm.mean()
    figures["seasonal"] = px.bar(seasonal, x="season", y="actual_rainfall_mm", title="Mean Daily Rainfall by Season", labels={"season": "Season", "actual_rainfall_mm": "Rainfall (mm)"})
    daily = frame.groupby("observation_date", as_index=False).actual_rainfall_mm.mean()
    figures["daily"] = px.line(daily, x="observation_date", y="actual_rainfall_mm", title="Daily Mean Rainfall (Available Period)", labels={"observation_date": "Date", "actual_rainfall_mm": "Rainfall (mm)"})
    if "departure_percentage" in frame:
        dep = frame.dropna(subset=["departure_percentage"])
        if not dep.empty:
            figures["departure"] = px.histogram(dep, x="departure_percentage", title="Rainfall Departure from Normal", labels={"departure_percentage": "Departure from Normal (%)"})
    if "rainfall_category" in frame:
        categories = frame.rainfall_category.fillna("Unknown").value_counts().rename_axis("category").reset_index(name="records")
        figures["categories"] = px.bar(categories, x="category", y="records", title="Rainfall Category Records", labels={"category": "Rainfall Category", "records": "Records"})
    return figures


def quality_summary(frame: pd.DataFrame) -> dict:
    result = {"rows": int(len(frame)), "missing_cells": int(frame.isna().sum().sum()),
              "missing_by_column": frame.isna().sum().astype(int).to_dict()}
    if "actual_rainfall_mm" in frame:
        rain = pd.to_numeric(frame.actual_rainfall_mm, errors="coerce")
        q1, q3 = rain.quantile([0.25, 0.75])
        result["negative_rainfall_rows"] = int((rain < 0).sum())
        result["iqr_outlier_rows"] = int(((rain < q1 - 1.5 * (q3 - q1)) | (rain > q3 + 1.5 * (q3 - q1))).sum())
    return result
