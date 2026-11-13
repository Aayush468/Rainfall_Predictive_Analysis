from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from sqlalchemy import text

from src.analytics.analysis import imd_daily_rain_category, quality_summary, rainfall_figures
from src.analytics.weather_summary import weatherapi_daily_frame, weatherapi_hourly_frame
from src.config import ROOT, settings
from src.database.session import create_schema, make_engine
from src.ingestion.run_pipeline import run_api

pio.templates.default = "plotly_dark"

st.set_page_config(page_title="Rainfall Predictive Project", layout="wide")
st.title("Rainfall Predictive Project")
st.caption("Observed rainfall, numerical forecasts, official outlooks, and machine-learning estimates are identified separately.")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#071522; color:#e5eef8; }
[data-testid="stHeader"] { background:rgba(7,21,34,.88); }
[data-testid="stSidebar"] { display:none; }
[data-testid="stSidebarCollapsedControl"] { display:none; }
.main .block-container { max-width:1500px; padding-top:1.4rem; }
[data-testid="stMetric"] { background:#12283a; border:1px solid #213d52; border-radius:15px; padding:12px 14px; min-width:0; }
[data-testid="stMetricLabel"] p { color:#9fb5c8 !important; font-size:clamp(.72rem, .9vw, .9rem) !important; line-height:1.25; white-space:normal; overflow-wrap:anywhere; }
[data-testid="stMetricValue"] { color:#eff7ff !important; font-size:clamp(1rem, 1.45vw, 1.45rem) !important; line-height:1.2; white-space:normal; overflow-wrap:anywhere; word-break:normal; }
[data-testid="stMetricValue"] div { font-size:inherit !important; line-height:inherit; white-space:inherit; overflow-wrap:inherit; }
[data-testid="stMetricDelta"] { font-size:clamp(.68rem, .8vw, .82rem) !important; line-height:1.2; white-space:normal; overflow-wrap:anywhere; }
[data-testid="stTextInput"] input { background:#142a3a; color:#edf5fc; border:1px solid #2c465a; border-radius:14px; }
[data-testid="stButton"] button { border-radius:12px; }
[data-testid="stButton"] button[kind="primary"] { background:#256a98; border:1px solid #56a7da; color:white; }
[data-testid="stAlert"] { background:#12283a; border:1px solid #29465b; }
[data-testid="stExpander"] { background:#102536; border:1px solid #263f52; border-radius:14px; }
[data-testid="stDataFrame"] { border:1px solid #263f52; border-radius:12px; }
.top-nav-label { color:#9fb5c8; font-size:12px; text-transform:uppercase; letter-spacing:.1em; margin:4px 0 8px; }
.nav-divider { border:0; border-top:1px solid #243a4c; margin:0 0 18px; }
[data-testid="stSegmentedControl"] { width:100%; }
[data-testid="stSegmentedControl"] button { border-radius:12px; font-weight:600; }
.weather-hero { position:relative; overflow:hidden; min-height:225px; padding:26px 30px; border-radius:24px;
  background:linear-gradient(112deg,#183850 0%,#29495e 52%,#536b73 100%); border:1px solid #426175; color:#f2f8ff; }
.weather-hero:after { content:""; position:absolute; width:250px; height:250px; right:7%; top:-115px; border-radius:50%;
  background:radial-gradient(circle,rgba(255,224,139,.55),rgba(255,224,139,.04) 68%,transparent 72%); }
.hero-place { font-size:18px; color:#c8d9e7; letter-spacing:.02em; }
.hero-temp { font-size:76px; line-height:1.05; font-weight:650; letter-spacing:-.06em; }
.hero-condition { font-size:19px; color:#e1edf6; }
.hero-summary { font-size:15px; color:#c1d2df; margin-top:12px; max-width:720px; }
.hourly-scroll { display:flex; gap:10px; overflow-x:auto; padding:6px 2px 14px; }
.hour-card { flex:0 0 105px; text-align:center; background:#12283a; border:1px solid #263f52; border-radius:15px; padding:13px 8px; }
.hour-time { color:#a9bfd1; font-size:13px; }
.hour-condition { color:#b9ccdc; font-size:11px; line-height:1.25; min-height:28px; margin:8px 0 4px; }
.hour-temp { font-size:18px; font-weight:600; color:#edf6ff; }
.hour-rain { font-size:12px; color:#86c7ee; margin-top:5px; }
.condition-card { min-height:115px; background:#12283a; border:1px solid #263f52; border-radius:16px; padding:14px 16px; }
.condition-label { color:#9fb5c8; font-size:13px; }
.condition-value { font-size:23px; font-weight:600; color:#edf6ff; margin-top:7px; }
.condition-detail { font-size:12px; color:#a9bfd1; margin-top:3px; }
[data-testid="stRadio"] label { color:#dce8f2 !important; }
@media (max-width:760px) {
  .main .block-container { padding:1rem .75rem 2rem; }
  .weather-hero { min-height:190px; padding:20px; }
  .hero-temp { font-size:62px; }
  .hour-card { flex-basis:88px; }
  [data-testid="stMetric"] { padding:10px 11px; }
  [data-testid="stMetricValue"] { font-size:1rem !important; }
  [data-testid="stMetricLabel"] p { font-size:.72rem !important; }
  [data-testid="stMetricDelta"] { font-size:.68rem !important; }
}
</style>
""", unsafe_allow_html=True)

page_labels = {
    "Overview": "Overview",
    "Rainfall analytics": "Rainfall Analytics",
    "Forecasts and warnings": "Forecasts & Warnings",
    "Live weather": "Live Weather",
    "Historical grid": "Historical Grid",
    "ML model": "ML Model",
}
page = st.segmented_control(
    "Dashboard pages",
    options=list(page_labels),
    format_func=page_labels.get,
    selection_mode="single",
    default="Overview",
    key="selected_dashboard_page",
    width="stretch",
    wrap=True,
    label_visibility="collapsed",
)
page = page or "Overview"
st.markdown("<hr class='nav-divider'>", unsafe_allow_html=True)


@st.cache_data(ttl=120, show_spinner=False)
def read_query(query: str) -> pd.DataFrame:
    return pd.read_sql(text(query), make_engine())


def load_table(query: str, label: str) -> pd.DataFrame:
    try:
        return read_query(query)
    except Exception as exc:
        st.warning(f"{label} is unavailable. Initialize the local database and load its data. Details: {exc}")
        return pd.DataFrame()


def weather_number(value, suffix: str, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "Unavailable"


def show_quality_summary(frame: pd.DataFrame) -> None:
    quality = quality_summary(frame)
    a, b, c, d = st.columns(4)
    a.metric("Daily Records", f"{quality['rows']:,}")
    b.metric("Missing Values", f"{quality['missing_cells']:,}")
    c.metric("Negative Rainfall Rows", f"{quality.get('negative_rainfall_rows', 0):,}")
    d.metric("Unusual Values Flagged", f"{quality.get('iqr_outlier_rows', 0):,}")
    st.caption("Values outside the 1.5x IQR range are flagged for review. Very heavy rain can be real, so these records are not removed automatically.")
    with st.expander("Missing Values by Field"):
        missing = pd.Series(quality["missing_by_column"], name="Missing Values").rename_axis("Field").reset_index()
        show_dataframe(missing, width="stretch", hide_index=True)


def display_column_name(column: object) -> str:
    tokens = str(column).replace("%", "percent").strip().split("_")
    units = {"mm": "mm", "pct": "%", "percent": "%", "c": "C", "kph": "km/h", "utc": "UTC"}
    unit = units.get(tokens[-1].lower()) if tokens else None
    words = tokens[:-1] if unit else tokens
    label = " ".join(word.upper() if word.lower() in {"id", "mae", "rmse", "r2", "ml"} else word.capitalize() for word in words)
    return f"{label} ({unit})" if unit else label


def show_dataframe(frame: pd.DataFrame, **kwargs) -> None:
    display = frame.copy()
    display.columns = [display_column_name(column) for column in display.columns]
    for column in display.select_dtypes(include=["object", "string"]).columns:
        display[column] = display[column].map(
            lambda value: project_text(value) if isinstance(value, str) else value
        )
    for column in display.columns:
        if column.lower() in {"data source", "source"}:
            display[column] = display[column].map(
                lambda value: "External Weather Data" if isinstance(value, str) and value in {
                    "Live Weather Service", "Numerical Weather Service", "Official Rainfall Service"
                } else value
            )
    st.dataframe(display, **kwargs)


def project_text(value: object) -> str:
    text = str(value)
    for name, replacement in (
        ("WeatherAPI.com", "Live Weather Service"),
        ("WeatherAPI", "Live Weather Service"),
        ("Open-Meteo", "Numerical Weather Service"),
        ("IMD", "Official Rainfall Service"),
        ("api.weatherapi.com", "the live weather service"),
    ):
        text = text.replace(name, replacement).replace(name.lower(), replacement)
    return text


@st.cache_resource(show_spinner=False)
def ensure_hosted_schema() -> None:
    create_schema(make_engine())


if getattr(settings, "auto_create_schema", False):
    try:
        ensure_hosted_schema()
    except Exception:
        st.error("Could not initialize the database. Check its connection details and schema permissions.")


observations = load_table("""
    SELECT s.state_name, d.district_name, r.observation_date, r.actual_rainfall_mm,
           r.normal_rainfall_mm, r.departure_percentage, r.rainfall_category, r.data_source
    FROM rainfall_observations r JOIN districts d USING (district_id)
    JOIN states s USING (state_id)
""", "District Observations")
states = load_table("""
    SELECT s.state_name, r.observation_date, r.actual_rainfall_mm, r.normal_rainfall_mm,
           r.departure_percentage, r.rainfall_category, r.data_source
    FROM state_rainfall_observations r JOIN states s USING (state_id)
""", "State Observations")
grid = load_table("""
    WITH selected_cells AS (
        SELECT grid_latitude, grid_longitude
        FROM rainfall_grid_observations
        WHERE rainfall_mm IS NOT NULL
        GROUP BY grid_latitude, grid_longitude
        ORDER BY COUNT(DISTINCT observation_date) DESC,
                 md5(grid_latitude::text || ',' || grid_longitude::text)
        LIMIT 200
    )
    SELECT g.grid_latitude, g.grid_longitude, AVG(g.rainfall_mm) AS rainfall_mm,
           MIN(g.observation_date) AS first_date, MAX(g.observation_date) AS last_date,
           COUNT(*) AS observation_count
    FROM rainfall_grid_observations g JOIN selected_cells c USING (grid_latitude, grid_longitude)
    WHERE g.rainfall_mm IS NOT NULL
    GROUP BY g.grid_latitude, g.grid_longitude
""", "Historical gridded observations")
grid_daily = load_table("""
    WITH selected_cells AS (
        SELECT grid_latitude, grid_longitude
        FROM rainfall_grid_observations
        WHERE rainfall_mm IS NOT NULL
        GROUP BY grid_latitude, grid_longitude
        ORDER BY COUNT(DISTINCT observation_date) DESC,
                 md5(grid_latitude::text || ',' || grid_longitude::text)
        LIMIT 200
    )
    SELECT g.observation_date, AVG(g.rainfall_mm) AS rainfall_mm,
           COUNT(*) AS cells_reporting
    FROM rainfall_grid_observations g JOIN selected_cells c USING (grid_latitude, grid_longitude)
    WHERE g.rainfall_mm IS NOT NULL
    GROUP BY g.observation_date
    ORDER BY g.observation_date
""", "Historical gridded observations")
grid_history = load_table("""
    WITH selected_cells AS (
        SELECT grid_latitude, grid_longitude
        FROM rainfall_grid_observations
        WHERE rainfall_mm IS NOT NULL
        GROUP BY grid_latitude, grid_longitude
        ORDER BY COUNT(DISTINCT observation_date) DESC,
                 md5(grid_latitude::text || ',' || grid_longitude::text)
        LIMIT 200
    )
    SELECT g.grid_latitude, g.grid_longitude, g.observation_date, g.rainfall_mm
    FROM rainfall_grid_observations g JOIN selected_cells c USING (grid_latitude, grid_longitude)
    WHERE g.rainfall_mm IS NOT NULL
    ORDER BY g.observation_date, g.grid_latitude, g.grid_longitude
""", "Historical grid detail")
district_places = load_table("""
    SELECT d.district_name, s.state_name, d.latitude, d.longitude
    FROM districts d JOIN states s USING (state_id)
    WHERE d.latitude IS NOT NULL AND d.longitude IS NOT NULL
""", "District location names")


def add_nearest_district(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the nearest district centroid name when the database has coordinates."""
    result = frame.copy()
    result["nearest_district"] = "Name unavailable"
    if result.empty or district_places.empty:
        return result
    places = district_places.dropna(subset=["latitude", "longitude"]).copy()
    if places.empty:
        return result
    place_lat = pd.to_numeric(places.latitude).to_numpy()
    place_lon = pd.to_numeric(places.longitude).to_numpy()
    for index, row in result.iterrows():
        lat, lon = float(row.grid_latitude), float(row.grid_longitude)
        lat1, lat2 = np.radians(lat), np.radians(place_lat)
        dlat = lat2 - lat1
        dlon = np.radians(place_lon - lon)
        haversine = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        nearest = int(np.argmin(haversine))
        place = places.iloc[nearest]
        result.at[index, "nearest_district"] = f"{place.district_name}, {place.state_name} (approx.)"
    return result


def coordinate_key(row) -> str:
    return f"{float(row.grid_latitude):.5f},{float(row.grid_longitude):.5f}"


def location_name_cache() -> dict[str, str]:
    cache = st.session_state.setdefault("forecast_location_names", {})
    path = ROOT / "data" / "geography" / "grid_district_names.csv"
    if path.exists():
        names = pd.read_csv(path).dropna(subset=["district_name", "state_name"])
        cache.update({
            f"{float(row.latitude):.5f},{float(row.longitude):.5f}":
            f"{row.district_name}, {row.state_name}"
            for row in names.itertuples(index=False) if str(row.district_name).strip()
        })
    return cache


def apply_resolved_place_names(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    cache = location_name_cache()
    lookup_path = ROOT / "data" / "geography" / "grid_district_names.csv"
    if "nearest_district" in result:
        result["nearest_district"] = result.apply(
            lambda row: cache.get(coordinate_key(row), row.nearest_district), axis=1
        )
    if lookup_path.exists() and {"grid_latitude", "grid_longitude"}.issubset(result.columns):
        lookup = pd.read_csv(lookup_path)
        distances = {
            f"{float(row.latitude):.5f},{float(row.longitude):.5f}": row.distance_to_district_centroid_km
            for row in lookup.itertuples(index=False)
        }
        result["district_centroid_distance_km"] = result.apply(
            lambda row: distances.get(coordinate_key(row)), axis=1
        )
    return result


if page == "Overview":
    st.subheader("Historical Rainfall Analysis")
    st.caption("This page gives a high-level view of observed rainfall over time. The main charts use a consistent historical grid sample, so they describe grid points rather than district totals or today's forecast.")
    if not grid.empty and not grid_daily.empty:
        daily = grid_daily.copy()
        daily["observation_date"] = pd.to_datetime(daily.observation_date)
        start_date, end_date = daily.observation_date.min(), daily.observation_date.max()
        a, b, c = st.columns(3)
        a.metric("Grid Data Period", f"{start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}")
        b.metric("Consistently Sampled Grid Cells", f"{len(grid):,}")
        c.metric("Grid Cell/Day Observations", f"{int(grid.observation_count.sum()):,}")
        st.caption("The time series uses the same 200 well-covered grid cells across years. Values are grid-cell averages, not district totals or live forecasts.")
        figures = rainfall_figures(daily.rename(columns={"rainfall_mm": "actual_rainfall_mm"}))
        if "daily" in figures:
            st.plotly_chart(figures["daily"], width="stretch")
        left, right = st.columns(2)
        if "monthly" in figures:
            left.plotly_chart(figures["monthly"], width="stretch")
        if "seasonal" in figures:
            right.plotly_chart(figures["seasonal"], width="stretch")
        map_frame = grid.rename(columns={"rainfall_mm": "mean_rainfall_mm"})
        st.plotly_chart(px.scatter_geo(
            map_frame, lat="grid_latitude", lon="grid_longitude", color="mean_rainfall_mm",
            hover_data={"grid_latitude": ":.2f", "grid_longitude": ":.2f"}, scope="asia",
            labels={"grid_latitude": "Grid Latitude", "grid_longitude": "Grid Longitude", "mean_rainfall_mm": "Mean Rainfall (mm)"},
            title="Mean Daily Rainfall by Grid Cell (mm/day)"), width="stretch")
        if observations.empty:
            st.info("District/state API data is not loaded; the analysis above uses the available gridded dataset.")
    else:
        a, b, c = st.columns(3)
        a.metric("District Observations", "Not loaded" if observations.empty else f"{len(observations):,}")
        b.metric("State Observations", "Not loaded" if states.empty else f"{len(states):,}")
        c.metric("Grid Observations", "Not loaded")
        st.info("No rainfall dataset is available yet. Import an official gridded rainfall dataset to populate the dashboard.")

elif page == "Rainfall analytics":
    st.subheader("Rainfall Analytics")
    st.caption("Use these charts to compare rainfall by date, month, season, state, or district when those records are loaded. Data-quality flags identify records to review; they do not automatically mean the rainfall value is wrong.")
    frame = observations if not observations.empty else states
    if frame.empty:
        if grid_daily.empty:
            st.info("Load verified rainfall observations to populate the analytics.")
        else:
            st.info("Analytics use the same 200 well-covered grid points across years. District and state observations are not connected, so results describe sampled grid points rather than administrative areas.")
            history = grid_history.copy()
            history["observation_date"] = pd.to_datetime(history.observation_date)
            first_date, last_date = history.observation_date.min().date(), history.observation_date.max().date()
            date_range = st.date_input("Analysis period", value=(first_date, last_date), min_value=first_date, max_value=last_date)
            if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
                start_date, end_date = map(pd.Timestamp, date_range)
                period_data = history[history.observation_date.between(start_date, end_date)].copy()
            else:
                period_data = history[history.observation_date.dt.date == date_range].copy()
            if period_data.empty:
                st.warning("No observations are available in this date range.")
            else:
                cell_options = period_data[["grid_latitude", "grid_longitude"]].drop_duplicates().reset_index(drop=True)
                cell_options = add_nearest_district(cell_options)
                cell_options = apply_resolved_place_names(cell_options)
                place_parts = cell_options.nearest_district.str.rsplit(", ", n=1, expand=True)
                cell_options["district_name"] = place_parts[0]
                cell_options["state_name"] = place_parts[1].fillna("Unknown state")
                cell_options = cell_options.sort_values(
                    ["state_name", "district_name", "grid_latitude", "grid_longitude"],
                    key=lambda values: values.str.casefold() if values.dtype == "object" else values,
                )
                cell_labels = {
                    f"{row.state_name} — {row.district_name} — {row.grid_latitude:.2f}, {row.grid_longitude:.2f}":
                    (row.grid_latitude, row.grid_longitude)
                    for row in cell_options.itertuples(index=False)
                }
                selected_label = st.selectbox("Location", ["All sampled locations", *cell_labels])
                if selected_label != "All sampled locations":
                    latitude, longitude = cell_labels[selected_label]
                    period_data = period_data[
                        (period_data.grid_latitude == latitude) & (period_data.grid_longitude == longitude)
                    ]
                rain = period_data.rainfall_mm
                a, b, c, d = st.columns(4)
                a.metric("Median Daily Rain", f"{rain.median():.1f} mm")
                b.metric("90th Percentile", f"{rain.quantile(.9):.1f} mm")
                c.metric("Wet Grid-Cell Days", f"{int((rain >= 0.1).sum()):,}")
                d.metric("Maximum Daily Rain", f"{rain.max():.1f} mm")

                daily_stats = period_data.groupby("observation_date").rainfall_mm.agg(
                    median="median", upper_90=lambda values: values.quantile(.9), mean="mean"
                ).reset_index()
                daily_long = daily_stats.melt(
                    id_vars="observation_date", value_vars=["median", "upper_90"],
                    var_name="Statistic", value_name="Rainfall (mm)",
                )
                st.plotly_chart(px.line(
                    daily_long, x="observation_date", y="Rainfall (mm)", color="Statistic", markers=False,
                    labels={"observation_date": "Date"}, title="Daily Rainfall Distribution Across Selected Locations",
                ), width="stretch")
                month_stats = period_data.assign(month=period_data.observation_date.dt.month).groupby("month").rainfall_mm.agg(
                    median="median", upper_90=lambda values: values.quantile(.9)
                ).reset_index()
                st.plotly_chart(px.line(
                    month_stats.melt(id_vars="month", var_name="Statistic", value_name="Rainfall (mm)"),
                    x="month", y="Rainfall (mm)", color="Statistic", markers=True,
                    labels={"month": "Calendar month"}, title="Seasonal Rainfall Distribution in Selected Period",
                ), width="stretch")
                cell_stats = period_data.groupby(["grid_latitude", "grid_longitude"], as_index=False).agg(
                    mean_rainfall_mm=("rainfall_mm", "mean"),
                    median_rainfall_mm=("rainfall_mm", "median"),
                    wet_days=("rainfall_mm", lambda values: int((values >= 0.1).sum())),
                    observation_days=("rainfall_mm", "count"),
                    max_daily_rain_mm=("rainfall_mm", "max"),
                )
                cell_stats = add_nearest_district(cell_stats)
                cell_stats = apply_resolved_place_names(cell_stats)
                if selected_label == "All sampled locations":
                    st.plotly_chart(px.scatter_geo(
                        cell_stats, lat="grid_latitude", lon="grid_longitude", color="median_rainfall_mm",
                        hover_name="nearest_district",
                        size="wet_days", hover_data={"median_rainfall_mm": ":.1f", "mean_rainfall_mm": ":.1f",
                                                      "wet_days": True, "max_daily_rain_mm": ":.1f",
                                                      "district_centroid_distance_km": ":.1f"},
                        scope="asia", color_continuous_scale="Blues",
                        labels={"median_rainfall_mm": "Median daily rain (mm)", "wet_days": "Wet days"},
                        title="Rainfall by Sampled Grid Location for Selected Period",
                    ), width="stretch")
                    st.markdown("#### Locations with Highest Rainfall in This Period")
                    show_dataframe(cell_stats.sort_values("median_rainfall_mm", ascending=False).head(20)[[
                        "nearest_district", "district_centroid_distance_km", "grid_latitude", "grid_longitude", "median_rainfall_mm",
                        "mean_rainfall_mm", "wet_days", "max_daily_rain_mm",
                    ]], width="stretch", hide_index=True)
                quality_frame = period_data.rename(columns={"rainfall_mm": "actual_rainfall_mm"})
                st.markdown("#### Data Quality for Selected Period")
                show_quality_summary(quality_frame)
    else:
        figures = rainfall_figures(frame)
        if not figures:
            st.info("No compatible rainfall series is available.")
        for figure in figures.values():
            st.plotly_chart(figure, width="stretch")
        st.markdown("#### Data Quality")
        show_quality_summary(frame)
        st.caption("Values outside the 1.5x IQR range are flagged for review; extreme rainfall can be valid and is not automatically removed.")

elif page == "Forecasts and warnings":
    st.subheader("Rainfall Forecasts and Warning Products")
    st.caption("This page separates numerical model estimates from official district forecasts and warnings. Read the issue time and source before interpreting a forecast; grid-point estimates are not district alerts.")
    model_forecasts = load_table("""
        SELECT grid_latitude, grid_longitude, forecast_date, rainfall_mm,
               precipitation_probability_pct, issued_at, data_source
        FROM grid_rainfall_forecasts
        ORDER BY forecast_date, grid_latitude, grid_longitude
    """, "Grid Forecasts")
    st.markdown("##### Numerical Rainfall Forecast (mm/day; model-based, not an observation)")
    if model_forecasts.empty:
        st.info("No grid forecast snapshot is stored. For a current place-based outlook, use **Live weather**.")
    else:
        latest_issue = pd.to_datetime(model_forecasts.issued_at).max()
        active_forecast = model_forecasts[pd.to_datetime(model_forecasts.issued_at) == latest_issue].copy()
        issue_age = pd.Timestamp.now(tz="UTC") - latest_issue
        st.caption(f"Issued {latest_issue:%Y-%m-%d %H:%M UTC}; Source: Numerical Weather Model. "
                   f"{'This stored run is over 24 hours old and should be treated as an archived snapshot.' if issue_age > pd.Timedelta(hours=24) else 'This stored run is less than 24 hours old.'} "
                   "Forecast source details are available in the project documentation.")
        active_forecast["forecast_date"] = pd.to_datetime(active_forecast.forecast_date).dt.date
        week_summary = active_forecast.groupby("forecast_date", as_index=False).agg(
            median_rainfall_mm=("rainfall_mm", "median"),
            p90_rainfall_mm=("rainfall_mm", lambda values: values.quantile(0.90)),
            peak_grid_rainfall_mm=("rainfall_mm", "max"),
            median_rain_probability_pct=("precipitation_probability_pct", "median"),
            p90_rain_probability_pct=("precipitation_probability_pct", lambda values: values.quantile(0.90)),
            cells_with_measurable_rain=("rainfall_mm", lambda values: int((values >= 0.1).sum())),
            cells_with_50pct_rain_chance=("precipitation_probability_pct", lambda values: int((values >= 50).sum())),
        )
        cells_per_day = active_forecast.groupby("forecast_date").size()
        week_summary["rainy_cells_percent"] = (
            100 * week_summary.cells_with_measurable_rain / week_summary.forecast_date.map(cells_per_day)
        )
        st.markdown("##### Seven-Day Outlook Across the Sampled Locations")
        widespread = week_summary.loc[week_summary.cells_with_measurable_rain.idxmax()]
        st.info(
            f"Rain is forecast at the most sampled locations on {pd.Timestamp(widespread.forecast_date):%A, %d %b}: "
            f"{int(widespread.cells_with_measurable_rain)} of {int(cells_per_day.loc[widespread.forecast_date])} points "
            f"({widespread.rainy_cells_percent:.0f}%). The table and chart show the median and 90th-percentile "
            "rainfall and rain chance for each date, so local variation remains visible. These are model estimates "
            "for sampled grid points, not district forecasts or official warnings."
        )
        rainfall_spread = week_summary.melt(
            id_vars="forecast_date", value_vars=["median_rainfall_mm", "p90_rainfall_mm"],
            var_name="sample_statistic", value_name="rainfall_mm",
        )
        st.plotly_chart(px.line(
            rainfall_spread, x="forecast_date", y="rainfall_mm", color="sample_statistic", markers=True,
            labels={"forecast_date": "Forecast Date", "rainfall_mm": "Rainfall (mm/day)",
                    "sample_statistic": "Across sampled locations"},
            title="Typical and Upper-End Forecast Rainfall"), width="stretch")
        week_display = week_summary.rename(columns={
            "forecast_date": "Date", "median_rainfall_mm": "Median rain (mm)",
            "p90_rainfall_mm": "90th percentile rain (mm)", "peak_grid_rainfall_mm": "Maximum (mm)",
            "median_rain_probability_pct": "Median rain chance (%)",
            "p90_rain_probability_pct": "90th percentile chance (%)",
            "cells_with_measurable_rain": "Points with measurable rain",
            "cells_with_50pct_rain_chance": "Points at 50%+ chance", "rainy_cells_percent": "Rainy points (%)",
        })
        show_dataframe(week_display, width="stretch", hide_index=True)
        days = sorted(active_forecast.forecast_date.unique())
        selected_day = st.selectbox("Choose Forecast Day", days,
                                    format_func=lambda day: pd.Timestamp(day).strftime("%A, %d %b %Y"))
        day_frame = active_forecast[active_forecast.forecast_date == selected_day].copy()
        day_frame = day_frame.dropna(subset=["rainfall_mm"])
        if day_frame.empty:
            st.warning("The selected forecast date has no valid rainfall amounts.")
            st.stop()
        day_frame["rainfall_category"] = day_frame.rainfall_mm.map(imd_daily_rain_category)
        day_frame = add_nearest_district(day_frame)
        day_frame = apply_resolved_place_names(day_frame)
        wet_cells = day_frame[day_frame.rainfall_mm >= 0.1]
        probability_cells = day_frame.dropna(subset=["precipitation_probability_pct"])
        rain_chance_threshold = st.slider(
            "Show locations with at least this rain chance",
            min_value=1, max_value=100, value=20, step=5, format="%d%%",
            help="Filters the sampled grid points for the selected forecast date.",
        )
        likely_cells = probability_cells[
            probability_cells.precipitation_probability_pct >= rain_chance_threshold
        ].sort_values(
            ["precipitation_probability_pct", "rainfall_mm"], ascending=False
        )
        district_parts = likely_cells.nearest_district.str.rsplit(", ", n=1, expand=True)
        likely_cells["district_name"] = district_parts[0]
        likely_cells["state_name"] = district_parts[1].fillna("Unknown state")
        likely_cells = likely_cells.sort_values(
            ["state_name", "district_name"], key=lambda values: values.str.casefold()
        )
        peak = day_frame.loc[day_frame.rainfall_mm.idxmax()]
        a, b, c, d = st.columns(4)
        a.metric("Median at Sampled Cells", f"{day_frame.rainfall_mm.median():.1f} mm")
        b.metric("Points with Measurable Rain", f"{len(wet_cells)}/{len(day_frame)} ({len(wet_cells) / len(day_frame):.0%})")
        c.metric("90th Percentile Rain", f"{day_frame.rainfall_mm.quantile(.9):.1f} mm")
        d.metric("Points at 50%+ Rain Chance", f"{int((probability_cells.precipitation_probability_pct >= 50).sum())}/{len(probability_cells)}")
        moderate_count = int((day_frame.rainfall_mm >= 15.6).sum())
        st.info(
            f"Plain-language outlook for {pd.Timestamp(selected_day):%A, %d %b}: "
            f"{len(wet_cells)} of {len(day_frame)} sampled points may get measurable rain. "
            f"The median is {day_frame.rainfall_mm.median():.1f} mm; the upper 10% of points are at or above "
            f"{day_frame.rainfall_mm.quantile(.9):.1f} mm. {moderate_count} points are forecast at 15.6 mm or more. "
            f"The highest sampled amount is {peak.rainfall_mm:.1f} mm near "
            f"{peak.grid_latitude:.2f}°N, {peak.grid_longitude:.2f}°E. "
            "Use this as a broad model outlook, not as a local guarantee."
        )
        st.caption(f"Highest amount is at {peak.grid_latitude:.2f}°N, {peak.grid_longitude:.2f}°E. "
                   "Rainfall amount bands follow standard daily thresholds; model calendar-day totals do not use the same 24-hour observation window. These are not official warnings.")
        st.plotly_chart(px.scatter_geo(
            day_frame, lat="grid_latitude", lon="grid_longitude", color="precipitation_probability_pct",
            hover_name="nearest_district" if "nearest_district" in day_frame else "rainfall_category",
        hover_data={"rainfall_mm": ":.1f", "precipitation_probability_pct": ":.0f",
                    "nearest_district": True, "district_centroid_distance_km": ":.1f",
                        "grid_latitude": ":.2f", "grid_longitude": ":.2f"},
            labels={"precipitation_probability_pct": "Rain chance (%)", "rainfall_mm": "Forecast rainfall (mm)",
                    "grid_latitude": "Grid Latitude", "grid_longitude": "Grid Longitude"},
            scope="asia", color_continuous_scale="Blues", range_color=(0, 100),
            title=f"Rain Chance by Sampled Grid Location - {selected_day}"), width="stretch")
        st.markdown(f"##### Sampled Locations with at Least {rain_chance_threshold}% Rain Chance")
        if likely_cells.empty:
            st.info("No sampled locations meet this rain-chance threshold, or probability data is unavailable for this date.")
        else:
            st.caption("District names use the nearest district coordinate supplied by the Open Admin Data API. Distance is from the grid point to that district coordinate; coordinates are retained for context.")
            st.caption(
                f"{len(likely_cells)} of {len(probability_cells)} sampled locations meet the threshold. "
                "Names show the nearest district with known coordinates; the grid point may be elsewhere in that district or nearby."
            )
            show_dataframe(likely_cells[[
                "state_name", "district_name", "district_centroid_distance_km", "grid_latitude", "grid_longitude", "precipitation_probability_pct",
                "rainfall_mm", "rainfall_category",
            ]].rename(columns={
                "state_name": "State", "district_name": "District",
                "grid_latitude": "Latitude", "grid_longitude": "Longitude",
                "district_centroid_distance_km": "Centroid distance (km)",
                "precipitation_probability_pct": "Rain chance (%)",
                "rainfall_mm": "Forecast rainfall (mm)", "rainfall_category": "Rainfall band",
            }), width="stretch", hide_index=True)
        with st.expander("How to read the rainfall amount bands"):
            st.markdown("Very light: trace–2.4 mm; light: 2.5–15.5 mm; moderate: 15.6–64.4 mm; heavy: 64.5–115.5 mm; very heavy: 115.6–204.4 mm; extremely heavy: ≥204.5 mm.")
    st.markdown("##### Official District Outlooks and Warnings")
    if settings.imd_auth_header and settings.imd_auth_value:
        if st.button("Refresh Official Records", key="refresh_official_imd"):
            with st.spinner("Fetching official products..."):
                try:
                    succeeded = run_api(["district_forecast", "state_rainfall", "district_rainfall", "current_weather", "district_warnings"])
                    read_query.clear()
                    st.success(f"Official data endpoints succeeded: {succeeded}/5")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Official data refresh failed: {project_text(exc)}")
    ingestion = load_table("""
        SELECT endpoint, ingested_at, status_code, records_received, error_message
        FROM ingestion_runs ORDER BY ingested_at DESC LIMIT 20
    """, "Ingestion status")
    if not ingestion.empty:
        st.markdown("##### Latest API Ingestion Attempts")
        show_dataframe(ingestion, width="stretch", hide_index=True)
    forecasts = load_table("""
        SELECT s.state_name, d.district_name, f.forecast_date, f.forecast_day,
               f.rainfall_distribution, f.distribution_percentage, f.data_source, f.created_at
        FROM rainfall_forecasts f JOIN districts d USING (district_id)
        JOIN states s USING (state_id) ORDER BY f.forecast_date, s.state_name, d.district_name
    """, "Official forecasts")
    warnings = load_table("""
        SELECT s.state_name, d.district_name, w.warning_date, w.forecast_day,
               w.warning_level, w.warning_description, w.data_source
        FROM district_warnings w JOIN districts d USING (district_id)
        JOIN states s USING (state_id) ORDER BY w.warning_date, s.state_name, d.district_name
    """, "Official warnings")
    st.markdown("##### Rainfall Distribution Forecast (Categorical; No Millimetre Conversion)")
    if forecasts.empty:
        pass
    else:
        show_dataframe(forecasts, width="stretch", hide_index=True)
    st.markdown("##### District Warning Levels")
    if warnings.empty:
        pass
    else:
        show_dataframe(warnings, width="stretch", hide_index=True)
    if forecasts.empty and warnings.empty:
        st.info("Official district forecasts and warnings have not been loaded. Configure approved access details in your local environment, then refresh official records. Grid-based weather estimates are not official district warnings.")

elif page == "Live weather":
    import html
    import plotly.graph_objects as go

    from src.analytics.weather_summary import weatherapi_next_hours_frame, weatherapi_nowcast_narrative
    from src.dashboard.radar import radar_map_html


    st.subheader("Weather Now and the Days Ahead")
    st.caption("Search and select a place to see current conditions, hourly rain chances, and the available daily outlook. Charts show forecast estimates, not official warnings.")
    weatherapi_key = getattr(settings, "weatherapi_key", "").strip()
    location_query = st.text_input("City or Place", value="New Delhi", key="weatherapi_location_query")
    stored_matches = st.session_state.get("weatherapi_locations", [])
    if st.session_state.get("weatherapi_search_query") != location_query:
        stored_matches = []
    selected_place = None

    def open_meteo_fallback(query):
        from src.ingestion.open_meteo_location import OpenMeteoLocationClient
        return OpenMeteoLocationClient().fetch(query, days=7)

    if weatherapi_key:
        st.caption("Current conditions and a short-range outlook are loaded when available. If the primary service cannot be reached, this page uses a clearly labeled model estimate.")
        if st.button("Search Locations"):
            try:
                from src.ingestion.weatherapi_client import WeatherAPIClient
                with st.spinner("Searching for locations..."):
                    stored_matches = WeatherAPIClient().search_locations(location_query)
                st.session_state["weatherapi_locations"] = stored_matches
                st.session_state["weatherapi_search_query"] = location_query
                if not stored_matches:
                    st.info("No matching places found. Try a more specific city or postcode.")
            except Exception as exc:
                try:
                    with st.spinner("Current conditions are unavailable; loading a model estimate..."):
                        payload, raw_path = open_meteo_fallback(location_query)
                    st.session_state["weatherapi_payload"] = payload
                    st.session_state["weatherapi_query"] = location_query
                    st.session_state["weatherapi_provider"] = "Model-Based Estimate"
                    st.session_state["weatherapi_raw_path"] = str(raw_path.relative_to(ROOT))
                    st.session_state["weatherapi_fetched_at"] = pd.Timestamp.now(tz="UTC")
                    st.warning(f"Current conditions are unavailable ({project_text(exc)}). Showing a model estimate instead.")
                except Exception as fallback_error:
                    st.error(f"Could not load current conditions or the model estimate ({project_text(fallback_error)}). Check internet access and try again.")
    else:
        st.caption("A model-based current estimate and up to 7 days of forecast are available without a key. This is not a station observation or an official warning.")
        if st.button("Load current estimate and 7-day outlook", type="primary"):
            try:
                with st.spinner("Finding the place and loading its forecast..."):
                    payload, raw_path = open_meteo_fallback(location_query)
                st.session_state["weatherapi_payload"] = payload
                st.session_state["weatherapi_query"] = location_query
                st.session_state["weatherapi_provider"] = "Model-Based Estimate"
                st.session_state["weatherapi_raw_path"] = str(raw_path.relative_to(ROOT))
                st.session_state["weatherapi_fetched_at"] = pd.Timestamp.now(tz="UTC")
            except Exception as exc:
                st.error(f"Could not load the location forecast: {project_text(exc)}")

    if stored_matches:
        def place_label(place):
            return ", ".join(part for part in [place.get("name"), place.get("region"), place.get("country")] if part)
        selected_place = st.selectbox("Choose the matching location", stored_matches, format_func=place_label, key="weatherapi_selected_place")
        if st.button("Load current conditions and 3-day forecast", type="primary"):
            try:
                from src.ingestion.weatherapi_client import WeatherAPIClient
                with st.spinner("Loading current conditions and forecast..."):
                    selected_location = f"{float(selected_place['lat'])},{float(selected_place['lon'])}"
                    payload, raw_path = WeatherAPIClient().fetch_location_weather(selected_location, days=3)
                st.session_state["weatherapi_payload"] = payload
                st.session_state["weatherapi_query"] = location_query
                st.session_state["weatherapi_provider"] = "Live Weather Service"
                st.session_state["weatherapi_raw_path"] = str(raw_path.relative_to(ROOT))
                st.session_state["weatherapi_fetched_at"] = pd.Timestamp.now(tz="UTC")
            except Exception as exc:
                try:
                    with st.spinner("Current conditions are unavailable; loading a model estimate..."):
                        payload, raw_path = open_meteo_fallback(location_query)
                    st.session_state["weatherapi_payload"] = payload
                    st.session_state["weatherapi_query"] = location_query
                    st.session_state["weatherapi_provider"] = "Model-Based Estimate"
                    st.session_state["weatherapi_raw_path"] = str(raw_path.relative_to(ROOT))
                    st.session_state["weatherapi_fetched_at"] = pd.Timestamp.now(tz="UTC")
                    st.warning(f"Current conditions are unavailable ({project_text(exc)}). Showing a model estimate instead.")
                except Exception as fallback_error:
                    st.error(f"Could not load current conditions or the model estimate ({project_text(fallback_error)}).")

    payload = st.session_state.get("weatherapi_payload")
    if payload and st.session_state.get("weatherapi_query") == location_query:
        provider = st.session_state.get("weatherapi_provider", "Model-Based Estimate")
        place, current = payload.get("location", {}), payload.get("current", {})
        next_hours = weatherapi_next_hours_frame(payload, count=3)
        daily = weatherapi_daily_frame(payload)
        if daily.empty:
            st.warning("The selected place has current conditions, but its forecast was empty.")
        else:
            def daily_label(day):
                row = daily[daily.date == day].iloc[0]
                name = "Today" if pd.Timestamp(day).date() == pd.Timestamp.now().date() else pd.Timestamp(day).strftime("%A")
                return (f"{name} | {str(row.condition).strip().capitalize()} | "
                        f"High {weather_number(row.temperature_high_c, ' C', 0)} / Low {weather_number(row.temperature_low_c, ' C', 0)} | "
                        f"Rain {row.rain_chance_pct:.0f}%")

            left, main = st.columns([0.29, 0.71], gap="medium")
            with left:
                st.markdown("### Daily Forecast")
                st.caption(f"Available Forecast Days: {len(daily)} | Source: {provider}")
                selected_date = st.radio(
                    "Choose a Day", daily.date.tolist(), format_func=daily_label,
                    key="weatherapi_day_selected", label_visibility="collapsed",
                )
                with st.expander("Rain Alert", expanded=False):
                    alert_enabled = st.toggle("Alert Me in This Dashboard", value=False, help="This is an in-page alert. It does not send phone notifications.")
                    threshold = st.slider("Rain Chance Threshold", 10, 100, 60, 5, format="%d%%", disabled=not alert_enabled)
                    if alert_enabled:
                        if not next_hours.empty and next_hours.rain_chance_pct.fillna(0).max() >= threshold:
                            alert_hour = next_hours.loc[next_hours.rain_chance_pct.fillna(0).idxmax()]
                            st.warning(f"Rain may arrive around {pd.Timestamp(alert_hour.time):%I %p} ({int(alert_hour.rain_chance_pct)}% chance).")
                        else:
                            st.success("No upcoming hour reaches this rain-chance threshold.")
                with st.expander("Recent Radar", expanded=False):
                    latitude, longitude = place.get("lat"), place.get("lon")
                    if latitude is not None and longitude is not None:
                        st.caption("Playback shows recent observed radar echoes, not future rainfall. Coverage varies by area.")
                        st.iframe(radar_map_html(latitude, longitude), height=510)
                        st.markdown("Radar: [RainViewer](https://www.rainviewer.com/api.html). Base map: OpenStreetMap.")
                    else:
                        st.info("Location coordinates were not returned, so radar cannot be centered here.")
            with main:
                location_name = html.escape(str(place.get("name", location_query)))
                region = html.escape(str(place.get("region", "")))
                country = html.escape(str(place.get("country", "")))
                condition = html.escape(str(current.get("condition", {}).get("text", "Conditions unavailable")).strip().capitalize())
                summary = html.escape(weatherapi_nowcast_narrative(payload, next_hours))
                temp = html.escape(weather_number(current.get("temp_c"), " C", 0))
                feels = html.escape(weather_number(current.get("feelslike_c"), " C", 0))
                st.markdown(
                    f"<div class='weather-hero'><div class='hero-place'>{location_name} | {region} | {country}</div>"
                    f"<div class='hero-temp'>{temp}</div>"
                    f"<div class='hero-condition'>{condition} | Feels like {feels}</div>"
                    f"<div class='hero-summary'>{summary}</div></div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"Current conditions from {provider} | Last updated locally: {current.get('last_updated', 'Time unavailable')}")

                st.markdown("#### Hourly Forecast")
                selected_hours = weatherapi_hourly_frame(payload, selected_date)
                if pd.Timestamp(selected_date).date() == pd.Timestamp.now().date() and not selected_hours.empty:
                    current_time = pd.to_datetime(current.get("last_updated"), errors="coerce")
                    if not pd.isna(current_time):
                        future_hours = selected_hours[selected_hours.time > current_time]
                        if not future_hours.empty:
                            selected_hours = future_hours
                visible_hours = selected_hours.head(12)
                if not visible_hours.empty:
                    cards = []
                    for _, hour in visible_hours.iterrows():
                        time_label = pd.Timestamp(hour.time).strftime("%I %p").lstrip("0")
                        chance = "-" if pd.isna(hour.rain_chance_pct) else f"{float(hour.rain_chance_pct):.0f}% rain"
                        amount = "-" if pd.isna(hour.precipitation_mm) else f"{float(hour.precipitation_mm):.1f} mm"
                        cards.append(
                            f"<div class='hour-card'><div class='hour-time'>{html.escape(time_label)}</div>"
                            f"<div class='hour-condition'>{html.escape(str(hour.condition).strip().capitalize())}</div>"
                            f"<div class='hour-temp'>{html.escape(weather_number(hour.temperature_c, ' C', 0))}</div>"
                            f"<div class='hour-rain'>{html.escape(chance)}<br>{html.escape(amount)}</div></div>"
                        )
                    st.markdown("<div class='hourly-scroll'>" + "".join(cards) + "</div>", unsafe_allow_html=True)
                    with st.expander("Interactive Hourly Rain Chart"):
                        hourly_chart = go.Figure()
                        colors = ["#94a3b8" if pd.isna(v) or v < 0.1 else "#66bb6a" if v < 2.5 else "#facc15" if v < 7.6 else "#f97316" if v < 15 else "#dc2626" for v in selected_hours.precipitation_mm]
                        hourly_chart.add_bar(x=selected_hours.time, y=selected_hours.precipitation_mm, name="Rain (mm/hour)", marker_color=colors)
                        hourly_chart.add_scatter(x=selected_hours.time, y=selected_hours.rain_chance_pct, name="Rain chance (%)", mode="lines+markers", yaxis="y2", line={"color": "#38bdf8", "width": 3})
                        hourly_chart.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis={"rangeslider": {"visible": True}}, yaxis={"title": "Rain (mm/hour)", "rangemode": "tozero"}, yaxis2={"title": "Chance (%)", "overlaying": "y", "side": "right", "range": [0, 100]}, legend={"orientation": "h"})
                        st.plotly_chart(hourly_chart, width="stretch", config={"displayModeBar": False})
                else:
                    st.info("No hourly forecast is available for this day.")

                st.markdown("#### Current Conditions")
                wind_direction = current.get("wind_degree")
                try:
                    compass = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][round(float(wind_direction) / 45) % 8]
                except (TypeError, ValueError):
                    compass = ""
                values = [
                    ("Wind", weather_number(current.get("wind_kph"), " km/h", 0), compass),
                    ("Humidity", weather_number(current.get("humidity"), "%", 0), f"Dew point {weather_number(current.get('dewpoint_c'), ' C', 0)}"),
                    ("Pressure", weather_number(current.get("pressure_mb"), " hPa", 0), "At current location"),
                    ("Rain Right Now", weather_number(current.get("precip_mm"), " mm"), f"UV Index {weather_number(current.get('uv'), '', 0)}"),
                ]
                condition_cols = st.columns(4)
                for column, (label, value, detail) in zip(condition_cols, values):
                    column.markdown(
                        f"<div class='condition-card'><div class='condition-label'>{html.escape(label)}</div>"
                        f"<div class='condition-value'>{html.escape(value)}</div>"
                        f"<div class='condition-detail'>{html.escape(detail)}</div></div>",
                        unsafe_allow_html=True,
                    )

                if len(daily) > 1:
                    st.markdown("#### Rain Outlook")
                    peak = daily.loc[daily.forecast_rain_mm.idxmax()]
                    st.caption(f"Wettest day: {pd.Timestamp(peak.date):%A, %d %b} | {peak.forecast_rain_mm:.1f} mm | {peak.rain_chance_pct:.0f}% chance")
                    with st.expander("Interactive Daily Rainfall Chart", expanded=False):
                        outlook = go.Figure()
                        outlook.add_bar(x=daily.date, y=daily.forecast_rain_mm, name="Rain (mm/day)", marker_color="#4f9bd8")
                        outlook.add_scatter(x=daily.date, y=daily.rain_chance_pct, name="Rain chance (%)", mode="lines+markers", yaxis="y2", line={"color": "#fbbf24", "width": 3})
                        outlook.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis={"rangeslider": {"visible": True}}, yaxis={"title": "Rain (mm/day)", "rangemode": "tozero"}, yaxis2={"title": "Chance (%)", "overlaying": "y", "side": "right", "range": [0, 100]}, legend={"orientation": "h"})
                        st.plotly_chart(outlook, width="stretch", config={"displayModeBar": False})
                st.caption(f"Source: {provider}. Rain labels are informational, not official warnings. Fetched {st.session_state.get('weatherapi_fetched_at', 'time unavailable')}.")
    elif payload:
        st.info("The place changed. Search or load weather again for this location.")
    else:
        st.info("Search a city to load current conditions and the local forecast.")

elif page == "Historical grid":
    st.subheader("Historical Gridded Rainfall")
    st.caption("This page explores observed daily rainfall at grid points across the loaded years. Each map point is a grid cell, not a district or weather station; historical totals are not a live forecast.")
    if grid.empty:
        st.info("Import an official rainfall NetCDF file with `python -m src.ingestion.run_pipeline --netcdf path-to-file.nc`.")
    else:
        st.write(f"Loaded records: {int(grid.observation_count.sum()):,}; period: {grid.first_date.min()} to {grid.last_date.max()}.")
        grid_daily["observation_date"] = pd.to_datetime(grid_daily.observation_date)
        grid_daily["month"] = grid_daily.observation_date.dt.month
        grid_daily["year"] = grid_daily.observation_date.dt.year
        monthly = grid_daily.groupby("month", as_index=False).rainfall_mm.mean()
        st.plotly_chart(px.line(monthly, x="month", y="rainfall_mm", labels={"month": "Month", "rainfall_mm": "Rainfall (mm)"}, title="Mean Grid-Cell Rainfall by Month (Loaded Grid Cells)"), width="stretch")
        st.plotly_chart(px.line(grid_daily, x="observation_date", y="rainfall_mm", labels={"observation_date": "Date", "rainfall_mm": "Rainfall (mm)"}, title="Mean Rainfall Across Loaded Grid Observations"), width="stretch")
        yearly = grid_daily.groupby("year", as_index=False).agg(
            mean_daily_rain_mm=("rainfall_mm", "mean"),
            peak_grid_average_mm=("rainfall_mm", "max"),
            wet_days=("rainfall_mm", lambda values: int((values >= 0.1).sum())),
        )
        st.markdown("##### Year-by-Year Summary for the Same Sampled Cells")
        st.plotly_chart(px.bar(yearly, x="year", y="mean_daily_rain_mm",
                               labels={"year": "Year", "mean_daily_rain_mm": "Mean Daily Rainfall (mm/day)"},
                               title="Average Daily Rainfall Across the Sample by Year (mm/day)"), width="stretch")
        show_dataframe(yearly, width="stretch", hide_index=True)
        map_frame = grid
        st.plotly_chart(px.scatter_geo(map_frame, lat="grid_latitude", lon="grid_longitude", color="rainfall_mm",
                                       hover_data={"grid_latitude": ":.2f", "grid_longitude": ":.2f"},
                                       labels={"grid_latitude": "Grid Latitude", "grid_longitude": "Grid Longitude", "rainfall_mm": "Rainfall (mm)"},
                                       scope="asia", title="Mean Rainfall by Grid Cell"), width="stretch")

else:
    st.subheader("Next-Day Rainfall Model")
    st.caption("This page reports how the machine-learning estimate performed on a later-in-time holdout set. MAE and RMSE are average error measures in millimetres; compare them with the previous-day baseline. Predictions are experimental, not official forecasts.")
    metrics_path = ROOT / "models" / "rainfall_predictive_project_metrics.json"
    if not metrics_path.exists():
        metrics_path = ROOT / "models" / "latest_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        device_display = str(metrics.get("training_device", "cpu")).upper()
        st.caption(f"Model: {metrics.get('model_version', 'Legacy')} | Training Device: {device_display} | "
                   f"Holdout: {metrics.get('test_target_start')} to {metrics.get('test_target_end')}. "
                   "The estimate is an ML output, not an official forecast.")
        baseline = metrics.get("baseline_previous_day", {})
        model_stats = metrics.get("selected_model_metrics", metrics.get("random_forest", {}))
        skill = metrics.get("skill_vs_previous_day_percent", {})
        selected_model_name = metrics.get("selected_model", "random_forest")
        model_display = {"random_forest": "Random Forest", "xgboost": "XGBoost"}.get(
            str(selected_model_name).lower(), str(selected_model_name).replace("_", " ").title()
        )
        a, b, c, d = st.columns(4)
        a.metric(f"{model_display} Average Error (MAE)", f"{model_stats.get('mae_mm', float('nan')):.2f} mm",
                 delta=f"{skill.get('mae', 0):.1f}% vs previous-day baseline")
        b.metric("Model Large-Error Score (RMSE)", f"{model_stats.get('rmse_mm', float('nan')):.2f} mm",
                 delta=f"{skill.get('rmse', 0):.1f}% vs baseline")
        c.metric("R-squared on Holdout", f"{model_stats.get('r2', float('nan')):.3f}",
                 help="1 is perfect; 0 matches a constant mean prediction; below 0 is worse than that mean baseline.")
        d.metric("Previous-Day Baseline MAE", f"{baseline.get('mae_mm', float('nan')):.2f} mm")
        if skill.get("mae", 0) > 0 and skill.get("rmse", 0) > 0:
            st.success("On the chronological holdout, the model improved both average and large-error scores over using yesterday's rainfall.")
        else:
            st.warning("The model did not beat the previous-day baseline on both scores. Treat it as experimental and do not rely on it for decisions.")
        profile = metrics.get("test_rainfall_profile", {})
        if profile:
            st.caption(f"Holdout sample: {profile.get('wet_grid_day_percent', 0):.1f}% of grid-cell days had at least 0.1 mm; "
                       f"{profile.get('heavy_grid_days_at_least_64_5mm', 0):,} grid-cell days had at least 64.5 mm.")
        importance = pd.DataFrame(metrics.get("top_feature_importance", []))
        if not importance.empty:
            importance["feature"] = importance.feature.map(display_column_name)
            st.plotly_chart(px.bar(importance.sort_values("importance"), x="importance", y="feature",
                                   labels={"importance": "Feature Importance", "feature": "Feature"},
                                   orientation="h", title=f"Features Most Used by {model_display}"), width="stretch")
        model_note = str(metrics.get("model_note", "Compare all model scores with a simple baseline.")).strip()
        model_note = project_text(model_note)
        st.caption(model_note[:1].upper() + model_note[1:])
        comparison_rows = []
        for label, stats in [("Previous-day baseline", baseline),
                             ("Random Forest (CPU)", metrics.get("random_forest", {})),
                             ("XGBoost", metrics.get("xgboost", {}))]:
            if stats:
                comparison_rows.append({"Model": label, **stats})
        if comparison_rows:
            st.markdown("##### Holdout Model Comparison (Lower MAE/RMSE Is Better)")
            show_dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)
    else:
        st.info("No trained model metrics are present. Train only with verified observations: `python -m src.ml.train path-to-observations.csv`.")
    predictions = load_table("""
        SELECT grid_latitude, grid_longitude, prediction_date, predicted_rainfall_mm,
               model_version, created_at
        FROM grid_ml_predictions ORDER BY prediction_date DESC
    """, "ML estimates")
    if not predictions.empty:
        latest_date = pd.to_datetime(predictions.prediction_date).max()
        latest = predictions[pd.to_datetime(predictions.prediction_date) == latest_date]
        st.markdown(f"#### Saved ML Estimates for {latest_date.date()} (mm)")
        st.plotly_chart(px.scatter_geo(latest, lat="grid_latitude", lon="grid_longitude",
                                       color="predicted_rainfall_mm", scope="asia",
                                       labels={"grid_latitude": "Grid Latitude", "grid_longitude": "Grid Longitude", "predicted_rainfall_mm": "Predicted Rainfall (mm)"},
                                       title="Predicted Rainfall by Grid Cell"), width="stretch")
        show_dataframe(latest, width="stretch", hide_index=True)
    else:
        st.info("No grid-cell predictions are saved. The available observations end on 2025-12-31, so a next-day estimate from this file would now be historical rather than current.")

