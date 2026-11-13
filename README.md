
# Rainfall Predictive Project

A data science application for rainfall history, current conditions, forecast
guidance, data quality, and next-day machine-learning estimates. The dashboard
identifies observed data, forecast products, and model estimates separately.

## Setup

The project uses a local Python virtual environment and PostgreSQL database.
Install dependencies, create your local environment file, and initialize the
database:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m src.database.init_db
```

Set the database credentials and any approved service credentials in `.env`.
Keep that file private and do not commit it.

## Load Rainfall Data

Import a downloaded gridded NetCDF rainfall file:

```powershell
.\.venv\Scripts\python.exe -m src.ingestion.run_pipeline --netcdf .\data\raw\rainfall_2025.nc
```

If the file contains more than one data variable, pass its rainfall variable
with `--variable`. Imported grid records are stored as grid observations and
are not assigned to districts without a verified geographic mapping.

## Train and Predict

Export verified historical observations, then train and evaluate a next-day
model using a chronological holdout:

```powershell
.\.venv\Scripts\python.exe -m src.ml.export_training_data --start-date 2015-01-01 --end-date 2025-12-31 --max-cells 200 --out .\data\processed\rainfall_predictive_project_training.csv
.\.venv\Scripts\python.exe -m src.ml.train .\data\processed\rainfall_predictive_project_training.csv
```

The evaluation compares model estimates with a previous-day baseline using
MAE, RMSE, R-squared, and feature importance. Predictions are estimates, not
official warnings. The default prediction output is
`data/processed/rainfall_predictive_project_predictions.csv`.

## Dashboard and Reports

Start the dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/dashboard/app.py
```

The six dashboard pages cover overview, rainfall analytics, forecasts and
warnings, live weather, historical grid data, and model performance. Tables
and charts use readable labels and explain the limits of the available data.

Generate standalone interactive figures:

```powershell
.\.venv\Scripts\python.exe -m src.analytics.generate_report
```

Report files are written under `reports/figures/` with the
`rainfall_predictive_project_` filename prefix.

## Free Hosting Preparation

The app is prepared for Streamlit Community Cloud. It reads hosted database
credentials from the deployment's private secrets settings and can create the
project schema on first startup. Create a PostgreSQL database, set
`DATABASE_URL` and `AUTO_CREATE_SCHEMA` from
`.streamlit/secrets.toml.example` in the host's secrets panel, then import
approved rainfall data into that database. Do not commit `.env`, secrets,
database exports, or raw data files.
=======
# Rainfall_Predictive_Analysis

