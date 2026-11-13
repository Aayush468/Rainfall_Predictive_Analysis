import pandas as pd

from src.ingestion.transform import (transform_district_forecast,
                                    transform_district_rainfall,
                                    transform_district_warnings,
                                    transform_state_rainfall,
                                    transform_current_weather)


def test_district_rainfall_parses_official_date_and_values():
    frame = transform_district_rainfall([{
        "OBJ_ID": "164", "District": "ADILABAD", "Date": "2023-01-31",
        "Daily Actual": "0.00", "Daily Normal": "1.70",
        "Daily Departure Per": "-100%", "Daily Category": "NR",
    }])
    assert frame.iloc[0].observation_date == pd.Timestamp("2023-01-31").date()
    assert frame.iloc[0].actual_rainfall_mm == 0
    assert frame.iloc[0].departure_percentage == -100


def test_state_rainfall_parses_documented_dd_mm_yyyy():
    frame = transform_state_rainfall([{
        "State": "DELHI", "Date": "31-05-2022", "Daily Actual": "0.00",
        "Daily Normal": "3.30", "Daily Departure Per": "-100%", "Daily Category": "NR",
    }])
    assert frame.iloc[0].observation_date == pd.Timestamp("2022-05-31").date()


def test_categorical_forecast_is_not_converted_to_mm():
    frame = transform_district_forecast([{
        "date_obs": "2026-06-06", "Obj_id": "573", "District": "NICOBAR",
        "State": "ANDAMAN AND NICOBAR", "day1_distribution": "Widespread",
        "day1_distribution_percentage": "Stations [76-100]%",
    }])
    assert frame.iloc[0].forecast_date == pd.Timestamp("2026-06-07").date()
    assert frame.iloc[0].rainfall_distribution == "Widespread"
    assert frame.iloc[0].distribution_percentage == "Stations [76-100]%"


def test_warning_retains_official_level_text():
    frame = transform_district_warnings([{
        "Obj_id": "573", "Date": "2026-06-06", "District": "NICOBAR",
        "Day_1": "Heavy Rain", "Day1_Color": "4",
    }])
    assert frame.iloc[0].warning_level == "4"
    assert frame.iloc[0].warning_description == "Heavy Rain"


def test_current_weather_timezone_is_parsed_as_documented_utc():
    frame = transform_current_weather([{
        "Station Id": "NDL", "Date of Observation": "2024-06-06",
        "Time of Observation": "07:00", "Temperature": "40.8",
        "Humidity": "20", "Wind Speed": "1", "Wind Direction": "189",
        "Last 24 hrs Rainfall": "0.0",
    }])
    assert frame.iloc[0].observation_datetime.isoformat() == "2024-06-06T07:00:00+00:00"
    assert frame.iloc[0].rainfall_mm == 0
