from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import text

from ..config import ROOT
from ..database.session import make_engine
from .analysis import rainfall_figures


def main() -> None:
    engine = make_engine()
    query = text("""
        SELECT s.state_name, d.district_name, r.observation_date,
               r.actual_rainfall_mm, r.normal_rainfall_mm,
               r.departure_percentage, r.rainfall_category
        FROM rainfall_observations r
        JOIN districts d USING (district_id)
        JOIN states s USING (state_id)
        WHERE r.actual_rainfall_mm IS NOT NULL
    """)
    frame = pd.read_sql(query, engine)
    output = ROOT / "reports" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    for name, figure in rainfall_figures(frame).items():
        figure.write_html(output / f"rainfall_predictive_project_{name}.html", include_plotlyjs="cdn")
    if frame.empty:
        print("No district rainfall observations are loaded; no figures generated.")
    else:
        print(f"Wrote {len(rainfall_figures(frame))} interactive figures to {output}")


if __name__ == "__main__":
    main()
