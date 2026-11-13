from __future__ import annotations

import logging
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sqlalchemy.dialects.postgresql import insert

from ..database.models import RainfallGridObservation
from ..database.session import create_schema, make_engine

logger = logging.getLogger(__name__)


def _coord(ds: xr.Dataset, candidates: tuple[str, ...], axis: str) -> str:
    for name in ds.coords:
        coord = ds[name]
        if name.lower() in candidates or str(coord.attrs.get("axis", "")).upper() == axis:
            return name
    raise ValueError(f"Cannot locate the {axis} coordinate in {list(ds.coords)}")


def import_netcdf(path: str | Path, variable: str | None = None, batch_size: int = 20_000,
                  max_cells: int | None = None) -> int:
    """Load an official IMD gridded NetCDF file, optionally sampling cells consistently."""
    if max_cells is not None and max_cells < 1:
        raise ValueError("max_cells must be at least 1")
    path = Path(path)
    with xr.open_dataset(path) as ds:
        time_name = _coord(ds, ("time", "date", "datetime"), "T")
        lat_name = _coord(ds, ("lat", "latitude", "y"), "Y")
        lon_name = _coord(ds, ("lon", "longitude", "x"), "X")
        if variable is None:
            eligible = [name for name, data in ds.data_vars.items() if {time_name, lat_name, lon_name}.issubset(data.dims)]
            if len(eligible) != 1:
                raise ValueError(f"Select --variable; gridded candidates are {eligible}")
            variable = eligible[0]
        if variable not in ds.data_vars:
            raise ValueError(f"Variable {variable!r} not found; available: {list(ds.data_vars)}")
        da = ds[variable].transpose(time_name, lat_name, lon_name).load()
        times = pd.to_datetime(ds[time_name].values).date
        latitudes = np.asarray(ds[lat_name].values, dtype=float)
        longitudes = np.asarray(ds[lon_name].values, dtype=float)
        values = np.asarray(da.values, dtype=float)

    selected_cells: set[tuple[int, int]] | None = None
    if max_cells is not None:
        valid_counts = np.isfinite(values).sum(axis=0)
        candidates = [(int(yi), int(xi)) for yi, xi in np.argwhere(valid_counts > 0)]
        candidates.sort(key=lambda pair: hashlib.md5(
            f"{latitudes[pair[0]]:.6f},{longitudes[pair[1]]:.6f}".encode("ascii")
        ).hexdigest())
        selected_cells = set(candidates[:max_cells])

    source = f"IMD 0.25-degree gridded rainfall NetCDF: {path.name}:{variable}"
    engine = make_engine()
    create_schema(engine)
    statement = insert(RainfallGridObservation).on_conflict_do_nothing(
        constraint="uq_grid_cell_date_source"
    ).returning(RainfallGridObservation.grid_observation_id)
    inserted = 0
    batch = []
    with engine.begin() as connection:
        for ti, day in enumerate(times):
            plane = values[ti]
            valid = np.isfinite(plane) & (plane >= 0)
            for yi, xi in np.argwhere(valid):
                if selected_cells is not None and (int(yi), int(xi)) not in selected_cells:
                    continue
                batch.append({
                    "grid_latitude": float(latitudes[yi]), "grid_longitude": float(longitudes[xi]),
                    "observation_date": day, "rainfall_mm": float(plane[yi, xi]), "data_source": source,
                })
                if len(batch) >= batch_size:
                    result = connection.execute(statement, batch)
                    inserted += len(result.scalars().all())
                    batch.clear()
        if batch:
            result = connection.execute(statement, batch)
            inserted += len(result.scalars().all())
    logger.info("Imported %d rainfall grid observations from %s", inserted, path)
    return inserted
