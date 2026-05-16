#!/usr/bin/env python
"""Preprocess CLARA-A3 daily OLR into anomalies ready for surprisal analysis."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import shutil
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
try:
    from zarr.codecs import BloscCodec, BloscShuffle

    _COMPRESSOR_KWARGS = {
        "compressors": (
            BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.bitshuffle),
        ),
    }
except ImportError:  # zarr < 3 fallback
    from numcodecs import Blosc

    _COMPRESSOR_KWARGS = {
        "compressor": Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE),
    }

import zarr
try:
    import s3fs  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    s3fs = None

DEFAULT_INPUT = "olr_tropics.zarr"
DEFAULT_OUTPUT = "olr_tropics_anomalies.zarr"
DEFAULT_VAR = "LW_flux"
DEFAULT_TIME_START = "1979-01-01"
DEFAULT_TIME_END = "2020-12-31"
DEFAULT_CHUNKS = (90, 60, 360)
DEFAULT_MAX_MISSING = 0.05
DEFAULT_SMOOTHING = 0  # in days; 0 disables smoothing


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="path to source OLR Zarr store")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="target Zarr store for anomalies")
    parser.add_argument("--variable", default=DEFAULT_VAR, help="name of the OLR variable to use")
    parser.add_argument("--start-date", default=DEFAULT_TIME_START, help="first date (inclusive) for the daily axis")
    parser.add_argument("--end-date", default=DEFAULT_TIME_END, help="last date (inclusive) for the daily axis")
    parser.add_argument("--max-missing-fraction", type=float, default=DEFAULT_MAX_MISSING, help="maximum missing-data fraction allowed for a grid cell")
    parser.add_argument("--drop-leap-day", dest="drop_leap_day", action="store_true", help="drop 29 Feb from the time axis before climatology (default)")
    parser.add_argument("--keep-leap-day", dest="drop_leap_day", action="store_false", help="keep 29 Feb as its own day")
    parser.add_argument("--seasonal-smoothing-days", type=int, default=DEFAULT_SMOOTHING, help="odd window length (days) for smoothing the daily climatology (0 disables)")
    parser.add_argument("--time-chunk", type=int, default=DEFAULT_CHUNKS[0], help="chunk size along time")
    parser.add_argument("--lat-chunk", type=int, default=DEFAULT_CHUNKS[1], help="chunk size along latitude")
    parser.add_argument("--lon-chunk", type=int, default=DEFAULT_CHUNKS[2], help="chunk size along longitude")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output store if present")
    parser.add_argument("--s3-endpoint", default=None, help="optional S3-compatible endpoint URL (e.g., Backblaze)")
    parser.set_defaults(drop_leap_day=True)
    return parser.parse_args()


def _is_s3_path(path: str) -> bool:
    return path.startswith("s3://")


def _require_s3fs() -> None:
    if s3fs is None:  # pragma: no cover - informative guard
        raise SystemExit("s3fs is required for s3:// paths; please install it first.")


def _as_s3_mapper(path: str, endpoint: str | None) -> Tuple["s3fs.S3Map", "s3fs.S3FileSystem", str]:
    _require_s3fs()
    key = path[len("s3://") :]
    client_kwargs = {"endpoint_url": endpoint} if endpoint else None
    fs = s3fs.S3FileSystem(client_kwargs=client_kwargs) if client_kwargs else s3fs.S3FileSystem()
    mapper = fs.get_mapper(key)
    return mapper, fs, key


def _prepare_input_store(path: str, endpoint: str | None) -> Tuple[Union[Path, "s3fs.S3Map"], str]:
    if _is_s3_path(path):
        mapper, fs, key = _as_s3_mapper(path, endpoint)
        if not fs.exists(key):
            raise SystemExit(f"Input store not found: {path}")
        return mapper, path
    input_path = Path(path)
    if not input_path.exists():
        raise SystemExit(f"Input store not found: {input_path}")
    return input_path, str(input_path)


def _prepare_output_store(path: str, endpoint: str | None, overwrite: bool) -> Tuple[Union[Path, "s3fs.S3Map"], str]:
    if _is_s3_path(path):
        mapper, fs, key = _as_s3_mapper(path, endpoint)
        if fs.exists(key):
            if not overwrite:
                raise SystemExit(
                    f"Output store {path} exists; pass --overwrite to replace it"
                )
            print(f"Removing existing store {path}")
            fs.rm(key, recursive=True)
        return mapper, path
    output_path = Path(path)
    if output_path.exists():
        if not overwrite:
            raise SystemExit(
                f"Output store {output_path} exists; pass --overwrite to replace it"
            )
        print(f"Removing existing store {output_path}")
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path, str(output_path)


def _target_time_index(start: str, end: str) -> pd.DatetimeIndex:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts < start_ts:
        raise ValueError("end date precedes start date")
    return pd.date_range(start_ts, end_ts, freq="D")


def _drop_leap_days(da: xr.DataArray) -> xr.DataArray:
    is_leap = (da.time.dt.month == 2) & (da.time.dt.day == 29)
    removed = int(is_leap.sum())
    if removed:
        print(f"    Dropping {removed} leap days to enforce a 365-day climatology")
        return da.sel(time=~is_leap)
    return da


def _ensure_units(da: xr.DataArray) -> None:
    units = (da.attrs.get("units") or "").strip().replace("^", "")
    valid = {"W m-2", "W m^-2", "W/m^2"}
    if units not in valid:
        raise ValueError(f"Unexpected units for {da.name}: {units!r}; expected W m^-2")


def _compute_mask(da: xr.DataArray, max_missing: float) -> tuple[xr.DataArray, xr.DataArray]:
    if not (0.0 <= max_missing < 1.0):
        raise ValueError("max_missing_fraction must be in [0, 1)")
    valid = da.notnull()
    valid_fraction = valid.sum("time") / da.sizes["time"]
    missing_fraction = 1.0 - valid_fraction
    mask = missing_fraction <= max_missing
    mask.name = "valid_mask"
    missing_fraction.name = "missing_fraction"
    missing_fraction.attrs["long_name"] = "fraction of missing days (after reindex)"
    return mask, missing_fraction


def _time_in_years(time: xr.DataArray) -> xr.DataArray:
    base = time.astype("datetime64[ns]")[0]
    dt = (time.astype("datetime64[ns]") - base).astype("timedelta64[ns]")
    years = dt / np.timedelta64(1, "D") / 365.2425
    out = xr.DataArray(years, dims=("time",), coords={"time": time}, name="time_years")
    return out


def _no_leap_dayofyear(time: xr.DataArray) -> xr.DataArray:
    doy = time.dt.dayofyear
    is_leap = time.dt.is_leap_year
    after_feb = time.dt.month > 2
    doy_noleap = xr.where(is_leap & after_feb, doy - 1, doy)
    return doy_noleap.astype("int16")


def _linear_detrend(da: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    print("  Fitting and removing linear trends")
    time_years = _time_in_years(da.time)
    coeffs = da.polyfit(dim="time", deg=1, skipna=True)
    trend = xr.polyval(time_years, coeffs.polyfit_coefficients)
    detrended = da - trend
    slope = coeffs.polyfit_coefficients.sel(degree=1, drop=True)
    intercept = coeffs.polyfit_coefficients.sel(degree=0, drop=True)
    slope.name = "linear_trend_slope"
    intercept.name = "linear_trend_intercept"
    slope.attrs.update({
        "long_name": "OLS slope of daily OLR",
        "units": f"{da.attrs.get('units', 'W m^-2')} per year",
    })
    intercept.attrs.update({
        "long_name": "OLS intercept of daily OLR",
        "units": da.attrs.get("units", "W m^-2"),
    })
    return detrended, slope, intercept


def _smooth_dayofyear(climatology: xr.DataArray, window: int) -> xr.DataArray:
    if window <= 1:
        return climatology
    if window % 2 == 0:
        raise ValueError("seasonal smoothing window must be odd")
    half = window // 2
    pad_left = climatology.isel(dayofyear=slice(-half, None))
    pad_right = climatology.isel(dayofyear=slice(0, half))
    padded = xr.concat([pad_left, climatology, pad_right], dim="dayofyear")
    smoothed = (
        padded.rolling(dayofyear=window, center=True, min_periods=window)
        .mean()
        .isel(dayofyear=slice(half, padded.sizes["dayofyear"] - half))
    )
    smoothed = smoothed.assign_coords(dayofyear=climatology.dayofyear)
    smoothed.attrs.update(climatology.attrs)
    return smoothed


def _daily_climatology(detrended: xr.DataArray, window: int, labels: xr.DataArray) -> xr.DataArray:
    print("  Computing daily climatology of detrended OLR")
    if labels.dims != ("time",):
        raise ValueError("Climatology labels must be a 1D array over time")
    clim = detrended.groupby(labels).mean("time", skipna=True)
    label_name = labels.name or "group"
    clim = clim.rename({label_name: "dayofyear"})
    clim.name = "seasonal_cycle"
    clim.attrs.update({
        "long_name": "Mean seasonal cycle of detrended daily OLR",
        "units": detrended.attrs.get("units", "W m^-2"),
    })
    return _smooth_dayofyear(clim, window)


def _build_output_dataset(
    anomalies: xr.DataArray,
    climatology: xr.DataArray,
    slope: xr.DataArray,
    intercept: xr.DataArray,
    mask: xr.DataArray,
    missing_fraction: xr.DataArray,
) -> xr.Dataset:
    anomalies.name = "olr_anomaly"
    anomalies.attrs.update({
        "long_name": "Daily OLR anomaly (detrended, seasonal cycle removed)",
        "units": climatology.attrs.get("units", "W m^-2"),
    })
    ds = xr.Dataset(
        data_vars={
            "olr_anomaly": anomalies,
            "seasonal_cycle": climatology,
            "linear_trend_slope": slope,
            "linear_trend_intercept": intercept,
            "valid_mask": mask,
            "missing_fraction": missing_fraction,
        }
    )
    return ds


def _encoding(chunks: Dict[str, int], day_chunks: int) -> Dict[str, Dict]:
    return {
        "olr_anomaly": {
            **_COMPRESSOR_KWARGS,
            "dtype": "float32",
            "chunks": (chunks["time"], chunks["lat"], chunks["lon"]),
        },
        "seasonal_cycle": {
            **_COMPRESSOR_KWARGS,
            "dtype": "float32",
            "chunks": (day_chunks, chunks["lat"], chunks["lon"]),
        },
        "linear_trend_slope": {
            **_COMPRESSOR_KWARGS,
            "dtype": "float32",
            "chunks": (chunks["lat"], chunks["lon"]),
        },
        "linear_trend_intercept": {
            **_COMPRESSOR_KWARGS,
            "dtype": "float32",
            "chunks": (chunks["lat"], chunks["lon"]),
        },
        "valid_mask": {
            **_COMPRESSOR_KWARGS,
            "dtype": "uint8",
            "chunks": (chunks["lat"], chunks["lon"]),
        },
        "missing_fraction": {
            **_COMPRESSOR_KWARGS,
            "dtype": "float32",
            "chunks": (chunks["lat"], chunks["lon"]),
        },
    }


def main() -> int:
    args = _parse_args()
    chunks = {"time": args.time_chunk, "lat": args.lat_chunk, "lon": args.lon_chunk}

    input_store, input_label = _prepare_input_store(args.input, args.s3_endpoint)
    output_store, output_label = _prepare_output_store(args.output, args.s3_endpoint, args.overwrite)

    print(f"Opening {input_label} and selecting {args.variable}")
    xr.set_options(keep_attrs=True)
    ds = xr.open_zarr(input_store, consolidated=True, chunks=chunks)
    if args.variable not in ds:
        raise SystemExit(f"Variable {args.variable!r} not found in {list(ds.data_vars)}")
    source_history = ds.attrs.get("history", "")
    source_time = pd.DatetimeIndex(ds["time"].values)
    olr = ds[args.variable].transpose("time", "lat", "lon")
    _ensure_units(olr)
    olr = olr.astype("float32").where(np.isfinite(olr))

    target_time = _target_time_index(args.start_date, args.end_date)
    missing_days = target_time.difference(source_time)
    del ds
    if missing_days.size:
        preview = ", ".join(str(ts.date()) for ts in missing_days[:5])
        extra = "" if missing_days.size <= 5 else ", ..."
        print(
            f"  Found {missing_days.size} missing days relative to target index (examples: {preview}{extra})"
        )
    else:
        print("  No missing days relative to target index")

    print("  Reindexing to continuous daily axis and inserting NaNs where necessary")
    olr = olr.reindex(time=target_time)

    if args.drop_leap_day:
        olr = _drop_leap_days(olr)

    dayofyear_vals = _no_leap_dayofyear(olr.time).compute().astype("int16")
    dayofyear_labels = xr.DataArray(
        dayofyear_vals,
        dims=("time",),
        coords={"time": olr.time},
        name="dayofyear_noleap",
    )
    olr = olr.assign_coords(dayofyear_noleap=dayofyear_labels)

    olr = olr.chunk(chunks)

    print("  Building static validity mask")
    mask, missing_fraction = _compute_mask(olr, args.max_missing_fraction)
    with ProgressBar():
        mask = mask.compute()
        missing_fraction = missing_fraction.compute()
    total_points = int(mask.sizes["lat"] * mask.sizes["lon"])
    valid_points = int(mask.sum())
    print(
        f"    Keeping {valid_points} / {total_points} grid cells ("
        f"{100.0 * valid_points / total_points:.2f}% valid)"
    )

    olr = olr.where(mask)

    detrended, slope, intercept = _linear_detrend(olr)

    climatology = _daily_climatology(detrended, args.seasonal_smoothing_days, dayofyear_labels)

    print("  Forming daily anomalies by removing the climatology")
    clim_for_anoms = climatology.rename(dayofyear="dayofyear_noleap")
    anomalies = detrended.groupby(dayofyear_labels) - clim_for_anoms
    anomalies = anomalies.chunk(chunks).astype("float32")
    day_chunk = min(chunks["time"], climatology.sizes["dayofyear"])
    climatology = climatology.chunk({"dayofyear": day_chunk, "lat": chunks["lat"], "lon": chunks["lon"]}).astype("float32")
    slope = slope.chunk({"lat": chunks["lat"], "lon": chunks["lon"]}).astype("float32")
    intercept = intercept.chunk({"lat": chunks["lat"], "lon": chunks["lon"]}).astype("float32")
    mask = mask.chunk({"lat": chunks["lat"], "lon": chunks["lon"]})
    missing_fraction = missing_fraction.astype("float32").chunk({"lat": chunks["lat"], "lon": chunks["lon"]})

    out = _build_output_dataset(anomalies, climatology, slope, intercept, mask, missing_fraction)

    history_line = (
        f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} detrended, removed seasonal cycle, "
        f"max_missing={args.max_missing_fraction}, drop_leap={args.drop_leap_day}"
    )
    out.attrs["history"] = history_line + "\n" + source_history
    out.attrs["source"] = input_label
    out.attrs["notes"] = (
        "OLR anomalies relative to 1979-2020 climatology with a fixed mask; "
        "suitable for surprisal/PDF estimation."
    )

    encoding = _encoding(chunks, day_chunks=day_chunk)

    print(f"Writing anomalies to {output_label}")
    with ProgressBar():
        out.to_zarr(output_store, mode="w", consolidated=False, encoding=encoding)

    print("Consolidating metadata")
    zarr.consolidate_metadata(output_store)

    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
