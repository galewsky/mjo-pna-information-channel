#!/usr/bin/env python3
"""
Extract pointwise surprisal time series from surprisal Zarr.

Reads olr_surprisal (bits) and writes a NetCDF for the requested latitude band.
"""

from __future__ import annotations

import argparse
import os

import dask.array as da
from dask.diagnostics import ProgressBar
import numpy as np
import s3fs
import xarray as xr
import zarr


def decode_time_manual(z_time_arr) -> np.ndarray:
    """Decode Zarr time array using units attribute if present."""
    vals = z_time_arr[:]
    attrs = z_time_arr.attrs.asdict()
    units = attrs.get("units", "")
    if "since" in units:
        try:
            parts = units.split(" since ")
            unit_str = parts[0].strip().lower()
            start_date_str = parts[1].strip()
            unit = "D" if "day" in unit_str else "s"
            start_date = np.datetime64(start_date_str)
            deltas = vals.astype("timedelta64[" + unit + "]")
            return (start_date + deltas).astype("datetime64[ns]")
        except Exception:
            pass
    try:
        return np.array(vals, dtype="datetime64[ns]")
    except Exception:
        return vals


def build_s3_mapper(s3_path: str, endpoint: str | None) -> s3fs.S3Map:
    key = os.environ.get("B2_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("B2_APP_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not key or not secret:
        raise SystemExit(
            "Missing credentials. Set B2_KEY_ID/B2_APP_KEY or "
            "AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY."
        )
    fs = s3fs.S3FileSystem(
        key=key,
        secret=secret,
        client_kwargs={"endpoint_url": endpoint} if endpoint else None,
    )
    return fs.get_mapper(s3_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="S3 path to surprisal zarr")
    parser.add_argument("--endpoint", default=None, help="S3 endpoint URL")
    parser.add_argument("--output", required=True, help="Output NetCDF path")
    parser.add_argument("--lat-min", type=float, default=-20.0, help="Minimum latitude")
    parser.add_argument("--lat-max", type=float, default=20.0, help="Maximum latitude")
    parser.add_argument("--chunks", default="auto", help="Dask chunks for time")
    args = parser.parse_args()

    print("Opening surprisal Zarr...", flush=True)
    mapper = build_s3_mapper(args.input, args.endpoint)
    z = zarr.open_group(mapper, mode="r")

    if "olr_surprisal" not in z:
        raise SystemExit(f"olr_surprisal not found. Vars: {list(z.array_keys())}")

    surp = da.from_zarr(z["olr_surprisal"])
    lat = z["lat"][:]
    lon = z["lon"][:]
    time_vals = decode_time_manual(z["time"])

    print("Building dataset...", flush=True)
    ds = xr.Dataset(
        data_vars={"olr_surprisal": (("time", "lat", "lon"), surp)},
        coords={"time": time_vals, "lat": lat, "lon": lon},
    )

    print(f"Selecting lat band {args.lat_min}..{args.lat_max}...", flush=True)
    if lat[0] < lat[-1]:
        ds_trop = ds.sel(lat=slice(args.lat_min, args.lat_max))
    else:
        ds_trop = ds.sel(lat=slice(args.lat_max, args.lat_min))

    surp_trop = ds_trop["olr_surprisal"]
    if args.chunks != "auto":
        surp_trop = surp_trop.chunk({"time": int(args.chunks)})

    out = surp_trop.to_dataset(name="olr_surprisal")
    encoding = {
        "olr_surprisal": {"zlib": True, "complevel": 4, "dtype": "float32"}
    }
    print(f"Writing NetCDF to {args.output}...", flush=True)
    with ProgressBar():
        out.to_netcdf(args.output, encoding=encoding)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
