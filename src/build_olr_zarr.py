#!/usr/bin/env python3
"""Subset the CM SAF CLARA-A3 OLR archive to the deep tropics and write Zarr."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List

import xarray as xr
from dask.diagnostics import ProgressBar
from numcodecs import Blosc
import zarr

DEFAULT_LAT_MIN = -30.0
DEFAULT_LAT_MAX = 30.0
DEFAULT_PATTERN = "OLRdm*.nc"
DEFAULT_OUTPUT = "olr_tropics.zarr"
DEFAULT_CHUNKS = {"time": 90, "lat": 60, "lon": 360}
DEFAULT_BATCH_SIZE = 180  # process roughly six months per write


def _lat_slice(ds: xr.Dataset, lat_min: float, lat_max: float):
    lat = ds["lat"].values
    if lat[0] <= lat[-1]:
        return slice(lat_min, lat_max)
    return slice(lat_max, lat_min)


def _subset_factory(lat_min: float, lat_max: float) -> Callable[[xr.Dataset], xr.Dataset]:
    def _subset(ds: xr.Dataset) -> xr.Dataset:
        return ds.sel(lat=_lat_slice(ds, lat_min, lat_max))

    return _subset


def _collect_encodings(example: Path, compressor: Blosc) -> Dict[str, Dict]:
    """Preserve packed encodings (scale/offset/fill) without forcing other dtypes."""

    with xr.open_dataset(example, decode_cf=False) as tmpl:
        encodings: Dict[str, Dict] = {}
        for name, var in tmpl.data_vars.items():
            enc: Dict[str, object] = {"compressor": compressor}
            fill_value = var.encoding.get("_FillValue", var.attrs.get("_FillValue"))
            if fill_value is not None:
                enc["_FillValue"] = fill_value
            scale_factor = var.encoding.get("scale_factor", var.attrs.get("scale_factor"))
            if scale_factor is not None:
                enc["scale_factor"] = scale_factor
            add_offset = var.encoding.get("add_offset", var.attrs.get("add_offset"))
            if add_offset is not None:
                enc["add_offset"] = add_offset
            if ("scale_factor" in enc or "add_offset" in enc) and "dtype" in var.encoding:
                enc["dtype"] = var.encoding["dtype"]
            encodings[name] = enc
    return encodings


def _batch(sequence: List[str], batch_size: int) -> Iterator[List[str]]:
    for idx in range(0, len(sequence), batch_size):
        yield sequence[idx : idx + batch_size]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="glob that matches the NetCDF inputs")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="path to the target Zarr store")
    parser.add_argument("--lat-min", type=float, default=DEFAULT_LAT_MIN, help="southern latitude bound (degrees)")
    parser.add_argument("--lat-max", type=float, default=DEFAULT_LAT_MAX, help="northern latitude bound (degrees)")
    parser.add_argument("--time-chunk", type=int, default=DEFAULT_CHUNKS["time"], help="chunk length along time")
    parser.add_argument("--lat-chunk", type=int, default=DEFAULT_CHUNKS["lat"], help="chunk length along latitude")
    parser.add_argument("--lon-chunk", type=int, default=DEFAULT_CHUNKS["lon"], help="chunk length along longitude")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="number of daily files to load at once")
    parser.add_argument(
        "--keep-vars",
        nargs="+",
        default=None,
        help="optional list of data variables to keep (defaults to everything)",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace any existing Zarr store")
    return parser.parse_args()


def _validate_keep_vars(dataset_vars: Iterable[str], keep_vars: Iterable[str]) -> List[str]:
    if keep_vars is None:
        return list(dataset_vars)
    missing = sorted(set(keep_vars) - set(dataset_vars))
    if missing:
        raise ValueError(f"Requested variables not found: {missing}")
    return list(keep_vars)


def main() -> int:
    args = _parse_args()
    workdir = Path.cwd()
    files = sorted(workdir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched pattern {args.pattern!r} in {workdir}")

    chunks = {"time": args.time_chunk, "lat": args.lat_chunk, "lon": args.lon_chunk}
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    encoding_template = _collect_encodings(files[0], compressor)

    target = Path(args.output)
    if target.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing store {target}. Pass --overwrite if intentional.")
    if target.exists() and args.overwrite:
        import shutil

        shutil.rmtree(target)

    subsetter = _subset_factory(args.lat_min, args.lat_max)
    file_groups = list(_batch([str(f) for f in files], args.batch_size))
    if not file_groups:
        raise SystemExit("No file groups were produced; check --batch-size")

    keep_list: List[str] | None = None
    total_batches = len(file_groups)
    for batch_index, group in enumerate(file_groups):
        mode = "w" if batch_index == 0 else "a"
        append_dim = None if batch_index == 0 else "time"

        first_name = Path(group[0]).name
        last_name = Path(group[-1]).name
        print(
            f"[{batch_index + 1}/{total_batches}] processing {len(group)} files "
            f"({first_name} -> {last_name})",
            flush=True,
        )

        ds = xr.open_mfdataset(
            group,
            combine="nested",
            concat_dim="time",
            parallel=False,
            chunks=chunks,
            engine="netcdf4",
            preprocess=subsetter,
            coords="minimal",
            compat="override",
            data_vars="minimal",
        )

        if keep_list is None:
            keep_list = _validate_keep_vars(ds.data_vars, args.keep_vars)
        ds = ds[keep_list]
        ds = ds.sortby("time").chunk(chunks)

        for coord in ("time", "time_bnds"):
            if coord in ds:
                ds[coord].encoding = {}

        if batch_index == 0:
            history_note = (
                f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} subset to lat [{args.lat_min}, {args.lat_max}] and "
                f"chunked as {chunks}"
            )
            ds.attrs["history"] = f"{history_note}\n" + ds.attrs.get("history", "")

        encoding = None
        if mode == "w":
            encoding = {name: encoding_template.get(name, {}) for name in ds.data_vars}

        with ProgressBar():
            ds.to_zarr(
                target,
                mode=mode,
                append_dim=append_dim,
                consolidated=False,
                compute=True,
                zarr_format=2,
                encoding=encoding,
            )

        ds.close()

    print("Consolidating metadata", flush=True)
    zarr.consolidate_metadata(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
