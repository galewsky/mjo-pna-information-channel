#!/usr/bin/env python
"""Season-conditioned kNN surprisal for CLARA-A3 daily OLR anomalies."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union

import numpy as np
import pandas as pd
import zarr
from numba import njit, prange
from tqdm import tqdm
try:  # zarr >= 3
    from zarr.codecs import BloscCodec, BloscShuffle
    _COMPRESSOR_KWARGS = {
        "compressors": (BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.bitshuffle),),
    }
except ImportError:  # older zarr -> fall back to numcodecs
    from numcodecs import Blosc

    _COMPRESSOR_KWARGS = {
        "compressor": Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE),
    }

# Keep threaded libraries from oversubscribing cores before Numba takes over
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

try:
    import s3fs  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    s3fs = None

SEASON_MONTHS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="olr_tropics_anomalies.zarr", help="path to the anomaly Zarr store")
    parser.add_argument("--output", default="olr_tropics_surprisal_knn.zarr", help="target Zarr store for surprisal")
    parser.add_argument("--k", type=int, default=5, help="number of neighbors for the 1-D kNN PDF estimator")
    parser.add_argument("--tile-lat", type=int, default=30, help="latitude size of processing tiles")
    parser.add_argument("--tile-lon", type=int, default=60, help="longitude size of processing tiles")
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=list(SEASON_MONTHS.keys()),
        help="ordered list of seasons to process (case-insensitive)",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace the output store if it exists")
    parser.add_argument("--max-time", type=int, default=None, help="optional limit on number of time steps (debug)")
    parser.add_argument("--s3-endpoint", default=None, help="optional S3-compatible endpoint URL")
    return parser.parse_args()


@njit(parallel=True, cache=True)
def _surprisal_knn_1d_vec(tile_vals: np.ndarray, k: int) -> np.ndarray:
    T, P = tile_vals.shape
    out = np.full((T, P), np.nan, dtype=np.float64)

    log2 = 0.6931471805599453
    logk = np.log(float(k))

    for p in prange(P):
        v = tile_vals[:, p]
        cnt = 0
        for t in range(T):
            if not np.isnan(v[t]):
                cnt += 1
        if cnt <= k:
            continue

        w = np.empty(cnt, dtype=np.float64)
        valid_times = np.empty(cnt, dtype=np.int64)
        c = 0
        for t in range(T):
            if not np.isnan(v[t]):
                w[c] = v[t]
                valid_times[c] = t
                c += 1

        sort_idx = np.argsort(w)
        w_sorted = w[sort_idx]
        N = cnt
        rk = np.empty(N, dtype=np.float64)
        tiny = np.finfo(np.float64).tiny
        for idx in range(N):
            left = tiny
            right = tiny
            if idx >= k:
                left = w_sorted[idx] - w_sorted[idx - k]
            if idx + k < N:
                right = w_sorted[idx + k] - w_sorted[idx]
            has_left = idx >= k
            has_right = idx + k < N
            if has_left and has_right:
                radius = 0.5 * (left if left > right else right)
            elif has_left:
                radius = 0.5 * left
            elif has_right:
                radius = 0.5 * right
            else:
                radius = tiny
            if radius <= tiny:
                radius = tiny
            rk[idx] = radius

        S_sorted = (np.log(2.0 * N * rk) - logk) / log2
        for r in range(N):
            original_idx = sort_idx[r]
            t = valid_times[original_idx]
            out[t, p] = S_sorted[r]
    return out


def _season_indices(times: np.ndarray, season_order: Iterable[str]) -> Dict[str, np.ndarray]:
    stamp = pd.to_datetime(times)
    months = stamp.month.values
    indices: Dict[str, np.ndarray] = {}
    for name in season_order:
        if name not in SEASON_MONTHS:
            raise ValueError(f"Unknown season {name!r}; valid keys: {sorted(SEASON_MONTHS)}")
        mask = np.isin(months, SEASON_MONTHS[name])
        indices[name] = np.nonzero(mask)[0]
    return indices


def _attrs_to_dict(attrs) -> Dict:
    if hasattr(attrs, "asdict"):
        return attrs.asdict()
    return dict(attrs)


def _copy_array(dst_root: zarr.Group, src_group: zarr.Group, name: str) -> zarr.Array | None:
    if name not in src_group:
        return None
    arr = src_group[name]
    data = arr[:]
    chunks = arr.chunks
    creator = getattr(dst_root, "create_array", None) or getattr(dst_root, "create_dataset")
    array = creator(
        name,
        data=data,
        chunks=chunks,
    )
    array.attrs.update(_attrs_to_dict(arr.attrs))
    return array


def _init_output_store(
    input_group: zarr.Group,
    output_store: Union[str, "s3fs.S3Map"],
    anom_arr: zarr.Array,
    args: argparse.Namespace,
) -> Tuple[zarr.Group, zarr.Array]:
    root = zarr.open_group(output_store, mode="w")

    # Copy coordinates and static fields for convenience
    for coord in ("time", "lat", "lon"):
        _copy_array(root, input_group, coord)
    for aux in ("valid_mask", "missing_fraction"):
        _copy_array(root, input_group, aux)

    creator = getattr(root, "create_array", None) or getattr(root, "create_dataset")
    out = creator(
        "olr_surprisal",
        shape=anom_arr.shape,
        chunks=anom_arr.chunks,
        dtype="float32",
        **_COMPRESSOR_KWARGS,
        fill_value=np.nan,
    )
    dims = _attrs_to_dict(anom_arr.attrs).get("_ARRAY_DIMENSIONS", ["time", "lat", "lon"])
    out.attrs.update({"_ARRAY_DIMENSIONS": dims})
    out.attrs.update(
        {
            "long_name": "Daily OLR surprisal (seasonal kNN)",
            "units": "bits",
            "k": args.k,
            "method": "symmetric 1-D kNN density estimate",
            "description": "Season-conditioned surprisal derived from daily OLR anomalies",
        }
    )

    return root, out


def _decode_time_axis(time_arr: zarr.Array, max_steps: int | None) -> np.ndarray:
    raw = time_arr[:max_steps]
    attrs = time_arr.attrs.asdict()
    units = (attrs.get("units") or "").lower()
    if units.startswith("days since"):
        base_str = units.split("since", 1)[1].strip()
        base = np.datetime64(base_str)
        offsets = raw.astype("timedelta64[D]")
        decoded = base + offsets
        return decoded.astype("datetime64[ns]")
    return pd.to_datetime(raw)


def _season_tile_loop(
    season: str,
    season_idx: np.ndarray,
    anom_arr: zarr.Array,
    out_arr: zarr.Array,
    valid_mask: np.ndarray,
    tile_lat: int,
    tile_lon: int,
    k: int,
    max_steps: int,
):
    n_time = season_idx.size
    if n_time <= k:
        print(f"  Skipping {season}: only {n_time} samples (<= k={k})")
        return

    nlat, nlon = valid_mask.shape
    lat_ranges = list(range(0, nlat, tile_lat))
    lon_ranges = list(range(0, nlon, tile_lon))
    total_tiles = len(lat_ranges) * len(lon_ranges)

    print(f"  Processing {season}: {n_time} samples across {total_tiles} tiles")
    idx_list = season_idx.tolist()

    with tqdm(total=total_tiles, desc=f"{season} tiles", unit="tile") as pbar:
        for lat0 in lat_ranges:
            lat1 = min(lat0 + tile_lat, nlat)
            lat_slice = slice(lat0, lat1)
            mask_lat = valid_mask[lat_slice, :]
            for lon0 in lon_ranges:
                lon1 = min(lon0 + tile_lon, nlon)
                lon_slice = slice(lon0, lon1)
                tile_mask = mask_lat[:, lon0:lon1]
                pbar.update(1)
                if not np.any(tile_mask):
                    continue

                full_tile = anom_arr[0:max_steps, lat_slice, lon_slice]
                tile = np.asarray(full_tile[idx_list, ...], dtype=np.float64)
                tlen = tile.shape[0]
                flat = tile.reshape(tlen, -1)
                mask_flat = tile_mask.reshape(-1)
                if not np.any(mask_flat):
                    continue
                invalid = ~mask_flat
                if np.any(invalid):
                    flat[:, invalid] = np.nan

                surprisal = _surprisal_knn_1d_vec(flat, k)
                surprisal = surprisal.reshape(tile.shape).astype(np.float32)
                out_arr.oindex[idx_list, lat_slice, lon_slice] = surprisal

                del full_tile, tile, flat, surprisal


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


def _prepare_input_store(path: str, endpoint: str | None) -> Tuple[Union[str, "s3fs.S3Map"], str]:
    if _is_s3_path(path):
        mapper, fs, key = _as_s3_mapper(path, endpoint)
        try:
            fs.ls(key)
        except FileNotFoundError:
            raise SystemExit(f"Input store not found: {path}")
        return mapper, path
    input_path = Path(path)
    if not input_path.exists():
        raise SystemExit(f"Input store not found: {input_path}")
    return str(input_path), str(input_path)


def _prepare_output_store(path: str, endpoint: str | None, overwrite: bool) -> Tuple[Union[str, "s3fs.S3Map"], str]:
    if _is_s3_path(path):
        mapper, fs, key = _as_s3_mapper(path, endpoint)
        exists = True
        try:
            fs.ls(key)
        except FileNotFoundError:
            exists = False
        if exists:
            if not overwrite:
                raise SystemExit(f"Output store {path} exists; pass --overwrite to replace it")
            print(f"Removing existing store {path}")
            fs.rm(key, recursive=True)
        return mapper, path
    output_path = Path(path)
    if output_path.exists():
        if not overwrite:
            raise SystemExit(f"Output store {output_path} exists; pass --overwrite to replace it")
        print(f"Removing existing store {output_path}")
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path), str(output_path)


def main() -> int:
    args = _parse_args()
    if args.k < 1:
        raise SystemExit("k must be >= 1")
    if args.tile_lat <= 0 or args.tile_lon <= 0:
        raise SystemExit("tile dimensions must be positive")

    season_order = [name.upper() for name in args.seasons]
    endpoint = getattr(args, "s3_endpoint", None)
    input_store, input_label = _prepare_input_store(args.input, endpoint)
    output_store, output_label = _prepare_output_store(args.output, endpoint, args.overwrite)

    print(f"Opening anomaly store {input_label}")
    input_group = zarr.open_group(input_store, mode="r")
    anom_arr = input_group["olr_anomaly"]
    valid_mask_arr = input_group["valid_mask"]
    time_arr = input_group["time"]

    total_time = anom_arr.shape[0]
    max_steps = total_time
    if args.max_time is not None and args.max_time < total_time:
        max_steps = args.max_time
        print(f"  Debug: restricting to first {max_steps} time steps")

    times = _decode_time_axis(time_arr, max_steps)
    base_indices = np.arange(max_steps, dtype=np.int64)
    season_idx = {
        name: base_indices[idx]
        for name, idx in _season_indices(times, season_order).items()
    }

    valid_mask = valid_mask_arr[:].astype(bool)

    root, out_arr = _init_output_store(input_group, output_store, anom_arr, args)

    history_line = (
        f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} seasonal kNN surprisal (k={args.k}) "
        f"from {input_label}"
    )
    root.attrs.update(
        {
            "history": history_line,
            "source": input_label,
            "notes": (
                "Surprisal computed season-by-season using 1-D kNN PDF estimates in anomaly space; "
                "values are in bits and share the same grid/time axes as the input anomalies."
            ),
            "seasons": {name: SEASON_MONTHS[name] for name in season_order},
        }
    )

    for season, idx in season_idx.items():
        if idx.size == 0:
            print(f"Skipping {season}: no samples")
            continue
        _season_tile_loop(
            season,
            idx,
            anom_arr,
            out_arr,
            valid_mask,
            args.tile_lat,
            args.tile_lon,
            args.k,
            max_steps,
        )

    print("Consolidating metadata")
    zarr.consolidate_metadata(output_store)
    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
