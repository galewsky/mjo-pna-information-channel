#!/usr/bin/env python3
"""
build_z500_phasebank_receiver.py

Construct phase-conditioned matched-filter templates for Z500 anomalies directly in
physical space. The driver is defined in surprisal space via phase labels and burst
intensity/masks, allowing weighted or binary gating. The output receiver NetCDF
contains the anomaly field, per-lag/per-phase templates, amplitudes, phase-matched
scores, and diagnostic metadata required for downstream scanning.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def decode_time_days_since_1979(time_da: xr.DataArray) -> pd.DatetimeIndex:
    values = time_da.values
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.to_datetime(values)
    base = np.datetime64("1979-01-01")
    days = values.astype("float64")
    if np.any(~np.isfinite(days)):
        raise ValueError("Non-finite values in time coordinate.")
    return pd.to_datetime(base + days.astype("timedelta64[D]"))


def drop_feb29(ds: xr.Dataset, time_name: str = "time") -> xr.Dataset:
    t = pd.to_datetime(ds[time_name].values)
    mask = ~((t.month == 2) & (t.day == 29))
    return ds.isel({time_name: np.where(mask)[0]})


def month_day_climatology(anom_src: xr.DataArray) -> xr.DataArray:
    clim = anom_src.groupby("time.dayofyear").mean("time")
    clim.attrs["dayofyear_encoding"] = "1..365 with Feb 29 dropped"
    return clim


def subtract_month_day_clim(da: xr.DataArray, clim_doy: xr.DataArray) -> xr.DataArray:
    return da.groupby("time.dayofyear") - clim_doy


def sqrt_coslat_weights(lat: xr.DataArray) -> xr.DataArray:
    lat_vals = lat.values.astype("float64")
    weights = np.sqrt(np.clip(np.cos(np.deg2rad(lat_vals)), a_min=0.0, a_max=None))
    return xr.DataArray(weights.astype("float32"), coords={lat.dims[0]: lat}, dims=lat.dims)


def parse_domain(domain_str: str) -> Tuple[float, float, float, float]:
    parts = [float(p.strip()) for p in domain_str.split(",")]
    if len(parts) != 4:
        raise ValueError("--domain must be 'lon0,lon1,lat0,lat1'")
    return parts[0], parts[1], parts[2], parts[3]


def parse_lags(expr: str) -> List[int]:
    if ":" in expr:
        parts = [p.strip() for p in expr.split(":")]
        if len(parts) != 3:
            raise ValueError("--lags must be 'a,b,c' or 'start:stop:step'")
        start, stop, step = [int(p) for p in parts]
        if step == 0:
            raise ValueError("Lag step cannot be zero.")
        if step > 0:
            return list(range(start, stop + 1, step))
        return list(range(start, stop - 1, step))
    return [int(p.strip()) for p in expr.split(",") if p.strip()]


def parse_window(expr: str) -> Tuple[str, str]:
    parts = [p.strip() for p in expr.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Window spec must be 'YYYY-MM-DD,YYYY-MM-DD'")
    return parts[0], parts[1]


def parse_chunks(expr: str) -> Optional[Dict[str, int]]:
    expr = expr.strip()
    if not expr:
        return None
    out: Dict[str, int] = {}
    for part in expr.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError("--chunks entries must be key=value")
        key, value = [p.strip() for p in part.split("=", 1)]
        out[key] = int(value)
    return out if out else None


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype("float32")
    series = pd.Series(values.astype("float64"))
    return series.rolling(window=window, center=True, min_periods=1).mean().values.astype("float32")


def build_lag_matrix_2d(series_mat: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    n_time, n_series = series_mat.shape
    out = np.full((n_time, len(lags), n_series), np.nan, dtype=series_mat.dtype)
    for j, lag in enumerate(lags):
        if lag > 0:
            out[lag:, j, :] = series_mat[:-lag, :]
        elif lag < 0:
            out[:lag, j, :] = series_mat[-lag:, :]
        else:
            out[:, j, :] = series_mat
    return out


def _flatten_space(z: xr.DataArray) -> Tuple[np.ndarray, np.ndarray]:
    lat = z["latitude"]
    lon = z["longitude"]
    z_np = np.asarray(z.values, dtype=np.float32)
    n_time, n_lat, n_lon = z_np.shape
    z_mat = z_np.reshape(n_time, -1)
    weights_lat = sqrt_coslat_weights(lat).values.astype(np.float32)
    weights_vec = np.repeat(weights_lat, n_lon)
    invalid = np.isnan(z_mat).any(axis=0)
    if np.any(invalid):
        z_mat[:, invalid] = 0.0
        weights_vec = weights_vec.copy()
        weights_vec[invalid] = 0.0
    return z_mat, weights_vec


def _datetime_mask(times: np.ndarray, start: str, end: str) -> np.ndarray:
    start_dt = np.datetime64(start)
    end_dt = np.datetime64(end)
    return (times >= start_dt) & (times <= end_dt)


def _compute_threshold(values: np.ndarray, kind: str, thresh: float, valid_mask: np.ndarray) -> float:
    valid_vals = values[valid_mask]
    if valid_vals.size == 0:
        raise ValueError("No finite values available to compute threshold.")
    if kind == "percentile":
        pct = float(thresh)
        if not (0.0 <= pct <= 100.0):
            raise ValueError("Percentile threshold must fall between 0 and 100.")
        return float(np.nanpercentile(valid_vals, pct))
    return float(thresh)


def _phase_mask_from_index(idx: np.ndarray, n_phase: int) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((idx.size, n_phase), dtype=np.uint8)
    valid = (idx >= 0) & (idx < n_phase)
    rows = np.where(valid)[0]
    cols = idx[valid]
    if rows.size:
        mask[rows, cols] = 1
    return mask, valid
def build_phase_mask_and_index(phase_vals: np.ndarray, n_phase: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase_vals = phase_vals.astype("float64")
    phase_idx = np.full(phase_vals.shape, -1, dtype=np.int16)
    valid = np.isfinite(phase_vals)
    idx_raw = np.rint(phase_vals[valid]).astype(np.int16) - 1  # ROMI phases are 1..n
    good = (idx_raw >= 0) & (idx_raw < n_phase)
    idx_full = np.full(idx_raw.shape, -1, dtype=np.int16)
    idx_full[good] = idx_raw[good]
    phase_idx[valid] = idx_full

    mask = np.zeros((phase_vals.size, n_phase), dtype=np.uint8)
    rows = np.where(phase_idx >= 0)[0]
    if rows.size:
        cols = phase_idx[rows]
        mask[rows, cols] = 1

    phase_values = (np.arange(n_phase) + 1).astype(np.int16)
    return mask, phase_idx, phase_values


def build_driver_matrix(values: np.ndarray, phase_mask: np.ndarray) -> np.ndarray:
    vals = values.astype(np.float32)
    vals = np.where(np.isfinite(vals), vals, np.nan)
    mask_bool = phase_mask.astype(bool)
    driver = np.where(mask_bool, vals[:, None], np.nan).astype(np.float32)
    return driver


def fit_phase_templates(
    z_mat: np.ndarray,
    weights: np.ndarray,
    driver_x: np.ndarray,
    lags: Sequence[int],
    train_mask: np.ndarray,
    min_train_n: int,
    min_on_train: int,
    min_off_train: int,
    strict: bool,
    binary_driver: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_lag = build_lag_matrix_2d(driver_x, lags).astype(np.float32)
    train_mask = train_mask.astype(bool)
    x_train = x_lag[train_mask, :, :]
    valid = np.isfinite(x_train)
    counts = valid.sum(axis=0)

    n_on = np.full_like(counts, np.nan, dtype=np.float32)
    n_off = np.full_like(counts, np.nan, dtype=np.float32)
    if binary_driver:
        on_mask = (x_train > 0.5) & valid
        off_mask = (x_train <= 0.5) & valid
        n_on = on_mask.sum(axis=0).astype(np.float32)
        n_off = off_mask.sum(axis=0).astype(np.float32)

    bad = counts < min_train_n
    if binary_driver:
        if min_on_train > 0:
            bad |= n_on < float(min_on_train)
        if min_off_train > 0:
            bad |= n_off < float(min_off_train)

    if strict and np.any(bad):
        lag_idx, phase_idx = np.argwhere(bad)[0]
        raise RuntimeError(
            f"Insufficient samples for lag={lags[int(lag_idx)]}, phase={phase_idx}: "
            f"n={counts[lag_idx, phase_idx]}, n_on={n_on[lag_idx, phase_idx]}, n_off={n_off[lag_idx, phase_idx]}"
        )
    elif np.any(bad):
        bad_pairs = [f"(lag={lags[i]},phase={j})" for i, j in zip(*np.where(bad))]
        print("WARNING: Filling templates with NaN due to insufficient samples for: " + ", ".join(bad_pairs))

    x_train_zeroed = np.where(valid, x_train, 0.0)
    counts_safe = np.maximum(counts, 1)
    means = x_train_zeroed.sum(axis=0) / counts_safe
    x_centered = (x_train_zeroed - means) * valid

    x_centered_2d = x_centered.reshape(x_centered.shape[0], -1)
    z_train = z_mat[train_mask, :]
    cov_num = x_centered_2d.T @ z_train
    var_den = (x_centered**2).sum(axis=0).reshape(-1)

    templates_flat = cov_num / np.maximum(var_den[:, None], 1e-9)
    bad_flat = bad.reshape(-1)
    templates_flat[bad_flat, :] = np.nan

    n_phase = driver_x.shape[1]
    templates = templates_flat.reshape(len(lags), n_phase, -1)

    z_weighted = z_mat * weights[None, :]
    templates_weighted = templates_flat * weights[None, :]
    amp_num = z_weighted @ templates_weighted.T
    amp_den = (templates_weighted**2).sum(axis=1)
    amp = amp_num / np.maximum(amp_den, 1e-9)
    amp[:, bad_flat] = np.nan
    amp = amp.reshape(z_mat.shape[0], len(lags), n_phase)

    template_energy = amp_den.reshape(len(lags), n_phase)

    return templates, amp, counts.astype(np.float32), n_on, n_off, template_energy


def phase_matched_score(amplitudes: np.ndarray, phase_idx: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    """Build y(t,L)=a(t,L,kappa(t-L)) on receiver days.

    For lag L, the relevant template phase is the source phase L days before
    the receiver field. This stored score is a convenience product; primary
    inference in phasebank_information_gain.py also pairs source day tau with
    receiver day tau+L directly from the full amplitude cube.
    """
    n_time, n_lag, n_phase = amplitudes.shape
    score = np.full((n_time, n_lag), np.nan, dtype=np.float32)
    for j, lag in enumerate(lags):
        source_idx = np.arange(n_time) - int(lag)
        valid = (source_idx >= 0) & (source_idx < n_time)
        source_phase = np.full(n_time, -1, dtype=np.int32)
        source_phase[valid] = phase_idx[source_idx[valid]]
        for p in range(n_phase):
            mask = source_phase == p
            if np.any(mask):
                score[mask, j] = amplitudes[mask, j, p]
    return score


ROMI_PHASE_FILE = "romi.cpcolr.1x.txt"
ROMI_AMP_MIN = 0.0
BURST_SMOOTH_DAYS = 10
BURST_GATE_KIND = "percentile"
BURST_GATE_PERCENTILE = 90.0


def smooth_intensity(values: np.ndarray, smooth_days: int) -> np.ndarray:
    vals = values.astype("float32")
    if smooth_days > 1:
        vals = rolling_mean(vals, smooth_days)
    return vals


def compute_burst_gate(intensity: np.ndarray, train_mask: np.ndarray) -> Tuple[np.ndarray, float]:
    valid_train = train_mask & np.isfinite(intensity)
    if not np.any(valid_train):
        raise ValueError("No finite intensity values in training window for burst gate computation.")
    if BURST_GATE_KIND != "percentile":
        raise ValueError("Only percentile burst gate is implemented in this build.")
    thresh = float(np.nanpercentile(intensity[valid_train], BURST_GATE_PERCENTILE))
    gate = np.full(intensity.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(intensity)
    gate[finite] = 0.0
    gate[finite & (intensity >= thresh)] = 1.0
    return gate, thresh


def load_romi_phase_table(path: Path, n_phase: int = 8) -> xr.Dataset:
    """Load ROMI daily index from text file and compute phase labels."""
    if not path.exists():
        raise FileNotFoundError(f"ROMI file not found: {path}")

    df = pd.read_csv(
        path,
        delim_whitespace=True,
        header=None,
        names=["year", "month", "day", "flag", "romi1", "romi2", "amp"],
    )
    time = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))
    romi1 = df["romi1"].to_numpy(dtype=np.float32)
    romi2 = df["romi2"].to_numpy(dtype=np.float32)
    amp = df["amp"].to_numpy(dtype=np.float32)

    theta = np.arctan2(romi2.astype(np.float64), romi1.astype(np.float64))
    theta_mod = np.mod(theta, 2.0 * np.pi)
    bin_width = 2.0 * np.pi / float(n_phase)
    phase = np.floor(theta_mod / bin_width).astype(np.int16) + 1

    ds = xr.Dataset(
        {
            "romi1": ("time", romi1),
            "romi2": ("time", romi2),
            "romi_amp": ("time", amp),
            "romi_phase": ("time", phase),
            "romi_theta": ("time", theta_mod.astype(np.float32)),
        },
        coords={"time": time},
    )
    ds.attrs["romi_source_file"] = str(path)
    ds.attrs["romi_phase_definition"] = "phase = floor(mod(atan2(romi2,romi1),2pi)/(2pi/8)) + 1"
    return ds


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Build phase-conditioned Z500 matched-filter receiver.")
    p.add_argument("--z500", required=True, help="Path to z500_global_raw.nc")
    p.add_argument("--mjo", required=True, help="Path to mjo_index_from_surprisal_timeseries.nc")
    p.add_argument("--out", required=True, help="Output NetCDF path")
    p.add_argument("--domain", default="120,300,20,80", help="lon0,lon1,lat0,lat1 in degE/degN")
    p.add_argument("--lags", default="0:30:1", help="Lag list as '0,5,10' or 'start:stop:step'")
    p.add_argument("--train", default="1991-01-01,2010-12-31", help="Training window for template regression")
    p.add_argument("--test", default="", help="Optional test window metadata")
    p.add_argument("--n_phase", type=int, default=8, help="Number of phase bins (default 8)")
    p.add_argument("--intensity_var", default="mjo_amp", help="Continuous burst intensity variable name")
    p.add_argument("--min_train_n", type=int, default=200, help="Minimum finite samples per lag/phase for training")
    p.add_argument("--min_on_train", type=int, default=50,
                   help="Minimum 'on' samples per lag/phase (binary modes only)")
    p.add_argument("--min_off_train", type=int, default=200,
                   help="Minimum 'off' samples per lag/phase (binary modes only)")
    p.add_argument("--chunks", default="time=90,latitude=45,longitude=90",
                   help="xarray chunk spec for Z500 (e.g., 'time=90,latitude=45,longitude=90')")
    p.add_argument("--engine", default="", help="Optional xarray backend engine (e.g., netcdf4)")
    p.add_argument("--strict", action="store_true", help="Fail if any lag/phase violates sample constraints")
    p.add_argument("--no_templates", action="store_true", help="Omit z500_template from output file")
    p.add_argument("--no_amp", action="store_true", help="Omit z500_amp from output file")
    p.add_argument("--include_score_energy", action="store_true", help="Store sqrt(sum_k amp^2) per lag as z500_score_energy")
    args = p.parse_args()

    lags = parse_lags(args.lags)
    if not lags:
        raise ValueError("No lags parsed from --lags")
    lon0, lon1, lat0, lat1 = parse_domain(args.domain)
    train_start, train_end = parse_window(args.train)
    test_window = args.test.strip()

    chunks = parse_chunks(args.chunks) if args.chunks else None

    print("[1/7] Loading MJO index and ROMI phase data...")
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=False)
    ds_driver = xr.open_dataset(args.mjo, decode_times=time_coder)
    ds_driver = ds_driver.assign_coords(time=decode_time_days_since_1979(ds_driver["time"]))
    ds_driver = drop_feb29(ds_driver, "time")

    romi_path = Path(args.mjo).resolve().parent / ROMI_PHASE_FILE
    ds_romi = load_romi_phase_table(romi_path, n_phase=args.n_phase)
    ds_romi = ds_romi.assign_coords(time=pd.to_datetime(ds_romi["time"].values))
    ds_romi = drop_feb29(ds_romi, "time")
    phase_da = ds_romi["romi_phase"]
    romi_amp_da = ds_romi["romi_amp"]
    romi1_da = ds_romi["romi1"]
    romi2_da = ds_romi["romi2"]
    phase_source = str(romi_path)

    if args.intensity_var not in ds_driver.data_vars:
        raise KeyError(f"Intensity variable '{args.intensity_var}' not found in driver file")
    intensity_da = ds_driver[args.intensity_var]

    print("[2/7] Loading Z500 dataset and selecting domain...")
    ds_z = xr.open_dataset(
        args.z500,
        decode_times=time_coder,
        chunks=chunks,
    )
    ds_z = ds_z.assign_coords(time=decode_time_days_since_1979(ds_z["time"]))
    ds_z = drop_feb29(ds_z, "time")
    z = ds_z["z500"]

    lat = z["latitude"]
    if float(lat[0]) > float(lat[-1]):
        lat_sel = slice(lat1, lat0)
    else:
        lat_sel = slice(lat0, lat1)
    lon = z["longitude"]
    if lon1 >= lon0:
        z = z.sel(longitude=slice(lon0, lon1), latitude=lat_sel)
    else:
        z = xr.concat([
            z.sel(longitude=slice(lon0, 360.0), latitude=lat_sel),
            z.sel(longitude=slice(0.0, lon1), latitude=lat_sel),
        ], dim="longitude")

    print("[3/7] Aligning Z500, ROMI phase, and driver timestamps...")
    align_items: List[Tuple[str, xr.DataArray]] = [
        ("z500", z),
        ("phase", phase_da),
        ("intensity", intensity_da),
        ("romi_amp", romi_amp_da),
        ("romi1", romi1_da),
        ("romi2", romi2_da),
    ]
    aligned = xr.align(*[item[1] for item in align_items], join="inner")
    aligned_map = {name: arr for (name, _), arr in zip(align_items, aligned)}
    z = aligned_map["z500"]
    phase_da = aligned_map["phase"]
    intensity_da = aligned_map["intensity"]
    romi_amp_da = aligned_map["romi_amp"]
    romi1_da = aligned_map["romi1"]
    romi2_da = aligned_map["romi2"]

    if z.sizes["time"] == 0:
        raise ValueError("No overlapping timestamps between Z500 and driver.")

    print("[4/7] Computing climatology and anomalies...")
    clim = month_day_climatology(z)
    z_anom = subtract_month_day_clim(z, clim).astype(np.float32)
    z_anom.name = "z500_anom"

    z_mat, weight_vec = _flatten_space(z_anom)
    times = z_anom["time"].values
    train_mask = _datetime_mask(times, train_start, train_end)

    print("[5/7] Building burst gate and driver matrices...")
    phase_vals = phase_da.values.astype(np.float64)
    intensity_vals = intensity_da.values.astype(np.float64)
    romi_amp_vals = romi_amp_da.values.astype(np.float64)
    romi1_vals = romi1_da.values.astype(np.float32)
    romi2_vals = romi2_da.values.astype(np.float32)

    intensity_smoothed = smooth_intensity(intensity_vals, BURST_SMOOTH_DAYS)
    burst_gate, gate_threshold = compute_burst_gate(intensity_smoothed, train_mask)

    phase_mask, phase_idx, phase_values = build_phase_mask_and_index(phase_vals, args.n_phase)
    romi_amp_mask = np.isfinite(romi_amp_vals) & (romi_amp_vals >= ROMI_AMP_MIN)
    phase_mask[~romi_amp_mask, :] = 0
    phase_idx[~romi_amp_mask] = -1

    driver_weighted = build_driver_matrix(intensity_smoothed, phase_mask)
    driver_gated = build_driver_matrix(burst_gate, phase_mask)
    driver_weighted[~romi_amp_mask, :] = np.nan
    driver_gated[~romi_amp_mask, :] = np.nan

    print("[6/7] Fitting templates and projecting amplitudes...")
    templates_w, amp_w, n_train_w, _, _, template_energy = fit_phase_templates(
        z_mat=z_mat,
        weights=weight_vec,
        driver_x=driver_weighted,
        lags=lags,
        train_mask=train_mask,
        min_train_n=args.min_train_n,
        min_on_train=0,
        min_off_train=0,
        strict=args.strict,
        binary_driver=False,
    )

    templates_g, amp_g, n_train_g, n_on_g, n_off_g, _ = fit_phase_templates(
        z_mat=z_mat,
        weights=weight_vec,
        driver_x=driver_gated,
        lags=lags,
        train_mask=train_mask,
        min_train_n=args.min_train_n,
        min_on_train=args.min_on_train,
        min_off_train=args.min_off_train,
        strict=args.strict,
        binary_driver=True,
    )

    print("[7/7] Computing phase-matched scores and writing output...")
    score_weighted = phase_matched_score(amp_w, phase_idx, lags)
    score_gated = phase_matched_score(amp_g, phase_idx, lags)
    if args.include_score_energy:
        energy = np.sum(np.where(np.isfinite(amp_w), amp_w**2, 0.0), axis=2)
        invalid_energy = ~np.any(np.isfinite(amp_w), axis=2)
        energy = np.sqrt(energy)
        energy[invalid_energy] = np.nan
    else:
        energy = None

    phase_coord = xr.IndexVariable("phase", phase_values)
    lag_coord = xr.IndexVariable("lag", np.array(lags, dtype=np.int32))

    template_da = xr.DataArray(
        templates_w.reshape(len(lags), phase_values.size, z_anom.sizes["latitude"], z_anom.sizes["longitude"]),
        coords={"lag": lag_coord, "phase": phase_coord, "latitude": z_anom["latitude"], "longitude": z_anom["longitude"]},
        dims=("lag", "phase", "latitude", "longitude"),
        name="z500_template_weighted",
    ).astype("float32")
    template_da.attrs["description"] = "Weighted driver template T(L,phase,lat,lon) = cov(z,x)/var(x)"

    amp_da = xr.DataArray(
        amp_w.astype(np.float32),
        coords={"time": z_anom["time"], "lag": lag_coord, "phase": phase_coord},
        dims=("time", "lag", "phase"),
        name="z500_amp_weighted",
    )
    amp_da.attrs["description"] = "Projection amplitude per lag/phase using sqrt(cos(lat)) weighting"

    template_g_da = xr.DataArray(
        templates_g.reshape(len(lags), phase_values.size, z_anom.sizes["latitude"], z_anom.sizes["longitude"]),
        coords={"lag": lag_coord, "phase": phase_coord, "latitude": z_anom["latitude"], "longitude": z_anom["longitude"]},
        dims=("lag", "phase", "latitude", "longitude"),
        name="z500_template_gated",
    ).astype("float32")
    template_g_da.attrs["description"] = "Binary gated template T(L,phase,lat,lon) derived from burst gate"

    score_w_da = xr.DataArray(
        score_weighted.astype(np.float32),
        coords={"time": z_anom["time"], "lag": lag_coord},
        dims=("time", "lag"),
        name="z500_score_weighted",
    )
    score_w_da.attrs["description"] = "Phase-matched score using weighted driver"

    score_g_da = xr.DataArray(
        score_gated.astype(np.float32),
        coords={"time": z_anom["time"], "lag": lag_coord},
        dims=("time", "lag"),
        name="z500_score_gated",
    )
    score_g_da.attrs["description"] = "Phase-matched score using gated driver"

    driver_intensity_da = xr.DataArray(
        intensity_smoothed.astype(np.float32), coords={"time": z_anom["time"]}, dims=("time",)
    )
    driver_intensity_da.attrs["description"] = "Smoothed burst intensity B(t)"

    burst_gate_da = xr.DataArray(
        burst_gate.astype(np.float32), coords={"time": z_anom["time"]}, dims=("time",)
    )
    burst_gate_da.attrs["description"] = "Binary burst gate g(t) derived inside builder"

    driver_phase_da = xr.DataArray(phase_vals.astype(np.float32), coords={"time": z_anom["time"]}, dims=("time",))
    driver_phase_da.attrs["description"] = "ROMI phase labels (1..n_phase)"

    driver_phase_index_da = xr.DataArray(phase_idx.astype(np.int16), coords={"time": z_anom["time"]}, dims=("time",))
    driver_phase_index_da.attrs["description"] = "Zero-based phase index used for template lookup (-1 invalid)"

    romi_amp_da_out = xr.DataArray(romi_amp_vals.astype(np.float32), coords={"time": z_anom["time"]}, dims=("time",))
    romi_amp_da_out.attrs["description"] = "ROMI amplitude from romi.cpcolr.1x.txt"

    romi1_da_out = xr.DataArray(romi1_vals, coords={"time": z_anom["time"]}, dims=("time",))
    romi1_da_out.attrs["description"] = "ROMI component 1"

    romi2_da_out = xr.DataArray(romi2_vals, coords={"time": z_anom["time"]}, dims=("time",))
    romi2_da_out.attrs["description"] = "ROMI component 2"

    phase_mask_da = xr.DataArray(
        phase_mask.astype(np.uint8), coords={"time": z_anom["time"], "phase": phase_coord}, dims=("time", "phase")
    )
    phase_mask_da.attrs["description"] = "Indicator I(phase(t)==k)"

    driver_weighted_da = xr.DataArray(
        driver_weighted.astype(np.float32), coords={"time": z_anom["time"], "phase": phase_coord}, dims=("time", "phase"),
    )
    driver_weighted_da.attrs["description"] = "x_k(t) = B(t) * I(phase=k)"

    driver_gated_da = xr.DataArray(
        driver_gated.astype(np.float32), coords={"time": z_anom["time"], "phase": phase_coord}, dims=("time", "phase"),
    )
    driver_gated_da.attrs["description"] = "x_k(t) = gate(t) * I(phase=k)"

    train_count_w_da = xr.DataArray(
        n_train_w.astype(np.float32), coords={"lag": lag_coord, "phase": phase_coord}, dims=("lag", "phase"),
    )
    train_count_w_da.attrs["description"] = "Finite weighted-driver samples per lag/phase"

    train_count_g_da = xr.DataArray(
        n_train_g.astype(np.float32), coords={"lag": lag_coord, "phase": phase_coord}, dims=("lag", "phase"),
    )
    train_count_g_da.attrs["description"] = "Finite gated-driver samples per lag/phase"

    train_on_da = xr.DataArray(n_on_g, coords={"lag": lag_coord, "phase": phase_coord}, dims=("lag", "phase"))
    train_on_da.attrs["description"] = "On-samples for gated driver"

    train_off_da = xr.DataArray(n_off_g, coords={"lag": lag_coord, "phase": phase_coord}, dims=("lag", "phase"))
    train_off_da.attrs["description"] = "Off-samples for gated driver"

    template_energy_da = xr.DataArray(
        template_energy.astype(np.float32), coords={"lag": lag_coord, "phase": phase_coord}, dims=("lag", "phase")
    )
    template_energy_da.attrs["description"] = "Template norm <T,T> using sqrt(cos(lat)) weighting"

    data_vars = {
        "z500_anom": z_anom,
        "driver_intensity": driver_intensity_da,
        "burst_gate": burst_gate_da,
        "driver_phase": driver_phase_da,
        "driver_phase_index": driver_phase_index_da,
        "driver_romi_amp": romi_amp_da_out,
        "driver_romi1": romi1_da_out,
        "driver_romi2": romi2_da_out,
        "driver_phase_mask": phase_mask_da,
        "driver_weighted": driver_weighted_da,
        "driver_gated": driver_gated_da,
        "train_count_weighted": train_count_w_da,
        "train_count_gated": train_count_g_da,
        "train_on_count_gated": train_on_da,
        "train_off_count_gated": train_off_da,
        "template_energy": template_energy_da,
        "z500_score_weighted": score_w_da,
        "z500_score_gated": score_g_da,
    }

    if not args.no_templates:
        data_vars["z500_template_weighted"] = template_da
        data_vars["z500_template_gated"] = template_g_da
    if not args.no_amp:
        data_vars["z500_amp_weighted"] = amp_da
    if energy is not None:
        score_energy_da = xr.DataArray(
            energy.astype(np.float32), coords={"time": z_anom["time"], "lag": lag_coord}, dims=("time", "lag"),
        )
        score_energy_da.attrs["description"] = "sqrt(sum_k z500_amp_weighted(t,L,k)^2)"
        data_vars["z500_score_energy"] = score_energy_da

    ds_out = xr.Dataset(data_vars)
    ds_out.attrs.update(
        {
            "domain_lon0_lon1_lat0_lat1": args.domain,
            "lag_list": ",".join(str(l) for l in lags),
            "train_window": args.train,
            "test_window": test_window,
            "driver_intensity_var": args.intensity_var,
            "phase_source": phase_source,
            "romi_phase_file": str(romi_path),
            "romi_phase_definition": ds_romi.attrs.get(
                "romi_phase_definition", "phase = floor(mod(atan2(romi2,romi1),2pi)/(2pi/8)) + 1"
            ),
            "romi_amp_min": ROMI_AMP_MIN,
            "burst_smooth_days": BURST_SMOOTH_DAYS,
            "burst_gate_kind": BURST_GATE_KIND,
            "burst_gate_percentile": BURST_GATE_PERCENTILE,
            "burst_gate_threshold": gate_threshold,
            "phase_values": ",".join(str(v) for v in phase_values),
            "leap_day_handling": "Feb 29 dropped before climatology/anomalies",
            "projection_weights": "sqrt(cos(lat)) weighting applied to templates/amplitudes",
            "score_definition_weighted": "score_w(t,L) = z500_amp_weighted(t,L, phase_index(t-L))",
            "score_definition_gated": "score_g(t,L) = z500_amp_gated(t,L, phase_index(t-L))",
            "template_normalization": "Templates scaled by cov(z,x)/var(x) with NaNs for insufficient samples",
        }
    )

    encoding = {var: {"zlib": True, "complevel": 5} for var in ds_out.data_vars}
    ds_out.to_netcdf(args.out, encoding=encoding, engine=args.engine or None)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
