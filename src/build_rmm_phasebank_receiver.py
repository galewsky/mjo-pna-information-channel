#!/usr/bin/env python3
"""
Build an RMM-based Z500 phasebank receiver from an existing no-leap Z500 anomaly cube.

This script reuses the preprocessed receiver-domain Z500 anomalies from the existing
phasebank receiver and swaps in the BOM RMM source representation:
  - phase labels from the RMM text file
  - 10-day centered running-mean RMM amplitude as the weighted driver
  - training-window q90 gate from the smoothed RMM amplitude as the gated driver

The output receiver is compact: it stores drivers, templates, amplitudes, and
phase-matched scores, but not the full Z500 anomaly cube.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from build_z500_phasebank_receiver import (
    build_lag_matrix_2d,
    parse_lags,
    parse_window,
    phase_matched_score,
    rolling_mean,
    sqrt_coslat_weights,
)
from mjo_index_from_surprisal import bursts_from_amplitude


RMM_COLUMNS = ["year", "month", "day", "rmm1", "rmm2", "phase", "amplitude", "meta"]


def iter_slices(n_items: int, chunk_size: int) -> Iterable[slice]:
    for start in range(0, n_items, chunk_size):
        yield slice(start, min(start + chunk_size, n_items))


def _bool_to_uint8(mask: np.ndarray) -> np.ndarray:
    return mask.astype(np.uint8, copy=False)


def load_rmm_text(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"RMM file not found: {path}")
    df = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=2,
        names=RMM_COLUMNS,
        usecols=range(8),
        engine="python",
    )
    df["time"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))
    return df


def align_rmm_to_time(
    rmm_df: pd.DataFrame,
    source_file: Path,
    time_index: pd.DatetimeIndex,
    amp_active_min: float,
    smooth_days: int,
    train_window: Tuple[str, str],
    gate_percentile: float,
) -> xr.Dataset:
    df = rmm_df.copy()
    leap_mask = (df["time"].dt.month == 2) & (df["time"].dt.day == 29)
    leap_days_removed = int(leap_mask.sum())
    df = df.loc[~leap_mask].copy()

    missing_mask = (
        (df["rmm1"].to_numpy(dtype=np.float64) > 1.0e35)
        | (df["rmm2"].to_numpy(dtype=np.float64) > 1.0e35)
        | (df["amplitude"].to_numpy(dtype=np.float64) > 1.0e35)
        | (df["phase"].to_numpy(dtype=np.float64) == 999.0)
        | df["meta"].astype(str).eq("Missing_value").to_numpy()
    )
    df["qc_valid"] = ~missing_mask
    df.loc[missing_mask, ["rmm1", "rmm2", "phase", "amplitude"]] = np.nan

    series = (
        df.set_index("time")[["rmm1", "rmm2", "phase", "amplitude", "qc_valid", "meta"]]
        .reindex(time_index)
    )

    rmm1 = series["rmm1"].to_numpy(dtype=np.float64)
    rmm2 = series["rmm2"].to_numpy(dtype=np.float64)
    phase_raw = series["phase"].to_numpy(dtype=np.float64)
    amp_raw = series["amplitude"].to_numpy(dtype=np.float64)
    qc_valid = series["qc_valid"].fillna(False).to_numpy(dtype=bool)
    meta = series["meta"].fillna("missing_after_reindex").astype(str).to_numpy(dtype=object)

    amp_calc = np.sqrt(rmm1**2 + rmm2**2)
    amp_consistency = np.isfinite(amp_raw) & np.isfinite(amp_calc)
    amp_absdiff_max = (
        float(np.nanmax(np.abs(amp_raw[amp_consistency] - amp_calc[amp_consistency])))
        if np.any(amp_consistency)
        else np.nan
    )

    amp_smooth = rolling_mean(amp_raw.astype(np.float32), smooth_days).astype(np.float64)
    active_mask = qc_valid & np.isfinite(amp_raw) & (amp_raw >= float(amp_active_min))

    phase_active = np.where(active_mask, phase_raw, np.nan)
    phase_index = np.full(time_index.size, -1, dtype=np.int16)
    valid_phase = np.isfinite(phase_active)
    phase_index[valid_phase] = np.rint(phase_active[valid_phase]).astype(np.int16) - 1
    bad_phase = (phase_index < 0) | (phase_index > 7)
    phase_index[bad_phase] = -1
    phase_active[bad_phase] = np.nan

    train_start, train_end = train_window
    train_mask = (time_index >= pd.Timestamp(train_start)) & (time_index <= pd.Timestamp(train_end))
    gate_base = amp_smooth.copy()
    gate_valid = train_mask & np.isfinite(gate_base)
    if not np.any(gate_valid):
        raise ValueError("No finite smoothed RMM amplitudes in training window.")
    gate_threshold = float(np.nanpercentile(gate_base[gate_valid], gate_percentile))
    gate = np.full(time_index.size, np.nan, dtype=np.float64)
    finite_gate = np.isfinite(gate_base)
    gate[finite_gate] = 0.0
    gate[finite_gate & (gate_base >= gate_threshold)] = 1.0

    amp_da = xr.DataArray(amp_raw.astype(np.float32), coords={"time": time_index}, dims=("time",), name="rmm_amplitude")
    burst_ds = bursts_from_amplitude(
        amp_da,
        thresh_kind="sigma",
        thresh_value=1.0,
        min_duration_days=7,
        min_separation_days=7,
    )
    burst_ds.attrs["burst_threshold_abs"] = float(np.nanmean(amp_raw) + np.nanstd(amp_raw))

    out = xr.Dataset(
        {
            "rmm1": ("time", rmm1.astype(np.float32)),
            "rmm2": ("time", rmm2.astype(np.float32)),
            "rmm_phase_raw": ("time", phase_raw.astype(np.float32)),
            "rmm_phase_active": ("time", phase_active.astype(np.float32)),
            "rmm_phase_index": ("time", phase_index.astype(np.int16)),
            "rmm_amplitude": ("time", amp_raw.astype(np.float32)),
            "rmm_amplitude_calc": ("time", amp_calc.astype(np.float32)),
            "rmm_amplitude_smoothed": ("time", amp_smooth.astype(np.float32)),
            "rmm_gate": ("time", gate.astype(np.float32)),
            "rmm_active_mask": ("time", _bool_to_uint8(active_mask)),
            "rmm_qc_valid": ("time", _bool_to_uint8(qc_valid)),
            "rmm_burst_mask": burst_ds["burst_mask"].astype(np.uint8),
            "rmm_burst_onset": burst_ds["burst_onset"].astype(np.uint8),
            "rmm_burst_id": burst_ds["burst_id"].astype(np.int32),
        },
        coords={"time": time_index},
    )
    out.attrs.update(
        {
            "rmm_source_file": str(source_file),
            "rmm_smooth_days": int(smooth_days),
            "rmm_active_amplitude_min": float(amp_active_min),
            "rmm_gate_percentile": float(gate_percentile),
            "rmm_gate_threshold": gate_threshold,
            "leap_days_removed_from_text": leap_days_removed,
            "qc_missing_rows_in_source": int(missing_mask.sum()),
            "aligned_missing_rows": int((~qc_valid).sum()),
            "rmm_amplitude_calc_max_absdiff": amp_absdiff_max,
            "burst_threshold_kind": burst_ds.attrs.get("burst_threshold_kind", "sigma"),
            "burst_threshold_value": burst_ds.attrs.get("burst_threshold_value", 1.0),
            "burst_threshold_abs": burst_ds.attrs.get("burst_threshold_abs", np.nan),
            "burst_min_duration_days": burst_ds.attrs.get("burst_min_duration_days", 7),
            "burst_min_separation_days": burst_ds.attrs.get("burst_min_separation_days", 7),
            "rmm_meta_counts": "; ".join(
                f"{k}:{int(v)}" for k, v in pd.Series(meta).value_counts(dropna=False).items()
            ),
        }
    )
    return out


def build_phase_mask(phase_index: np.ndarray, n_phase: int = 8) -> np.ndarray:
    mask = np.zeros((phase_index.size, n_phase), dtype=np.uint8)
    valid = (phase_index >= 0) & (phase_index < n_phase)
    rows = np.where(valid)[0]
    if rows.size:
        mask[rows, phase_index[rows]] = 1
    return mask


def build_driver_matrix(values: np.ndarray, phase_mask: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    out = np.full((phase_mask.shape[0], phase_mask.shape[1]), np.nan, dtype=np.float32)
    finite = np.isfinite(vals)
    for p in range(phase_mask.shape[1]):
        mask_p = phase_mask[:, p].astype(bool) & finite
        out[mask_p, p] = vals[mask_p]
    return out


def prepare_driver_terms(
    driver_x: np.ndarray,
    lags: Sequence[int],
    train_mask: np.ndarray,
    *,
    binary_driver: bool,
    min_train_n: int,
    min_on_train: int,
    min_off_train: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_lag = build_lag_matrix_2d(driver_x, lags).astype(np.float32)
    x_train = x_lag[train_mask, :, :]
    valid = np.isfinite(x_train)
    counts = valid.sum(axis=0).astype(np.float32)

    if binary_driver:
        on_mask = (x_train > 0.5) & valid
        off_mask = (x_train <= 0.5) & valid
        n_on = on_mask.sum(axis=0).astype(np.float32)
        n_off = off_mask.sum(axis=0).astype(np.float32)
    else:
        n_on = np.full(counts.shape, np.nan, dtype=np.float32)
        n_off = np.full(counts.shape, np.nan, dtype=np.float32)

    bad = counts < float(min_train_n)
    if binary_driver:
        bad |= n_on < float(min_on_train)
        bad |= n_off < float(min_off_train)

    counts_safe = np.maximum(counts, 1.0)
    x_zero = np.where(np.isfinite(x_lag), x_lag, 0.0)
    means = np.where(
        counts_safe > 0.0,
        np.nansum(np.where(np.isfinite(x_train), x_train, 0.0), axis=0) / counts_safe,
        0.0,
    ).astype(np.float32)
    x_centered = (x_zero - means[None, :, :]) * np.isfinite(x_lag)
    var_den = np.sum((x_centered[train_mask, :, :].astype(np.float64)) ** 2, axis=0)
    return x_centered.astype(np.float32), counts, n_on, n_off, bad | (var_den <= 0.0)


def flatten_chunk(z_chunk: np.ndarray) -> np.ndarray:
    return np.asarray(z_chunk, dtype=np.float32).reshape(z_chunk.shape[0], -1)


def fit_templates_chunked(
    z_da: xr.DataArray,
    x_centered: np.ndarray,
    train_mask: np.ndarray,
    bad: np.ndarray,
    chunk_time: int,
) -> np.ndarray:
    n_time, n_lag, n_phase = x_centered.shape
    n_pair = n_lag * n_phase
    n_space = z_da.sizes["latitude"] * z_da.sizes["longitude"]
    cov_num = np.zeros((n_pair, n_space), dtype=np.float64)
    x2d = x_centered.reshape(n_time, n_pair)

    train_idx = np.where(train_mask)[0]
    for chunk in iter_slices(train_idx.size, chunk_time):
        idx = train_idx[chunk]
        start = int(idx[0])
        stop = int(idx[-1]) + 1
        z_chunk = flatten_chunk(z_da.isel(time=slice(start, stop)).values)
        cov_num += x2d[start:stop, :].T @ z_chunk

    bad_flat = bad.reshape(n_pair)
    return cov_num, bad_flat


def project_templates_chunked(
    z_da: xr.DataArray,
    weights_vec: np.ndarray,
    templates_flat: np.ndarray,
    bad_flat: np.ndarray,
    chunk_time: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_time = z_da.sizes["time"]
    n_pair = templates_flat.shape[0]
    weights = weights_vec.astype(np.float32, copy=False)
    tpl_w = templates_flat.astype(np.float32, copy=False) * weights[None, :]
    amp_den = np.sum(tpl_w.astype(np.float64) ** 2, axis=1)
    amp = np.full((n_time, n_pair), np.nan, dtype=np.float32)

    for chunk in iter_slices(n_time, chunk_time):
        start = int(chunk.start)
        stop = int(chunk.stop)
        z_chunk = flatten_chunk(z_da.isel(time=slice(start, stop)).values)
        z_weighted = z_chunk * weights[None, :]
        amp_num = z_weighted @ tpl_w.T
        amp_chunk = amp_num / np.maximum(amp_den[None, :], 1.0e-9)
        amp[start:stop, :] = amp_chunk.astype(np.float32)

    amp[:, bad_flat] = np.nan
    return amp, amp_den


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_receiver", required=True, help="Existing receiver NetCDF containing z500_anom")
    p.add_argument("--rmm_txt", required=True, help="BOM RMM text file")
    p.add_argument("--out_receiver", required=True, help="Output compact RMM receiver NetCDF")
    p.add_argument("--out_driver", default="", help="Optional RMM driver NetCDF")
    p.add_argument("--train", default="1991-01-01,2010-12-31", help="Training window")
    p.add_argument("--test", default="2011-01-01,2020-12-31", help="Test window metadata")
    p.add_argument("--lags", default="0:30:1", help="Lag specification")
    p.add_argument("--smooth_days", type=int, default=10)
    p.add_argument("--gate_percentile", type=float, default=90.0)
    p.add_argument("--amp_active_min", type=float, default=1.0)
    p.add_argument("--min_train_n", type=int, default=200)
    p.add_argument("--min_on_train", type=int, default=20)
    p.add_argument("--min_off_train", type=int, default=100)
    p.add_argument("--chunk_time", type=int, default=120, help="Time chunk for chunked matrix operations")
    p.add_argument("--engine", default="", help="Optional NetCDF engine for output")
    args = p.parse_args()

    lags = parse_lags(args.lags)
    train_window = parse_window(args.train)
    _ = parse_window(args.test)

    base_receiver = Path(args.base_receiver)
    rmm_txt = Path(args.rmm_txt)
    out_receiver = Path(args.out_receiver)
    out_driver = Path(args.out_driver) if args.out_driver else None
    out_receiver.parent.mkdir(parents=True, exist_ok=True)
    if out_driver is not None:
        out_driver.parent.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading base receiver and aligning RMM time axis...")
    ds_base = xr.open_dataset(base_receiver)
    if "z500_anom" not in ds_base:
        raise KeyError(f"{base_receiver} is missing z500_anom")
    z_da = ds_base["z500_anom"]
    time_index = pd.DatetimeIndex(pd.to_datetime(ds_base["time"].values))

    rmm_df = load_rmm_text(rmm_txt)
    rmm_ds = align_rmm_to_time(
        rmm_df,
        source_file=rmm_txt,
        time_index=time_index,
        amp_active_min=args.amp_active_min,
        smooth_days=args.smooth_days,
        train_window=train_window,
        gate_percentile=args.gate_percentile,
    )

    if time_index.size != 10950:
        print(f"WARNING: overlapping no-leap days = {time_index.size} (expected 10950 for 1991-2020)")
    else:
        print(f"Calendar alignment check: overlapping no-leap days = {time_index.size}")

    print("[2/6] Building RMM phase masks and drivers...")
    phase_index = rmm_ds["rmm_phase_index"].values.astype(np.int16)
    phase_mask = build_phase_mask(phase_index, n_phase=8)
    intensity = rmm_ds["rmm_amplitude_smoothed"].values.astype(np.float32)
    gate = rmm_ds["rmm_gate"].values.astype(np.float32)

    driver_weighted = build_driver_matrix(intensity, phase_mask)
    driver_gated = build_driver_matrix(gate, phase_mask)
    train_mask = (time_index >= pd.Timestamp(train_window[0])) & (time_index <= pd.Timestamp(train_window[1]))

    xw_centered, train_count_w, _, _, bad_w = prepare_driver_terms(
        driver_weighted,
        lags=lags,
        train_mask=train_mask,
        binary_driver=False,
        min_train_n=args.min_train_n,
        min_on_train=0,
        min_off_train=0,
    )
    xg_centered, train_count_g, train_on_g, train_off_g, bad_g = prepare_driver_terms(
        driver_gated,
        lags=lags,
        train_mask=train_mask,
        binary_driver=True,
        min_train_n=args.min_train_n,
        min_on_train=args.min_on_train,
        min_off_train=args.min_off_train,
    )

    if np.any(bad_w):
        pairs = [f"(lag={lags[i]},phase={j + 1})" for i, j in np.argwhere(bad_w)]
        print("WARNING: weighted templates underpowered for " + ", ".join(pairs))
    if np.any(bad_g):
        pairs = [f"(lag={lags[i]},phase={j + 1})" for i, j in np.argwhere(bad_g)]
        print("WARNING: gated templates underpowered for " + ", ".join(pairs))

    print("[3/6] Fitting weighted and gated templates...")
    cov_w, bad_flat_w = fit_templates_chunked(
        z_da=z_da,
        x_centered=xw_centered,
        train_mask=train_mask,
        bad=bad_w,
        chunk_time=args.chunk_time,
    )
    cov_g, bad_flat_g = fit_templates_chunked(
        z_da=z_da,
        x_centered=xg_centered,
        train_mask=train_mask,
        bad=bad_g,
        chunk_time=args.chunk_time,
    )

    var_den_w = np.sum((xw_centered[train_mask, :, :].astype(np.float64)) ** 2, axis=0).reshape(-1)
    var_den_g = np.sum((xg_centered[train_mask, :, :].astype(np.float64)) ** 2, axis=0).reshape(-1)
    templates_w_flat = cov_w / np.maximum(var_den_w[:, None], 1.0e-9)
    templates_g_flat = cov_g / np.maximum(var_den_g[:, None], 1.0e-9)
    templates_w_flat[bad_flat_w, :] = np.nan
    templates_g_flat[bad_flat_g, :] = np.nan

    weights_lat = sqrt_coslat_weights(z_da["latitude"]).values.astype(np.float32)
    weights_vec = np.repeat(weights_lat, z_da.sizes["longitude"])

    print("[4/6] Projecting full-record amplitudes and phase-matched scores...")
    amp_w_flat, energy_w = project_templates_chunked(
        z_da=z_da,
        weights_vec=weights_vec,
        templates_flat=templates_w_flat,
        bad_flat=bad_flat_w,
        chunk_time=args.chunk_time,
    )
    amp_g_flat, energy_g = project_templates_chunked(
        z_da=z_da,
        weights_vec=weights_vec,
        templates_flat=templates_g_flat,
        bad_flat=bad_flat_g,
        chunk_time=args.chunk_time,
    )

    amp_w = amp_w_flat.reshape(time_index.size, len(lags), 8)
    amp_g = amp_g_flat.reshape(time_index.size, len(lags), 8)
    score_w = phase_matched_score(amp_w, phase_index, lags)
    score_g = phase_matched_score(amp_g, phase_index, lags)

    print("[5/6] Writing compact RMM driver and receiver files...")
    lag_coord = xr.IndexVariable("lag", np.asarray(lags, dtype=np.int32))
    phase_coord = xr.IndexVariable("phase", np.arange(1, 9, dtype=np.int16))
    lat_coord = z_da["latitude"]
    lon_coord = z_da["longitude"]

    receiver_ds = xr.Dataset(
        {
            "driver_intensity": xr.DataArray(
                intensity.astype(np.float32), coords={"time": time_index}, dims=("time",), name="driver_intensity"
            ),
            "burst_gate": xr.DataArray(gate.astype(np.float32), coords={"time": time_index}, dims=("time",), name="burst_gate"),
            "burst_mask": rmm_ds["rmm_burst_mask"].astype(np.uint8).rename("burst_mask"),
            "burst_onset": rmm_ds["rmm_burst_onset"].astype(np.uint8).rename("burst_onset"),
            "burst_id": rmm_ds["rmm_burst_id"].astype(np.int32).rename("burst_id"),
            "driver_phase": xr.DataArray(
                rmm_ds["rmm_phase_active"].values.astype(np.float32),
                coords={"time": time_index},
                dims=("time",),
                name="driver_phase",
            ),
            "driver_phase_index": xr.DataArray(
                phase_index.astype(np.int16), coords={"time": time_index}, dims=("time",), name="driver_phase_index"
            ),
            "driver_phase_mask": xr.DataArray(
                phase_mask.astype(np.uint8),
                coords={"time": time_index, "phase": phase_coord},
                dims=("time", "phase"),
                name="driver_phase_mask",
            ),
            "driver_weighted": xr.DataArray(
                driver_weighted.astype(np.float32),
                coords={"time": time_index, "phase": phase_coord},
                dims=("time", "phase"),
                name="driver_weighted",
            ),
            "driver_gated": xr.DataArray(
                driver_gated.astype(np.float32),
                coords={"time": time_index, "phase": phase_coord},
                dims=("time", "phase"),
                name="driver_gated",
            ),
            "driver_rmm1": rmm_ds["rmm1"].rename("driver_rmm1"),
            "driver_rmm2": rmm_ds["rmm2"].rename("driver_rmm2"),
            "driver_rmm_amp": rmm_ds["rmm_amplitude"].rename("driver_rmm_amp"),
            "driver_rmm_phase_raw": rmm_ds["rmm_phase_raw"].rename("driver_rmm_phase_raw"),
            "driver_active_mask": rmm_ds["rmm_active_mask"].rename("driver_active_mask"),
            "driver_qc_valid": rmm_ds["rmm_qc_valid"].rename("driver_qc_valid"),
            "z500_template_weighted": xr.DataArray(
                templates_w_flat.reshape(len(lags), 8, z_da.sizes["latitude"], z_da.sizes["longitude"]).astype(np.float32),
                coords={"lag": lag_coord, "phase": phase_coord, "latitude": lat_coord, "longitude": lon_coord},
                dims=("lag", "phase", "latitude", "longitude"),
                name="z500_template_weighted",
            ),
            "z500_template_gated": xr.DataArray(
                templates_g_flat.reshape(len(lags), 8, z_da.sizes["latitude"], z_da.sizes["longitude"]).astype(np.float32),
                coords={"lag": lag_coord, "phase": phase_coord, "latitude": lat_coord, "longitude": lon_coord},
                dims=("lag", "phase", "latitude", "longitude"),
                name="z500_template_gated",
            ),
            "z500_amp_weighted": xr.DataArray(
                amp_w.astype(np.float32),
                coords={"time": time_index, "lag": lag_coord, "phase": phase_coord},
                dims=("time", "lag", "phase"),
                name="z500_amp_weighted",
            ),
            "z500_amp_gated": xr.DataArray(
                amp_g.astype(np.float32),
                coords={"time": time_index, "lag": lag_coord, "phase": phase_coord},
                dims=("time", "lag", "phase"),
                name="z500_amp_gated",
            ),
            "z500_score_weighted": xr.DataArray(
                score_w.astype(np.float32),
                coords={"time": time_index, "lag": lag_coord},
                dims=("time", "lag"),
                name="z500_score_weighted",
            ),
            "z500_score_gated": xr.DataArray(
                score_g.astype(np.float32),
                coords={"time": time_index, "lag": lag_coord},
                dims=("time", "lag"),
                name="z500_score_gated",
            ),
            "train_count_weighted": xr.DataArray(
                train_count_w.astype(np.float32), coords={"lag": lag_coord, "phase": phase_coord}, dims=("lag", "phase")
            ),
            "train_count_gated": xr.DataArray(
                train_count_g.astype(np.float32), coords={"lag": lag_coord, "phase": phase_coord}, dims=("lag", "phase")
            ),
            "train_on_count_gated": xr.DataArray(
                train_on_g.astype(np.float32), coords={"lag": lag_coord, "phase": phase_coord}, dims=("lag", "phase")
            ),
            "train_off_count_gated": xr.DataArray(
                train_off_g.astype(np.float32), coords={"lag": lag_coord, "phase": phase_coord}, dims=("lag", "phase")
            ),
            "template_energy": xr.DataArray(
                energy_w.reshape(len(lags), 8).astype(np.float32),
                coords={"lag": lag_coord, "phase": phase_coord},
                dims=("lag", "phase"),
                name="template_energy",
            ),
            "template_energy_gated": xr.DataArray(
                energy_g.reshape(len(lags), 8).astype(np.float32),
                coords={"lag": lag_coord, "phase": phase_coord},
                dims=("lag", "phase"),
                name="template_energy_gated",
            ),
        }
    )
    receiver_ds.attrs.update(
        {
            "base_receiver_file": str(base_receiver),
            "rmm_source_file": str(rmm_txt),
            "train_window": args.train,
            "test_window": args.test,
            "lag_list": ",".join(str(v) for v in lags),
            "leap_day_handling": "Aligned to existing no-leap receiver axis (Feb 29 removed)",
            "driver_intensity_definition": f"{args.smooth_days}-day centered running mean of RMM amplitude",
            "driver_gate_percentile": float(args.gate_percentile),
            "driver_gate_threshold": float(rmm_ds.attrs["rmm_gate_threshold"]),
            "driver_phase_definition": "RMM file phase labels, active only where daily amplitude >= 1.0",
            "driver_active_amplitude_min": float(args.amp_active_min),
            "projection_weights": "sqrt(cos(lat)) weighting applied to projections",
            "template_normalization": "Templates scaled by cov(z,x)/var(x); underpowered lag/phase cells set to NaN",
            "n_time": int(time_index.size),
            "n_active_days_total": int(np.sum(rmm_ds["rmm_active_mask"].values > 0)),
            "n_qc_invalid_days": int(np.sum(rmm_ds["rmm_qc_valid"].values == 0)),
        }
    )

    for name, desc in [
        ("driver_intensity", "Smoothed RMM amplitude B_RMM(t)"),
        ("burst_gate", "q90 gate from smoothed RMM amplitude"),
        ("driver_phase", "RMM phase labels (1..8) on active days only"),
        ("driver_phase_index", "Zero-based RMM phase index used for template lookup (-1 inactive/invalid)"),
        ("driver_weighted", "x_k^(w)(t) = B_RMM(t) * I_k(t)"),
        ("driver_gated", "x_k^(g)(t) = g_RMM(t) * I_k(t)"),
        ("z500_amp_weighted", "Weighted-template projection amplitudes a^(w)"),
        ("z500_amp_gated", "Gated-template projection amplitudes a^(g)"),
        ("z500_score_weighted", "Phase-matched weighted receiver score"),
        ("z500_score_gated", "Phase-matched gated receiver score"),
    ]:
        receiver_ds[name].attrs["description"] = desc

    encoding = {name: {"zlib": True, "complevel": 4} for name in receiver_ds.data_vars}
    receiver_ds.to_netcdf(out_receiver, encoding=encoding, engine=args.engine or None)

    if out_driver is not None:
        driver_ds = xr.Dataset(
            {
                "rmm1": rmm_ds["rmm1"],
                "rmm2": rmm_ds["rmm2"],
                "amplitude": rmm_ds["rmm_amplitude"],
                "amplitude_smoothed": rmm_ds["rmm_amplitude_smoothed"],
                "phase_raw": rmm_ds["rmm_phase_raw"],
                "phase_active": rmm_ds["rmm_phase_active"],
                "phase_index": rmm_ds["rmm_phase_index"],
                "gate": rmm_ds["rmm_gate"],
                "active_mask": rmm_ds["rmm_active_mask"],
                "qc_valid": rmm_ds["rmm_qc_valid"],
                "burst_mask": rmm_ds["rmm_burst_mask"],
                "burst_onset": rmm_ds["rmm_burst_onset"],
                "burst_id": rmm_ds["rmm_burst_id"],
            }
        )
        driver_ds.attrs.update(rmm_ds.attrs)
        driver_encoding = {name: {"zlib": True, "complevel": 4} for name in driver_ds.data_vars}
        driver_ds.to_netcdf(out_driver, encoding=driver_encoding, engine=args.engine or None)

    print(f"Wrote {out_receiver}")
    if out_driver is not None:
        print(f"Wrote {out_driver}")

    print("[6/6] Key counts")
    eval_mask = (time_index >= pd.Timestamp("2011-01-01")) & (time_index <= pd.Timestamp("2020-12-31"))
    phase_counts_all = pd.Series(rmm_ds["rmm_phase_raw"].values[eval_mask]).value_counts().sort_index()
    phase_counts_active = pd.Series(rmm_ds["rmm_phase_active"].values[eval_mask]).value_counts().sort_index()
    gate_on = (rmm_ds["rmm_gate"].values > 0.5) & eval_mask
    print("  Eval phase counts (all days, raw phase labels):")
    for ph in range(1, 9):
        print(f"    Phase {ph}: {int(phase_counts_all.get(float(ph), 0))}")
    print("  Eval phase counts (active days only, amplitude >= 1):")
    for ph in range(1, 9):
        print(f"    Phase {ph}: {int(phase_counts_active.get(float(ph), 0))}")
    print("  Eval gate-on counts by active phase:")
    for ph in range(1, 9):
        mask = (rmm_ds["rmm_phase_index"].values == (ph - 1)) & eval_mask
        n_phase = int(mask.sum())
        n_on = int(np.sum(mask & gate_on))
        frac = (n_on / n_phase) if n_phase else np.nan
        print(f"    Phase {ph}: n_on={n_on}, n_phase={n_phase}, frac={frac:.3f}" if n_phase else f"    Phase {ph}: n_on=0, n_phase=0, frac=nan")


if __name__ == "__main__":
    main()
