#!/usr/bin/env python3
"""
phasebank_information_gain.py

Phase-resolved information gain evaluation for phasebank receivers with
red-noise-safe nulls and lag-hunting correction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed


@dataclass
class LagData:
    lag_value: int
    lag_index: int
    w_indices: np.ndarray
    y: np.ndarray
    bins: Optional[np.ndarray]
    n_on: int
    n_off: int
    n_total: int
    p_on: float
    n_bins: int
    q90_shift: float
    underpowered: bool


@dataclass
class PhaseResult:
    phase: int
    lags: np.ndarray
    metrics: Dict[str, np.ndarray]
    n_on: np.ndarray
    n_off: np.ndarray
    n_total: np.ndarray
    p_on: np.ndarray
    n_bins: np.ndarray
    q90_shift: np.ndarray
    underpowered: np.ndarray
    max_stat_obs: float
    best_lag_obs: int
    max_stat_null: np.ndarray
    n_lags_used: int
    min_n: float
    median_n: float


def parse_window(expr: str) -> Tuple[str, str]:
    parts = [p.strip() for p in expr.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Window spec must be 'YYYY-MM-DD,YYYY-MM-DD'")
    return parts[0], parts[1]


def parse_lags(expr: str) -> List[int]:
    if not expr:
        return []
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


def parse_int_list(expr: str) -> List[int]:
    if not expr:
        return []
    return [int(p.strip()) for p in expr.split(",") if p.strip()]


def parse_value_list(expr: str) -> List[str]:
    if not expr:
        return []
    return [p.strip() for p in expr.split(",") if p.strip()]


def rolling_mean(values: np.ndarray, window: int, centered: bool) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float64, copy=False)
    series = pd.Series(values.astype(np.float64))
    return series.rolling(window=window, center=centered, min_periods=1).mean().values


def compute_gate(
    intensity: np.ndarray,
    times: np.ndarray,
    train_window: Tuple[str, str],
    percentile: float,
    smooth_days: int,
    smooth_centered: bool,
    assume_smoothed: bool,
) -> Tuple[np.ndarray, float, np.ndarray]:
    vals = intensity.astype(np.float64, copy=False)
    if not assume_smoothed:
        vals = rolling_mean(vals, smooth_days, smooth_centered)
    train_start, train_end = train_window
    train_mask = (times >= np.datetime64(train_start)) & (times <= np.datetime64(train_end))
    valid_train = train_mask & np.isfinite(vals)
    if not np.any(valid_train):
        raise ValueError("No finite intensity values in gate training window.")
    threshold = float(np.nanpercentile(vals[valid_train], percentile))
    gate = np.full(vals.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(vals)
    gate[finite] = 0.0
    gate[finite & (vals >= threshold)] = 1.0
    return gate, threshold, vals


def compute_bins_fd(values: np.ndarray, min_bins: int, max_bins: int) -> Optional[np.ndarray]:
    vals = values[np.isfinite(values)]
    if vals.size < 2:
        return None
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return None
    q25, q75 = np.nanpercentile(vals, [25, 75])
    iqr = q75 - q25
    if iqr <= 0:
        n_bins = int(np.clip(np.sqrt(vals.size), min_bins, max_bins))
    else:
        bin_width = 2.0 * iqr * (vals.size ** (-1.0 / 3.0))
        if bin_width <= 0:
            n_bins = int(np.clip(np.sqrt(vals.size), min_bins, max_bins))
        else:
            n_bins = int(np.ceil((vmax - vmin) / bin_width))
            n_bins = int(np.clip(n_bins, min_bins, max_bins))
    return np.linspace(vmin, vmax, n_bins + 1, dtype=np.float64)


def compute_bins_fixed(values: np.ndarray, n_bins: int) -> Optional[np.ndarray]:
    vals = values[np.isfinite(values)]
    if vals.size < 2:
        return None
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return None
    return np.linspace(vmin, vmax, n_bins + 1, dtype=np.float64)


def hist_probs(values: np.ndarray, bins: np.ndarray, alpha: float) -> np.ndarray:
    counts, _ = np.histogram(values, bins=bins)
    m = bins.size - 1
    return (counts + alpha) / (counts.sum() + alpha * m)


def entropy_bits(probs: np.ndarray) -> float:
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def digitize_y(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Map values onto fixed histogram bins, preserving the rightmost edge."""
    out = np.digitize(values, bins[1:-1], right=False)
    return out.astype(np.int64, copy=False)


def joint_information_bits(
    y: np.ndarray,
    x_bins: np.ndarray,
    y_bins: np.ndarray,
    n_x: int,
    n_y: int,
    alpha: float,
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate I(Y;X) from one smoothed joint count table.

    Earlier revisions estimated information as H(Y)-H(Y|X) from separately
    smoothed histograms. Separate smoothing can make the plug-in difference
    slightly negative even though mutual information is nonnegative. Building
    one joint table and deriving both marginals from it keeps the estimator
    internally consistent and prevents impossible negative MI/IG cells.
    """
    valid = np.isfinite(y) & np.isfinite(x_bins) & np.isfinite(y_bins)
    if not np.any(valid):
        empty = np.full((n_x, n_y), np.nan)
        return np.nan, np.nan, np.nan, empty, np.full(n_x, np.nan), np.full(n_y, np.nan)

    xb = x_bins[valid].astype(np.int64, copy=False)
    yb = y_bins[valid].astype(np.int64, copy=False)
    in_range = (xb >= 0) & (xb < n_x) & (yb >= 0) & (yb < n_y)
    if not np.any(in_range):
        empty = np.full((n_x, n_y), np.nan)
        return np.nan, np.nan, np.nan, empty, np.full(n_x, np.nan), np.full(n_y, np.nan)

    counts = np.zeros((n_x, n_y), dtype=np.float64)
    np.add.at(counts, (xb[in_range], yb[in_range]), 1.0)
    counts += float(alpha)

    pxy = counts / counts.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    h_y = entropy_bits(py)
    with np.errstate(divide="ignore", invalid="ignore"):
        py_given_x = np.divide(pxy, px[:, None], where=px[:, None] > 0)
    h_y_given_x = 0.0
    for i in range(n_x):
        if px[i] > 0:
            h_y_given_x += px[i] * entropy_bits(py_given_x[i])

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(pxy, px[:, None] * py[None, :], where=(px[:, None] * py[None, :]) > 0)
        terms = np.where(pxy > 0, pxy * np.log2(ratio), 0.0)
    mi = float(np.nansum(terms))
    if mi < 0 and abs(mi) < 1e-12:
        mi = 0.0
    return max(mi, 0.0), h_y, h_y_given_x, pxy, px, py


def ig_kl_from_samples(
    y: np.ndarray,
    w: np.ndarray,
    bins: np.ndarray,
    alpha: float,
) -> Tuple[float, float, float, float, float]:
    on_mask = w > 0.5
    off_mask = ~on_mask
    if not np.any(on_mask) or not np.any(off_mask):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    y_bins = digitize_y(y, bins)
    x_bins = on_mask.astype(np.int64)
    ig, h_all, h_cond, pxy, px, py = joint_information_bits(
        y=y,
        x_bins=x_bins,
        y_bins=y_bins,
        n_x=2,
        n_y=bins.size - 1,
        alpha=alpha,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        py_given_x = np.divide(pxy, px[:, None], where=px[:, None] > 0)
        kl_off = float(np.nansum(np.where(py_given_x[0] > 0, py_given_x[0] * np.log2(py_given_x[0] / py), 0.0)))
        kl_on = float(np.nansum(np.where(py_given_x[1] > 0, py_given_x[1] * np.log2(py_given_x[1] / py), 0.0)))

    return ig, kl_on, kl_off, h_all, h_cond


def mi_discrete_from_samples(
    y: np.ndarray,
    w_bins: np.ndarray,
    bins: np.ndarray,
    alpha: float,
) -> Tuple[float, float, float]:
    valid = np.isfinite(y) & np.isfinite(w_bins)
    y = y[valid]
    w_bins = w_bins[valid].astype(int)
    if y.size == 0:
        return np.nan, np.nan, np.nan

    y_bins = digitize_y(y, bins)
    n_x = int(np.nanmax(w_bins)) + 1
    mi, h_all, h_cond, _, _, _ = joint_information_bits(
        y=y,
        x_bins=w_bins,
        y_bins=y_bins,
        n_x=n_x,
        n_y=bins.size - 1,
        alpha=alpha,
    )
    return mi, h_all, h_cond


def discretize_quantiles(values: np.ndarray, n_bins: int) -> Optional[np.ndarray]:
    vals = values[np.isfinite(values)]
    if vals.size < 2:
        return None
    edges = np.quantile(vals, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return None
    return edges


def digitize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if edges.size < 2:
        return np.full(values.shape, np.nan, dtype=np.float64)
    bins = np.digitize(values, edges[1:-1], right=False)
    return bins.astype(np.float64)


def block_permute(w: np.ndarray, block_len: int, rng: np.random.Generator) -> np.ndarray:
    n = w.size
    blocks = [(i, min(i + block_len, n)) for i in range(0, n, block_len)]
    order = rng.permutation(len(blocks))
    out = np.empty_like(w)
    pos = 0
    for idx in order:
        start, end = blocks[idx]
        block = w[start:end]
        out[pos : pos + block.size] = block
        pos += block.size
    return out


def build_lag_data_binary(
    t_phase: np.ndarray,
    w_full: np.ndarray,
    y_source: np.ndarray,
    lag_value: int,
    min_on: int,
    min_off: int,
    min_total: int,
    bin_method: str,
    min_bins: int,
    max_bins: int,
    fixed_bins: int,
) -> LagData:
    t_shift = t_phase + lag_value
    valid_time = (t_shift >= 0) & (t_shift < y_source.size)
    if not np.any(valid_time):
        return LagData(
            lag_value=lag_value,
            lag_index=-1,
            w_indices=np.array([], dtype=int),
            y=np.array([], dtype=np.float64),
            bins=None,
            n_on=0,
            n_off=0,
            n_total=0,
            p_on=np.nan,
            n_bins=0,
            q90_shift=np.nan,
            underpowered=True,
        )
    t_valid = t_phase[valid_time]
    w_valid = w_full[valid_time]
    y_valid = y_source[t_shift[valid_time]]
    finite = np.isfinite(y_valid) & np.isfinite(w_valid)
    y_valid = y_valid[finite].astype(np.float64, copy=False)
    w_valid = w_valid[finite].astype(np.float64, copy=False)
    w_indices = np.where(valid_time)[0][finite]

    on_mask = w_valid > 0.5
    n_on = int(on_mask.sum())
    n_off = int((~on_mask).sum())
    n_total = int(w_valid.size)
    p_on = float(on_mask.mean()) if n_total else np.nan

    underpowered = (n_on < min_on) or (n_off < min_off) or (n_total < min_total)

    if bin_method == "fixed":
        bins = compute_bins_fixed(y_valid, fixed_bins)
    else:
        bins = compute_bins_fd(y_valid, min_bins=min_bins, max_bins=max_bins)

    if bins is None or bins.size < 2:
        bins = None
        underpowered = True

    q90 = float(np.nanpercentile(y_valid, 90)) if n_total else np.nan
    p_exceed_on = float(np.mean(y_valid[on_mask] > q90)) if n_on else np.nan
    p_exceed_all = float(np.mean(y_valid > q90)) if n_total else np.nan
    q90_shift = p_exceed_on - p_exceed_all if np.isfinite(p_exceed_on) else np.nan

    return LagData(
        lag_value=lag_value,
        lag_index=-1,
        w_indices=w_indices,
        y=y_valid,
        bins=bins,
        n_on=n_on,
        n_off=n_off,
        n_total=n_total,
        p_on=p_on,
        n_bins=(bins.size - 1) if bins is not None else 0,
        q90_shift=q90_shift,
        underpowered=underpowered,
    )


def build_lag_data_continuous(
    t_phase: np.ndarray,
    w_full: np.ndarray,
    y_source: np.ndarray,
    lag_value: int,
    min_total: int,
    bin_method: str,
    min_bins: int,
    max_bins: int,
    fixed_bins: int,
) -> LagData:
    t_shift = t_phase + lag_value
    valid_time = (t_shift >= 0) & (t_shift < y_source.size)
    if not np.any(valid_time):
        return LagData(
            lag_value=lag_value,
            lag_index=-1,
            w_indices=np.array([], dtype=int),
            y=np.array([], dtype=np.float64),
            bins=None,
            n_on=0,
            n_off=0,
            n_total=0,
            p_on=np.nan,
            n_bins=0,
            q90_shift=np.nan,
            underpowered=True,
        )
    t_valid = t_phase[valid_time]
    w_valid = w_full[valid_time]
    y_valid = y_source[t_shift[valid_time]]
    finite = np.isfinite(y_valid) & np.isfinite(w_valid)
    y_valid = y_valid[finite].astype(np.float64, copy=False)
    w_valid = w_valid[finite].astype(np.float64, copy=False)
    w_indices = np.where(valid_time)[0][finite]

    n_total = int(w_valid.size)
    underpowered = n_total < min_total

    if bin_method == "fixed":
        bins = compute_bins_fixed(y_valid, fixed_bins)
    else:
        bins = compute_bins_fd(y_valid, min_bins=min_bins, max_bins=max_bins)

    if bins is None or bins.size < 2:
        bins = None
        underpowered = True

    return LagData(
        lag_value=lag_value,
        lag_index=-1,
        w_indices=w_indices,
        y=y_valid,
        bins=bins,
        n_on=0,
        n_off=0,
        n_total=n_total,
        p_on=np.nan,
        n_bins=(bins.size - 1) if bins is not None else 0,
        q90_shift=np.nan,
        underpowered=underpowered,
    )


def run_phase(
    phase_index: int,
    phase_label: int,
    lag_values: np.ndarray,
    lag_indices: np.ndarray,
    amp: Optional[np.ndarray],
    score: Optional[np.ndarray],
    external: Optional[np.ndarray],
    window_mode: str,
    w_full: np.ndarray,
    t_full: np.ndarray,
    w_edges: Optional[np.ndarray],
    null_mode: str,
    block_len: int,
    n_null: int,
    alpha: float,
    min_on: int,
    min_off: int,
    min_total: int,
    bin_method: str,
    min_bins: int,
    max_bins: int,
    fixed_bins: int,
    underpowered_policy: str,
    two_sided: bool,
    seed: int,
) -> PhaseResult:
    idx_phase = np.where(t_full)[0]
    if idx_phase.size == 0:
        empty = np.full(lag_values.size, np.nan, dtype=np.float64)
        return PhaseResult(
            phase=phase_label,
            lags=lag_values,
            metrics={"stat_bits": empty, "H_uncond": empty, "H_cond": empty, "KL_on_bits": empty, "KL_off_bits": empty},
            n_on=np.zeros(lag_values.size, dtype=np.int32),
            n_off=np.zeros(lag_values.size, dtype=np.int32),
            n_total=np.zeros(lag_values.size, dtype=np.int32),
            p_on=np.full(lag_values.size, np.nan, dtype=np.float64),
            n_bins=np.zeros(lag_values.size, dtype=np.int32),
            q90_shift=np.full(lag_values.size, np.nan, dtype=np.float64),
            underpowered=np.ones(lag_values.size, dtype=bool),
            max_stat_obs=np.nan,
            best_lag_obs=-1,
            max_stat_null=np.full(n_null, np.nan, dtype=np.float64),
            n_lags_used=0,
            min_n=np.nan,
            median_n=np.nan,
        )

    if window_mode == "continuous_intensity" and w_edges is None:
        empty = np.full(lag_values.size, np.nan, dtype=np.float64)
        return PhaseResult(
            phase=phase_label,
            lags=lag_values,
            metrics={"stat_bits": empty, "H_uncond": empty, "H_cond": empty, "KL_on_bits": empty, "KL_off_bits": empty},
            n_on=np.zeros(lag_values.size, dtype=np.int32),
            n_off=np.zeros(lag_values.size, dtype=np.int32),
            n_total=np.zeros(lag_values.size, dtype=np.int32),
            p_on=np.full(lag_values.size, np.nan, dtype=np.float64),
            n_bins=np.zeros(lag_values.size, dtype=np.int32),
            q90_shift=np.full(lag_values.size, np.nan, dtype=np.float64),
            underpowered=np.ones(lag_values.size, dtype=bool),
            max_stat_obs=np.nan,
            best_lag_obs=-1,
            max_stat_null=np.full(n_null, np.nan, dtype=np.float64),
            n_lags_used=0,
            min_n=np.nan,
            median_n=np.nan,
        )

    t_phase = idx_phase.astype(int)
    w_phase = w_full[idx_phase].astype(np.float64, copy=False)

    lag_data: List[LagData] = []
    for lag_value, lag_index in zip(lag_values, lag_indices):
        if amp is not None:
            y_source = amp[:, lag_index, phase_index]
        elif score is not None:
            y_source = score[:, lag_index]
        elif external is not None:
            y_source = external
        else:
            raise ValueError("No target source available.")
        if window_mode == "continuous_intensity":
            ld = build_lag_data_continuous(
                t_phase=t_phase,
                w_full=w_phase,
                y_source=y_source,
                lag_value=int(lag_value),
                min_total=min_total,
                bin_method=bin_method,
                min_bins=min_bins,
                max_bins=max_bins,
                fixed_bins=fixed_bins,
            )
        else:
            ld = build_lag_data_binary(
                t_phase=t_phase,
                w_full=w_phase,
                y_source=y_source,
                lag_value=int(lag_value),
                min_on=min_on,
                min_off=min_off,
                min_total=min_total,
                bin_method=bin_method,
                min_bins=min_bins,
                max_bins=max_bins,
                fixed_bins=fixed_bins,
            )
        ld.lag_index = int(lag_index)
        lag_data.append(ld)

    stat_obs = np.full(lag_values.size, np.nan, dtype=np.float64)
    kl_on = np.full(lag_values.size, np.nan, dtype=np.float64)
    kl_off = np.full(lag_values.size, np.nan, dtype=np.float64)
    h_all = np.full(lag_values.size, np.nan, dtype=np.float64)
    h_cond = np.full(lag_values.size, np.nan, dtype=np.float64)
    n_on = np.zeros(lag_values.size, dtype=np.int32)
    n_off = np.zeros(lag_values.size, dtype=np.int32)
    n_total = np.zeros(lag_values.size, dtype=np.int32)
    p_on = np.full(lag_values.size, np.nan, dtype=np.float64)
    n_bins = np.zeros(lag_values.size, dtype=np.int32)
    q90_shift = np.full(lag_values.size, np.nan, dtype=np.float64)
    underpowered = np.zeros(lag_values.size, dtype=bool)

    for i, ld in enumerate(lag_data):
        n_on[i] = ld.n_on
        n_off[i] = ld.n_off
        n_total[i] = ld.n_total
        p_on[i] = ld.p_on
        n_bins[i] = ld.n_bins
        q90_shift[i] = ld.q90_shift
        underpowered[i] = ld.underpowered

        if ld.bins is None or ld.y.size == 0:
            continue

        if window_mode == "continuous_intensity":
            if w_edges is None:
                continue
            w_bins = digitize_with_edges(w_phase[ld.w_indices], w_edges)
            mi, h_u, h_c = mi_discrete_from_samples(ld.y, w_bins, ld.bins, alpha)
            stat_obs[i] = mi
            h_all[i] = h_u
            h_cond[i] = h_c
        else:
            ig, k_on, k_off, h_u, h_c = ig_kl_from_samples(ld.y, w_phase[ld.w_indices], ld.bins, alpha)
            stat_obs[i] = ig
            kl_on[i] = k_on
            kl_off[i] = k_off
            h_all[i] = h_u
            h_cond[i] = h_c

    if underpowered_policy == "nan":
        stat_obs[underpowered] = np.nan
        kl_on[underpowered] = np.nan
        kl_off[underpowered] = np.nan
        h_all[underpowered] = np.nan
        h_cond[underpowered] = np.nan

    if underpowered_policy == "skip_phase" and np.any(underpowered):
        return PhaseResult(
            phase=phase_label,
            lags=lag_values,
            metrics={
                "stat_bits": np.full(lag_values.size, np.nan, dtype=np.float64),
                "H_uncond": np.full(lag_values.size, np.nan, dtype=np.float64),
                "H_cond": np.full(lag_values.size, np.nan, dtype=np.float64),
                "KL_on_bits": np.full(lag_values.size, np.nan, dtype=np.float64),
                "KL_off_bits": np.full(lag_values.size, np.nan, dtype=np.float64),
            },
            n_on=n_on,
            n_off=n_off,
            n_total=n_total,
            p_on=p_on,
            n_bins=n_bins,
            q90_shift=q90_shift,
            underpowered=underpowered,
            max_stat_obs=np.nan,
            best_lag_obs=-1,
            max_stat_null=np.full(n_null, np.nan, dtype=np.float64),
            n_lags_used=0,
            min_n=float(np.nanmin(n_total)) if n_total.size else np.nan,
            median_n=float(np.nanmedian(n_total)) if n_total.size else np.nan,
        )

    stat_for_inference = np.abs(stat_obs) if two_sided else stat_obs
    valid_for_inference = np.isfinite(stat_for_inference) & ~underpowered if underpowered_policy == "flag_only" else np.isfinite(stat_for_inference)
    if np.any(valid_for_inference):
        best_idx = int(np.nanargmax(np.where(valid_for_inference, stat_for_inference, np.nan)))
        max_stat_obs = float(stat_for_inference[best_idx])
        best_lag_obs = int(lag_values[best_idx])
        n_lags_used = int(valid_for_inference.sum())
    else:
        max_stat_obs = np.nan
        best_lag_obs = -1
        n_lags_used = 0

    max_stat_null = np.full(n_null, np.nan, dtype=np.float64)
    if n_null > 0 and n_lags_used > 0:
        rng = np.random.default_rng(seed + 1000 * phase_index)
        for r in range(n_null):
            if null_mode == "blockperm":
                w_perm = block_permute(w_phase, block_len, rng)
            elif null_mode == "circshift":
                shift = int(rng.integers(0, w_phase.size))
                w_perm = np.roll(w_phase, shift)
            else:
                raise ValueError(f"Unknown null mode: {null_mode}")

            if r == 0 and window_mode == "binary_gate":
                duty_diff = float(np.nanmean(w_perm) - np.nanmean(w_phase))
                if abs(duty_diff) > 1e-6:
                    print(f"WARNING: phase {phase_label} null duty cycle drift: {duty_diff:.6f}")

            stat_null = np.full(lag_values.size, np.nan, dtype=np.float64)
            for i, ld in enumerate(lag_data):
                if ld.bins is None or ld.y.size == 0:
                    continue
                if underpowered_policy == "flag_only" and ld.underpowered:
                    continue
                if window_mode == "continuous_intensity":
                    if w_edges is None:
                        continue
                    w_bins = digitize_with_edges(w_perm[ld.w_indices], w_edges)
                    mi, _, _ = mi_discrete_from_samples(ld.y, w_bins, ld.bins, alpha)
                    stat_null[i] = mi
                else:
                    ig, _, _, _, _ = ig_kl_from_samples(ld.y, w_perm[ld.w_indices], ld.bins, alpha)
                    stat_null[i] = ig
            stat_null_for_inference = np.abs(stat_null) if two_sided else stat_null
            max_stat_null[r] = float(np.nanmax(stat_null_for_inference)) if np.any(np.isfinite(stat_null_for_inference)) else np.nan

    return PhaseResult(
        phase=phase_label,
        lags=lag_values,
        metrics={
            "stat_bits": stat_obs,
            "H_uncond": h_all,
            "H_cond": h_cond,
            "KL_on_bits": kl_on,
            "KL_off_bits": kl_off,
        },
        n_on=n_on,
        n_off=n_off,
        n_total=n_total,
        p_on=p_on,
        n_bins=n_bins,
        q90_shift=q90_shift,
        underpowered=underpowered,
        max_stat_obs=max_stat_obs,
        best_lag_obs=best_lag_obs,
        max_stat_null=max_stat_null,
        n_lags_used=n_lags_used,
        min_n=float(np.nanmin(n_total)) if n_total.size else np.nan,
        median_n=float(np.nanmedian(n_total)) if n_total.size else np.nan,
    )


def bh_fdr(p_vals: np.ndarray) -> np.ndarray:
    p = np.asarray(p_vals, dtype=np.float64)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not np.any(valid):
        return out
    p_valid = p[valid]
    order = np.argsort(p_valid)
    p_sorted = p_valid[order]
    m = float(p_sorted.size)
    q_sorted = p_sorted * m / np.arange(1, p_sorted.size + 1, dtype=np.float64)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_valid = np.empty_like(p_valid, dtype=np.float64)
    q_valid[order] = np.minimum(q_sorted, 1.0)
    out[valid] = q_valid
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Phasebank information gain evaluation.")
    p.add_argument("--receiver_nc", required=True, help="Phasebank receiver NetCDF")
    p.add_argument("--test", required=True, help="Test window YYYY-MM-DD,YYYY-MM-DD")
    p.add_argument("--lags", default="", help="Lag list as '0,5,10' or 'start:stop:step'")
    p.add_argument("--phases", default="", help="Phase list as '1,2,3' (default all)")
    p.add_argument("--out_prefix", default="info_gain")
    p.add_argument("--target", choices=["amp_phase", "score_phase_matched", "external"], default="amp_phase")
    p.add_argument("--amp_var", default="z500_amp_weighted")
    p.add_argument("--score_var", default="z500_score_weighted")
    p.add_argument("--target_file", default="", help="External target NetCDF (required for target=external)")
    p.add_argument("--target_var", default="", help="Variable name for external target")

    p.add_argument("--gate_source", choices=["from_file", "recompute"], default="from_file")
    p.add_argument("--gate_percentile", type=float, default=90.0)
    p.add_argument("--gate_train", default="", help="Gate training window YYYY-MM-DD,YYYY-MM-DD")
    p.add_argument("--intensity_var", default="driver_intensity")
    p.add_argument("--smooth_days", type=int, default=10)
    p.add_argument("--no_smooth_centered", action="store_true")
    p.add_argument("--assume_intensity_already_smoothed", action="store_true")
    p.add_argument("--window_mode", choices=["binary_gate", "continuous_intensity"], default="binary_gate")
    p.add_argument("--phase_var", default="driver_phase_index")
    p.add_argument("--phase_is_one_based", action="store_true")
    p.add_argument("--condition_file", default="", help="Optional NetCDF holding a conditioning time series.")
    p.add_argument("--condition_var", default="", help="Variable used to restrict analysis samples by time.")
    p.add_argument(
        "--condition_values",
        default="",
        help="Comma-separated values to keep from condition_var; if omitted, uses nonzero/True samples.",
    )

    p.add_argument("--bin_method", choices=["freedman_diaconis", "fixed"], default="freedman_diaconis")
    p.add_argument("--bins_min", type=int, default=10)
    p.add_argument("--bins_max", type=int, default=80)
    p.add_argument("--fixed_bins", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--mi_method", choices=["hist_discretize", "ksg"], default="hist_discretize")
    p.add_argument("--w_bins", type=int, default=5)
    p.add_argument("--include_gate_edge", action="store_true", help="Include burst gate threshold as an additional W-bin edge (continuous mode)")

    p.add_argument("--null_mode", choices=["blockperm", "circshift"], default="blockperm")
    p.add_argument("--block_len", type=int, default=60)
    p.add_argument("--n_null", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--two_sided", action="store_true")

    p.add_argument("--min_on", type=int, default=20)
    p.add_argument("--min_off", type=int, default=100)
    p.add_argument("--min_total", type=int, default=200)
    p.add_argument("--underpowered_policy", choices=["nan", "flag_only", "skip_phase"], default="flag_only")

    p.add_argument("--float32", action="store_true")
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--backend", choices=["loky", "threading"], default="loky")
    p.add_argument("--save_null_dist", action="store_true")
    p.add_argument("--global_test", action="store_true")
    args = p.parse_args()

    if args.mi_method == "ksg":
        raise NotImplementedError("KSG MI is not implemented in this build.")

    ds_full = xr.open_dataset(args.receiver_nc)
    time_full = ds_full["time"].values

    if args.target == "external":
        if not args.target_file or not args.target_var:
            raise ValueError("--target_file and --target_var are required for target=external.")
        ds_t = xr.open_dataset(args.target_file)
        if args.target_var not in ds_t:
            raise KeyError(f"{args.target_var} not found in {args.target_file}")
        aligned = xr.align(ds_full["time"], ds_t[args.target_var], join="inner")
        time_aligned = aligned[0].values
        target_aligned = aligned[1].sel(time=time_aligned)
        ds_full = ds_full.sel(time=time_aligned)
        time_full = ds_full["time"].values
        external_full = target_aligned.values
    else:
        external_full = None

    condition_mask_full = np.ones(time_full.shape, dtype=bool)
    condition_values = parse_value_list(args.condition_values)
    if args.condition_var:
        if args.condition_file:
            ds_cond = xr.open_dataset(args.condition_file)
            if args.condition_var not in ds_cond:
                raise KeyError(f"{args.condition_var} not found in {args.condition_file}")
            cond_da = ds_cond[args.condition_var]
        else:
            if args.condition_var not in ds_full:
                raise KeyError(f"{args.condition_var} not found in receiver file.")
            cond_da = ds_full[args.condition_var]

        if "time" not in cond_da.coords:
            raise ValueError("Condition variable must have a time coordinate.")

        cond_aligned = cond_da.reindex(time=ds_full["time"].values)
        cond_vals = cond_aligned.values

        if condition_values:
            if np.asarray(cond_vals).dtype.kind in {"U", "S", "O"}:
                cond_str = np.asarray(cond_vals).astype(str)
                condition_mask_full = np.isin(cond_str, np.asarray(condition_values, dtype=object))
            else:
                keep_vals = np.asarray([float(v) for v in condition_values], dtype=np.float64)
                cond_num = np.asarray(cond_vals, dtype=np.float64)
                condition_mask_full = np.isfinite(cond_num) & np.isin(cond_num, keep_vals)
        else:
            if np.asarray(cond_vals).dtype.kind == "b":
                condition_mask_full = np.asarray(cond_vals, dtype=bool)
            else:
                cond_num = np.asarray(cond_vals, dtype=np.float64)
                condition_mask_full = np.isfinite(cond_num) & (cond_num != 0.0)

    test_start, test_end = parse_window(args.test)
    test_mask = (time_full >= np.datetime64(test_start)) & (time_full <= np.datetime64(test_end))
    if not np.any(test_mask):
        raise ValueError("No timestamps in test window after alignment.")

    gate_threshold = np.nan
    intensity_full = None
    if args.window_mode == "continuous_intensity" or args.gate_source == "recompute":
        if args.intensity_var not in ds_full:
            raise KeyError(f"{args.intensity_var} not found in receiver file.")
        intensity_full = ds_full[args.intensity_var].values.astype(np.float64, copy=False)

    gate_train_str = ""
    if args.gate_source == "recompute":
        if args.gate_train:
            gate_train = parse_window(args.gate_train)
        else:
            train_attr = ds_full.attrs.get("train_window", "").strip()
            if train_attr:
                gate_train = parse_window(train_attr)
            else:
                train_start = time_full.min()
                train_end = np.datetime64(test_start) - np.timedelta64(1, "D")
                if train_end < train_start:
                    raise ValueError("No pre-test period available for gate training. Provide --gate_train.")
                gate_train = (str(train_start)[:10], str(train_end)[:10])
        gate_train_str = ",".join(gate_train)
        gate_full, gate_threshold, intensity_smoothed = compute_gate(
            intensity=intensity_full,
            times=time_full,
            train_window=gate_train,
            percentile=args.gate_percentile,
            smooth_days=args.smooth_days,
            smooth_centered=not args.no_smooth_centered,
            assume_smoothed=args.assume_intensity_already_smoothed,
        )
        intensity_full = intensity_smoothed

        if "burst_gate" in ds_full:
            train_mask = (time_full >= np.datetime64(gate_train[0])) & (time_full <= np.datetime64(gate_train[1]))
            gate_file = ds_full["burst_gate"].values.astype(np.float64, copy=False)
            valid = np.isfinite(gate_full) & np.isfinite(gate_file) & train_mask
            if np.any(valid):
                mismatch = float(np.mean(gate_full[valid] != gate_file[valid]))
                print(f"Gate recompute mismatch rate (train window): {mismatch:.4f}")
    else:
        if "burst_gate" not in ds_full:
            raise KeyError("burst_gate not found in receiver file.")
        gate_full = ds_full["burst_gate"].values.astype(np.float64, copy=False)
        if args.gate_train:
            gate_train = parse_window(args.gate_train)
        else:
            train_attr = ds_full.attrs.get("train_window", "").strip()
            gate_train = parse_window(train_attr) if train_attr else ("", "")
        gate_train_str = ",".join(gate_train) if gate_train[0] else (args.gate_train or ds_full.attrs.get("train_window", ""))

    if args.window_mode == "continuous_intensity":
        if intensity_full is None:
            raise ValueError("Intensity series required for continuous window mode.")
        if not args.assume_intensity_already_smoothed and args.gate_source != "recompute":
            intensity_full = rolling_mean(intensity_full, args.smooth_days, not args.no_smooth_centered)

    # Retrieve or reuse the gate threshold so continuous W bins can include the
    # same edge as the binary gate in data-processing comparisons.
    gate_thr_attr = np.nan
    try:
        if np.isfinite(gate_threshold):
            gate_thr_attr = float(gate_threshold)
        elif "burst_gate_threshold" in ds_full.attrs:
            gate_thr_attr = float(ds_full.attrs.get("burst_gate_threshold"))
    except Exception:
        gate_thr_attr = np.nan

    ds = ds_full.isel(time=np.where(test_mask)[0])
    time = ds["time"].values

    if args.target == "amp_phase":
        if args.amp_var not in ds:
            raise KeyError(f"{args.amp_var} not found in receiver.")
        amp = ds[args.amp_var].values
        score = None
        external = None
    elif args.target == "score_phase_matched":
        if args.score_var not in ds:
            raise KeyError(f"{args.score_var} not found in receiver.")
        score = ds[args.score_var].values
        amp = None
        external = None
    else:
        amp = None
        score = None
        external = external_full[test_mask]

    phase_idx = ds[args.phase_var].values.astype(np.int16)
    if args.phase_is_one_based:
        phase_idx = np.where(phase_idx > 0, phase_idx - 1, phase_idx)
    lags_all = ds["lag"].values.astype(int)
    phases_all = ds["phase"].values.astype(int)
    gate = gate_full[test_mask].astype(np.float64, copy=False)
    condition_mask = condition_mask_full[test_mask]

    intensity = None
    if intensity_full is not None:
        intensity = intensity_full[test_mask].astype(np.float64, copy=False)

    if args.float32:
        if amp is not None:
            amp = amp.astype(np.float32)
        if score is not None:
            score = score.astype(np.float32)
        if external is not None:
            external = external.astype(np.float32)
        gate = gate.astype(np.float32)
        if intensity is not None:
            intensity = intensity.astype(np.float32)

    if args.lags:
        lag_values = np.array(parse_lags(args.lags), dtype=int)
    else:
        lag_values = lags_all.astype(int)
    lag_to_index = {int(v): i for i, v in enumerate(lags_all)}
    lag_indices = np.array([lag_to_index[int(v)] for v in lag_values], dtype=int)

    if args.phases:
        phase_values = np.array(parse_int_list(args.phases), dtype=int)
    else:
        phase_values = phases_all.astype(int)
    phase_to_index = {int(v): i for i, v in enumerate(phases_all)}
    phase_indices = np.array([phase_to_index[int(v)] for v in phase_values], dtype=int)

    if args.bin_method == "freedman_diaconis":
        bin_method = "freedman_diaconis"
    else:
        bin_method = "fixed"

    for phase_label, phase_index in zip(phase_values, phase_indices):
        phase_mask = (phase_idx == phase_index) & condition_mask
        if args.window_mode == "continuous_intensity":
            valid = phase_mask & np.isfinite(intensity)
            frac_on = np.nan
        else:
            valid = phase_mask & np.isfinite(gate)
            frac_on = float(np.nanmean(gate[valid])) if np.any(valid) else np.nan
        if np.isfinite(frac_on) and frac_on < 0.03:
            print(f"WARNING: phase {phase_label} burst gate fraction is low: {frac_on:.3f}")

    results = []
    for phase_label, phase_index in zip(phase_values, phase_indices):
        phase_mask = (phase_idx == phase_index) & condition_mask
        if args.window_mode == "continuous_intensity":
            window_series = intensity
            phase_valid = phase_mask & np.isfinite(window_series)
            w_edges = discretize_quantiles(window_series[phase_valid], args.w_bins)
            if w_edges is None:
                print(f"WARNING: phase {phase_label} has insufficient intensity variability for W bins.")
            # Optionally insert the gate threshold as an additional edge so G is a function of W-bins
            if args.include_gate_edge and w_edges is not None and np.isfinite(gate_thr_attr):
                try:
                    w_edges = np.unique(np.concatenate([w_edges, np.asarray([gate_thr_attr], dtype=np.float64)]))
                    w_edges = np.sort(w_edges)
                except Exception:
                    pass
        else:
            window_series = gate
            phase_valid = phase_mask & np.isfinite(window_series)
            w_edges = None

        results.append(
            delayed(run_phase)(
                phase_index=int(phase_index),
                phase_label=int(phase_label),
                lag_values=lag_values,
                lag_indices=lag_indices,
                amp=amp,
                score=score,
                external=external,
                window_mode=args.window_mode,
                w_full=window_series,
                t_full=phase_valid,
                w_edges=w_edges,
                null_mode=args.null_mode,
                block_len=args.block_len,
                n_null=args.n_null,
                alpha=args.alpha,
                min_on=args.min_on,
                min_off=args.min_off,
                min_total=args.min_total,
                bin_method=bin_method,
                min_bins=args.bins_min,
                max_bins=args.bins_max,
                fixed_bins=args.fixed_bins,
                underpowered_policy=args.underpowered_policy,
                two_sided=args.two_sided,
                seed=args.seed,
            )
        )

    phase_results = Parallel(n_jobs=args.n_jobs, backend=args.backend)(results)

    for res in phase_results:
        under_count = int(res.underpowered.sum())
        total_lags = res.lags.size
        print(
            f"Phase {res.phase}: lags_used={res.n_lags_used}/{total_lags}, "
            f"underpowered={under_count}, min_n={res.min_n:.1f}, median_n={res.median_n:.1f}"
        )

    rows: List[Dict[str, object]] = []
    for res in phase_results:
        for i, lag in enumerate(res.lags):
            row = {
                "phase": res.phase,
                "lag": int(lag),
                "n_on": int(res.n_on[i]),
                "n_off": int(res.n_off[i]),
                "n_total": int(res.n_total[i]),
                "p_on": float(res.p_on[i]),
                "H_uncond": float(res.metrics["H_uncond"][i]),
                "H_cond": float(res.metrics["H_cond"][i]),
                "IG_bits": float(res.metrics["stat_bits"][i]) if args.window_mode == "binary_gate" else np.nan,
                "MI_bits": float(res.metrics["stat_bits"][i]) if args.window_mode == "continuous_intensity" else np.nan,
                "KL_on_bits": float(res.metrics["KL_on_bits"][i]),
                "KL_off_bits": float(res.metrics["KL_off_bits"][i]),
                "q90_shift": float(res.q90_shift[i]),
                "n_bins": int(res.n_bins[i]),
                "underpowered": int(res.underpowered[i]),
                "bestlag_phase": int(lag == res.best_lag_obs),
                "window_mode": args.window_mode,
                "gate_source": args.gate_source,
                "gate_percentile": args.gate_percentile,
                "gate_threshold": gate_threshold,
                "gate_train": gate_train_str,
                "block_len": args.block_len,
                "n_null": args.n_null,
                "bin_method": args.bin_method,
                "bins_min": args.bins_min,
                "bins_max": args.bins_max,
                "fixed_bins": args.fixed_bins,
                "test_window": args.test,
                "target": args.target,
                "underpowered_policy": args.underpowered_policy,
                "two_sided": int(args.two_sided),
                "min_on": args.min_on,
                "min_off": args.min_off,
                "min_total": args.min_total,
                "condition_file": args.condition_file,
                "condition_var": args.condition_var,
                "condition_values": args.condition_values,
            }
            if args.window_mode == "continuous_intensity":
                row["w_bins"] = args.w_bins
            rows.append(row)

    obs_path = f"{args.out_prefix}_observed.csv"
    pd.DataFrame(rows).to_csv(obs_path, index=False)

    max_obs = np.array([res.max_stat_obs for res in phase_results], dtype=np.float64)
    best_lag_obs = np.array([res.best_lag_obs for res in phase_results], dtype=int)
    max_null = np.stack([res.max_stat_null for res in phase_results], axis=0)
    p_phase = (1.0 + np.sum(max_null >= max_obs[:, None], axis=1)) / (args.n_null + 1.0)
    p_phase = np.where(np.isfinite(max_obs), p_phase, np.nan)
    q_phase = bh_fdr(p_phase)

    summary_rows: List[Dict[str, object]] = []
    for res, maxobs, bestlag, p_val, q_val in zip(phase_results, max_obs, best_lag_obs, p_phase, q_phase):
        summary_rows.append(
            {
                "phase": int(res.phase),
                "max_stat_obs": float(maxobs),
                "best_lag_obs": int(bestlag),
                "p_phase": float(p_val),
                "q_phase": float(q_val),
                "n_lags_used": int(res.n_lags_used),
                "min_n": float(res.min_n),
                "median_n": float(res.median_n),
                "n_null": args.n_null,
                "block_len": args.block_len,
                "null_mode": args.null_mode,
                "window_mode": args.window_mode,
                "underpowered_policy": args.underpowered_policy,
                "two_sided": int(args.two_sided),
                "condition_file": args.condition_file,
                "condition_var": args.condition_var,
                "condition_values": args.condition_values,
            }
        )

    if args.global_test:
        if np.any(np.isfinite(max_obs)):
            global_obs = float(np.nanmax(max_obs))
            global_null = np.nanmax(max_null, axis=0)
            p_global = (1.0 + np.sum(global_null >= global_obs)) / (args.n_null + 1.0)
            for row in summary_rows:
                row["p_global"] = float(p_global)
        else:
            for row in summary_rows:
                row["p_global"] = np.nan

    summary_path = f"{args.out_prefix}_nullsummary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    if args.save_null_dist:
        np.savez(
            f"{args.out_prefix}_nulldist.npz",
            max_null=max_null,
            max_obs=max_obs,
            best_lag_obs=best_lag_obs,
            lags=lag_values,
            phases=phase_values,
            window_mode=args.window_mode,
        )

    if args.gate_source == "recompute" and args.window_mode == "binary_gate":
        mean_gate = float(np.nanmean(gate))
        print(f"Gate duty cycle (test window): {mean_gate:.4f}")

    print(f"Wrote {obs_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
