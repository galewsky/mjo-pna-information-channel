#!/usr/bin/env python3
"""
evaluate_phasebank_skill.py

Phase-resolved evaluation of a phasebank receiver with red-noise-safe nulls.
Computes observed metrics per (lag, phase) and phase-wise lag-hunting-corrected
p-values using block permutation or circular shift of the burst gate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed


# ---------------------------------------------
# Helper functions for histogram metrics (JSD)
# ---------------------------------------------
def _compute_bins_fd(values: np.ndarray, min_bins: int = 10, max_bins: int = 80) -> Optional[np.ndarray]:
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
        width = 2.0 * iqr * (vals.size ** (-1.0 / 3.0))
        if width <= 0:
            n_bins = int(np.clip(np.sqrt(vals.size), min_bins, max_bins))
        else:
            n_bins = int(np.ceil((vmax - vmin) / width))
            n_bins = int(np.clip(n_bins, min_bins, max_bins))
    return np.linspace(vmin, vmax, n_bins + 1, dtype=np.float64)


def _hist_probs(values: np.ndarray, bins: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    counts, _ = np.histogram(values, bins=bins)
    m = bins.size - 1
    return (counts + alpha) / (counts.sum() + alpha * m)


def _entropy_bits(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p)))


@dataclass
class PhaseResult:
    phase: int
    lags: np.ndarray
    stats: Dict[str, np.ndarray]
    n_on: np.ndarray
    n_off: np.ndarray
    maxstat_obs: float
    bestlag_obs: int
    maxstat_null: np.ndarray


def parse_window(expr: str) -> Tuple[str, str]:
    parts = [p.strip() for p in expr.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Window spec must be 'YYYY-MM-DD,YYYY-MM-DD'")
    return parts[0], parts[1]


def rolling_mean(values: np.ndarray, window: int, centered: bool) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float64, copy=False)
    series = pd.Series(values.astype(np.float64))
    return series.rolling(window=window, center=centered, min_periods=1).mean().values


def recompute_gate(
    intensity: np.ndarray,
    times: np.ndarray,
    train_window: Tuple[str, str],
    percentile: float,
    smooth_days: int,
    smooth_centered: bool,
    assume_smoothed: bool,
) -> Tuple[np.ndarray, float]:
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
    return gate, threshold


def rankdata_avg(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = 0.5 * (i + j + 2)
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    on_mask = labels.astype(bool)
    n_on = int(on_mask.sum())
    n_off = int(labels.size - n_on)
    if n_on == 0 or n_off == 0:
        return np.nan
    ranks = rankdata_avg(scores)
    rank_sum = float(ranks[on_mask].sum())
    return (rank_sum - n_on * (n_on + 1) / 2.0) / (n_on * n_off)


def compute_stats(
    a_k: np.ndarray,
    gate: np.ndarray,
    stat_name: str,
    *,
    a_all_idx: Optional[np.ndarray] = None,
    pfa: float = 0.1,
    alpha_hist: float = 0.5,
    q: float = 0.2,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    on_mask = gate > 0.5
    off_mask = ~on_mask

    a_on = np.where(on_mask[:, None], a_k, np.nan)
    a_off = np.where(off_mask[:, None], a_k, np.nan)

    n_on = np.sum(np.isfinite(a_on), axis=0).astype(np.int32)
    n_off = np.sum(np.isfinite(a_off), axis=0).astype(np.int32)

    mu_on = np.nanmean(a_on, axis=0)
    mu_off = np.nanmean(a_off, axis=0)
    diff = mu_on - mu_off

    std_off = np.nanstd(a_off, axis=0, ddof=1)
    snr = diff / std_off

    var_on = np.nanvar(a_on, axis=0, ddof=1)
    var_off = np.nanvar(a_off, axis=0, ddof=1)
    pooled = ((n_on - 1) * var_on + (n_off - 1) * var_off) / np.maximum(n_on + n_off - 2, 1)
    cohen_d = diff / np.sqrt(pooled)

    stats: Dict[str, np.ndarray] = {
        "diff": diff,
        "std_off": std_off,
        "snr": snr,
        "cohen_d": cohen_d,
    }

    # Extended stats: AUC already below; add SINR/selectivity, Pd@Pfa, JSD
    if stat_name in ("sinr", "selectivity"):
        if a_all_idx is None:
            raise ValueError("sinr/selectivity requires a_all_idx to compute across-phase energy.")
        # Compute sample-wise transforms, then aggregate like other stats
        # Shapes: a_k: (n_samples, n_lags), a_all_idx: (n_samples, n_lags, n_phase)
        ak2 = a_k**2
        energy_all = np.nansum(a_all_idx**2, axis=2)  # (n_samples, n_lags)
        if stat_name == "sinr":
            denom = np.maximum(energy_all - ak2, 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                x_mat = np.where(denom > 0.0, ak2 / denom, np.nan)
            stats["sinr"] = np.nanmean(np.where(on_mask[:, None], x_mat, np.nan), axis=0) - np.nanmean(
                np.where(off_mask[:, None], x_mat, np.nan), axis=0
            )
        else:  # selectivity
            with np.errstate(divide="ignore", invalid="ignore"):
                x_mat = np.where(energy_all > 0.0, np.abs(a_k) / np.sqrt(energy_all), np.nan)
            stats["selectivity"] = np.nanmean(np.where(on_mask[:, None], x_mat, np.nan), axis=0) - np.nanmean(
                np.where(off_mask[:, None], x_mat, np.nan), axis=0
            )

    if stat_name == "pd_at_pfa":
        # One-sided detection on amplitudes: threshold from OFF distribution
        pfa = float(pfa)
        pd_vals = np.full(a_k.shape[1], np.nan, dtype=np.float64)
        for j in range(a_k.shape[1]):
            off_vals = a_k[:, j][off_mask]
            off_vals = off_vals[np.isfinite(off_vals)]
            on_vals = a_k[:, j][on_mask]
            on_vals = on_vals[np.isfinite(on_vals)]
            if off_vals.size == 0 or on_vals.size == 0:
                continue
            thr = float(np.nanpercentile(off_vals, 100.0 * (1.0 - pfa)))
            pd_vals[j] = float(np.mean(on_vals > thr))
        stats["pd_at_pfa"] = pd_vals

    if stat_name == "jsd":
        jsd_vals = np.full(a_k.shape[1], np.nan, dtype=np.float64)
        for j in range(a_k.shape[1]):
            on_vals = a_k[:, j][on_mask]
            off_vals = a_k[:, j][off_mask]
            on_vals = on_vals[np.isfinite(on_vals)]
            off_vals = off_vals[np.isfinite(off_vals)]
            if on_vals.size < 2 or off_vals.size < 2:
                continue
            all_vals = np.concatenate([on_vals, off_vals])
            bins = _compute_bins_fd(all_vals)
            if bins is None or bins.size < 2:
                continue
            p_on = _hist_probs(on_vals, bins, alpha_hist)
            p_off = _hist_probs(off_vals, bins, alpha_hist)
            m = 0.5 * (p_on + p_off)
            h_m = _entropy_bits(m)
            h_on = _entropy_bits(p_on)
            h_off = _entropy_bits(p_off)
            jsd_vals[j] = h_m - 0.5 * (h_on + h_off)
        stats["jsd"] = jsd_vals

    if stat_name in ("outage", "lcr", "afd"):
        q = float(q)
        outage_vals = np.full(a_k.shape[1], np.nan, dtype=np.float64)
        lcr_vals = np.full(a_k.shape[1], np.nan, dtype=np.float64)
        afd_vals = np.full(a_k.shape[1], np.nan, dtype=np.float64)
        for j in range(a_k.shape[1]):
            series = a_k[:, j]
            off_vals = series[off_mask]
            off_vals = off_vals[np.isfinite(off_vals)]
            if off_vals.size == 0:
                continue
            thr = float(np.nanquantile(off_vals, q))
            on_vals = series[on_mask]
            on_vals = on_vals[np.isfinite(on_vals)]
            if on_vals.size == 0:
                continue
            below = on_vals < thr
            outage_vals[j] = float(np.mean(below))
            # LCR: count threshold crossings in on-series
            if on_vals.size > 1:
                transitions = np.count_nonzero(below[1:] != below[:-1])
                lcr_vals[j] = transitions / (on_vals.size - 1)
                # AFD: average run length below
                runs = []
                run = 0
                for b in below:
                    if b:
                        run += 1
                    elif run > 0:
                        runs.append(run)
                        run = 0
                if run > 0:
                    runs.append(run)
                afd_vals[j] = float(np.mean(runs)) if runs else 0.0
            else:
                lcr_vals[j] = 0.0
                afd_vals[j] = float(int(below[0]))
        stats["outage"] = outage_vals
        stats["lcr"] = lcr_vals
        stats["afd"] = afd_vals

    if stat_name == "auc":
        auc_vals = np.full(a_k.shape[1], np.nan, dtype=np.float64)
        for j in range(a_k.shape[1]):
            vals = a_k[:, j]
            valid = np.isfinite(vals)
            if not np.any(valid):
                continue
            auc_vals[j] = auc_from_scores(vals[valid], gate[valid])
        stats["auc"] = auc_vals

    median_on = np.nanmedian(a_on, axis=0)
    median_off = np.nanmedian(a_off, axis=0)
    stats["median_on"] = median_on
    stats["median_off"] = median_off
    stats["diff_median"] = median_on - median_off

    return stats, n_on, n_off


def lagged_phase_amplitudes(
    a: np.ndarray,
    source_idx: np.ndarray,
    lags: np.ndarray,
    phase_index: int,
    *,
    include_all_phases: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return receiver amplitudes paired as source day tau -> receiver day tau+L.

    The phasebank template for lag L and phase k describes a source in phase k
    on day tau and the receiver field on day tau+L. Event skill must therefore
    evaluate the gate label at tau against the matched-filter amplitude at
    tau+L. Earlier versions used a[source day, L, k], which mixed source-day
    labels with same-day receiver amplitudes for nonzero lags.
    """
    n_src = source_idx.size
    n_lag = lags.size
    out = np.full((n_src, n_lag), np.nan, dtype=np.float64)
    out_all = (
        np.full((n_src, n_lag, a.shape[2]), np.nan, dtype=np.float64)
        if include_all_phases
        else None
    )
    for j, lag in enumerate(lags.astype(int)):
        receiver_idx = source_idx + lag
        valid = (receiver_idx >= 0) & (receiver_idx < a.shape[0])
        if not np.any(valid):
            continue
        out[valid, j] = a[receiver_idx[valid], j, phase_index]
        if out_all is not None:
            out_all[valid, j, :] = a[receiver_idx[valid], j, :]
    return out, out_all


def stat_for_max(stat_name: str, stats: Dict[str, np.ndarray], two_sided: bool) -> np.ndarray:
    if stat_name == "auc":
        base = stats["auc"] - 0.5
    elif stat_name in ("jsd", "pd_at_pfa"):
        # Non-negative by definition
        base = stats[stat_name]
        return base
    elif stat_name in ("outage", "lcr", "afd"):
        # Smaller is better; invert for max selection
        base = -stats[stat_name]
        return base
    else:
        base = stats[stat_name]
    if two_sided:
        return np.abs(base)
    return base


def block_permute_gate(gate: np.ndarray, block_len: int, rng: np.random.Generator) -> np.ndarray:
    n = gate.size
    blocks = [(i, min(i + block_len, n)) for i in range(0, n, block_len)]
    order = rng.permutation(len(blocks))
    out = np.empty_like(gate)
    pos = 0
    for idx in order:
        start, end = blocks[idx]
        block = gate[start:end]
        out[pos : pos + block.size] = block
        pos += block.size
    return out


def run_phase(
    phase_index: int,
    phase_label: int,
    lags: np.ndarray,
    a: np.ndarray,
    p: np.ndarray,
    gate: np.ndarray,
    null_mode: str,
    block_len: int,
    n_null: int,
    stat_name: str,
    two_sided: bool,
    seed: int,
    pfa: float,
    alpha_hist: float,
    q: float,
) -> PhaseResult:
    idx = np.where((p == phase_index) & np.isfinite(gate))[0]
    if idx.size == 0:
        empty = np.full(lags.size, np.nan, dtype=np.float64)
        return PhaseResult(
            phase=phase_label,
            lags=lags,
            stats={stat_name: empty},
            n_on=np.zeros(lags.size, dtype=np.int32),
            n_off=np.zeros(lags.size, dtype=np.int32),
            maxstat_obs=np.nan,
            bestlag_obs=-1,
            maxstat_null=np.full(n_null, np.nan, dtype=np.float64),
        )

    include_all = stat_name in ("sinr", "selectivity")
    a_k, a_all_idx = lagged_phase_amplitudes(
        a,
        idx,
        lags,
        phase_index,
        include_all_phases=include_all,
    )
    g_k = gate[idx]
    stats, n_on, n_off = compute_stats(
        a_k,
        g_k,
        stat_name,
        a_all_idx=a_all_idx,
        pfa=pfa,
        alpha_hist=alpha_hist,
    )
    stat_vec = stat_for_max(stat_name, stats, two_sided)
    maxstat_obs = float(np.nanmax(stat_vec))
    bestlag_obs = int(lags[int(np.nanargmax(stat_vec))])

    rng = np.random.default_rng(seed + 1000 * phase_index)
    maxstat_null = np.full(n_null, np.nan, dtype=np.float64)
    for r in range(n_null):
        if null_mode == "blockperm":
            g_null = block_permute_gate(g_k, block_len, rng)
        elif null_mode == "circshift":
            shift = int(rng.integers(0, g_k.size))
            g_null = np.roll(g_k, shift)
        else:
            raise ValueError(f"Unknown null mode: {null_mode}")
        stats_null, _, _ = compute_stats(
            a_k,
            g_null,
            stat_name,
            a_all_idx=a_all_idx,
            pfa=pfa,
            alpha_hist=alpha_hist,
        )
        stat_null = stat_for_max(stat_name, stats_null, two_sided)
        maxstat_null[r] = np.nanmax(stat_null)

    return PhaseResult(
        phase=phase_label,
        lags=lags,
        stats=stats,
        n_on=n_on,
        n_off=n_off,
        maxstat_obs=maxstat_obs,
        bestlag_obs=bestlag_obs,
        maxstat_null=maxstat_null,
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
    p = argparse.ArgumentParser(description="Evaluate phasebank receiver skill with red-noise-safe nulls.")
    p.add_argument("--receiver_nc", required=True, help="Path to phasebank receiver NetCDF")
    p.add_argument("--train", default="", help="Train window (YYYY-MM-DD,YYYY-MM-DD)")
    p.add_argument("--test", default="", help="Test window (YYYY-MM-DD,YYYY-MM-DD)")
    p.add_argument("--amp_var", default="z500_amp_weighted", help="Amplitude cube variable to evaluate")
    p.add_argument("--gate_var", default="burst_gate", help="Binary gate variable")
    p.add_argument("--gate_source", choices=["from_file", "recompute"], default="from_file")
    p.add_argument("--gate_percentile", type=float, default=85.0)
    p.add_argument("--gate_train", default="", help="Gate training window YYYY-MM-DD,YYYY-MM-DD")
    p.add_argument("--intensity_var", default="driver_intensity")
    p.add_argument("--smooth_days", type=int, default=10)
    p.add_argument("--no_smooth_centered", action="store_true")
    p.add_argument("--assume_intensity_already_smoothed", action="store_true")
    p.add_argument("--phase_var", default="driver_phase_index", help="Phase index variable")
    p.add_argument("--phase_is_one_based", action="store_true", help="Interpret phase_var as 1..N instead of 0..N-1")
    p.add_argument("--null", choices=["blockperm", "circshift"], default="blockperm")
    p.add_argument("--block_len", type=int, default=60)
    p.add_argument("--n_null", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--stat",
        choices=["snr", "diff", "cohen_d", "auc", "sinr", "selectivity", "pd_at_pfa", "jsd", "outage", "lcr", "afd"],
        default="snr",
    )
    p.add_argument("--pfa", type=float, default=0.1, help="False alarm rate for Pd@Pfa metric")
    p.add_argument("--q", type=float, default=0.2, help="Quantile threshold for outage/LCR/AFD (on unconditional Y)")
    p.add_argument("--two_sided", action="store_true", help="Use abs(stat) for lag hunting")
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--backend", choices=["loky", "threading"], default="loky")
    p.add_argument("--float32", action="store_true", help="Cast arrays to float32")
    p.add_argument("--output_prefix", default="phasebank_eval")
    p.add_argument("--save_null_dist", action="store_true", help="Save null max stats to NPZ")
    p.add_argument("--global_test", action="store_true", help="Compute global max across phases/lags")
    args = p.parse_args()

    ds = xr.open_dataset(args.receiver_nc)
    if args.amp_var not in ds:
        raise KeyError(f"{args.amp_var} not found; phase-resolved evaluation requires amplitudes.")
    if args.gate_source == "from_file" and args.gate_var not in ds:
        raise KeyError(f"{args.gate_var} not found in receiver.")
    if args.phase_var not in ds:
        raise KeyError(f"{args.phase_var} not found in receiver.")

    time = ds["time"].values
    if args.test:
        test_start, test_end = parse_window(args.test)
        test_mask = (time >= np.datetime64(test_start)) & (time <= np.datetime64(test_end))
    else:
        test_mask = np.ones(time.shape, dtype=bool)

    if args.gate_source == "recompute":
        if args.intensity_var not in ds:
            raise KeyError(f"{args.intensity_var} not found in receiver file.")
        if args.gate_train:
            gate_train = parse_window(args.gate_train)
        else:
            train_attr = ds.attrs.get("train_window", "").strip()
            gate_train = parse_window(train_attr) if train_attr else ("", "")
        if not gate_train[0]:
            raise ValueError("Provide --gate_train or store train_window in the receiver file.")
        gate_full, gate_threshold = recompute_gate(
            ds[args.intensity_var].values.astype(np.float64, copy=False),
            time,
            gate_train,
            args.gate_percentile,
            args.smooth_days,
            not args.no_smooth_centered,
            args.assume_intensity_already_smoothed,
        )
    else:
        gate_full = ds[args.gate_var].values
        gate_threshold = np.nan
        gate_train = ("", "")

    ds_test = ds.isel(time=np.where(test_mask)[0])
    a = ds_test[args.amp_var].values
    gate = gate_full[test_mask]
    phase_idx = ds_test[args.phase_var].values.astype(np.int16)
    if args.phase_is_one_based:
        phase_idx = np.where(phase_idx > 0, phase_idx - 1, phase_idx)
    lags = ds_test["lag"].values.astype(int)
    phases = ds_test["phase"].values.astype(int)

    if args.float32:
        a = a.astype(np.float32)
        gate = gate.astype(np.float32)

    if np.any(phase_idx < 0):
        valid = phase_idx >= 0
        a = a[valid, :, :]
        gate = gate[valid]
        phase_idx = phase_idx[valid]

    frac_on = []
    for k in range(phases.size):
        idx_k = (phase_idx == k) & np.isfinite(gate)
        frac = float(np.nanmean(gate[idx_k])) if np.any(idx_k) else np.nan
        frac_on.append(frac)
    for phase_label, frac in zip(phases, frac_on):
        if np.isfinite(frac) and frac < 0.03:
            print(f"WARNING: phase {phase_label} burst gate fraction is low: {frac:.3f}")

    results = Parallel(n_jobs=args.n_jobs, backend=args.backend)(
        delayed(run_phase)(
            phase_index=k,
            phase_label=int(phases[k]),
            lags=lags,
            a=a,
            p=phase_idx,
            gate=gate,
            null_mode=args.null,
            block_len=args.block_len,
            n_null=args.n_null,
            stat_name=args.stat,
            two_sided=args.two_sided,
            seed=args.seed,
            pfa=args.pfa,
            alpha_hist=0.5,
            q=args.q,
        )
        for k in range(phases.size)
    )

    rows: List[Dict[str, object]] = []
    for res in results:
        stats = res.stats
        for i, lag in enumerate(res.lags):
            rows.append(
                {
                    "phase": res.phase,
                    "lag": int(lag),
                    "n_on": int(res.n_on[i]),
                    "n_off": int(res.n_off[i]),
                    "diff": float(stats["diff"][i]),
                    "std_off": float(stats["std_off"][i]),
                    "snr": float(stats["snr"][i]),
                    "cohen_d": float(stats["cohen_d"][i]),
                    "median_on": float(stats["median_on"][i]),
                    "median_off": float(stats["median_off"][i]),
                    "diff_median": float(stats["diff_median"][i]),
                    "auc": float(stats.get("auc", np.nan)[i]) if "auc" in stats else np.nan,
                    "sinr": float(stats.get("sinr", np.nan)[i]) if "sinr" in stats else np.nan,
                    "selectivity": float(stats.get("selectivity", np.nan)[i]) if "selectivity" in stats else np.nan,
                    "pd_at_pfa": float(stats.get("pd_at_pfa", np.nan)[i]) if "pd_at_pfa" in stats else np.nan,
                    "jsd": float(stats.get("jsd", np.nan)[i]) if "jsd" in stats else np.nan,
                    "outage": float(stats.get("outage", np.nan)[i]) if "outage" in stats else np.nan,
                    "lcr": float(stats.get("lcr", np.nan)[i]) if "lcr" in stats else np.nan,
                    "afd": float(stats.get("afd", np.nan)[i]) if "afd" in stats else np.nan,
                    "bestlag_phase": int(lag == res.bestlag_obs),
                    "train_window": args.train,
                    "test_window": args.test,
                    "block_len": args.block_len,
                    "n_null": args.n_null,
                    "null_mode": args.null,
                    "stat": args.stat,
                    "two_sided": int(args.two_sided),
                    "amp_var": args.amp_var,
                    "gate_var": args.gate_var,
                    "gate_source": args.gate_source,
                    "gate_percentile": args.gate_percentile,
                    "gate_threshold": gate_threshold,
                    "gate_train": ",".join(gate_train) if gate_train[0] else "",
                    "phase_var": args.phase_var,
                    "pfa": args.pfa,
                    "q": args.q,
                }
            )

    df_obs = pd.DataFrame(rows)
    obs_path = f"{args.output_prefix}_observed.csv"
    df_obs.to_csv(obs_path, index=False)

    maxstat_obs = np.array([res.maxstat_obs for res in results], dtype=np.float64)
    bestlag_obs = np.array([res.bestlag_obs for res in results], dtype=int)
    maxstat_null = np.stack([res.maxstat_null for res in results], axis=0)
    p_phase = (1.0 + np.sum(maxstat_null >= maxstat_obs[:, None], axis=1)) / (args.n_null + 1.0)
    q_phase = bh_fdr(p_phase)

    summary_rows: List[Dict[str, object]] = []
    for phase_label, maxobs, bestlag, p_val, q_val in zip(phases, maxstat_obs, bestlag_obs, p_phase, q_phase):
        summary_rows.append(
            {
                "phase": int(phase_label),
                "maxstat_obs": float(maxobs),
                "best_lag_obs": int(bestlag),
                "p_phase": float(p_val),
                "q_phase": float(q_val),
                "n_null": args.n_null,
                "null_mode": args.null,
                "block_len": args.block_len,
                "stat": args.stat,
                "two_sided": int(args.two_sided),
                "amp_var": args.amp_var,
                "gate_var": args.gate_var,
                "gate_source": args.gate_source,
                "gate_percentile": args.gate_percentile,
                "gate_threshold": gate_threshold,
                "gate_train": ",".join(gate_train) if gate_train[0] else "",
                "phase_var": args.phase_var,
            }
        )

    if args.global_test:
        global_obs = float(np.nanmax(maxstat_obs))
        global_null = np.nanmax(maxstat_null, axis=0)
        p_global = (1.0 + np.sum(global_null >= global_obs)) / (args.n_null + 1.0)
        for row in summary_rows:
            row["p_global"] = float(p_global)

    summary_path = f"{args.output_prefix}_nullsummary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    if args.save_null_dist:
        npz_path = f"{args.output_prefix}_null_dist.npz"
        np.savez(
            npz_path,
            maxstat_null=maxstat_null,
            maxstat_obs=maxstat_obs,
            bestlag_obs=bestlag_obs,
            lags=lags,
            phases=phases,
            stat=args.stat,
            two_sided=int(args.two_sided),
        )

    print(f"Wrote {obs_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
