#!/usr/bin/env python3
"""
Generate manuscript figures for the phasebank MJO-PNA teleconnection study.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from matplotlib.lines import Line2D


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", default="manuscript_results", help="Directory with analysis CSV outputs")
    p.add_argument("--output_dir", default="manuscript_results/figures", help="Directory to save figures")
    p.add_argument(
        "--receiver_nc",
        default="z500_receiver_phasebank_Bint_domain120_300_20_80_train1991_2010.nc",
        help="Phasebank receiver NetCDF for template/anomaly figures",
    )
    p.add_argument("--mjo_timeseries", default="mjo_index_from_surprisal_timeseries.nc", help="MJO index NetCDF")
    p.add_argument(
        "--surprisal_nc",
        default="olr_tropics_pointwise_info.nc",
        help="NetCDF with olr_surprisal(time,lat,lon)",
    )
    p.add_argument(
        "--surprisal_phase_nc",
        default="olr_surprisal_phase_composites.nc",
        help="Precomputed NetCDF with per-phase mean surprisal composites",
    )
    p.add_argument("--mjo_phase_composites", default="mjo_phase_composites.png", help="Existing phase-composite figure")
    p.add_argument(
        "--mjo_burst_climatology",
        default="mjo_surprisal_burst_climatology.png",
        help="Existing burst-climatology figure",
    )
    return p.parse_args()


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "savefig.dpi": 220,
        }
    )


def plot_telecom_summary(results_dir: Path, out_path: Path) -> None:
    cap = pd.read_csv(results_dir / "telecom_capacity_throughput.csv").set_index("phase")
    eff = pd.read_csv(results_dir / "telecom_coding_efficiency.csv").set_index("phase")
    outg = pd.read_csv(results_dir / "telecom_outage_table.csv").set_index("phase")
    phases = np.arange(1, 9)

    spectral = cap.loc[phases, "spectral_eff_bits_per_symbol"].values.astype(np.float64)
    throughput = cap.loc[phases, "throughput_bits_per_day"].values.astype(np.float64)
    coding = eff.loc[phases, "coding_efficiency"].values.astype(np.float64)
    outage = outg.loc[phases, "outage_at_pfa"].values.astype(np.float64)

    colors = ["#4c78a8"] * phases.size
    colors[4] = "#d62728"

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2), constrained_layout=True)
    panels = [
        ("A", "Spectral efficiency", spectral, "bits/symbol", None),
        ("B", "Realized throughput", throughput, "bits/day", None),
        ("C", "Coding efficiency", coding, "fraction of H_L(G|k)", None),
        ("D", "Outage probability", outage, "probability", (0.0, 1.0)),
    ]

    for ax, (panel, title, values, ylabel, ylim) in zip(axes.ravel(), panels):
        ax.bar(phases, values, color=colors, edgecolor="white", linewidth=0.7)
        ax.set_title(title)
        ax.set_xlabel("MJO phase")
        ax.set_ylabel(ylabel)
        ax.set_xticks(phases)
        ax.grid(axis="y", alpha=0.25)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(0.0, max(0.01, 1.08 * np.nanmax(values)))
        ax.text(
            -0.12,
            1.08,
            panel,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
        )

    fig.suptitle("Four-Metric Phase Scorecard")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_diversity_gain(results_dir: Path, out_path: Path) -> None:
    df = pd.read_csv(results_dir / "telecom_diversity_gain.csv").set_index("phase")
    phases = np.arange(1, 9)
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.bar(phases, df.loc[phases, "diversity_gain_bits"], color="#2ca02c")
    ax.axhline(0.0, color="0.2", lw=0.8)
    ax.set_title("Diversity Gain from (W_t, W_{t-1}) vs W_t")
    ax.set_xlabel("MJO phase")
    ax.set_ylabel("ΔMI (bits)")
    ax.set_xticks(phases)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _phase_from_radians(rad: np.ndarray) -> np.ndarray:
    angle = np.mod(rad, 2.0 * np.pi)
    phase = np.floor(angle / (np.pi / 4.0)).astype(int) + 1
    phase = np.clip(phase, 1, 8)
    return phase


def _load_heatmap(csv_path: Path, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    mat = df.pivot(index="phase", columns="lag", values=value_col).sort_index()
    return mat


def _rolling_mean_centered(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float64)
    series = pd.Series(values.astype(np.float64))
    return series.rolling(window=window, center=True, min_periods=1).mean().values


def _compute_fd_bins(values: np.ndarray, min_bins: int = 10, max_bins: int = 80) -> np.ndarray | None:
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


def _hist_probs(values: np.ndarray, bins: np.ndarray, alpha: float) -> np.ndarray:
    counts, _ = np.histogram(values, bins=bins)
    m = bins.size - 1
    return (counts + alpha) / (counts.sum() + alpha * m)


def _entropy_bits(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p)))


def _phase_lag_continuous_samples(
    ds: xr.Dataset,
    phase_label: int,
    lag: int,
    test_start: str = "2011-01-01",
    test_end: str = "2020-12-31",
    smooth_days: int = 10,
    w_bins: int = 5,
    alpha: float = 0.5,
) -> Dict[str, object]:
    time_full = pd.to_datetime(ds["time"].values)
    test_mask = (time_full >= pd.Timestamp(test_start)) & (time_full <= pd.Timestamp(test_end))
    ds_t = ds.isel(time=np.where(test_mask)[0])
    time = pd.to_datetime(ds_t["time"].values)

    phase_idx = ds_t["driver_phase_index"].values.astype(np.int16)
    intensity_raw = ds_t["driver_intensity"].values.astype(np.float64)
    intensity = _rolling_mean_centered(intensity_raw, smooth_days)
    y_source = ds_t["z500_amp_weighted"].sel(lag=lag, phase=phase_label).values.astype(np.float64)

    phase_index = phase_label - 1
    phase_mask = (phase_idx == phase_index) & np.isfinite(intensity)
    t_phase = np.where(phase_mask)[0].astype(int)
    t_shift = t_phase + int(lag)
    in_bounds = (t_shift >= 0) & (t_shift < y_source.size)
    t_phase = t_phase[in_bounds]
    t_shift = t_shift[in_bounds]

    w_valid = intensity[t_phase]
    y_valid = y_source[t_shift]
    finite = np.isfinite(w_valid) & np.isfinite(y_valid)
    t_phase = t_phase[finite]
    t_shift = t_shift[finite]
    w_valid = w_valid[finite]
    y_valid = y_valid[finite]

    w_edges = np.quantile(intensity[phase_mask], np.linspace(0.0, 1.0, w_bins + 1))
    w_edges = np.unique(w_edges)
    if w_edges.size < 2:
        raise ValueError("Insufficient variability in phase-conditioned intensity to define W bins.")
    w_bin = np.digitize(w_valid, w_edges[1:-1], right=False).astype(int)

    y_bins = _compute_fd_bins(y_valid, min_bins=10, max_bins=80)
    if y_bins is None or y_bins.size < 2:
        raise ValueError("Unable to construct Y histogram bins for worked-example diagnostics.")

    p_all = _hist_probs(y_valid, y_bins, alpha)
    h_all = _entropy_bits(p_all)
    counts_w = np.bincount(w_bin, minlength=np.max(w_bin) + 1)
    total = int(counts_w.sum())
    if total == 0:
        raise ValueError("No valid samples for worked-example diagnostics.")

    h_cond = 0.0
    h_by_bin = np.full(counts_w.size, np.nan, dtype=np.float64)
    mean_by_bin = np.full(counts_w.size, np.nan, dtype=np.float64)
    std_by_bin = np.full(counts_w.size, np.nan, dtype=np.float64)
    for b, count in enumerate(counts_w):
        if count == 0:
            continue
        yy = y_valid[w_bin == b]
        p_yw = _hist_probs(yy, y_bins, alpha)
        h_b = _entropy_bits(p_yw)
        h_by_bin[b] = h_b
        mean_by_bin[b] = float(np.nanmean(yy))
        std_by_bin[b] = float(np.nanstd(yy, ddof=1))
        h_cond += (count / total) * h_b
    mi = h_all - h_cond

    return {
        "dataset_test": ds_t,
        "time_test": time,
        "phase_mask": phase_mask,
        "source_indices": t_phase,
        "receiver_indices": t_shift,
        "source_times": time[t_phase],
        "receiver_times": time[t_shift],
        "intensity": intensity,
        "y_source_full": y_source,
        "w": w_valid,
        "y": y_valid,
        "w_edges": w_edges,
        "w_bin": w_bin,
        "y_bins": y_bins,
        "counts_w": counts_w,
        "h_by_bin": h_by_bin,
        "mean_by_bin": mean_by_bin,
        "std_by_bin": std_by_bin,
        "H_all": h_all,
        "H_cond": h_cond,
        "MI": mi,
    }


def plot_workflow(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    ax.set_axis_off()

    boxes = [
        (0.02, 0.60, 0.15, 0.28, "1. OLR\nanomalies"),
        (0.20, 0.60, 0.17, 0.28, "2. Surprisal\nconversion"),
        (0.40, 0.60, 0.17, 0.28, "3. WK MJO\nindex + phase"),
        (0.60, 0.60, 0.18, 0.28, "4. Z500 phasebank\nreceiver"),
        (0.81, 0.60, 0.17, 0.28, "5. Info gain / MI\nlag scans"),
        (0.30, 0.16, 0.20, 0.28, "6. Null tests\n(block perm)"),
        (0.56, 0.16, 0.26, 0.28, "7. ENSO-conditioned\nteleconnection skill"),
    ]
    for x, y, w, h, text in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", linewidth=1.2, facecolor="#e9f2fb")
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center")

    arrows = [
        ((0.17, 0.74), (0.20, 0.74)),
        ((0.37, 0.74), (0.40, 0.74)),
        ((0.57, 0.74), (0.60, 0.74)),
        ((0.78, 0.74), (0.81, 0.74)),
        ((0.89, 0.60), (0.74, 0.44)),
        ((0.50, 0.30), (0.56, 0.30)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.2, color="0.3"))

    ax.text(0.02, 0.05, "Pipeline scope for this manuscript: 1991-2020 receiver, 2011-2020 independent test window")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_surprisal_fields(surprisal_nc: Path, mjo_timeseries: Path, out_path: Path) -> None:
    ds_s = xr.open_dataset(surprisal_nc)
    ds_i = xr.open_dataset(mjo_timeseries)
    try:
        if "olr_surprisal" not in ds_s:
            raise KeyError(f"{surprisal_nc} is missing 'olr_surprisal'")
        if "mjo_amp" not in ds_i or "burst_mask" not in ds_i:
            raise KeyError(f"{mjo_timeseries} must contain 'mjo_amp' and 'burst_mask'")

        # Plot on a 0.5-degree grid for faster figure generation while preserving structure.
        surp = ds_s["olr_surprisal"].isel(lat=slice(None, None, 2), lon=slice(None, None, 2))
        amp, burst = xr.align(ds_i["mjo_amp"], ds_i["burst_mask"], join="inner")
        amp_vals = amp.values.astype(float)
        burst_vals = burst.values.astype(bool)
        valid = burst_vals & np.isfinite(amp_vals)
        if not np.any(valid):
            raise ValueError("No finite burst days found in mjo_timeseries.")

        peak_idx = int(np.nanargmax(np.where(valid, amp_vals, np.nan)))
        peak_time = pd.to_datetime(amp["time"].values[peak_idx])
        peak_amp = float(amp_vals[peak_idx])

        climatology = surp.mean("time")
        snapshot = surp.sel(time=peak_time, method="nearest")
        snapshot_time = pd.to_datetime(snapshot["time"].values)

        lon = climatology["lon"].values.astype(float)
        lat = climatology["lat"].values.astype(float)
        cvals = climatology.values.astype(float)
        svals = snapshot.values.astype(float)
        finite_snapshot = np.isfinite(svals)
        snapshot_max = float(np.nanmax(svals))
        max_flat = int(np.nanargmax(svals))
        max_ilat, max_ilon = np.unravel_index(max_flat, svals.shape)
        max_lat = float(snapshot["lat"].values[max_ilat])
        max_lon = float(snapshot["lon"].values[max_ilon])

        local_series = surp.isel(lat=max_ilat, lon=max_ilon).load()
        local_vals = local_series.values.astype(float)
        local_time = pd.to_datetime(local_series["time"].values)
        local_finite = np.isfinite(local_vals)

        def _season(month: int) -> str:
            if month in (12, 1, 2):
                return "DJF"
            if month in (3, 4, 5):
                return "MAM"
            if month in (6, 7, 8):
                return "JJA"
            return "SON"

        peak_season = _season(snapshot_time.month)
        local_season_mask = np.array([_season(int(m)) == peak_season for m in local_time.month]) & local_finite
        amp_finite = np.isfinite(amp_vals)
        diag_rows = [
            {
                "metric": "source_amplitude_peak",
                "value": peak_amp,
                "n": int(np.sum(amp_finite)),
                "count_ge": int(np.sum(amp_vals[amp_finite] >= peak_amp)),
                "count_gt": int(np.sum(amp_vals[amp_finite] > peak_amp)),
                "frequency_ge": float(np.sum(amp_vals[amp_finite] >= peak_amp) / np.sum(amp_finite)),
                "lat": np.nan,
                "lon": np.nan,
            },
            {
                "metric": "pointwise_snapshot_max",
                "value": snapshot_max,
                "n": int(np.sum(local_finite)),
                "count_ge": int(np.sum(local_vals[local_finite] >= snapshot_max)),
                "count_gt": int(np.sum(local_vals[local_finite] > snapshot_max)),
                "frequency_ge": float(np.sum(local_vals[local_finite] >= snapshot_max) / np.sum(local_finite)),
                "lat": max_lat,
                "lon": max_lon,
            },
            {
                "metric": f"pointwise_snapshot_max_same_season_{peak_season}",
                "value": snapshot_max,
                "n": int(np.sum(local_season_mask)),
                "count_ge": int(np.sum(local_vals[local_season_mask] >= snapshot_max)),
                "count_gt": int(np.sum(local_vals[local_season_mask] > snapshot_max)),
                "frequency_ge": float(np.sum(local_vals[local_season_mask] >= snapshot_max) / np.sum(local_season_mask)),
                "lat": max_lat,
                "lon": max_lon,
            },
        ]
        for threshold in (8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0):
            count = int(np.sum(svals[finite_snapshot] >= threshold))
            total = int(np.sum(finite_snapshot))
            diag_rows.append(
                {
                    "metric": f"snapshot_grid_ge_{threshold:g}_bits",
                    "value": threshold,
                    "n": total,
                    "count_ge": count,
                    "count_gt": int(np.sum(svals[finite_snapshot] > threshold)),
                    "frequency_ge": float(count / total),
                    "lat": np.nan,
                    "lon": np.nan,
                }
            )
        pd.DataFrame(diag_rows).to_csv(out_path.with_name(f"{out_path.stem}_exceedance.csv"), index=False)

        cmin = float(np.nanpercentile(cvals[np.isfinite(cvals)], 1.0))
        cmax = float(np.nanpercentile(cvals[np.isfinite(cvals)], 99.0))
        smin = float(np.nanpercentile(svals[np.isfinite(svals)], 1.0))
        smax = float(np.nanpercentile(svals[np.isfinite(svals)], 99.0))
        if not np.isfinite(cmin) or not np.isfinite(cmax) or cmin >= cmax:
            cmin, cmax = float(np.nanmin(cvals)), float(np.nanmax(cvals))
        if not np.isfinite(smin) or not np.isfinite(smax) or smin >= smax:
            smin, smax = float(np.nanmin(svals)), float(np.nanmax(svals))

        data_crs = ccrs.PlateCarree()
        proj = ccrs.Robinson(central_longitude=180.0)
        fig = plt.figure(figsize=(8.0, 6.8))
        gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 0.055, 1.0, 0.055], hspace=0.05)
        ax0 = fig.add_subplot(gs[0, 0], projection=proj)
        cax0 = fig.add_subplot(gs[1, 0])
        ax1 = fig.add_subplot(gs[2, 0], projection=proj)
        cax1 = fig.add_subplot(gs[3, 0])

        im0 = ax0.pcolormesh(
            lon,
            lat,
            cvals,
            shading="auto",
            cmap="viridis",
            vmin=cmin,
            vmax=cmax,
            transform=data_crs,
        )
        ax0.set_extent([float(lon.min()), float(lon.max()), -25.0, 25.0], crs=data_crs)
        ax0.set_aspect("auto")
        ax0.coastlines(resolution="110m", linewidth=0.6, color="k")
        ax0.gridlines(draw_labels=False, linewidth=0.3, color="0.6", alpha=0.5)
        ax0.text(
            0.5,
            0.98,
            "A) Mean OLR surprisal (1979--2020)",
            transform=ax0.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
        cbar0 = fig.colorbar(im0, cax=cax0, orientation="horizontal")
        cbar0.set_label("bits", labelpad=2)
        cbar0.ax.xaxis.set_ticks_position("top")
        cbar0.ax.xaxis.set_label_position("top")

        im1 = ax1.pcolormesh(
            lon,
            lat,
            svals,
            shading="auto",
            cmap="viridis",
            vmin=smin,
            vmax=smax,
            transform=data_crs,
        )
        ax1.set_extent([float(lon.min()), float(lon.max()), -25.0, 25.0], crs=data_crs)
        ax1.set_aspect("auto")
        ax1.coastlines(resolution="110m", linewidth=0.6, color="k")
        ax1.gridlines(draw_labels=False, linewidth=0.3, color="0.6", alpha=0.5)
        ax1.text(
            0.5,
            0.98,
            f"B) Burst snapshot ({snapshot_time:%Y-%m-%d})",
            transform=ax1.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
        ax1.text(
            0.01,
            0.99,
            f"Peak A(t) = {peak_amp:.2f}",
            transform=ax1.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.8},
        )
        cbar1 = fig.colorbar(im1, cax=cax1, orientation="horizontal")
        cbar1.set_label("bits")

        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(
            f"Surprisal fields figure: peak burst day {snapshot_time:%Y-%m-%d}, "
            f"peak amplitude {peak_amp:.3f}, output={out_path}"
        )
    finally:
        ds_s.close()
        ds_i.close()


def plot_burst_diagnostics(mjo_timeseries: Path, out_path: Path) -> None:
    ds = xr.open_dataset(mjo_timeseries)
    ds = ds.sel(time=slice("2014-01-01", "2017-12-31"))

    time = pd.to_datetime(ds["time"].values)
    amp = ds["mjo_amp"].values.astype(float)
    burst = ds["burst_mask"].values.astype(bool)
    phase = _phase_from_radians(ds["mjo_phase_rad"].values.astype(float))
    phase_counts = np.array([(phase[burst] == p).sum() for p in range(1, 9)], dtype=int)

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 6.2), gridspec_kw={"height_ratios": [2.1, 1.0]})

    ax = axes[0]
    ax.plot(time, amp, color="#1f77b4", lw=1.0, label="MJO amplitude")
    ax.axhline(1.0, color="0.3", ls="--", lw=1.0, label="Amp = 1")
    ax.fill_between(time, 0, amp, where=burst, color="#d62728", alpha=0.18, label="Burst days")
    ax.set_ylabel("Amplitude")
    ax.set_title("Surprisal-Derived MJO Amplitude and Burst Mask (2014-2017)")
    ax.legend(loc="upper right", frameon=False, ncol=3)

    ax = axes[1]
    ax.bar(np.arange(1, 9), phase_counts, color="#4c78a8")
    ax.set_xlabel("MJO phase")
    ax.set_ylabel("Burst-day count")
    ax.set_xticks(np.arange(1, 9))
    ax.set_title("Phase distribution of burst days (same period)")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_surprisal_phase_composites(phase_nc: Path, out_path: Path) -> None:
    """Render compact per-phase OLR surprisal anomaly composites over the active MJO region.

    Expects variables: mjo_phase01_mean_surprisal .. mjo_phase08_mean_surprisal
    with coordinates lat (-20..20) and lon (-180..180 or 0..360).
    We show anomalies relative to the across-phase mean to enhance structure.
    """
    ds = xr.open_dataset(phase_nc)
    # Collect in phase order
    vars_by_phase = [f"mjo_phase{p:02d}_mean_surprisal" for p in range(1, 9)]
    for name in vars_by_phase:
        if name not in ds:
            raise KeyError(f"{phase_nc} missing {name}")

    # Active MJO longitudes: 40°E–178°E (east edge near Fiji to avoid seam artifacts)
    lon = ds["lon"].values.astype(float)
    if lon.min() < 0:
        lon_mod = lon.copy()
        # Convert to 0..360 for easy slicing
        lon_mod = (lon_mod + 360.0) % 360.0
    else:
        lon_mod = lon
    # Build index mask for 40..178E to stay west of the dateline
    lon_mask = (lon_mod >= 40.0) & (lon_mod <= 178.0)
    lon_sel = lon[lon_mask]
    lat_sel = ds["lat"].values.astype(float)

    fields = []
    for name in vars_by_phase:
        arr = ds[name].sel(lon=lon_sel)
        fields.append(arr)

    # Compute anomalies: subtract the mean across phases (approximate active-day mean)
    import numpy as np
    stack = np.stack([f.values for f in fields], axis=0)
    mean_all = np.nanmean(stack, axis=0)
    anoms = [np.asarray(f.values, dtype=float) - mean_all for f in fields]

    # Diverging scale centered at 0 using robust 98th percentile of |anomaly|
    vals = np.concatenate([np.ravel(a) for a in anoms])
    vals = vals[np.isfinite(vals)]
    vmax = float(np.nanpercentile(np.abs(vals), 98))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 0.5

    data_crs = ccrs.PlateCarree()
    proj = ccrs.PlateCarree(central_longitude=120.0)

    # 4 rows x 2 columns uses vertical space better for this wide domain
    fig = plt.figure(figsize=(8.0, 7.8))
    gs = fig.add_gridspec(5, 2, height_ratios=[1.0, 1.0, 1.0, 1.0, 0.08], hspace=0.06, wspace=0.04)
    axes = [fig.add_subplot(gs[i//2, i%2], projection=proj) for i in range(8)]

    # Plot
    mappable = None
    for ax, a_vals, ph in zip(axes, anoms, range(1, 9)):
        im = ax.pcolormesh(
            lon_sel,
            lat_sel,
            a_vals,
            shading="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            transform=data_crs,
        )
        ax.set_extent([40.0, 178.0, -20.0, 20.0], crs=data_crs)
        ax.set_aspect('auto')
        ax.coastlines(resolution="110m", linewidth=0.5)
        ax.gridlines(draw_labels=False, linewidth=0.3, color="0.6", alpha=0.4)
        ax.text(0.02, 0.98, f"Phase {ph}", transform=ax.transAxes, ha="left", va="top",
                fontsize=9, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7})
        mappable = im

    cax = fig.add_subplot(gs[4, :])
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_label("OLR surprisal anomaly (bits)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _fd_bins(values: np.ndarray, min_bins: int = 20, max_bins: int = 100) -> np.ndarray:
    vals = values[np.isfinite(values)]
    if vals.size < 2:
        return np.linspace(-1, 1, 41)
    q25, q75 = np.nanpercentile(vals, [25, 75])
    iqr = q75 - q25
    width = 2.0 * iqr * (vals.size ** (-1.0 / 3.0))
    if not np.isfinite(width) or width <= 0:
        n_bins = int(np.clip(np.sqrt(vals.size), min_bins, max_bins))
    else:
        n_bins = int(np.clip(np.ceil((vals.max() - vals.min()) / width), min_bins, max_bins))
    return np.linspace(vals.min(), vals.max(), n_bins + 1)


def plot_phase_channel_mechanics(
    receiver_nc: Path,
    results_dir: Path,
    out_path: Path,
    high_phase: int = 5,
    low_phase: int = 1,
) -> None:
    """Compare amplitude distributions for a high-MI phase and a low-MI phase.

    For each phase k, we plot the distribution of a_{L,k}(t) on phase-on days
    versus the unconditional distribution, and annotate discrete KL divergence
    D_KL(P_on || P_all) (bits).
    """
    ds = xr.open_dataset(receiver_nc)
    df_mi = pd.read_csv(results_dir / "info_gain_cont_gatealigned_nullsummary.csv").set_index("phase")
    l_high = int(df_mi.loc[high_phase, "best_lag_obs"]) if high_phase in df_mi.index else 24
    l_low = int(df_mi.loc[low_phase, "best_lag_obs"]) if low_phase in df_mi.index else 3

    def _phase_data(phase: int, lag: int) -> Tuple[np.ndarray, np.ndarray]:
        amp = ds["z500_amp_weighted"].sel(phase=phase, lag=lag).values.astype(float)
        on_mask = (ds["driver_phase_index"].values.astype(int) == (phase - 1))
        a_on = amp[on_mask & np.isfinite(amp)]
        a_all = amp[np.isfinite(amp)]
        return a_on, a_all

    a_on_hi, a_all_hi = _phase_data(high_phase, l_high)
    a_on_lo, a_all_lo = _phase_data(low_phase, l_low)

    # Common x-limits based on pooled percentiles
    pool = np.concatenate([a_all_hi, a_all_lo])
    xlo, xhi = np.nanpercentile(pool, [1, 99])
    bins = _fd_bins(pool)
    bins = bins[(bins >= xlo) & (bins <= xhi)]
    if bins.size < 20:
        bins = np.linspace(xlo, xhi, 41)

    def _hist_p(values: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(values, bins=bins)
        alpha = 0.5
        m = bins.size - 1
        return (counts + alpha) / (counts.sum() + alpha * m)

    def _kl(p: np.ndarray, q: np.ndarray) -> float:
        mask = (p > 0) & (q > 0)
        return float(np.sum(p[mask] * (np.log2(p[mask]) - np.log2(q[mask]))))

    p_on_hi = _hist_p(a_on_hi)
    p_all_hi = _hist_p(a_all_hi)
    p_on_lo = _hist_p(a_on_lo)
    p_all_lo = _hist_p(a_all_lo)
    dkl_hi = _kl(p_on_hi, p_all_hi)
    dkl_lo = _kl(p_on_lo, p_all_lo)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)
    for ax, phase, lag, p_on, p_all, dkl in [
        (axes[0], high_phase, l_high, p_on_hi, p_all_hi, dkl_hi),
        (axes[1], low_phase, l_low, p_on_lo, p_all_lo, dkl_lo),
    ]:
        centers = 0.5 * (bins[:-1] + bins[1:])
        width = np.diff(bins)
        ax.bar(centers, p_all, width=width, color="#bbbbbb", alpha=0.6, label="All days")
        ax.bar(centers, p_on, width=width, color="#1f77b4", alpha=0.6, label=f"Phase {phase} on")
        ax.set_xlim(xlo, xhi)
        ax.set_xlabel("Receiver amplitude a")
        ax.set_ylabel("Probability")
        ax.set_title(f"Phase {phase}, lag {lag} d: KL={dkl:.2f} bits")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Amplitude distributions: on-phase vs climatology")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_weighted_vs_gated_summary(results_dir: Path, out_path: Path) -> None:
    """Compare per-phase max MI (gate-aligned, weighted) vs max IG (binary gate, weighted receiver)."""
    mi = pd.read_csv(results_dir / "info_gain_cont_gatealigned_nullsummary.csv").set_index("phase")
    ig = pd.read_csv(results_dir / "info_gain_p85_nullsummary.csv").set_index("phase")
    phases = np.arange(1, 9, dtype=int)
    mi_max = np.array([mi.loc[p, "max_stat_obs"] if p in mi.index else np.nan for p in phases])
    ig_max = np.array([ig.loc[p, "max_stat_obs"] if p in ig.index else np.nan for p in phases])

    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    x = np.arange(phases.size)
    w = 0.38
    ax.bar(x - w / 2, mi_max, width=w, label="MI (continuous)")
    ax.bar(x + w / 2, ig_max, width=w, label="IG (p85 gate)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in phases])
    ax.set_xlabel("MJO phase")
    ax.set_ylabel("Max bits over lags")
    ax.set_title("Continuous MI vs Binary-Gate IG by Phase")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_mjo_index_burst_definition(mjo_timeseries: Path, receiver_nc: Path, out_path: Path) -> None:
    ds_idx = xr.open_dataset(mjo_timeseries)
    ds_rcv = xr.open_dataset(receiver_nc)
    try:
        required_idx = ["mjo_amp", "burst_mask"]
        required_rcv = ["driver_intensity", "burst_gate"]
        for name in required_idx:
            if name not in ds_idx:
                raise KeyError(f"{mjo_timeseries} is missing '{name}'")
        for name in required_rcv:
            if name not in ds_rcv:
                raise KeyError(f"{receiver_nc} is missing '{name}'")

        amp_da, burst_da, intensity_da, gate_da = xr.align(
            ds_idx["mjo_amp"],
            ds_idx["burst_mask"],
            ds_rcv["driver_intensity"],
            ds_rcv["burst_gate"],
            join="inner",
        )
        time = pd.to_datetime(amp_da["time"].values)
        amp = amp_da.values.astype(float)
        burst = burst_da.values.astype(bool)
        intensity = intensity_da.values.astype(float)
        gate = gate_da.values.astype(float)
        gate_on = np.isfinite(gate) & (gate >= 0.5)

        gate_pct = float(ds_rcv.attrs.get("burst_gate_percentile", np.nan))
        gate_thr = float(ds_rcv.attrs.get("burst_gate_threshold", np.nan))
        smooth_days = int(ds_rcv.attrs.get("burst_smooth_days", 10))

        def _parse_window(
            attr_name: str,
            default_start: pd.Timestamp,
            default_end: pd.Timestamp,
        ) -> tuple[pd.Timestamp, pd.Timestamp]:
            raw = ds_rcv.attrs.get(attr_name)
            if isinstance(raw, str):
                parts = [part.strip() for part in raw.split(",") if part.strip()]
                if len(parts) >= 2:
                    try:
                        return pd.Timestamp(parts[0]), pd.Timestamp(parts[1])
                    except ValueError:
                        pass
            return default_start, default_end

        train_start, train_end = _parse_window(
            "train_window",
            pd.Timestamp(time[0]),
            pd.Timestamp(time[-1]),
        )
        test_start_default = train_end + pd.Timedelta(days=1)
        test_start, test_end = _parse_window(
            "test_window",
            test_start_default,
            pd.Timestamp(time[-1]),
        )

        y_candidates = [1.0]
        if np.any(np.isfinite(amp)):
            y_candidates.append(float(np.nanmax(amp)))
        if np.any(np.isfinite(intensity)):
            y_candidates.append(float(np.nanmax(intensity)))
        if np.isfinite(gate_thr):
            y_candidates.append(float(gate_thr))
        y_max = 1.05 * max(y_candidates)
        gate_tick_top = 0.055 * y_max

        blue = "#1f4e79"
        gray = "#6a6a6a"
        threshold_color = "0.10"
        gate_color = "#8f1d1d"
        burst_color = "#d68686"
        train_bg = "#f4f0e8"
        eval_bg = "#edf4f8"

        def shade_boolean_intervals(
            ax: plt.Axes,
            t: np.ndarray,
            mask: np.ndarray,
            *,
            color: str,
            alpha: float,
            zorder: int,
        ) -> None:
            if t.size == 0 or not np.any(mask):
                return
            idx = np.flatnonzero(mask)
            splits = np.where(np.diff(idx) > 1)[0] + 1
            for group in np.split(idx, splits):
                start = pd.Timestamp(t[group[0]]) - pd.Timedelta(hours=12)
                end = pd.Timestamp(t[group[-1]]) + pd.Timedelta(hours=12)
                ax.axvspan(start, end, color=color, alpha=alpha, ec="none", zorder=zorder)

        fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.6), sharey=True)
        fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.18, hspace=0.26)

        ax = axes[0]
        ax.axvspan(train_start, train_end + pd.Timedelta(days=1), color=train_bg, alpha=0.55, zorder=0)
        ax.axvspan(test_start, test_end + pd.Timedelta(days=1), color=eval_bg, alpha=0.60, zorder=0)
        ax.vlines(time[gate_on], 0.0, gate_tick_top, color=gate_color, lw=0.45, alpha=0.85, zorder=4)
        ax.plot(time, intensity, color=blue, lw=1.35, zorder=3)
        if np.isfinite(gate_thr):
            ax.axhline(gate_thr, color=threshold_color, ls=(0, (4, 3)), lw=1.0, zorder=2)
        ax.axvline(test_start, color="0.45", lw=0.8, alpha=0.65, zorder=2)
        ax.set_xlim(pd.Timestamp(time[0]), pd.Timestamp(time[-1]))
        ax.set_ylim(0.0, y_max)
        ax.set_title(f"Full record ({time[0].year}-{time[-1].year})", loc="left", pad=8)
        ax.set_ylabel("Amplitude")
        ax.grid(axis="y", alpha=0.18, lw=0.5)
        ax.margins(x=0)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        train_mid = train_start + (train_end - train_start) / 2
        test_mid = test_start + (test_end - test_start) / 2
        window_label_kws = dict(
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            color="0.35",
            fontsize=8.3,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.4),
        )
        ax.text(train_mid, 0.97, f"Training ({train_start.year}-{train_end.year})", **window_label_kws)
        ax.text(test_mid, 0.97, f"Evaluation ({test_start.year}-{test_end.year})", **window_label_kws)
        ax.text(
            -0.10,
            1.04,
            "(A)",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
            fontsize=11,
            clip_on=False,
        )

        zoom_start = pd.Timestamp("2014-01-01")
        zoom_end = pd.Timestamp("2017-12-31")
        zmask = (time >= zoom_start) & (time <= zoom_end)
        if int(np.sum(zmask)) < 200:
            zoom_end = pd.Timestamp(time[-1])
            zoom_start = zoom_end - pd.Timedelta(days=1460)
            zmask = (time >= zoom_start) & (time <= zoom_end)

        tz = time[zmask]
        ampz = amp[zmask]
        intz = intensity[zmask]
        burstz = burst[zmask]
        gatez = gate_on[zmask]

        ax = axes[1]
        shade_boolean_intervals(ax, tz, burstz, color=burst_color, alpha=0.20, zorder=1)
        ax.vlines(tz[gatez], 0.0, gate_tick_top, color=gate_color, lw=0.75, alpha=0.90, zorder=5)
        ax.plot(tz, ampz, color=gray, lw=0.85, alpha=0.95, zorder=3)
        ax.plot(tz, intz, color=blue, lw=1.65, zorder=4)
        if np.isfinite(gate_thr):
            ax.axhline(gate_thr, color=threshold_color, ls=(0, (4, 3)), lw=1.0, zorder=3)
        ax.set_xlim(zoom_start, zoom_end)
        ax.set_ylim(0.0, y_max)
        ax.set_title(f"Detail window ({zoom_start.year}-{zoom_end.year})", loc="left", pad=8)
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Date")
        ax.grid(axis="y", alpha=0.18, lw=0.5)
        ax.margins(x=0)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.text(
            -0.10,
            1.04,
            "(B)",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
            fontsize=11,
            clip_on=False,
        )


        gate_pct_label = int(round(gate_pct)) if np.isfinite(gate_pct) else 90
        legend_handles = [
            Line2D([0], [0], color=blue, lw=1.6, label=f"B(t): {smooth_days}-day centered mean"),
            Line2D([0], [0], color=gray, lw=0.9, label="A(t): daily amplitude"),
            Line2D(
                [0],
                [0],
                color=threshold_color,
                lw=1.0,
                ls=(0, (4, 3)),
                label=f"Training threshold $q_{{{gate_pct_label}}}^{{\\rm train}}$",
            ),
            Line2D([0], [0], color=gate_color, marker="|", linestyle="None", markersize=10, label="B(t) gate-on days"),
            patches.Patch(facecolor=burst_color, alpha=0.20, edgecolor="none", label="A(t) burst intervals"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            ncol=3,
            frameon=False,
            columnspacing=1.4,
            handlelength=2.4,
        )

        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    finally:
        ds_idx.close()
        ds_rcv.close()


def plot_heatmap(csv_path: Path, value_col: str, title: str, cbar_label: str, out_path: Path) -> None:
    mat = _load_heatmap(csv_path, value_col=value_col)
    arr = mat.values
    vmax = np.nanpercentile(np.abs(arr), 98)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = np.nanmax(np.abs(arr))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(10.0, 3.8))
    im = ax.imshow(arr, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)
    ax.set_title(title)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("MJO phase")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index.astype(int))

    lag_vals = mat.columns.astype(int).to_numpy()
    if lag_vals.size > 8:
        idx = np.linspace(0, lag_vals.size - 1, 8).astype(int)
        ax.set_xticks(idx)
        ax.set_xticklabels(lag_vals[idx])
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_phase_curves(results_dir: Path, out_path: Path) -> None:
    df_ig = pd.read_csv(results_dir / "info_gain_p85_observed.csv")
    df_mi = pd.read_csv(results_dir / "info_gain_cont_gatealigned_observed.csv")
    phases = [5, 6, 7, 8]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.5), sharex=True)
    axes = axes.ravel()
    for ax, ph in zip(axes, phases):
        s_ig = df_ig[df_ig["phase"] == ph].sort_values("lag")
        s_mi = df_mi[df_mi["phase"] == ph].sort_values("lag")
        ax.plot(s_ig["lag"], s_ig["IG_bits"], lw=1.8, color="#1f77b4", label="IG bits (p85 gate)")
        ax.plot(s_mi["lag"], s_mi["MI_bits"], lw=1.8, color="#d62728", label="MI bits (continuous)")
        ax.set_title(f"Phase {ph}")
        ax.grid(alpha=0.2, lw=0.6)
    axes[2].set_xlabel("Lag (days)")
    axes[3].set_xlabel("Lag (days)")
    axes[0].set_ylabel("Bits")
    axes[2].set_ylabel("Bits")
    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle("Lag-Dependent Information Transfer for Phases 5-8")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_significance(results_dir: Path, out_path: Path) -> pd.DataFrame:
    metrics = {
        "IG p85 block": "info_gain_p85_nullsummary.csv",
        "IG p85 circ": "info_gain_p85_cshift_nullsummary.csv",
        "MI cont block": "info_gain_cont_gatealigned_nullsummary.csv",
        "MI cont circ": "info_gain_cont_cshift_nullsummary.csv",
        "AUC block": "phasebank_skill_auc_nullsummary.csv",
        "SNR block": "phasebank_skill_snr_nullsummary.csv",
    }
    phases = np.arange(1, 9)
    records = []
    fig, ax = plt.subplots(figsize=(10.0, 4.2))

    colors = {
        "IG p85 block": "#1f77b4",
        "IG p85 circ": "#1f77b4",
        "MI cont block": "#d62728",
        "MI cont circ": "#d62728",
        "AUC block": "#2ca02c",
        "SNR block": "#9467bd",
    }
    markers = {
        "IG p85 block": "o",
        "IG p85 circ": "o",
        "MI cont block": "s",
        "MI cont circ": "s",
        "AUC block": "^",
        "SNR block": "D",
    }
    linestyles = {
        "IG p85 block": "-",
        "IG p85 circ": "--",
        "MI cont block": "-",
        "MI cont circ": "--",
        "AUC block": ":",
        "SNR block": ":",
    }

    for name, fn in metrics.items():
        df = pd.read_csv(results_dir / fn).set_index("phase")
        qvals = np.array([df.loc[p, "q_phase"] if p in df.index else np.nan for p in phases], dtype=float)
        y = -np.log10(np.clip(qvals, 1e-6, 1.0))
        ax.plot(
            phases,
            y,
            marker=markers[name],
            lw=1.5,
            ls=linestyles[name],
            color=colors[name],
            label=name,
        )
        for p, qv in zip(phases, qvals):
            records.append({"phase": int(p), "metric": name, "q_phase": float(qv)})

    ax.axhline(-np.log10(0.05), color="0.2", ls="--", lw=1.0, label="q = 0.05")
    ax.set_xticks(phases)
    ax.set_xlabel("MJO phase")
    ax.set_ylabel("-log10(FDR q)")
    ax.set_title("Phase-Level FDR Significance Across Metrics and Nulls")
    ax.legend(loc="upper right", frameon=False, ncol=3)
    ax.grid(alpha=0.2, lw=0.6)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    out_df = pd.DataFrame(records)
    out_df.to_csv(results_dir / "phase_significance_long.csv", index=False)
    return out_df


def _enso_matrix(results_dir: Path, suffix: str = "_mt60") -> Tuple[np.ndarray, np.ndarray]:
    state_files = {
        "El Nino": results_dir / f"info_gain_cont_elnino{suffix}_nullsummary.csv",
        "Neutral": results_dir / f"info_gain_cont_neutral{suffix}_nullsummary.csv",
        "La Nina": results_dir / f"info_gain_cont_lanina{suffix}_nullsummary.csv",
    }
    phases = np.arange(1, 9)
    max_mi = np.full((phases.size, len(state_files)), np.nan, dtype=float)
    pvals = np.full_like(max_mi, np.nan)

    for j, (_, path) in enumerate(state_files.items()):
        df = pd.read_csv(path).set_index("phase")
        for i, ph in enumerate(phases):
            if ph not in df.index:
                continue
            max_mi[i, j] = df.loc[ph, "max_stat_obs"]
            pvals[i, j] = df.loc[ph, "p_phase"]
    return max_mi, pvals


def plot_enso_conditioned(results_dir: Path, out_path: Path) -> None:
    max_mi, pvals = _enso_matrix(results_dir, suffix="_mt60")
    states = ["El Nino", "Neutral", "La Nina"]
    phases = np.arange(1, 9)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), constrained_layout=True)

    im0 = axes[0].imshow(max_mi, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Max MI by Phase and ENSO State")
    axes[0].set_xlabel("ENSO state")
    axes[0].set_ylabel("MJO phase")
    axes[0].set_xticks(np.arange(len(states)))
    axes[0].set_xticklabels(states, rotation=20)
    axes[0].set_yticks(np.arange(phases.size))
    axes[0].set_yticklabels(phases)
    c0 = fig.colorbar(im0, ax=axes[0], pad=0.02)
    c0.set_label("Bits")

    score = -np.log10(np.clip(pvals, 1e-6, 1.0))
    im1 = axes[1].imshow(score, aspect="auto", origin="lower", cmap="cividis")
    axes[1].set_title("-log10(p) for ENSO-Conditioned MI")
    axes[1].set_xlabel("ENSO state")
    axes[1].set_ylabel("MJO phase")
    axes[1].set_xticks(np.arange(len(states)))
    axes[1].set_xticklabels(states, rotation=20)
    axes[1].set_yticks(np.arange(phases.size))
    axes[1].set_yticklabels(phases)
    c1 = fig.colorbar(im1, ax=axes[1], pad=0.02)
    c1.set_label("-log10(p)")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_receiver_templates(receiver_nc: Path, results_dir: Path, out_path: Path) -> None:
    ds = xr.open_dataset(receiver_nc)
    df = pd.read_csv(results_dir / "info_gain_p85_nullsummary.csv").set_index("phase")
    phases = [3, 5, 6, 8]

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.6), constrained_layout=True)
    axes = axes.ravel()

    vmax = 0.0
    fields = []
    for ph in phases:
        lag = int(df.loc[ph, "best_lag_obs"])
        tpl = ds["z500_template_weighted"].sel(phase=ph, lag=lag)
        fields.append((ph, lag, tpl))
        vmax = max(vmax, float(np.nanpercentile(np.abs(tpl.values), 99)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    mappable = None
    for ax, (ph, lag, tpl) in zip(axes, fields):
        mappable = ax.pcolormesh(
            tpl["longitude"].values,
            tpl["latitude"].values,
            tpl.values,
            shading="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(f"Phase {ph}, lag {lag} d")
        ax.set_xlabel("Longitude (degE)")
        ax.set_ylabel("Latitude (degN)")

    cbar = fig.colorbar(mappable, ax=axes, pad=0.02, shrink=0.9)
    cbar.set_label("Weighted template loading (m per source-index unit)")
    fig.suptitle("Phasebank Receiver Templates at IG-Optimal Lags")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_matched_filter_schematic(receiver_nc: Path, results_dir: Path, out_path: Path) -> None:
    ds = xr.open_dataset(receiver_nc)
    df = pd.read_csv(results_dir / "info_gain_p85_nullsummary.csv").set_index("phase")
    phase = 5
    lag = int(df.loc[phase, "best_lag_obs"])

    amp = ds["z500_amp_weighted"].sel(phase=phase, lag=lag)
    t_idx = int(np.nanargmax(np.abs(amp.values)))
    tstamp = pd.to_datetime(ds["time"].values[t_idx]).strftime("%Y-%m-%d")

    template = ds["z500_template_weighted"].sel(phase=phase, lag=lag)
    snapshot = ds["z500_anom"].isel(time=t_idx)

    lat = ds["latitude"].values.astype(float)
    w_lat = np.sqrt(np.clip(np.cos(np.deg2rad(lat)), 0.0, None))
    w2 = np.repeat(w_lat[:, None], ds["longitude"].size, axis=1)

    tvec = template.values.astype(float).reshape(-1)
    zvec = snapshot.values.astype(float).reshape(-1)
    wvec = w2.reshape(-1)
    valid = np.isfinite(tvec) & np.isfinite(zvec) & np.isfinite(wvec)
    t_w = tvec[valid] * wvec[valid]
    z_w = zvec[valid] * wvec[valid]
    denom = np.sum(t_w * t_w)
    a_hat = float(np.sum(z_w * t_w) / denom) if denom > 0 else np.nan
    corr = float(np.corrcoef(t_w, z_w)[0, 1]) if t_w.size > 2 else np.nan

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)

    vmax_t = float(np.nanpercentile(np.abs(template.values), 99))
    vmax_t = vmax_t if np.isfinite(vmax_t) and vmax_t > 0 else 1.0
    im0 = axes[0].pcolormesh(
        template["longitude"].values,
        template["latitude"].values,
        template.values,
        shading="auto",
        cmap="RdBu_r",
        vmin=-vmax_t,
        vmax=vmax_t,
    )
    axes[0].set_title(f"Template T (phase {phase}, lag {lag} d)")
    axes[0].set_xlabel("Longitude (degE)")
    axes[0].set_ylabel("Latitude (degN)")
    fig.colorbar(im0, ax=axes[0], pad=0.01)

    vmax_z = float(np.nanpercentile(np.abs(snapshot.values), 99))
    vmax_z = vmax_z if np.isfinite(vmax_z) and vmax_z > 0 else 1.0
    im1 = axes[1].pcolormesh(
        snapshot["longitude"].values,
        snapshot["latitude"].values,
        snapshot.values,
        shading="auto",
        cmap="RdBu_r",
        vmin=-vmax_z,
        vmax=vmax_z,
    )
    axes[1].set_title(f"Z500 anomaly snapshot ({tstamp})")
    axes[1].set_xlabel("Longitude (degE)")
    axes[1].set_ylabel("Latitude (degN)")
    fig.colorbar(im1, ax=axes[1], pad=0.01)

    n_plot = min(9000, t_w.size)
    if t_w.size > n_plot:
        rng = np.random.default_rng(0)
        idx = rng.choice(t_w.size, size=n_plot, replace=False)
        tx = t_w[idx]
        zx = z_w[idx]
    else:
        tx = t_w
        zx = z_w
    axes[2].scatter(tx, zx, s=3, alpha=0.25, color="#1f77b4")
    xline = np.linspace(np.nanpercentile(tx, 1), np.nanpercentile(tx, 99), 100)
    axes[2].plot(xline, a_hat * xline, color="#d62728", lw=2.0, label=rf"$\hat{{a}}={a_hat:.2f}$")
    axes[2].set_title(f"Weighted projection: corr={corr:.2f}")
    axes[2].set_xlabel(r"$T\sqrt{\cos\phi}$")
    axes[2].set_ylabel(r"$Z\sqrt{\cos\phi}$")
    axes[2].legend(frameon=False, loc="upper left")
    axes[2].grid(alpha=0.2, lw=0.5)

    fig.suptitle("Matched Filter Concept in the Phasebank Receiver")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_phase5_worked_example(
    receiver_nc: Path,
    results_dir: Path,
    maps_out: Path,
    series_out: Path,
    summary_csv_out: Path,
) -> None:
    ds = xr.open_dataset(receiver_nc)
    try:
        phase = 5
        lag_summary = pd.read_csv(results_dir / "info_gain_cont_gatealigned_nullsummary.csv").set_index("phase")
        lag = int(lag_summary.loc[phase, "best_lag_obs"])

        sample = _phase_lag_continuous_samples(
            ds,
            phase_label=phase,
            lag=lag,
            test_start="2011-01-01",
            test_end="2020-12-31",
            smooth_days=10,
            w_bins=5,
            alpha=0.5,
        )

        j_strong = int(np.argmax(np.abs(sample["y"])))
        j_weak = int(np.argmin(np.abs(sample["y"])))

        ds_t = sample["dataset_test"]
        lat = ds_t["latitude"].values.astype(np.float64)
        lon = ds_t["longitude"].values.astype(np.float64)
        w_lat = np.sqrt(np.clip(np.cos(np.deg2rad(lat)), a_min=0.0, a_max=None))
        w2 = np.repeat(w_lat[:, None], lon.size, axis=1)
        template = ds_t["z500_template_weighted"].sel(phase=phase, lag=lag).values.astype(np.float64)

        def _projection_terms(idx_pair: int) -> Dict[str, object]:
            recv_idx = int(sample["receiver_indices"][idx_pair])
            snapshot = ds_t["z500_anom"].isel(time=recv_idx).values.astype(np.float64)
            valid = np.isfinite(template) & np.isfinite(snapshot) & np.isfinite(w2)
            tw = template[valid] * w2[valid]
            zw = snapshot[valid] * w2[valid]
            numerator = float(np.sum(zw * tw))
            denominator = float(np.sum(tw * tw))
            amp_hat = numerator / denominator if denominator > 0 else np.nan
            contribution = np.full(template.shape, np.nan, dtype=np.float64)
            contribution[valid] = snapshot[valid] * template[valid] * (w2[valid] ** 2)
            return {
                "snapshot": snapshot,
                "contribution": contribution,
                "numerator": numerator,
                "denominator": denominator,
                "amp_hat": amp_hat,
            }

        strong = _projection_terms(j_strong)
        weak = _projection_terms(j_weak)

        # Figure 11: real-data map-based worked example
        fig = plt.figure(figsize=(14.0, 8.0))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.22, hspace=0.22)
        ax_tpl = fig.add_subplot(gs[:, 0])
        ax_str = fig.add_subplot(gs[0, 1])
        ax_str_c = fig.add_subplot(gs[0, 2])
        ax_weak = fig.add_subplot(gs[1, 1])
        ax_weak_c = fig.add_subplot(gs[1, 2])

        v_tpl = float(np.nanpercentile(np.abs(template), 99))
        v_tpl = v_tpl if np.isfinite(v_tpl) and v_tpl > 0 else 1.0
        im_tpl = ax_tpl.pcolormesh(lon, lat, template, shading="auto", cmap="RdBu_r", vmin=-v_tpl, vmax=v_tpl)
        ax_tpl.set_title(f"Template $T_{{L={lag},\\,k=5}}$ (real data)")
        ax_tpl.set_xlabel("Longitude (degE)")
        ax_tpl.set_ylabel("Latitude (degN)")
        fig.colorbar(im_tpl, ax=ax_tpl, pad=0.01, fraction=0.046).set_label("Weighted template loading (m per source-index unit)")

        snapshots = [strong["snapshot"], weak["snapshot"]]
        v_snap = float(np.nanpercentile(np.abs(np.concatenate([arr.ravel() for arr in snapshots])), 99))
        v_snap = v_snap if np.isfinite(v_snap) and v_snap > 0 else 1.0

        im_str = ax_str.pcolormesh(lon, lat, strong["snapshot"], shading="auto", cmap="RdBu_r", vmin=-v_snap, vmax=v_snap)
        ax_str.set_title(
            "Strong-fit receiver snapshot\n"
            f"{sample['receiver_times'][j_strong].strftime('%Y-%m-%d')}, "
            f"$\\hat{{a}}={strong['amp_hat']:.2f}$"
        )
        ax_str.set_xlabel("Longitude (degE)")
        ax_str.set_ylabel("Latitude (degN)")

        im_weak = ax_weak.pcolormesh(lon, lat, weak["snapshot"], shading="auto", cmap="RdBu_r", vmin=-v_snap, vmax=v_snap)
        ax_weak.set_title(
            "Weak-fit receiver snapshot\n"
            f"{sample['receiver_times'][j_weak].strftime('%Y-%m-%d')}, "
            f"$\\hat{{a}}={weak['amp_hat']:.3f}$"
        )
        ax_weak.set_xlabel("Longitude (degE)")
        ax_weak.set_ylabel("Latitude (degN)")
        fig.colorbar(im_weak, ax=[ax_str, ax_weak], pad=0.01, fraction=0.046).set_label("Z500 anomaly (m)")

        contribs = [strong["contribution"], weak["contribution"]]
        v_contrib = float(np.nanpercentile(np.abs(np.concatenate([arr.ravel() for arr in contribs])), 99))
        v_contrib = v_contrib if np.isfinite(v_contrib) and v_contrib > 0 else 1.0
        im_sc = ax_str_c.pcolormesh(
            lon,
            lat,
            strong["contribution"],
            shading="auto",
            cmap="RdBu_r",
            vmin=-v_contrib,
            vmax=v_contrib,
        )
        ax_str_c.set_title(r"Strong-fit local numerator terms $Z\,T\,\cos\phi$")
        ax_str_c.set_xlabel("Longitude (degE)")
        ax_str_c.set_ylabel("Latitude (degN)")

        im_wc = ax_weak_c.pcolormesh(
            lon,
            lat,
            weak["contribution"],
            shading="auto",
            cmap="RdBu_r",
            vmin=-v_contrib,
            vmax=v_contrib,
        )
        ax_weak_c.set_title(r"Weak-fit local numerator terms $Z\,T\,\cos\phi$")
        ax_weak_c.set_xlabel("Longitude (degE)")
        ax_weak_c.set_ylabel("Latitude (degN)")
        fig.colorbar(im_wc, ax=[ax_str_c, ax_weak_c], pad=0.01, fraction=0.046).set_label("Contribution (m$^2$)")

        fig.suptitle(f"Worked Example (Real Data): Phase 5, Continuous Mode, Best-Lag {lag} Days", y=0.98)
        fig.savefig(maps_out, bbox_inches="tight")
        plt.close(fig)

        # Figure 12: time-series and entropy/MI diagnostics for the same phase-lag pair
        fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.0), constrained_layout=True)
        ax1, ax2, ax3, ax4 = axes.ravel()

        time_test = sample["time_test"]
        ax1.plot(time_test, sample["intensity"], color="0.65", lw=0.9, label="Smoothed source intensity B(t)")
        ax1.scatter(
            time_test[sample["phase_mask"]],
            sample["intensity"][sample["phase_mask"]],
            s=8,
            color="#1f77b4",
            alpha=0.35,
            label="Phase-5 source days",
        )
        ax1.scatter(
            [sample["source_times"][j_strong], sample["source_times"][j_weak]],
            [sample["w"][j_strong], sample["w"][j_weak]],
            s=45,
            color=["#d62728", "#2ca02c"],
            edgecolor="k",
            linewidth=0.4,
            zorder=5,
        )
        ax1.set_title("Source side (test window): intensity and phase-5 sampling")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Intensity")
        ax1.legend(frameon=False, loc="upper right")

        ax2.plot(time_test, sample["y_source_full"], color="#8c564b", lw=0.9)
        ax2.scatter(
            [sample["receiver_times"][j_strong], sample["receiver_times"][j_weak]],
            [sample["y"][j_strong], sample["y"][j_weak]],
            s=45,
            color=["#d62728", "#2ca02c"],
            edgecolor="k",
            linewidth=0.4,
            zorder=5,
        )
        ax2.axhline(0.0, color="0.2", lw=0.8, ls="--")
        ax2.set_title(rf"Receiver side: amplitude series $a_{{L={lag},k=5}}(t)$")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Matched-filter amplitude (m)")

        cmap = plt.get_cmap("viridis", int(sample["counts_w"].size))
        sc = ax3.scatter(sample["w"], sample["y"], c=sample["w_bin"], s=18, alpha=0.7, cmap=cmap, edgecolor="none")
        ax3.scatter(
            [sample["w"][j_strong], sample["w"][j_weak]],
            [sample["y"][j_strong], sample["y"][j_weak]],
            s=70,
            color=["#d62728", "#2ca02c"],
            edgecolor="k",
            linewidth=0.5,
            zorder=6,
        )
        ax3.axhline(0.0, color="0.25", lw=0.8, ls="--")
        ax3.set_title(rf"Sample pairs used for MI: $W(\tau)$ versus $Y(\tau+{lag})$")
        ax3.set_xlabel("Source intensity W")
        ax3.set_ylabel("Receiver amplitude Y")
        cbar = fig.colorbar(sc, ax=ax3, pad=0.01, fraction=0.046)
        cbar.set_label("W quantile bin index")

        ax4.hist(sample["y"], bins=sample["y_bins"], histtype="step", lw=2.0, color="k", label="All Y")
        for b in range(sample["counts_w"].size):
            if sample["counts_w"][b] == 0:
                continue
            yy = sample["y"][sample["w_bin"] == b]
            lo = sample["w_edges"][b]
            hi = sample["w_edges"][b + 1]
            ax4.hist(
                yy,
                bins=sample["y_bins"],
                histtype="step",
                lw=1.2,
                label=f"W bin {b + 1}: [{lo:.2f}, {hi:.2f})",
            )
        ax4.set_title("Conditional Y distributions used in entropy calculation")
        ax4.set_xlabel("Receiver amplitude Y")
        ax4.set_ylabel("Count")
        ax4.legend(frameon=False, fontsize=7, ncol=1)
        ax4.text(
            0.02,
            0.98,
            f"H(Y) = {sample['H_all']:.4f} bits\n"
            f"H(Y|W_b) = {sample['H_cond']:.4f} bits\n"
            f"MI = H(Y)-H(Y|W_b) = {sample['MI']:.4f} bits\n"
            f"n = {sample['y'].size}, Y bins = {sample['y_bins'].size - 1}",
            transform=ax4.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.7"},
        )

        fig.savefig(series_out, bbox_inches="tight")
        plt.close(fig)

        summary_rows = [
            {
                "entry": f"global_phase5_lag{lag}_cont",
                "phase": phase,
                "lag_days": lag,
                "test_start": "2011-01-01",
                "test_end": "2020-12-31",
                "n_samples": int(sample["y"].size),
                "n_y_bins": int(sample["y_bins"].size - 1),
                "h_uncond_bits": float(sample["H_all"]),
                "h_cond_bits": float(sample["H_cond"]),
                "mi_bits": float(sample["MI"]),
            },
            {
                "entry": "strong_fit_event",
                "phase": phase,
                "lag_days": lag,
                "source_date": sample["source_times"][j_strong].strftime("%Y-%m-%d"),
                "receiver_date": sample["receiver_times"][j_strong].strftime("%Y-%m-%d"),
                "source_intensity": float(sample["w"][j_strong]),
                "receiver_amplitude": float(sample["y"][j_strong]),
                "projection_numerator": float(strong["numerator"]),
                "projection_denominator": float(strong["denominator"]),
                "projection_a_hat": float(strong["amp_hat"]),
            },
            {
                "entry": "weak_fit_event",
                "phase": phase,
                "lag_days": lag,
                "source_date": sample["source_times"][j_weak].strftime("%Y-%m-%d"),
                "receiver_date": sample["receiver_times"][j_weak].strftime("%Y-%m-%d"),
                "source_intensity": float(sample["w"][j_weak]),
                "receiver_amplitude": float(sample["y"][j_weak]),
                "projection_numerator": float(weak["numerator"]),
                "projection_denominator": float(weak["denominator"]),
                "projection_a_hat": float(weak["amp_hat"]),
            },
        ]
        pd.DataFrame(summary_rows).to_csv(summary_csv_out, index=False)
    finally:
        ds.close()


def plot_null_robustness(results_dir: Path, out_path: Path) -> None:
    ig_bp = np.load(results_dir / "info_gain_p85_nulldist.npz")
    ig_cs = np.load(results_dir / "info_gain_p85_cshift_nulldist.npz")
    mi_bp = np.load(results_dir / "info_gain_cont_gatealigned_nulldist.npz")
    mi_cs = np.load(results_dir / "info_gain_cont_cshift_nulldist.npz")

    phases_ig = ig_bp["phases"].astype(int)
    i5_ig = int(np.where(phases_ig == 5)[0][0])
    phases_mi = mi_bp["phases"].astype(int)
    i5_mi = int(np.where(phases_mi == 5)[0][0])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)

    ax = axes[0]
    ax.hist(ig_bp["max_null"][i5_ig], bins=35, alpha=0.55, color="#1f77b4", label="Block permutation")
    ax.hist(ig_cs["max_null"][i5_ig], bins=35, alpha=0.55, color="#ff7f0e", label="Circular shift")
    ax.axvline(float(ig_bp["max_obs"][i5_ig]), color="k", lw=2, label="Observed")
    ax.set_title("Phase 5 max-IG nulls")
    ax.set_xlabel("Max IG over lags (bits)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.hist(mi_bp["max_null"][i5_mi], bins=35, alpha=0.55, color="#1f77b4", label="Block permutation")
    ax.hist(mi_cs["max_null"][i5_mi], bins=35, alpha=0.55, color="#ff7f0e", label="Circular shift")
    ax.axvline(float(mi_bp["max_obs"][i5_mi]), color="k", lw=2, label="Observed")
    ax.set_title("Phase 5 max-MI nulls")
    ax.set_xlabel("Max MI over lags (bits)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)

    fig.suptitle("Red-Noise-Robust Null Comparison")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def copy_existing_stage_figures(output_dir: Path, paths: Iterable[Path]) -> None:
    for src in paths:
        if src.exists():
            shutil.copy2(src, output_dir / src.name)


def export_codebooks(receiver_nc: Path, results_dir: Path) -> Tuple[Path, Path]:
    """Write weighted and gated codebook templates to standalone NetCDF files."""

    specs = (
        ("z500_template_weighted", results_dir / "phasebank_codebook_weighted.nc", "Weighted-intensity template codebook"),
        ("z500_template_gated", results_dir / "phasebank_codebook_gated.nc", "Binary burst-gated template codebook"),
    )

    written: list[Path] = []
    with xr.open_dataset(receiver_nc) as ds:
        lag_list = ds.attrs.get("lag_list", "")
        phase_values = ds.attrs.get("phase_values", "")
        for var_name, out_path, desc in specs:
            if var_name not in ds:
                raise ValueError(
                    f"Receiver NetCDF {receiver_nc} is missing required codebook variable '{var_name}'. "
                    "Rebuild the receiver to ensure both template sets are available."
                )
            cb_ds = ds[[var_name]].copy()
            cb_ds.attrs.update(
                {
                    "codebook_type": desc,
                    "source_receiver_nc": str(receiver_nc),
                    "lag_list": lag_list,
                    "phase_values": phase_values,
                }
            )
            cb_ds[var_name].attrs.setdefault("description", desc)
            encoding = {var_name: {"zlib": True, "complevel": 4}}
            cb_ds.to_netcdf(out_path, encoding=encoding)
            written.append(out_path)

    return tuple(written)


def build_summary_tables(results_dir: Path) -> None:
    phase = pd.DataFrame({"phase": np.arange(1, 9, dtype=int)})

    def merge_metric(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        keep = df[["phase", "best_lag_obs", "p_phase", "q_phase"]].copy()
        keep.columns = ["phase", f"{prefix}_best_lag", f"{prefix}_p", f"{prefix}_q"]
        return keep

    t_ig = merge_metric(pd.read_csv(results_dir / "info_gain_p85_nullsummary.csv"), "ig_p85")
    t_mi = merge_metric(pd.read_csv(results_dir / "info_gain_cont_gatealigned_nullsummary.csv"), "mi_cont_ga")
    # Keep circshift variants if present (optional)
    t_ig_cs = merge_metric(pd.read_csv(results_dir / "info_gain_p85_cshift_nullsummary.csv"), "ig_p85_cshift")
    t_mi_cs = merge_metric(pd.read_csv(results_dir / "info_gain_cont_cshift_nullsummary.csv"), "mi_cont_cshift")
    t_auc = merge_metric(pd.read_csv(results_dir / "phasebank_skill_auc_nullsummary.csv"), "auc")
    t_snr = merge_metric(pd.read_csv(results_dir / "phasebank_skill_snr_nullsummary.csv"), "snr")

    out = (
        phase.merge(t_ig, on="phase")
        .merge(t_mi, on="phase")
        .merge(t_ig_cs, on="phase")
        .merge(t_mi_cs, on="phase")
        .merge(t_auc, on="phase")
        .merge(t_snr, on="phase")
    )
    out.to_csv(results_dir / "phase_significance_summary.csv", index=False)

    frames = []
    state_files = {
        "ElNino": results_dir / "info_gain_cont_elnino_mt60_nullsummary.csv",
        "Neutral": results_dir / "info_gain_cont_neutral_mt60_nullsummary.csv",
        "LaNina": results_dir / "info_gain_cont_lanina_mt60_nullsummary.csv",
    }
    for state, path in state_files.items():
        df = pd.read_csv(path).copy()
        df["enso_state"] = state
        frames.append(df[["enso_state", "phase", "max_stat_obs", "best_lag_obs", "p_phase", "q_phase", "n_lags_used"]])
    pd.concat(frames, ignore_index=True).to_csv(results_dir / "enso_conditioned_mi_summary_mt60.csv", index=False)


def write_phase_summary_table_tex(results_dir: Path, out_tex: Path) -> None:
    """Write a LaTeX table summarizing MI and IG by phase: max bits, best lag, and FDR q."""
    mi = pd.read_csv(results_dir / "info_gain_cont_gatealigned_nullsummary.csv").set_index("phase")
    ig = pd.read_csv(results_dir / "info_gain_p85_nullsummary.csv").set_index("phase")
    phases = np.arange(1, 9, dtype=int)
    rows = []
    for p in phases:
        mi_max = mi.loc[p, "max_stat_obs"] if p in mi.index else np.nan
        mi_lag = mi.loc[p, "best_lag_obs"] if p in mi.index else np.nan
        mi_q = mi.loc[p, "q_phase"] if p in mi.index else np.nan
        ig_max = ig.loc[p, "max_stat_obs"] if p in ig.index else np.nan
        ig_lag = ig.loc[p, "best_lag_obs"] if p in ig.index else np.nan
        ig_q = ig.loc[p, "q_phase"] if p in ig.index else np.nan
        rows.append((p, mi_max, mi_lag, mi_q, ig_max, ig_lag, ig_q))

    def _fmt_bits(x):
        return "" if not np.isfinite(x) else f"{x:.3f}"

    def _fmt_lag(x):
        return "" if not np.isfinite(x) else f"{int(x)}"

    def _fmt_q(x):
        if not np.isfinite(x):
            return ""
        # compact scientific/decimal formatting
        return f"{x:.2g}"

    lines = []
    lines.append("% Auto-generated by make_phasebank_manuscript_figures.write_phase_summary_table_tex")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{r r r r r r r}")
    lines.append("\\hline")
    lines.append("Phase & MI$_\\mathrm{max}$ (bits) & MI best lag (d) & MI $q_{\\rm block}$ & IG$_\\mathrm{max}$ (bits) & IG best lag (d) & IG $q_{\\rm block}$")
    lines.append("\\\\")
    lines.append("\\hline")
    for p, mi_max, mi_lag, mi_q, ig_max, ig_lag, ig_q in rows:
        row = f"{p} & {_fmt_bits(mi_max)} & {_fmt_lag(mi_lag)} & {_fmt_q(mi_q)} & {_fmt_bits(ig_max)} & {_fmt_lag(ig_lag)} & {_fmt_q(ig_q)}"
        lines.append(row)
        lines.append("\\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Per-phase summary of continuous-intensity Mutual Information (MI; continuous bins include the p85 gate edge) and binary-gated Information Gain (IG; p85 gate): maximum value over lags, best lag, and block-permutation FDR-adjusted q-value.}")
    lines.append("\\label{tab:phase_summary}")
    lines.append("\\end{table}")
    out_tex.write_text("\n".join(lines))


def write_duty_cycle_table_tex(receiver_nc: Path, results_dir: Path, out_tex: Path) -> None:
    """Write a LaTeX table summarizing phase occupancy and gate duty cycle.

    Columns:
      phase, P(phase=k), P(G=1 & phase=k), P(G=1 | phase=k), conditional
      binary entropy h2(P(G=1 | phase=k)), counts.

    The main IG analysis is phase-conditioned, so its source entropy ceiling is
    the conditional gate entropy within each phase, not the smaller entropy of
    the joint "phase and gate" event.
    """
    import numpy as np
    import xarray as xr

    ds = xr.open_dataset(receiver_nc)
    phase_idx = ds["driver_phase_index"].values.astype(int)
    intensity = ds["driver_intensity"].values.astype(float)
    time = pd.to_datetime(ds["time"].values)
    train_start, train_end = [pd.to_datetime(x.strip()) for x in ds.attrs.get("train_window", "1991-01-01,2010-12-31").split(",")]
    test_start, test_end = [pd.to_datetime(x.strip()) for x in ds.attrs.get("test_window", "2011-01-01,2020-12-31").split(",")]

    train_mask = (time >= train_start) & (time <= train_end) & np.isfinite(intensity)
    gate_threshold = float(np.nanpercentile(intensity[train_mask], 85.0))
    gate = np.full(intensity.shape, np.nan, dtype=float)
    finite_intensity = np.isfinite(intensity)
    gate[finite_intensity] = 0.0
    gate[finite_intensity & (intensity >= gate_threshold)] = 1.0

    # Sanitize and restrict to the held-out evaluation window used for MI/IG.
    test_mask = (time >= test_start) & (time <= test_end)
    finite = test_mask & np.isfinite(phase_idx) & np.isfinite(gate) & (phase_idx >= 0)
    phase_idx = phase_idx[finite]
    gate = gate[finite]
    n = phase_idx.size

    def h2(p: float) -> float:
        if p <= 0 or p >= 1 or not np.isfinite(p):
            return 0.0
        return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))

    lines = []
    lines.append("% Auto-generated duty cycle table")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{r r r r r r}")
    lines.append("\\hline")
    lines.append("Phase & $P(\\text{phase}=k)$ & $P(G=1,\\,\\text{phase}=k)$ & $P(G=1\\mid\\text{phase}=k)$ & $H_{\\rm eval}(G\\mid k)$ (bits) & $N_k$\\\\")
    lines.append("\\hline")
    for k in range(8):
        mask_k = phase_idx == k
        nk = int(mask_k.sum())
        p_phase = nk / n if n else np.nan
        p_onk = float(((gate > 0.5) & mask_k).sum()) / n if n else np.nan
        p_on_given_k = float(((gate > 0.5) & mask_k).sum()) / nk if nk else np.nan
        h_bound = h2(p_on_given_k)
        def fmt(x, f="{:.3f}"):
            return "" if not np.isfinite(x) else f.format(x)
        lines.append(
            f"{k+1} & {fmt(p_phase)} & {fmt(p_onk)} & {fmt(p_on_given_k)} & {fmt(h_bound)} & {nk}\\\\"
        )
    # Also report global gate fraction and bound
    p1 = float((gate > 0.5).sum()) / n if n else np.nan
    h_global = h2(p1)
    lines.append("\\hline")
    lines.append(f"\\multicolumn{{6}}{{l}}{{p85 gate: $q_{{85}}^{{\\rm train}}={gate_threshold:.3f}$, global $P(G=1)={fmt(p1)}$, $H(G)={fmt(h_global)}$ bits}}\\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Duty cycle and binary-entropy bounds for the p85 gate over all 2011--2020 evaluation-window source days. $P(\\text{phase}=k)$ is phase occupancy; $P(G=1,\\,\\text{phase}=k)$ is the joint phase--gate frequency; $P(G=1\\mid\\text{phase}=k)$ is the on-fraction within phase $k$. $H_{\\rm eval}(G\\mid k)=h_2[P(G=1\\mid\\text{phase}=k)]$ is the all-window conditional binary entropy. The telecom scorecard uses the corresponding lag-paired entropy at each IG-optimal lag, so small differences from this duty-cycle table arise from lag-edge sample exclusion.}")
    lines.append("\\label{tab:duty_cycle}")
    lines.append("\\end{table}")
    out_tex.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    _configure_matplotlib()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_workflow(output_dir / "fig01_workflow.png")
    plot_surprisal_fields(
        Path(args.surprisal_nc),
        Path(args.mjo_timeseries),
        output_dir / "fig_surprisal_fields.png",
    )
    # Note: this figure is now logically Fig. 3 in the paper after inserting the phase composites as Fig. 2.
    plot_mjo_index_burst_definition(
        Path(args.mjo_timeseries),
        Path(args.receiver_nc),
        output_dir / "fig03_mjo_index_burst_definition.png",
    )
    # Per-phase surprisal composites over active MJO region
    try:
        plot_surprisal_phase_composites(Path(args.surprisal_phase_nc), output_dir / "fig02a_mjo_surprisal_phase_composites.png")
    except Exception as exc:
        print(f"WARNING: Could not render surprisal phase composites: {exc}")
    # Mechanics: amplitude distributions and KL for a high-MI phase vs a low-MI phase
    try:
        plot_phase_channel_mechanics(
            Path(args.receiver_nc),
            results_dir,
            output_dir / "fig_channel_mechanics_phase5_vs1.png",
            high_phase=5,
            low_phase=1,
        )
    except Exception as exc:
        print(f"WARNING: Could not render channel mechanics figure: {exc}")

    # MI vs IG summary across phases
    try:
        plot_weighted_vs_gated_summary(results_dir, output_dir / "fig_mi_vs_ig_by_phase.png")
    except Exception as exc:
        print(f"WARNING: Could not render weighted-vs-gated summary: {exc}")
    plot_burst_diagnostics(Path(args.mjo_timeseries), output_dir / "fig02_burst_diagnostics.png")
    plot_heatmap(
        results_dir / "info_gain_p85_observed.csv",
        value_col="IG_bits",
        title="Binary-Gate Information Gain (IG, p85) by Phase and Lag",
        cbar_label="IG (bits)",
        out_path=output_dir / "fig03_ig_p85_heatmap.png",
    )
    plot_heatmap(
        results_dir / "info_gain_cont_gatealigned_observed.csv",
        value_col="MI_bits",
        title="Continuous-Intensity Mutual Information by Phase and Lag",
        cbar_label="MI (bits)",
        out_path=output_dir / "fig04_mi_cont_heatmap.png",
    )
    plot_significance(results_dir, output_dir / "fig05_phase_significance.png")
    plot_phase_curves(results_dir, output_dir / "fig06_phase_curves_5_8.png")
    plot_enso_conditioned(results_dir, output_dir / "fig07_enso_conditioned_mi.png")
    plot_receiver_templates(Path(args.receiver_nc), results_dir, output_dir / "fig08_receiver_templates.png")
    # Codebook snapshots with map projection for clarity
    try:
        def _best_lag_for_phase(results_dir: Path, phase: int) -> int:
            df = pd.read_csv(results_dir / "info_gain_cont_gatealigned_nullsummary.csv").set_index("phase")
            return int(df.loc[phase, "best_lag_obs"])

        def plot_codebook_phase5_cartopy(results_dir: Path, out_path: Path, layout: str = "stack") -> None:
            phase = 5
            lag = _best_lag_for_phase(results_dir, phase)

            ds_w = xr.open_dataset(results_dir / "phasebank_codebook_weighted.nc")
            ds_g = xr.open_dataset(results_dir / "phasebank_codebook_gated.nc")

            tpl_w = ds_w["z500_template_weighted"].sel(phase=phase, lag=lag)
            tpl_g = ds_g["z500_template_gated"].sel(phase=phase, lag=lag)

            lon = tpl_w["longitude"].values.astype(float)
            lat = tpl_w["latitude"].values.astype(float)
            tw = tpl_w.values.astype(float)
            tg = tpl_g.values.astype(float)

            # Robust panel-specific limits to avoid washing out the weaker field
            vmax_w = float(np.nanpercentile(np.abs(tw), 99))
            vmax_g = float(np.nanpercentile(np.abs(tg), 99))
            if not np.isfinite(vmax_w) or vmax_w <= 0:
                vmax_w = 1.0
            if not np.isfinite(vmax_g) or vmax_g <= 0:
                vmax_g = 1.0

            data_crs = ccrs.PlateCarree()
            # Use Plate Carree for direct longitude/latitude interpretation.
            proj = ccrs.PlateCarree(central_longitude=210.0)

            if layout == "stack":
                fig = plt.figure(figsize=(8.0, 10.0))
                # Extra vertical spacing and taller colorbar bands for legibility
                gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 0.20, 1.0, 0.20], hspace=0.32)
                ax0 = fig.add_subplot(gs[0, 0], projection=proj)
                cax0 = fig.add_subplot(gs[1, 0])
                ax1 = fig.add_subplot(gs[2, 0], projection=proj)
                cax1 = fig.add_subplot(gs[3, 0])
            else:
                fig = plt.figure(figsize=(11.5, 4.6))
                gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.06], width_ratios=[1.0, 1.0], hspace=0.04, wspace=0.10)
                ax0 = fig.add_subplot(gs[0, 0], projection=proj)
                ax1 = fig.add_subplot(gs[0, 1], projection=proj)
                cax0 = fig.add_subplot(gs[1, 0])
                cax1 = fig.add_subplot(gs[1, 1])

            extent = [120.0, 300.0, 20.0, 80.0]
            for ax in (ax0, ax1):
                ax.set_extent(extent, crs=data_crs)
                ax.coastlines(resolution="110m", linewidth=0.7, color="k")
                gl = ax.gridlines(draw_labels=False, linewidth=0.4, color="0.6", alpha=0.5)

            im0 = ax0.pcolormesh(lon, lat, tw, shading="auto", cmap="RdBu_r", vmin=-vmax_w, vmax=vmax_w, transform=data_crs)
            ax0.text(
                0.5,
                0.985,
                f"Weighted template: phase {phase}, lag {lag} d",
                transform=ax0.transAxes,
                ha="center",
                va="top",
                fontsize=11,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            )

            im1 = ax1.pcolormesh(lon, lat, tg, shading="auto", cmap="RdBu_r", vmin=-vmax_g, vmax=vmax_g, transform=data_crs)
            ax1.text(
                0.5,
                0.985,
                f"p90 gated diagnostic: phase {phase}, lag {lag} d",
                transform=ax1.transAxes,
                ha="center",
                va="top",
                fontsize=11,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            )

            cbar0 = fig.colorbar(im0, cax=cax0, orientation="horizontal")
            cbar0.set_label("Weighted template loading (m per source-index unit)")
            cbar0.ax.xaxis.set_ticks_position("bottom")
            cbar0.ax.xaxis.set_label_position("bottom")
            cbar0.ax.tick_params(labelsize=9, pad=2)

            cbar1 = fig.colorbar(im1, cax=cax1, orientation="horizontal")
            cbar1.set_label("p90 gated-template loading (m per binary-gate unit)")
            cbar1.ax.xaxis.set_ticks_position("bottom")
            cbar1.ax.xaxis.set_label_position("bottom")
            cbar1.ax.tick_params(labelsize=9, pad=2)
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

        # Write a new stacked version with independent scales
        plot_codebook_phase5_cartopy(results_dir, output_dir / "fig_codebook_phase5_templates_stack.png", layout="stack")
    except Exception as exc:
        print(f"WARNING: Failed to render cartopy codebook snapshots: {exc}")
    plot_matched_filter_schematic(Path(args.receiver_nc), results_dir, output_dir / "fig09_matched_filter_schematic.png")
    plot_null_robustness(results_dir, output_dir / "fig10_null_robustness.png")
    plot_phase5_worked_example(
        Path(args.receiver_nc),
        results_dir,
        output_dir / "fig11_phase5_worked_example_maps.png",
        output_dir / "fig12_phase5_worked_example_timeseries.png",
        results_dir / "phase5_worked_example_values.csv",
    )
    codebook_paths = export_codebooks(Path(args.receiver_nc), results_dir)

    copy_existing_stage_figures(
        output_dir,
        [Path(args.mjo_phase_composites), Path(args.mjo_burst_climatology)],
    )
    build_summary_tables(results_dir)
    print(f"Wrote figures to {output_dir}")
    print("Codebook NetCDF files:")
    for path in codebook_paths:
        print(f"  - {path}")
    # Write a LaTeX table with per-phase MI/IG summary
    write_phase_summary_table_tex(results_dir, results_dir / "phase_summary_table.tex")
    write_duty_cycle_table_tex(Path(args.receiver_nc), results_dir, results_dir / "duty_cycle_table.tex")
    # Telecom-style summary panel
    try:
        scorecard_path = output_dir / "fig18_four_metric_scorecard.png"
        plot_telecom_summary(results_dir, scorecard_path)
        shutil.copyfile(scorecard_path, output_dir / "fig_telecom_summary.png")
    except Exception as exc:
        print(f"WARNING: Could not render telecom summary: {exc}")


if __name__ == "__main__":
    main()
