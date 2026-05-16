#!/usr/bin/env python3
"""
Generate additional reviewer-requested manuscript figures and diagnostics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.patches import Rectangle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", default="manuscript_results")
    p.add_argument("--fig_dir", default="manuscript_results/figures")
    p.add_argument("--mjo_index_nc", default="mjo_index_from_surprisal.nc")
    p.add_argument("--receiver_nc", default="z500_receiver_phasebank_Bint_domain120_300_20_80_train1991_2010.nc")
    p.add_argument("--diag_csv", default="manuscript_results/additional_figures_diagnostics.csv")
    p.add_argument("--summary_md", default="manuscript_results/additional_figures_summary_2026-03-03.md")
    return p.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "savefig.dpi": 240,
        }
    )


def fd_bins(values: np.ndarray, min_bins: int = 16, max_bins: int = 80) -> np.ndarray:
    vals = values[np.isfinite(values)]
    if vals.size < 2:
        return np.linspace(-1.0, 1.0, 31)
    q25, q75 = np.nanpercentile(vals, [25, 75])
    iqr = q75 - q25
    if iqr <= 0:
        n_bins = int(np.clip(np.sqrt(vals.size), min_bins, max_bins))
    else:
        width = 2.0 * iqr * (vals.size ** (-1.0 / 3.0))
        if width <= 0:
            n_bins = int(np.clip(np.sqrt(vals.size), min_bins, max_bins))
        else:
            n_bins = int(np.clip(np.ceil((vals.max() - vals.min()) / width), min_bins, max_bins))
    return np.linspace(vals.min(), vals.max(), n_bins + 1)


def smooth_2d(arr: np.ndarray, win_y: int, win_x: int) -> np.ndarray:
    out = arr.astype(np.float64, copy=True)
    if win_y > 1:
        ky = np.ones(win_y, dtype=np.float64) / float(win_y)
        out = np.apply_along_axis(lambda v: np.convolve(v, ky, mode="same"), 0, out)
    if win_x > 1:
        kx = np.ones(win_x, dtype=np.float64) / float(win_x)
        out = np.apply_along_axis(lambda v: np.convolve(v, kx, mode="same"), 1, out)
    return out


def weighted_pattern_corr(a: np.ndarray, b: np.ndarray, weights_2d: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b) & np.isfinite(weights_2d)
    if np.count_nonzero(mask) < 10:
        return np.nan
    wa = np.sqrt(weights_2d[mask])
    av = a[mask]
    bv = b[mask]
    av = av - np.average(av, weights=wa)
    bv = bv - np.average(bv, weights=wa)
    denom = np.sqrt(np.sum(wa * av * av) * np.sum(wa * bv * bv))
    if denom <= 0:
        return np.nan
    return float(np.sum(wa * av * bv) / denom)


def auc_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    valid = np.isfinite(scores) & np.isfinite(labels)
    scores = scores[valid]
    labels = labels[valid]
    n_on = int(np.sum(labels == 1))
    n_off = int(np.sum(labels == 0))
    if n_on == 0 or n_off == 0:
        return np.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg_rank = 0.5 * (i + j + 2)
            ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    rank_sum = np.sum(ranks[labels == 1])
    return float((rank_sum - n_on * (n_on + 1) / 2.0) / (n_on * n_off))


def roc_curve(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    valid = np.isfinite(scores) & np.isfinite(labels)
    scores = scores[valid]
    labels = labels[valid]
    pos = int(np.sum(labels == 1))
    neg = int(np.sum(labels == 0))
    if pos == 0 or neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    order = np.argsort(scores)[::-1]
    s = scores[order]
    y = labels[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    change = np.where(np.diff(s))[0]
    idx = np.r_[change, y.size - 1]
    tpr = np.r_[0.0, tp[idx] / pos, 1.0]
    fpr = np.r_[0.0, fp[idx] / neg, 1.0]
    return fpr, tpr


def select_test(ds: xr.Dataset, start: str = "2011-01-01", end: str = "2020-12-31") -> xr.Dataset:
    return ds.sel(time=slice(start, end))


def plot_wk_spectrum(mjo_index_nc: Path, out_path: Path) -> Dict[str, float]:
    """Compute and plot a Wheeler–Kiladis spectrum following WK99.

    Implementation notes aligned with the spec:
      - Use daily `olr_surprisal_anom` over 15S–15N, all longitudes
      - Symmetric/antisymmetric decomposition across the equator BEFORE spectral analysis
      - Segment time series into 96-day windows with 65-day overlap (step=31)
      - Detrend to order 0 per-segment (remove segment mean at each longitude)
      - Apply a split-cosine-bell (Tukey) 10% taper in time to each segment
      - 2D FFT per segment and per latitude-pair; power averaged across latitudes and segments
      - Background: average symmetric and antisymmetric raw power, then heavy 1-2-1 smoothing
    """

    def _tukey(n: int, alpha: float = 0.1) -> np.ndarray:
        if n <= 0:
            return np.array([], dtype=np.float64)
        if alpha <= 0.0:
            return np.ones(n, dtype=np.float64)
        if alpha >= 1.0:
            return np.hanning(n).astype(np.float64)
        w = np.ones(n, dtype=np.float64)
        edge = int(np.floor(alpha * (n - 1) / 2.0))
        if edge >= 1:
            k = np.arange(edge, dtype=np.float64)
            w0 = 0.5 * (1.0 + np.cos(np.pi * (2.0 * k / (alpha * (n - 1)) - 1.0)))
            w[:edge] = w0
            w[-edge:] = w0[::-1]
        return w

    def _smooth_121_1d(a: np.ndarray, axis: int) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        out = np.take(a, indices=range(a.shape[axis]), axis=axis).copy()
        # Build slices
        slc_c = [slice(None)] * a.ndim
        slc_l = [slice(None)] * a.ndim
        slc_r = [slice(None)] * a.ndim
        # interior
        slc_c[axis] = slice(1, -1)
        slc_l[axis] = slice(0, -2)
        slc_r[axis] = slice(2, None)
        out[tuple(slc_c)] = (
            0.25 * a[tuple(slc_l)] + 0.5 * a[tuple(slc_c)] + 0.25 * a[tuple(slc_r)]
        )
        # Endpoints: hold fixed (copy from input)
        slc0 = [slice(None)] * a.ndim
        slc0[axis] = 0
        slcN = [slice(None)] * a.ndim
        slcN[axis] = -1
        out[tuple(slc0)] = a[tuple(slc0)]
        out[tuple(slcN)] = a[tuple(slcN)]
        return out

    def _smooth_background_121(field: np.ndarray, freq_passes: int = 40, wav_passes: int = 20) -> np.ndarray:
        bg = np.asarray(field, dtype=np.float64)
        for _ in range(max(0, int(freq_passes))):
            bg = _smooth_121_1d(bg, axis=0)
        for _ in range(max(0, int(wav_passes))):
            bg = _smooth_121_1d(bg, axis=1)
        return bg

    # Load data and subset
    with xr.open_dataset(mjo_index_nc) as ds:
        da = ds["olr_surprisal_anom"].sel(lat=slice(-15.0, 15.0))
        # Optional longitude coarsening by factor 2 for runtime; no temporal subsampling
        da = da.coarsen(lon=2, boundary="trim").mean()
        da = da.transpose("time", "lat", "lon")
        data = da.values.astype(np.float64)
        lat = da["lat"].values.astype(np.float64)
        lon = da["lon"].values.astype(np.float64)

    nt, nlat, nlon = data.shape

    # Build latitude pairing for sym/anti
    lat_map = {round(float(v), 5): i for i, v in enumerate(lat)}
    pos_idx: List[int] = []
    neg_idx: List[int] = []
    eq_idx: int | None = None
    for i, v in enumerate(lat):
        if abs(v) < 1e-6:
            eq_idx = i
        elif v > 0:
            j = lat_map.get(round(-v, 5))
            if j is not None:
                pos_idx.append(i)
                neg_idx.append(j)

    n_pairs = len(pos_idx)
    if n_pairs == 0 and eq_idx is None:
        raise RuntimeError("No symmetric latitude pairs found within 15S–15N")

    # Segment indices: 96-day windows, 65-day overlap => step=31
    seg_len = 96
    seg_step = 31
    windows = list(range(0, max(nt - seg_len + 1, 0), seg_step))
    if nt >= seg_len and (nt - seg_len) % seg_step != 0:
        # Include the last partial step that still yields a full 96-day window
        last_start = nt - seg_len
        if windows and last_start > windows[-1]:
            windows.append(last_start)

    tuk = _tukey(seg_len, alpha=0.1).reshape(seg_len, 1)

    # Accumulators (positive frequencies only)
    # Compute axes once from the first segment
    if nt < seg_len:
        raise RuntimeError("Time series is shorter than 96 days; cannot compute WK spectrum")

    freqs_full = np.fft.fftfreq(seg_len, d=1.0)  # cycles/day
    # Use strictly positive frequencies to avoid DC artifact
    fpos = freqs_full > 0.0
    freqs = freqs_full[fpos]
    k_full = np.fft.fftfreq(nlon, d=1.0 / nlon)
    k = np.fft.fftshift(k_full)

    sym_accum = np.zeros((freqs.size, nlon), dtype=np.float64)
    asym_accum = np.zeros_like(sym_accum)
    n_segments = 0

    for start in windows:
        end = start + seg_len
        seg = data[start:end, :, :]  # (time, lat, lon)

        # Per-latitude symmetric/antisymmetric segments, detrend (remove mean over time at each lon), taper
        seg_sym_power = np.zeros((freqs.size, nlon), dtype=np.float64)
        seg_asym_power = np.zeros_like(seg_sym_power)
        n_lat_contrib_sym = 0
        n_lat_contrib_asym = 0

        # Paired latitudes (phi, -phi)
        for ip, ineg in zip(pos_idx, neg_idx):
            x_pos = seg[:, ip, :]
            x_neg = seg[:, ineg, :]
            sym_block = 0.5 * (x_pos + x_neg)
            asym_block = 0.5 * (x_pos - x_neg)
            # Detrend to order 0 over time at each longitude (remove segment mean)
            sym_block = sym_block - np.mean(sym_block, axis=0, keepdims=True)
            asym_block = asym_block - np.mean(asym_block, axis=0, keepdims=True)
            # Taper in time (10% Tukey)
            sym_t = sym_block * tuk
            asym_t = asym_block * tuk
            # 2D FFT and power
            Fs = np.fft.fft2(sym_t, axes=(0, 1))
            Fa = np.fft.fft2(asym_t, axes=(0, 1))
            Ps = np.abs(Fs) ** 2
            Pa = np.abs(Fa) ** 2
            # Positive frequencies only; keep all wavenumbers
            Ps = Ps[fpos, :]
            Pa = Pa[fpos, :]
            # Shift wavenumber to center 0
            Ps = np.fft.fftshift(Ps, axes=1)
            Pa = np.fft.fftshift(Pa, axes=1)
            seg_sym_power += Ps
            seg_asym_power += Pa
            n_lat_contrib_sym += 1
            n_lat_contrib_asym += 1

        # Equator row contributes to symmetric only (no antisymmetric component)
        if eq_idx is not None:
            x_eq = seg[:, eq_idx, :]
            x_eq = x_eq - np.mean(x_eq, axis=0, keepdims=True)
            x_eq = x_eq * tuk
            Feq = np.fft.fft2(x_eq, axes=(0, 1))
            Peq = np.abs(Feq) ** 2
            Peq = Peq[fpos, :]
            Peq = np.fft.fftshift(Peq, axes=1)
            seg_sym_power += Peq
            n_lat_contrib_sym += 1

        if n_lat_contrib_sym > 0:
            seg_sym_power /= float(n_lat_contrib_sym)
        if n_lat_contrib_asym > 0:
            seg_asym_power /= float(n_lat_contrib_asym)

        sym_accum += seg_sym_power
        asym_accum += seg_asym_power
        n_segments += 1

    if n_segments == 0:
        raise RuntimeError("No segments produced for WK spectrum (check time length)")

    # Average across segments
    sym_power = sym_accum / float(n_segments)
    asym_power = asym_accum / float(n_segments)

    # Construct propagation-agnostic background: sym+anti average, then heavy 1-2-1 smoothing
    bg_raw = 0.5 * (sym_power + asym_power)
    bg_sm = _smooth_background_121(bg_raw, freq_passes=40, wav_passes=20)
    # Ensure strictly positive background
    bg_pos = np.where(bg_sm > 0.0, bg_sm, np.nan)
    if not np.isfinite(bg_pos).any():
        raise RuntimeError("Background spectrum became non-positive everywhere")
    bg_min = float(np.nanpercentile(bg_pos, 1.0))
    bg = np.where(bg_sm > 0.0, bg_sm, max(bg_min, 1e-12))

    ratio = sym_power / bg

    # Plot limits and masks
    fmask = (freqs >= 0.0) & (freqs <= 0.10)
    kmask = (k >= -15) & (k <= 15)
    f = freqs[fmask]
    kk = k[kmask]
    p_sym = sym_power[np.ix_(fmask, kmask)]
    r_sym = ratio[np.ix_(fmask, kmask)]

    # Diagnostics within MJO band: k=1..5 (eastward), f in [1/96, 1/30]
    band_k = (kk >= 1) & (kk <= 5)
    band_f = (f >= 1.0 / 96.0) & (f <= 1.0 / 30.0)
    band = r_sym[np.ix_(band_f, band_k)]
    if np.all(~np.isfinite(band)) or band.size == 0:
        peak_f = np.nan
        peak_k = np.nan
        peak_period = np.nan
        band_mean = np.nan
    else:
        ipk = int(np.nanargmax(band))
        i_f, i_k = np.unravel_index(ipk, band.shape)
        peak_f = float(f[band_f][i_f])
        peak_k = float(kk[band_k][i_k])
        peak_period = float(1.0 / peak_f) if peak_f > 0 else np.nan
        band_mean = float(np.nanmean(band))
    # Additional diagnostics for MJO-band values in plot space
    mjo_log2_vals = np.log2(np.maximum(band, 1e-12)) if band.size else np.array([np.nan])
    mjo_log2_min = float(np.nanmin(mjo_log2_vals)) if np.isfinite(mjo_log2_vals).any() else np.nan
    mjo_log2_max = float(np.nanmax(mjo_log2_vals)) if np.isfinite(mjo_log2_vals).any() else np.nan
    mjo_log2_mean = float(np.nanmean(mjo_log2_vals)) if np.isfinite(mjo_log2_vals).any() else np.nan

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), constrained_layout=True)

    # Do not mask the lowest frequency bins: the MJO band spans the first 3 bins for 96-day segments
    p_sym_plot = p_sym
    r_sym_plot = r_sym

    vmin = float(np.nanpercentile(p_sym_plot[p_sym_plot > 0], 1.0)) if np.any(p_sym_plot > 0) else 1e-12
    logp = np.log10(np.maximum(p_sym_plot, vmin))
    im0 = axes[0].pcolormesh(kk, f, logp, shading="auto", cmap="magma")
    axes[0].set_title("A) WK Spectrum: log10 raw symmetric power")
    axes[0].set_xlabel("Zonal wavenumber (eastward > 0)")
    axes[0].set_ylabel("Frequency (cycles/day)")
    fig.colorbar(im0, ax=axes[0], pad=0.02, label="log10(power)")

    # Panel B in log2(ratio) with sequential colormap emphasizing >1.0
    log2ratio = np.log2(np.maximum(r_sym_plot, 1e-12))
    vmax_log2 = 0.8  # tighten to highlight 1.1–1.5 (0.14–0.58 in log2)
    log2ratio_plot = np.clip(log2ratio, 0.0, vmax_log2)
    im1 = axes[1].pcolormesh(kk, f, log2ratio_plot, shading="auto",
                              cmap="magma", vmin=0.0, vmax=vmax_log2)
    # Overlay interpretable contour levels in linear ratio units
    cont_levels = [lv for lv in [1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0, 5.0, 7.0] if np.isfinite(lv)]
    try:
        # Emphasize the 1.1 contour per WK convention
        axes[1].contour(kk, f, r_sym, levels=[1.1], colors="white", linewidths=1.2, alpha=0.9)
        axes[1].contour(kk, f, r_sym, levels=cont_levels, colors="k", linewidths=0.6, alpha=0.9)
    except Exception:
        pass
    axes[1].set_title("B) Normalized symmetric power (log2 ratio)")
    axes[1].set_xlabel("Zonal wavenumber (eastward > 0)")
    axes[1].set_ylabel("Frequency (cycles/day)")
    cbar = fig.colorbar(im1, ax=axes[1], pad=0.02)
    cbar.set_label("log2(P_sym / P_bg)")

    for ax in axes:
        ax.add_patch(
            Rectangle(
                (1.0, 1.0 / 96.0),
                4.0,
                (1.0 / 30.0) - (1.0 / 96.0),
                fill=False,
                edgecolor="cyan",
                linewidth=1.5,
            )
        )
        ax.text(1.2, 0.034, "MJO band: k=1..5, 30–96 d", color="cyan", fontsize=8)
        ax.set_xlim(-15, 15)
        ax.set_ylim(0.005, 0.10)
        ax.grid(alpha=0.25)

    if np.isfinite(peak_k) and np.isfinite(peak_f):
        axes[1].plot([peak_k], [peak_f], "ko", markersize=5)
        axes[1].text(
            peak_k + 0.4,
            peak_f + 0.002,
            f"peak: k={peak_k:.1f}, period={peak_period:.1f} d",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
        )

    fig.suptitle("Wheeler–Kiladis Spectrum of Surprisal (15S–15N)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "wk_peak_k": float(peak_k),
        "wk_peak_freq_cpd": float(peak_f),
        "wk_peak_period_days": float(peak_period),
        "wk_band_mean_power_ratio": float(band_mean),
        "wk_segments_used": float(n_segments),
        "wk_time_samples": float(nt),
        "wk_lat_pairs": float(n_pairs),
        "wk_lon_count": float(nlon),
        "wk_mjo_log2_min": mjo_log2_min,
        "wk_mjo_log2_max": mjo_log2_max,
        "wk_mjo_log2_mean": mjo_log2_mean,
    }


def _lag_corr(x: np.ndarray, y: np.ndarray, lags: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
    vals = []
    lgs = []
    for lag in lags:
        if lag > 0:
            a = x[:-lag]
            b = y[lag:]
        elif lag < 0:
            a = x[-lag:]
            b = y[:lag]
        else:
            a = x
            b = y
        m = np.isfinite(a) & np.isfinite(b)
        if np.count_nonzero(m) < 10:
            vals.append(np.nan)
        else:
            vals.append(np.corrcoef(a[m], b[m])[0, 1])
        lgs.append(lag)
    return np.asarray(lgs, dtype=int), np.asarray(vals, dtype=np.float64)


def plot_eof_patterns_and_variance(mjo_index_nc: Path, out_path: Path) -> Dict[str, float]:
    with xr.open_dataset(mjo_index_nc) as ds:
        da = ds["olr_surprisal_mjo"]
        lon_da = (da["lon"] + 360.0) % 360.0
        da = da.assign_coords(lon=lon_da).sortby("lon")
        da = da.sel(lat=slice(-15.0, 15.0), lon=slice(60.0, 180.0))
        da = da.transpose("time", "lat", "lon")
        da = da.coarsen(lat=4, lon=4, boundary="trim").mean()  # 2-degree grid for stable eigen spectrum

        arr = da.values.astype(np.float64)
        arr = arr - np.nanmean(arr, axis=0, keepdims=True)
        arr = np.nan_to_num(arr, nan=0.0)
        ntime, nlat, nlon = arr.shape
        lat = da["lat"].values.astype(np.float64)
        lon = da["lon"].values.astype(np.float64)

        wlat = np.sqrt(np.clip(np.cos(np.deg2rad(lat)), 0.0, None))
        w2d = wlat[:, None] * np.ones((1, nlon), dtype=np.float64)
        x = arr.reshape(ntime, nlat * nlon)
        wflat = w2d.reshape(-1)
        xw = x * wflat[None, :]

        cov = (xw.T @ xw) / max(ntime - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        varfrac = eigvals / np.sum(eigvals)

        eof1 = (eigvecs[:, 0].reshape(nlat, nlon)) / np.where(w2d > 0, w2d, np.nan)
        eof2 = (eigvecs[:, 1].reshape(nlat, nlon)) / np.where(w2d > 0, w2d, np.nan)
        pcs = xw @ eigvecs[:, :2]

        corr0 = float(np.corrcoef(pcs[:, 0], pcs[:, 1])[0, 1])
        lags, cc = _lag_corr(pcs[:, 0], pcs[:, 1], range(-20, 21))
        i_peak = int(np.nanargmax(np.abs(cc)))
        lag_peak = int(lags[i_peak])
        cc_peak = float(cc[i_peak])

    # Sign convention for stable visual orientation.
    if np.nanmean(eof1[:, lon <= 100.0]) < 0:
        eof1 *= -1.0
        pcs[:, 0] *= -1.0
    if np.nanmean(eof2[:, (lon >= 90.0) & (lon <= 130.0)]) < 0:
        eof2 *= -1.0
        pcs[:, 1] *= -1.0

    vmax = float(np.nanpercentile(np.abs(np.r_[eof1.ravel(), eof2.ravel()]), 99.0))
    vmax = max(vmax, 1e-6)

    fig = plt.figure(figsize=(12.2, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.30, wspace=0.22)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    im1 = ax1.contourf(lon, lat, eof1, levels=np.linspace(-vmax, vmax, 21), cmap="RdBu_r", extend="both")
    ax1.set_title(f"A) EOF1 ({100.0 * varfrac[0]:.1f}% variance)")
    ax1.set_xlabel("Longitude (E)")
    ax1.set_ylabel("Latitude")
    ax1.grid(alpha=0.25)
    fig.colorbar(im1, ax=ax1, pad=0.01, shrink=0.9)

    im2 = ax2.contourf(lon, lat, eof2, levels=np.linspace(-vmax, vmax, 21), cmap="RdBu_r", extend="both")
    ax2.set_title(f"B) EOF2 ({100.0 * varfrac[1]:.1f}% variance)")
    ax2.set_xlabel("Longitude (E)")
    ax2.set_ylabel("Latitude")
    ax2.grid(alpha=0.25)
    fig.colorbar(im2, ax=ax2, pad=0.01, shrink=0.9)

    n_show = 10
    ax3.bar(np.arange(1, n_show + 1), 100.0 * varfrac[:n_show], color="#4c78a8", alpha=0.8, label="Individual mode")
    ax3.plot(np.arange(1, n_show + 1), 100.0 * np.cumsum(varfrac[:n_show]), color="#d62728", marker="o", label="Cumulative")
    ax3.set_xticks(np.arange(1, n_show + 1))
    ax3.set_xlabel("EOF mode number")
    ax3.set_ylabel("Explained variance (%)")
    ax3.set_title(
        f"C) Scree (coarsened 2° domain) | EOF1-EOF2 sum = {100.0 * np.sum(varfrac[:2]):.1f}%, "
        f"corr(PC1,PC2)={corr0:.3f}, max|lag-corr|={cc_peak:.3f} at {lag_peak:+d} d"
    )
    ax3.grid(alpha=0.25)
    ax3.legend(frameon=False, ncol=2)

    fig.suptitle("EOF Structure of WK-Filtered Surprisal Field (60-180E, 15S-15N)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "eof1_var_frac_pct": float(100.0 * varfrac[0]),
        "eof2_var_frac_pct": float(100.0 * varfrac[1]),
        "eof12_var_frac_pct": float(100.0 * np.sum(varfrac[:2])),
        "pc1_pc2_corr0": corr0,
        "pc1_pc2_maxabs_lag_days": float(lag_peak),
        "pc1_pc2_maxabs_lag_corr": cc_peak,
    }


def _prep_lat_lon_field(field: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if lat[0] > lat[-1]:
        return field[::-1, :], lat[::-1]
    return field, lat


def plot_phase5_lag_evolution(codebook_nc: Path, out_path: Path, mi_csv: Path | None = None) -> Dict[str, float]:
    # Load weighted templates for Phase 5 at requested lags (revised: 4 panels)
    lags = [12, 16, 20, 24]
    with xr.open_dataset(codebook_nc) as ds:
        tpl = ds["z500_template_weighted"].sel(phase=5, lag=lags)
        lon = ds["longitude"].values.astype(np.float64)
        lat = ds["latitude"].values.astype(np.float64)
        fields = tpl.values.astype(np.float64)  # shape (lag, lat, lon)

    # Load MI by lag for Phase 5 (continuous-intensity)
    import os
    mi_path_candidates = []
    if mi_csv is not None:
        mi_path_candidates.append(Path(mi_csv))
    mi_path_candidates.extend([
        Path("manuscript_results/info_gain_cont_gatealigned_observed.csv"),
        Path("info_gain_cont_gatealigned_observed.csv"),
    ])
    mi_df = None
    for pth in mi_path_candidates:
        if pth.exists():
            try:
                df = pd.read_csv(pth)
                if {"phase", "lag", "MI_bits"}.issubset(df.columns):
                    mi_df = df
                    break
            except Exception:
                pass
    if mi_df is None:
        raise FileNotFoundError("Could not locate MI-by-lag CSV (info_gain_cont_gatealigned_observed.csv)")
    mi5 = mi_df.loc[mi_df["phase"] == 5, ["lag", "MI_bits"]].dropna().groupby("lag").mean().reset_index()
    mi_map = {int(row.lag): float(row.MI_bits) for _, row in mi5.iterrows()}
    mi_vals = np.array([mi_map.get(L, np.nan) for L in lags], dtype=np.float64)
    if not np.isfinite(mi_vals).any():
        raise RuntimeError("No MI values found for Phase 5 in MI CSV")
    peak_row = mi5.loc[mi5["MI_bits"].idxmax()]
    mi_ref = float(peak_row["MI_bits"])
    peak_lag = int(peak_row["lag"])
    if not np.isfinite(mi_ref) or mi_ref <= 0:
        mi_ref = float(np.nanmax(mi_vals))
        peak_lag = int(lags[int(np.nanargmax(mi_vals))])
    # Clamp MI at 0 to avoid negative weights from small-sample noise
    w = np.where(np.isfinite(mi_vals) & (mi_ref > 0), np.clip(mi_vals, 0.0, None) / mi_ref, 0.0)

    # MI-weighted templates
    fields_weighted = (w[:, None, None] * fields).astype(np.float64)

    # Shared color limits based on 99th percentile of |L=24 weighted panel| to avoid saturation by outliers
    try:
        idx24 = lags.index(24)
        abs_vals = np.abs(fields_weighted[idx24]).ravel()
    except Exception:
        abs_vals = np.abs(fields_weighted.reshape(fields_weighted.shape[0], -1)).ravel()
    finite_abs = abs_vals[np.isfinite(abs_vals)]
    if finite_abs.size == 0:
        vmax = 1.0
    else:
        vmax = float(np.nanpercentile(finite_abs, 99.0))
        vmax = max(vmax, 1e-6)
    levels = np.linspace(-vmax, vmax, 23)

    # Weighted correlation diagnostics vs L=24 (information-weighted)
    wlat = np.sqrt(np.clip(np.cos(np.deg2rad(lat)), 0.0, None))
    w2d = wlat[:, None] * np.ones((1, lon.size), dtype=np.float64)
    ref = fields_weighted[idx24]
    corr = [weighted_pattern_corr(fields_weighted[i], ref, w2d) for i in range(fields_weighted.shape[0])]

    # Projection setup (Cartopy Plate Carrée if available)
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        # Center map on the Pacific to avoid antimeridian seam issues
        proj = ccrs.PlateCarree(central_longitude=210)
        use_cartopy = True
    except Exception:
        proj = None
        use_cartopy = False

    # 2x2 grid with minimal whitespace; horizontal colorbar at bottom
    fig = plt.figure(figsize=(10.5, 6.4))
    gs = fig.add_gridspec(2, 2, wspace=0.04, hspace=0.04)

    axes_maps = []
    letters = ["(A)", "(B)", "(C)", "(D)"]
    for i, L in enumerate(lags):
        r, c = divmod(i, 2)
        ax = fig.add_subplot(gs[r, c], projection=proj) if use_cartopy else fig.add_subplot(gs[r, c])
        if use_cartopy:
            # Clip to receiver domain (Taiwan to eastern Canada)
            ax.set_extent([120, 300, 20, 75], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="0.4")
            im = ax.contourf(lon, lat, fields_weighted[i], levels=levels, cmap="RdBu_r",
                             extend="both", transform=ccrs.PlateCarree())
        else:
            fldp, latp = _prep_lat_lon_field(fields_weighted[i], lat)
            im = ax.contourf(lon, latp, fldp, levels=levels, cmap="RdBu_r", extend="both")
            ax.set_xlim(120, 300)
            ax.set_ylim(20, 75)
        ax.set_xticks([])
        ax.set_yticks([])
        # Annotate with panel letter and MI at this lag (clamped at 0)
        mi_here = mi_map.get(L, np.nan)
        mi_disp = float(max(mi_here, 0.0)) if np.isfinite(mi_here) else 0.0
        ax.text(
            0.02,
            0.98,
            f"{letters[i]}  Lag {L} d | MI = {mi_disp:.3f} bits",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
        )
        axes_maps.append(ax)

    cbar = fig.colorbar(im, ax=axes_maps, orientation='horizontal', pad=0.06, fraction=0.06)
    cbar.set_label("MI-scaled weighted template (m per source-index unit)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    out: Dict[str, float] = {
        "phase5_mi_peak_bits": float(mi_ref),
        "phase5_mi_peak_lag": float(peak_lag),
        "phase5_weighted_vmax_m": vmax,
    }
    for i, L in enumerate(lags):
        out[f"phase5_mi_bits_lag{L}"] = float(mi_map.get(L, np.nan))
        out[f"phase5_weight_corr_vs24_lag{L}"] = float(corr[i]) if np.isfinite(corr[i]) else np.nan
    return out


def plot_phase6_snapshot(codebook_nc: Path, ig_summary_csv: Path | None, out_path: Path,
                         mi_csv: Path | None = None, ig_csv: Path | None = None,
                         normalise_mode: str = "unit_variance") -> Dict[str, float]:
    # Load MI (continuous) and IG (gated) CSVs
    mi_path_candidates = []
    if mi_csv is not None:
        mi_path_candidates.append(Path(mi_csv))
    mi_path_candidates.extend([
        Path("manuscript_results/info_gain_cont_gatealigned_observed.csv"),
        Path("info_gain_cont_gatealigned_observed.csv"),
    ])
    ig_path_candidates = []
    if ig_csv is not None:
        ig_path_candidates.append(Path(ig_csv))
    if ig_summary_csv is not None:
        ig_path_candidates.append(Path(ig_summary_csv))
    ig_path_candidates.extend([
        Path("manuscript_results/info_gain_p85_observed.csv"),
        Path("manuscript_results/info_gain_v2_p85_observed.csv"),
        Path("info_gain_p85_observed.csv"),
        Path("info_gain_v2_p85_observed.csv"),
    ])
    mi_df = None
    for pth in mi_path_candidates:
        if pth.exists():
            try:
                df = pd.read_csv(pth)
                if {"phase", "lag", "MI_bits"}.issubset(df.columns):
                    mi_df = df
                    break
            except Exception:
                pass
    if mi_df is None:
        raise FileNotFoundError("MI CSV (continuous) not found")
    ig_df = None
    for pth in ig_path_candidates:
        if pth.exists():
            try:
                df = pd.read_csv(pth)
                if {"phase", "lag", "IG_bits"}.issubset(df.columns):
                    ig_df = df
                    break
            except Exception:
                pass
    if ig_df is None:
        raise FileNotFoundError("IG CSV (p85 observed) not found")

    # Select metric-specific lags used in the manuscript scorecard.
    def value_at_lag(df, phase, lag, col):
        sub = df[(df["phase"] == phase) & (df["lag"] == lag)].dropna(subset=[col])
        if sub.empty:
            return None
        return float(sub.iloc[0][col])

    lag5_mi = 25
    lag5_ig = 13
    lag6_mi = 8
    lag6_ig = 8
    mi5 = value_at_lag(mi_df, 5, lag5_mi, "MI_bits")
    mi6 = value_at_lag(mi_df, 6, lag6_mi, "MI_bits")
    ig5 = value_at_lag(ig_df, 5, lag5_ig, "IG_bits")
    ig6 = value_at_lag(ig_df, 6, lag6_ig, "IG_bits")
    if None in (mi5, mi6, ig5, ig6):
        raise RuntimeError("Could not retrieve MI/IG values at requested lags (P5 MI:25, P5 IG:13, P6:8)")

    # Global MI reference (peak across all phases for scaling)
    mi_global_peak = float(mi_df["MI_bits"].max())
    if not np.isfinite(mi_global_peak) or mi_global_peak <= 0:
        mi_global_peak = 1.0

    # Load templates
    with xr.open_dataset(codebook_nc) as ds:
        lon = ds["longitude"].values.astype(np.float64)
        lat = ds["latitude"].values.astype(np.float64)
        t5w = ds["z500_template_weighted"].sel(phase=5, lag=lag5_mi).values.astype(np.float64)
        t5g = ds["z500_template_gated"].sel(phase=5, lag=lag5_ig).values.astype(np.float64)
        t6w = ds["z500_template_weighted"].sel(phase=6, lag=lag6_mi).values.astype(np.float64)
        t6g = ds["z500_template_gated"].sel(phase=6, lag=lag6_ig).values.astype(np.float64)

    # Normalization: compute area-weighted spatial standard deviation
    def area_weighted_rms(field: np.ndarray, latitudes: np.ndarray) -> float:
        w = np.clip(np.cos(np.deg2rad(latitudes)), 0.0, None)
        if np.all(w == 0) or not np.isfinite(field).any():
            return 1.0
        ww = w[:, None]
        num = float(np.nansum(ww * (field ** 2)))
        den = float(np.nansum(ww))
        rms = np.sqrt(num / max(den, 1e-12))
        return float(max(rms, 1e-12))

    if normalise_mode == "unit_variance":
        n5w = area_weighted_rms(t5w, lat)
        n5g = area_weighted_rms(t5g, lat)
        n6w = area_weighted_rms(t6w, lat)
        n6g = area_weighted_rms(t6g, lat)
    elif normalise_mode == "phase5_reference":
        n_ref = area_weighted_rms(t5w, lat)
        n5w = n5g = n6w = n6g = n_ref
    else:
        n5w = n5g = n6w = n6g = 1.0

    # Scale by bits relative to global MI peak, after normalization
    s5w = mi5 / mi_global_peak if mi_global_peak > 0 else 1.0
    s6w = mi6 / mi_global_peak if mi_global_peak > 0 else 1.0
    s5g = ig5 / mi_global_peak if mi_global_peak > 0 else 1.0
    s6g = ig6 / mi_global_peak if mi_global_peak > 0 else 1.0
    p5w = s5w * (t5w / n5w)
    p5g = s5g * (t5g / n5g)
    p6w = s6w * (t6w / n6w)
    p6g = s6g * (t6g / n6g)

    # Shared color limits from the 99th percentile of |Phase 5 weighted| to avoid saturation
    finite_abs = np.abs(p5w).ravel()
    finite_abs = finite_abs[np.isfinite(finite_abs)]
    if finite_abs.size == 0:
        vmax = 1.0
    else:
        vmax = float(np.nanpercentile(finite_abs, 99.0))
        vmax = max(vmax, 1e-6)
    levels = np.linspace(-vmax, vmax, 23)

    # Projection setup (Cartopy if available)
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        # Center map on the Pacific to avoid antimeridian seam issues
        proj = ccrs.PlateCarree(central_longitude=210)
        use_cartopy = True
    except Exception:
        proj = None
        use_cartopy = False

    # Layout 2x2 with minimal whitespace; horizontal colorbar at bottom
    fig = plt.figure(figsize=(10.5, 6.4))
    gs = fig.add_gridspec(2, 2, wspace=0.04, hspace=0.04)
    entries = [
        (0, 0, p5w, "(A)", f"Phase 5 weighted, lag {lag5_mi} d | MI = {mi5:.3f} bits"),
        (0, 1, p5g, "(B)", f"Phase 5 p90 gated diag., lag {lag5_ig} d | p85 IG = {ig5:.3f} bits"),
        (1, 0, p6w, "(C)", f"Phase 6 weighted, lag {lag6_mi} d | MI = {mi6:.3f} bits"),
        (1, 1, p6g, "(D)", f"Phase 6 p90 gated diag., lag {lag6_ig} d | p85 IG = {ig6:.3f} bits"),
    ]
    axes = []
    for r, c, fld, letter, label in entries:
        if use_cartopy:
            ax = fig.add_subplot(gs[r, c], projection=proj)
            # Clip to receiver domain (Taiwan to eastern Canada)
            ax.set_extent([120, 300, 20, 75], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="0.4")
            im = ax.contourf(lon, lat, fld, levels=levels, cmap="RdBu_r",
                             extend="both", transform=ccrs.PlateCarree())
        else:
            ax = fig.add_subplot(gs[r, c])
            fldp, latp = _prep_lat_lon_field(fld, lat)
            im = ax.contourf(lon, latp, fldp, levels=levels, cmap="RdBu_r", extend="both")
            ax.set_xlim(120, 300)
            ax.set_ylim(20, 75)
        ax.set_xticks([])
        ax.set_yticks([])
        # Panel letter and annotation inside panel
        ax.text(
            0.02,
            0.98,
            f"{letter}  {label}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
        )
        axes.append(ax)

    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.06, fraction=0.06)
    if normalise_mode == "unit_variance":
        cbar.set_label("MI/IG-scaled normalized template loading (dimensionless)")
    else:
        cbar.set_label("MI/IG-scaled template loading")
    # No figure title per revision
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "p5_mi_bits": mi5,
        "p6_mi_bits": mi6,
        "p5_ig_bits": ig5,
        "p6_ig_bits": ig6,
        "p5_mi_lag": float(lag5_mi),
        "p6_mi_lag": float(lag6_mi),
        "p5_ig_lag": float(lag5_ig),
        "p6_ig_lag": float(lag6_ig),
        "mi_global_peak_bits": mi_global_peak,
        "phase5_weighted_vmax_m": vmax,
    }


def _phase_gate_samples_shifted(
    ds_test: xr.Dataset, phase: int, lag: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phase_idx = ds_test["driver_phase_index"].values.astype(np.int16)
    gate = ds_test["burst_gate"].values.astype(np.float64)
    amp = ds_test["z500_amp_weighted"].sel(lag=lag, phase=phase).values.astype(np.float64)
    t_phase = np.where((phase_idx == (phase - 1)) & np.isfinite(gate))[0].astype(int)
    t_shift = t_phase + int(lag)
    in_bounds = (t_shift >= 0) & (t_shift < amp.size)
    t_phase = t_phase[in_bounds]
    t_shift = t_shift[in_bounds]
    scores = amp[t_shift]
    labels = (gate[t_phase] > 0.5).astype(np.int8)
    valid = np.isfinite(scores) & np.isfinite(labels)
    scores = scores[valid]
    labels = labels[valid]
    on = scores[labels == 1]
    off = scores[labels == 0]
    all_scores = scores
    return on, off, all_scores, labels


def plot_receiver_pdfs(
    receiver_nc: Path,
    ig_summary_csv: Path,
    out_path: Path,
) -> Dict[str, float]:
    ig = pd.read_csv(ig_summary_csv).set_index("phase")
    phase_signal = 5
    lag_signal = int(ig.loc[phase_signal, "best_lag_obs"])

    with xr.open_dataset(receiver_nc) as ds:
        ds_test = select_test(ds)
        on_s, off_s, scores_s, labels_s = _phase_gate_samples_shifted(ds_test, phase_signal, lag_signal)
        # Pick a null phase from 1-4 with adequate sample size and near-random discrimination.
        candidates: List[Tuple[float, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for ph in [1, 2, 3, 4]:
            lag = int(ig.loc[ph, "best_lag_obs"])
            on, off, scores, labels = _phase_gate_samples_shifted(ds_test, ph, lag)
            auc = auc_rank(scores, labels)
            if on.size >= 15 and off.size >= 150 and np.isfinite(auc):
                candidates.append((abs(auc - 0.5), ph, lag, on, off, scores, labels))
        if candidates:
            candidates.sort(key=lambda r: r[0])
            _, phase_null, lag_null, on_n, off_n, scores_n, labels_n = candidates[0]
        else:
            phase_null = 3
            lag_null = int(ig.loc[phase_null, "best_lag_obs"])
            on_n, off_n, scores_n, labels_n = _phase_gate_samples_shifted(ds_test, phase_null, lag_null)

    pooled = np.r_[scores_s, scores_n]
    xlo, xhi = np.nanpercentile(pooled, [1.0, 99.0])
    bins = fd_bins(pooled, min_bins=20, max_bins=80)
    bins = bins[(bins >= xlo) & (bins <= xhi)]
    if bins.size < 12:
        bins = np.linspace(xlo, xhi, 31)

    def cohend(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 2 or b.size < 2:
            return np.nan
        va = np.var(a, ddof=1)
        vb = np.var(b, ddof=1)
        pooled_v = ((a.size - 1) * va + (b.size - 1) * vb) / max(a.size + b.size - 2, 1)
        return float((np.mean(a) - np.mean(b)) / np.sqrt(max(pooled_v, 1e-12)))

    auc_s = auc_rank(scores_s, labels_s)
    auc_n = auc_rank(scores_n, labels_n)
    d_s = cohend(on_s, off_s)
    d_n = cohend(on_n, off_n)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), constrained_layout=True, sharey=True)
    panels = [
        (axes[0], phase_signal, lag_signal, on_s, off_s, auc_s, d_s),
        (axes[1], phase_null, lag_null, on_n, off_n, auc_n, d_n),
    ]
    for ax, phase, lag, on, off, auc, dval in panels:
        ax.hist(off, bins=bins, density=True, alpha=0.35, color="#9e9e9e", label="Gate-off")
        ax.hist(on, bins=bins, density=True, alpha=0.45, color="#1f77b4", label="Gate-on")
        ax.axvline(np.mean(off), color="#595959", lw=1.2, ls="--")
        ax.axvline(np.mean(on), color="#08306b", lw=1.2, ls="--")
        ax.set_xlim(xlo, xhi)
        ax.set_xlabel("Receiver amplitude")
        ax.grid(alpha=0.22)
        ax.set_title(
            f"Phase {phase}, lag {lag} d\n"
            f"n_on={on.size}, n_off={off.size}, AUC={auc:.3f}, d={dval:.2f}"
        )
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)
    fig.suptitle("Receiver PDFs: Gate-on vs Gate-off")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "pdf_signal_phase": float(phase_signal),
        "pdf_signal_lag": float(lag_signal),
        "pdf_null_phase": float(phase_null),
        "pdf_null_lag": float(lag_null),
        "pdf_signal_auc": float(auc_s),
        "pdf_null_auc": float(auc_n),
        "pdf_signal_cohend": float(d_s),
        "pdf_null_cohend": float(d_n),
        "pdf_signal_n_on": float(on_s.size),
        "pdf_signal_n_off": float(off_s.size),
        "pdf_null_n_on": float(on_n.size),
        "pdf_null_n_off": float(off_n.size),
    }


def plot_four_metric_scorecard(
    capacity_csv: Path, coding_csv: Path, outage_csv: Path, out_path: Path
) -> Dict[str, float]:
    cap = pd.read_csv(capacity_csv).set_index("phase")
    cod = pd.read_csv(coding_csv).set_index("phase")
    out = pd.read_csv(outage_csv).set_index("phase")
    phases = np.arange(1, 9, dtype=int)

    spectral = cap.loc[phases, "spectral_eff_bits_per_symbol"].values.astype(np.float64)
    throughput = cap.loc[phases, "throughput_bits_per_day"].values.astype(np.float64)
    coding = cod.loc[phases, "coding_efficiency"].values.astype(np.float64)
    outage = out.loc[phases, "outage_at_pfa"].values.astype(np.float64)

    colors = ["#4c78a8"] * phases.size
    colors[4] = "#d62728"  # highlight phase 5

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2), constrained_layout=True)
    specs = [
        ("A", "Spectral efficiency", spectral, "bits/symbol", None),
        ("B", "Realized throughput", throughput, "bits/day", None),
        ("C", "Coding efficiency", coding, "fraction of H(G)", None),
        ("D", "Outage probability", outage, "probability", (0.0, 1.0)),
    ]
    for ax, (panel, title, vals, ylab, ylim) in zip(axes.ravel(), specs):
        ax.bar(phases, vals, color=colors, edgecolor="white", linewidth=0.7)
        ax.set_title(title)
        ax.set_xlabel("MJO phase")
        ax.set_ylabel(ylab)
        ax.set_xticks(phases)
        ax.grid(axis="y", alpha=0.25)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(0.0, max(0.01, 1.08 * np.nanmax(vals)))
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

    return {
        "scorecard_top_spectral_phase": float(phases[int(np.nanargmax(spectral))]),
        "scorecard_top_throughput_phase": float(phases[int(np.nanargmax(throughput))]),
        "scorecard_top_coding_phase": float(phases[int(np.nanargmax(coding))]),
        "scorecard_lowest_outage_phase": float(phases[int(np.nanargmin(outage))]),
    }


def plot_roc_curves(
    receiver_nc: Path, ig_summary_csv: Path, out_path: Path, phases: List[int] | None = None
) -> Dict[str, float]:
    if phases is None:
        phases = [5, 6, 2]
    ig = pd.read_csv(ig_summary_csv).set_index("phase")

    auc_diag: Dict[str, float] = {}
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    with xr.open_dataset(receiver_nc) as ds:
        ds_test = select_test(ds)
        for phase in phases:
            lag = int(ig.loc[phase, "best_lag_obs"]) if phase in ig.index else 8
            _, _, scores, labels = _phase_gate_samples_shifted(ds_test, phase, lag)
            fpr, tpr = roc_curve(scores, labels)
            auc = auc_rank(scores, labels)
            ax.plot(fpr, tpr, lw=2.0, label=f"Phase {phase}, lag {lag} d (AUC={auc:.3f}, n_on={int(labels.sum())})")
            auc_diag[f"roc_auc_phase{phase}"] = float(auc)
            auc_diag[f"roc_lag_phase{phase}"] = float(lag)
            auc_diag[f"roc_n_on_phase{phase}"] = float(np.sum(labels == 1))
            auc_diag[f"roc_n_off_phase{phase}"] = float(np.sum(labels == 0))

    ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="No-skill")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC Curves for Contrasting Phases")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return auc_diag


def write_summary(summary_path: Path, figure_paths: Dict[str, Path], diag: Dict[str, float]) -> None:
    lines: List[str] = []
    lines.append("# Additional Reviewer Figures Summary (Generated 2026-03-03)")
    lines.append("")
    lines.append("## Figure Files")
    lines.append("")
    for key, path in figure_paths.items():
        lines.append(f"- `{key}`: `{path}`")
    lines.append("")
    lines.append("## Quantitative Diagnostics")
    lines.append("")
    lines.append("### 1) WK Spectrum")
    lines.append(
        f"- Peak enhanced power in retained band: `k={diag.get('wk_peak_k', np.nan):.2f}`, "
        f"`f={diag.get('wk_peak_freq_cpd', np.nan):.4f}` cpd "
        f"(period `{diag.get('wk_peak_period_days', np.nan):.1f}` days)."
    )
    lines.append(f"- Mean power/background ratio inside retained band: `{diag.get('wk_band_mean_power_ratio', np.nan):.3f}`.")
    if np.isfinite(diag.get('wk_segments_used', np.nan)):
        lines.append(f"- Welch segments used: `{diag.get('wk_segments_used', np.nan):.0f}` (96 d windows; 65 d overlap).")
    lines.append("")
    lines.append("### 2) EOF Structure")
    lines.append(
        f"- Variance explained: EOF1 `{diag.get('eof1_var_frac_pct', np.nan):.2f}%`, "
        f"EOF2 `{diag.get('eof2_var_frac_pct', np.nan):.2f}%`, "
        f"EOF1+EOF2 `{diag.get('eof12_var_frac_pct', np.nan):.2f}%`."
    )
    lines.append(
        f"- PC orthogonality check: corr(PC1,PC2) at lag 0 = `{diag.get('pc1_pc2_corr0', np.nan):.3f}`; "
        f"max abs lead-lag corr = `{diag.get('pc1_pc2_maxabs_lag_corr', np.nan):.3f}` at "
        f"{diag.get('pc1_pc2_maxabs_lag_days', np.nan):+.0f} days."
    )
    lines.append("")
    lines.append("### 3) Phase-Template Diagnostics")
    # Phase 5 MI-weighted evolution summary
    if np.isfinite(diag.get('phase5_mi_peak_bits', np.nan)):
        lines.append(
            f"- Phase 5 MI peak: `{diag.get('phase5_mi_peak_bits', np.nan):.3f}` bits at lag "
            f"`{diag.get('phase5_mi_peak_lag', np.nan):.0f}` days."
        )
        # Report MI at plotted lags if available
        items = []
        for L in [0, 4, 8, 12, 16, 20, 24]:
            key = f'phase5_mi_bits_lag{L}'
            if key in diag and np.isfinite(diag.get(key, np.nan)):
                items.append(f"lag{L}:{diag.get(key):.3f}")
        if items:
            lines.append("- MI at plotted lags (bits): " + ", ".join(items) + ".")
    # Companion 2×2 MI/IG-weighted snapshot summary
    if np.isfinite(diag.get('p5_mi_bits', np.nan)) or np.isfinite(diag.get('p6_ig_bits', np.nan)):
        lines.append(
            f"- 2×2 snapshot scaling: MI global peak `{diag.get('mi_global_peak_bits', np.nan):.3f}` bits; "
            f"Phase 5 weighted `{diag.get('p5_mi_bits', np.nan):.3f}` (lag {diag.get('p5_mi_lag', np.nan):.0f}), "
            f"Phase 5 gated `{diag.get('p5_ig_bits', np.nan):.3f}` (lag {diag.get('p5_ig_lag', np.nan):.0f}); "
            f"Phase 6 weighted `{diag.get('p6_mi_bits', np.nan):.3f}` (lag {diag.get('p6_mi_lag', np.nan):.0f}), "
            f"Phase 6 gated `{diag.get('p6_ig_bits', np.nan):.3f}` (lag {diag.get('p6_ig_lag', np.nan):.0f})."
        )
    lines.append("")
    lines.append("### 4) Receiver PDFs and ROC")
    lines.append(
        f"- PDF contrast (signal phase {diag.get('pdf_signal_phase', np.nan):.0f}, lag {diag.get('pdf_signal_lag', np.nan):.0f}): "
        f"AUC `{diag.get('pdf_signal_auc', np.nan):.3f}`, Cohen's d `{diag.get('pdf_signal_cohend', np.nan):.2f}`."
    )
    lines.append(
        f"- PDF contrast (null phase {diag.get('pdf_null_phase', np.nan):.0f}, lag {diag.get('pdf_null_lag', np.nan):.0f}): "
        f"AUC `{diag.get('pdf_null_auc', np.nan):.3f}`, Cohen's d `{diag.get('pdf_null_cohend', np.nan):.2f}`."
    )
    roc_keys = sorted([k for k in diag if k.startswith("roc_auc_phase")])
    for key in roc_keys:
        phase = key.replace("roc_auc_phase", "")
        lines.append(
            f"- ROC phase {phase}: AUC `{diag[key]:.3f}` "
            f"(lag {diag.get(f'roc_lag_phase{phase}', np.nan):.0f}, "
            f"n_on {diag.get(f'roc_n_on_phase{phase}', np.nan):.0f}, "
            f"n_off {diag.get(f'roc_n_off_phase{phase}', np.nan):.0f})."
        )
    lines.append("")
    lines.append("### 5) Four-Metric Scorecard")
    lines.append(
        f"- Top spectral-efficiency phase: `{diag.get('scorecard_top_spectral_phase', np.nan):.0f}`; "
        f"top throughput phase: `{diag.get('scorecard_top_throughput_phase', np.nan):.0f}`."
    )
    lines.append(
        f"- Top coding-efficiency phase: `{diag.get('scorecard_top_coding_phase', np.nan):.0f}`; "
        f"lowest outage phase: `{diag.get('scorecard_lowest_outage_phase', np.nan):.0f}`."
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- WK spectrum follows WK99: sym/anti decomposition (15S–15N), 96-day overlapping segments (step 31), 10% Tukey time taper, no temporal subsampling; background = (sym+anti)/2 smoothed by repeated 1–2–1 filters (freq×40, wav×20).")
    lines.append("- EOF scree computed on a coarsened 2-degree grid over 60-180E, 15S-15N; map panels and scree are internally consistent.")
    lines.append("- ROC/PDF diagnostics follow the manuscript convention: within-phase gate-on/off classification using `burst_gate` and phase-matched receiver amplitudes.")
    lines.append("")
    summary_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    results_dir = Path(args.results_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    mjo_index_nc = Path(args.mjo_index_nc)
    receiver_nc = Path(args.receiver_nc)
    diag_csv = Path(args.diag_csv)
    summary_md = Path(args.summary_md)

    figure_paths = {
        "wk_spectrum": fig_dir / "fig13_wk_surprisal_spectrum.png",
        "eof_patterns": fig_dir / "fig14_eof_patterns_variance.png",
        "phase5_lag_evolution": fig_dir / "fig15_phase5_codebook_lag_evolution.png",
        "phase6_companion": fig_dir / "fig16_phase6_companion_snapshot.png",
        "receiver_pdfs": fig_dir / "fig17_receiver_pdf_gate_on_off.png",
        "four_metric_scorecard": fig_dir / "fig18_four_metric_scorecard.png",
        "roc_curves": fig_dir / "fig19_roc_curves_phase_contrast.png",
    }

    diag: Dict[str, float] = {}
    diag.update(plot_wk_spectrum(mjo_index_nc, figure_paths["wk_spectrum"]))
    diag.update(plot_eof_patterns_and_variance(mjo_index_nc, figure_paths["eof_patterns"]))
    # Attempt to use MI CSVs if available; function falls back to defaults
    mi_csv = Path("manuscript_results/info_gain_cont_gatealigned_observed.csv")
    if not mi_csv.exists():
        mi_csv = Path("info_gain_cont_gatealigned_observed.csv")
    diag.update(plot_phase5_lag_evolution(receiver_nc, figure_paths["phase5_lag_evolution"], mi_csv=mi_csv))
    ig_csv = results_dir / "info_gain_p85_observed.csv"
    if not ig_csv.exists():
        ig_csv = Path("info_gain_p85_observed.csv")
    diag.update(
        plot_phase6_snapshot(
            receiver_nc,
            ig_csv,
            figure_paths["phase6_companion"],
            mi_csv=mi_csv,
            ig_csv=ig_csv,
        )
    )
    diag.update(
        plot_receiver_pdfs(
            receiver_nc,
            results_dir / "info_gain_p85_nullsummary.csv",
            figure_paths["receiver_pdfs"],
        )
    )
    diag.update(
        plot_four_metric_scorecard(
            results_dir / "telecom_capacity_throughput.csv",
            results_dir / "telecom_coding_efficiency.csv",
            results_dir / "telecom_outage_table.csv",
            figure_paths["four_metric_scorecard"],
        )
    )
    null_phase = int(diag.get("pdf_null_phase", 3.0))
    roc_phases = [5, 6, null_phase]
    if len(set(roc_phases)) < 3:
        for cand in [3, 2, 4, 7]:
            if cand not in roc_phases:
                roc_phases.append(cand)
            if len(set(roc_phases)) >= 3:
                break
    roc_phases = list(dict.fromkeys(roc_phases))[:3]
    diag.update(
        plot_roc_curves(
            receiver_nc,
            results_dir / "info_gain_p85_nullsummary.csv",
            figure_paths["roc_curves"],
            phases=roc_phases,
        )
    )

    df_diag = pd.DataFrame({"metric": list(diag.keys()), "value": list(diag.values())})
    df_diag.to_csv(diag_csv, index=False)
    write_summary(summary_md, figure_paths, diag)

    print(f"Wrote diagnostics CSV: {diag_csv}")
    print(f"Wrote summary markdown: {summary_md}")
    for key, path in figure_paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
