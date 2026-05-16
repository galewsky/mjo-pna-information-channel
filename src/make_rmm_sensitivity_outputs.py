#!/usr/bin/env python3
"""
Assemble figures, tables, and machine-readable summary products for the RMM sensitivity test.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def binary_entropy(p: float) -> float:
    if not np.isfinite(p) or p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def load_heatmap(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return df.pivot(index="phase", columns="lag", values=value_col).sort_index().sort_index(axis=1)


def load_mask(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return df.pivot(index="phase", columns="lag", values=value_col).sort_index().sort_index(axis=1)


def plot_heatmap(
    df: pd.DataFrame,
    value_col: str,
    out_path: Path,
    *,
    title: str,
    cbar_label: str,
    vmin: float = 0.0,
    vmax: float | None = None,
    cmap: str = "viridis",
    underpowered_col: str = "underpowered",
) -> None:
    mat = load_heatmap(df, value_col)
    arr = mat.values.astype(float)
    if vmax is None:
        vmax = float(np.nanmax(arr))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = max(vmin + 1.0e-6, 1.0)

    fig, ax = plt.subplots(figsize=(10.0, 4.0))
    im = ax.imshow(arr, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)

    ax.set_title(title)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Phase")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index.astype(int))
    lag_vals = mat.columns.astype(int).to_numpy()
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(lag_vals)
    if mat.shape[1] > 10:
        keep = np.arange(0, mat.shape[1], 5)
        ax.set_xticks(keep)
        ax.set_xticklabels(lag_vals[keep])

    if underpowered_col in df.columns:
        up = load_mask(df, underpowered_col).fillna(0).values.astype(bool)
        if np.any(up):
            xx = np.arange(mat.shape[1] + 1) - 0.5
            yy = np.arange(mat.shape[0] + 1) - 0.5
            overlay = np.ma.masked_where(~up, np.ones_like(up, dtype=float))
            ax.pcolor(xx, yy, overlay, hatch="///", alpha=0.0, edgecolor="0.35", linewidth=0.0)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def summarize_metric(obs_csv: Path, sum_csv: Path, value_col: str) -> pd.DataFrame:
    df_obs = pd.read_csv(obs_csv)
    df_sum = pd.read_csv(sum_csv).set_index("phase")
    rows = []
    for phase in sorted(df_sum.index.astype(int)):
        lag = int(df_sum.loc[phase, "best_lag_obs"])
        row = df_obs[(df_obs["phase"] == phase) & (df_obs["lag"] == lag)]
        value = float(row.iloc[0][value_col]) if not row.empty else np.nan
        rows.append(
            {
                "phase": int(phase),
                "best_lag": lag,
                "value": value,
                "q_phase": float(df_sum.loc[phase, "q_phase"]) if "q_phase" in df_sum.columns else np.nan,
                "p_phase": float(df_sum.loc[phase, "p_phase"]) if "p_phase" in df_sum.columns else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("phase").reset_index(drop=True)


def build_scorecard(
    receiver_nc: Path,
    mi_obs_csv: Path,
    mi_sum_csv: Path,
    ig_obs_csv: Path,
    ig_sum_csv: Path,
    pd_csv: Path,
    outage_csv: Path,
    auc_sum_csv: Path,
    snr_sum_csv: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ds = xr.open_dataset(receiver_nc)
    time = pd.to_datetime(ds["time"].values)
    eval_mask = (time >= pd.Timestamp("2011-01-01")) & (time <= pd.Timestamp("2020-12-31"))
    phase_idx = ds["driver_phase_index"].values.astype(int)
    phase_raw = ds["driver_rmm_phase_raw"].values.astype(float) if "driver_rmm_phase_raw" in ds else np.where(phase_idx >= 0, phase_idx + 1, np.nan)
    gate = ds["burst_gate"].values.astype(float)

    mi_best = summarize_metric(mi_obs_csv, mi_sum_csv, "MI_bits").set_index("phase")
    ig_best = summarize_metric(ig_obs_csv, ig_sum_csv, "IG_bits").set_index("phase")
    df_pd = pd.read_csv(pd_csv)
    df_out = pd.read_csv(outage_csv)
    df_auc = pd.read_csv(auc_sum_csv).set_index("phase")
    df_snr = pd.read_csv(snr_sum_csv).set_index("phase")

    rows = []
    duty_rows = []
    for phase in range(1, 9):
        idx = phase - 1
        occ_active = float(np.mean((phase_idx[eval_mask] == idx)))
        occ_raw = float(np.mean(phase_raw[eval_mask] == float(phase)))
        active_mask = eval_mask & (phase_idx == idx)
        p_on = float(np.nanmean(gate[active_mask])) if np.any(active_mask) else np.nan
        h_g = binary_entropy(p_on)

        mi_val = float(mi_best.loc[phase, "value"]) if phase in mi_best.index else np.nan
        lag_mi = int(mi_best.loc[phase, "best_lag"]) if phase in mi_best.index else -1
        ig_val = float(ig_best.loc[phase, "value"]) if phase in ig_best.index else np.nan
        lag_ig = int(ig_best.loc[phase, "best_lag"]) if phase in ig_best.index else -1

        row_pd = df_pd[(df_pd["phase"] == phase) & (df_pd["lag"] == lag_ig)]
        row_out = df_out[(df_out["phase"] == phase) & (df_out["lag"] == lag_ig)]
        pd_at_pfa = float(row_pd.iloc[0]["pd_at_pfa"]) if not row_pd.empty else np.nan
        outage_q20 = float(row_out.iloc[0]["outage"]) if not row_out.empty else np.nan

        auc_q = float(df_auc.loc[phase, "q_phase"]) if phase in df_auc.index else np.nan
        snr_q = float(df_snr.loc[phase, "q_phase"]) if phase in df_snr.index else np.nan

        rows.append(
            {
                "phase": phase,
                "spectral_efficiency": mi_val,
                "throughput": occ_active * mi_val if np.isfinite(mi_val) else np.nan,
                "throughput_raw_phase_occupancy": occ_raw * mi_val if np.isfinite(mi_val) else np.nan,
                "coding_efficiency": (ig_val / h_g) if (np.isfinite(ig_val) and h_g > 0.0) else np.nan,
                "outage_at_pfa": (1.0 - pd_at_pfa) if np.isfinite(pd_at_pfa) else np.nan,
                "outage_q20": outage_q20,
                "lag_mi": lag_mi,
                "lag_ig": lag_ig,
                "mi_bits": mi_val,
                "ig_bits": ig_val,
                "phase_occupancy_active_days_all_eval": occ_active,
                "phase_occupancy_raw_days_all_eval": occ_raw,
                "gate_duty_cycle_within_phase": p_on,
                "H_G_bits": h_g,
                "auc_q": auc_q,
                "snr_q": snr_q,
            }
        )
        duty_rows.append(
            {
                "phase": phase,
                "phase_days_raw": int(np.sum(eval_mask & (phase_raw == float(phase)))),
                "phase_days_active": int(np.sum(active_mask)),
                "occupancy_raw_days_all_eval": occ_raw,
                "occupancy_active_days_all_eval": occ_active,
                "gate_duty_cycle_within_phase": p_on,
                "H_G_bits": h_g,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(duty_rows)


def plot_grouped_bars(
    phases: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    out_path: Path,
    *,
    left_label: str,
    right_label: str,
    title: str,
    ylabel: str,
    left_color: str = "#1f77b4",
    right_color: str = "#ff7f0e",
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    x = np.arange(phases.size)
    width = 0.38
    ax.bar(x - width / 2, left, width=width, color=left_color, label=left_label)
    ax.bar(x + width / 2, right, width=width, color=right_color, label=right_label)
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_xlabel("Phase")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_scorecard(score_df: pd.DataFrame, out_path: Path) -> None:
    phases = score_df["phase"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True)
    panels = [
        ("spectral_efficiency", "Spectral efficiency", "bits per symbol", "#1f77b4"),
        ("throughput", "Realized throughput", "bits per day", "#2ca02c"),
        ("coding_efficiency", "Coding efficiency", "IG / H(G)", "#ff7f0e"),
        ("outage_at_pfa", "Outage at Pfa = 0.1", "1 - Pd", "#d62728"),
    ]
    for ax, (col, title, ylabel, color) in zip(axes.ravel(), panels):
        ax.bar(phases, score_df[col].to_numpy(dtype=float), color=color)
        ax.set_title(title)
        ax.set_xlabel("Phase")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_direct_comparison(
    surp_mi: pd.DataFrame,
    rmm_mi: pd.DataFrame,
    surp_ig: pd.DataFrame,
    rmm_ig: pd.DataFrame,
    out_path: Path,
) -> None:
    mats = {
        "surp_mi": load_heatmap(surp_mi, "MI_bits"),
        "rmm_mi": load_heatmap(rmm_mi, "MI_bits"),
        "surp_ig": load_heatmap(surp_ig, "IG_bits"),
        "rmm_ig": load_heatmap(rmm_ig, "IG_bits"),
    }
    vmax_mi = float(np.nanmax([np.nanmax(mats["surp_mi"].values), np.nanmax(mats["rmm_mi"].values)]))
    vmax_ig = float(np.nanmax([np.nanmax(mats["surp_ig"].values), np.nanmax(mats["rmm_ig"].values)]))
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.5), constrained_layout=True)
    entries = [
        ("surp_mi", axes[0, 0], "Surprisal MI", vmax_mi),
        ("rmm_mi", axes[0, 1], "RMM MI", vmax_mi),
        ("surp_ig", axes[1, 0], "Surprisal IG", vmax_ig),
        ("rmm_ig", axes[1, 1], "RMM IG", vmax_ig),
    ]
    for key, ax, title, vmax in entries:
        mat = mats[key]
        im = ax.imshow(mat.values.astype(float), aspect="auto", origin="lower", vmin=0.0, vmax=vmax, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("Phase")
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_yticklabels(mat.index.astype(int))
        lag_vals = mat.columns.astype(int).to_numpy()
        keep = np.arange(0, mat.shape[1], 5)
        ax.set_xticks(keep)
        ax.set_xticklabels(lag_vals[keep])
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("bits")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_null_histograms(
    ig_bp_npz: Path,
    ig_cs_npz: Path,
    mi_bp_npz: Path,
    mi_cs_npz: Path,
    out_path: Path,
    phase: int = 5,
) -> None:
    ig_bp = np.load(ig_bp_npz)
    ig_cs = np.load(ig_cs_npz)
    mi_bp = np.load(mi_bp_npz)
    mi_cs = np.load(mi_cs_npz)
    i_ig = int(np.where(ig_bp["phases"].astype(int) == phase)[0][0])
    i_mi = int(np.where(mi_bp["phases"].astype(int) == phase)[0][0])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.hist(ig_bp["max_null"][i_ig], bins=35, alpha=0.55, color="#1f77b4", label="Block permutation")
    ax.hist(ig_cs["max_null"][i_ig], bins=35, alpha=0.55, color="#ff7f0e", label="Circular shift")
    ax.axvline(float(ig_bp["max_obs"][i_ig]), color="k", lw=2, label="Observed")
    ax.set_title(f"Phase {phase} max-IG nulls")
    ax.set_xlabel("Bits")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.hist(mi_bp["max_null"][i_mi], bins=35, alpha=0.55, color="#1f77b4", label="Block permutation")
    ax.hist(mi_cs["max_null"][i_mi], bins=35, alpha=0.55, color="#ff7f0e", label="Circular shift")
    ax.axvline(float(mi_bp["max_obs"][i_mi]), color="k", lw=2, label="Observed")
    ax.set_title(f"Phase {phase} max-MI nulls")
    ax.set_xlabel("Bits")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_tex_table_main(tex_path: Path) -> pd.DataFrame:
    lines = tex_path.read_text().splitlines()
    rows = []
    in_table = False
    for line in lines:
        if r"\label{tab:main}" in line:
            in_table = True
            continue
        if in_table and r"\midrule" in line:
            continue
        if in_table and r"\bottomrule" in line:
            break
        if in_table and "&" in line and "\\" in line:
            clean = line.replace("\\", "").strip()
            parts = [p.strip() for p in clean.split("&")]
            if len(parts) >= 9 and parts[0].isdigit():
                rows.append(
                    {
                        "phase": int(parts[0]),
                        "ig_lag_tex": int(parts[1]),
                        "ig_p_tex": float(parts[2]),
                        "mi_lag_tex": int(parts[3]),
                        "mi_p_tex": float(parts[4]),
                    }
                )
    return pd.DataFrame(rows).sort_values("phase").reset_index(drop=True)


def verify_manuscript_baseline(tex_path: Path, mi_sum_csv: Path, ig_sum_csv: Path, out_path: Path) -> None:
    tex = parse_tex_table_main(tex_path).set_index("phase")
    mi = pd.read_csv(mi_sum_csv).set_index("phase")
    ig = pd.read_csv(ig_sum_csv).set_index("phase")
    rows = []
    for phase in range(1, 9):
        rows.append(
            {
                "phase": phase,
                "mi_lag_tex": int(tex.loc[phase, "mi_lag_tex"]),
                "mi_lag_csv": int(mi.loc[phase, "best_lag_obs"]),
                "mi_p_tex": float(tex.loc[phase, "mi_p_tex"]),
                "mi_p_csv": float(mi.loc[phase, "p_phase"]),
                "mi_match": (int(tex.loc[phase, "mi_lag_tex"]) == int(mi.loc[phase, "best_lag_obs"]))
                and np.isclose(float(tex.loc[phase, "mi_p_tex"]), float(mi.loc[phase, "p_phase"]), atol=5.0e-4),
                "ig_lag_tex": int(tex.loc[phase, "ig_lag_tex"]),
                "ig_lag_csv": int(ig.loc[phase, "best_lag_obs"]),
                "ig_p_tex": float(tex.loc[phase, "ig_p_tex"]),
                "ig_p_csv": float(ig.loc[phase, "p_phase"]),
                "ig_match": (int(tex.loc[phase, "ig_lag_tex"]) == int(ig.loc[phase, "best_lag_obs"]))
                and np.isclose(float(tex.loc[phase, "ig_p_tex"]), float(ig.loc[phase, "p_phase"]), atol=5.0e-4),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)


def save_metric_arrays(obs_csv: Path, value_col: str, out_npz: Path, null_bp_npz: Path, null_cs_npz: Path) -> None:
    df = pd.read_csv(obs_csv)
    mat = load_heatmap(df, value_col).values.astype(np.float32)
    bp = np.load(null_bp_npz)
    cs = np.load(null_cs_npz)
    np.savez(
        out_npz,
        metric=mat,
        phases=load_heatmap(df, value_col).index.astype(int).to_numpy(),
        lags=load_heatmap(df, value_col).columns.astype(int).to_numpy(),
        null_blockperm=bp["max_null"].astype(np.float32),
        null_circshift=cs["max_null"].astype(np.float32),
        max_obs_blockperm=bp["max_obs"].astype(np.float32),
        max_obs_circshift=cs["max_obs"].astype(np.float32),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", default="results/rmm_sensitivity")
    p.add_argument("--receiver_nc", default="results/rmm_sensitivity/rmm_phasebank_receiver_q90.nc")
    p.add_argument("--surprisal_mi_obs", default="manuscript_results/info_gain_cont_observed.csv")
    p.add_argument("--surprisal_mi_sum", default="manuscript_results/info_gain_cont_nullsummary.csv")
    p.add_argument("--surprisal_ig_obs", default="manuscript_results/info_gain_p85_observed.csv")
    p.add_argument("--surprisal_ig_sum", default="manuscript_results/info_gain_p85_nullsummary.csv")
    p.add_argument("--tex_path", default="phasebank_teleconnection_study.tex")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rmm_mi_obs = results_dir / "info_gain_cont_observed.csv"
    rmm_mi_sum = results_dir / "info_gain_cont_nullsummary.csv"
    rmm_mi_bp = results_dir / "info_gain_cont_nulldist.npz"
    rmm_mi_cs = results_dir / "info_gain_cont_cshift_nulldist.npz"
    rmm_ig_obs = results_dir / "info_gain_p90_observed.csv"
    rmm_ig_sum = results_dir / "info_gain_p90_nullsummary.csv"
    rmm_ig_bp = results_dir / "info_gain_p90_nulldist.npz"
    rmm_ig_cs = results_dir / "info_gain_p90_cshift_nulldist.npz"

    verify_manuscript_baseline(
        tex_path=Path(args.tex_path),
        mi_sum_csv=Path(args.surprisal_mi_sum),
        ig_sum_csv=Path(args.surprisal_ig_sum),
        out_path=results_dir / "baseline_verification_table1.csv",
    )

    df_rmm_mi = pd.read_csv(rmm_mi_obs)
    df_rmm_ig = pd.read_csv(rmm_ig_obs)
    df_surp_mi = pd.read_csv(args.surprisal_mi_obs)
    df_surp_ig = pd.read_csv(args.surprisal_ig_obs)

    vmax_mi = float(np.nanmax([df_rmm_mi["MI_bits"].max(), df_surp_mi["MI_bits"].max()]))
    vmax_ig = float(np.nanmax([df_rmm_ig["IG_bits"].max(), df_surp_ig["IG_bits"].max()]))
    plot_heatmap(
        df_rmm_mi,
        "MI_bits",
        figures_dir / "rmm_mi_heatmap.png",
        title="RMM Continuous-Intensity MI by Phase and Lag",
        cbar_label="MI (bits)",
        vmin=0.0,
        vmax=vmax_mi,
    )
    plot_heatmap(
        df_rmm_ig,
        "IG_bits",
        figures_dir / "rmm_ig_heatmap.png",
        title="RMM Binary-Gate IG by Phase and Lag",
        cbar_label="IG (bits)",
        vmin=0.0,
        vmax=vmax_ig,
    )

    rmm_mi_best = summarize_metric(rmm_mi_obs, rmm_mi_sum, "MI_bits")
    rmm_ig_best = summarize_metric(rmm_ig_obs, rmm_ig_sum, "IG_bits")
    plot_grouped_bars(
        rmm_mi_best["phase"].to_numpy(),
        rmm_mi_best["value"].to_numpy(dtype=float),
        rmm_ig_best["value"].to_numpy(dtype=float),
        figures_dir / "rmm_mi_ig_grouped_bars.png",
        left_label="Max MI",
        right_label="Max IG",
        title="RMM Per-Phase Max MI and Max IG",
        ylabel="Bits",
    )

    score_df, duty_df = build_scorecard(
        receiver_nc=Path(args.receiver_nc),
        mi_obs_csv=rmm_mi_obs,
        mi_sum_csv=rmm_mi_sum,
        ig_obs_csv=rmm_ig_obs,
        ig_sum_csv=rmm_ig_sum,
        pd_csv=results_dir / "phasebank_skill_pd_p10_observed.csv",
        outage_csv=results_dir / "phasebank_skill_outage_q20_observed.csv",
        auc_sum_csv=results_dir / "phasebank_skill_auc_nullsummary.csv",
        snr_sum_csv=results_dir / "phasebank_skill_snr_nullsummary.csv",
    )
    score_df.to_csv(results_dir / "rmm_scorecard_metrics.csv", index=False)
    duty_df.to_csv(results_dir / "rmm_gate_duty_cycle.csv", index=False)
    plot_scorecard(score_df, figures_dir / "rmm_four_metric_scorecard.png")

    plot_direct_comparison(
        surp_mi=df_surp_mi,
        rmm_mi=df_rmm_mi,
        surp_ig=df_surp_ig,
        rmm_ig=df_rmm_ig,
        out_path=figures_dir / "surprisal_vs_rmm_heatmap_comparison.png",
    )

    surp_mi_best = summarize_metric(Path(args.surprisal_mi_obs), Path(args.surprisal_mi_sum), "MI_bits")
    surp_ig_best = summarize_metric(Path(args.surprisal_ig_obs), Path(args.surprisal_ig_sum), "IG_bits")
    ratio_df = pd.DataFrame(
        {
            "phase": surp_mi_best["phase"],
            "mi_ig_ratio_surp": surp_mi_best["value"].to_numpy(dtype=float) / surp_ig_best["value"].to_numpy(dtype=float),
            "mi_ig_ratio_rmm": rmm_mi_best["value"].to_numpy(dtype=float) / rmm_ig_best["value"].to_numpy(dtype=float),
        }
    )
    ratio_df.to_csv(results_dir / "rmm_vs_surprisal_mi_ig_ratio.csv", index=False)
    plot_grouped_bars(
        ratio_df["phase"].to_numpy(),
        ratio_df["mi_ig_ratio_surp"].to_numpy(dtype=float),
        ratio_df["mi_ig_ratio_rmm"].to_numpy(dtype=float),
        figures_dir / "surprisal_vs_rmm_mi_ig_ratio.png",
        left_label="Surprisal MI/IG",
        right_label="RMM MI/IG",
        title="Per-Phase MI/IG Ratio Comparison",
        ylabel="MI / IG",
        left_color="#4c78a8",
        right_color="#f58518",
    )

    comparison = pd.DataFrame(
        {
            "Phase": surp_mi_best["phase"].to_numpy(dtype=int),
            "MI_surp": surp_mi_best["value"].to_numpy(dtype=float),
            "MI_RMM": rmm_mi_best["value"].to_numpy(dtype=float),
            "IG_surp": surp_ig_best["value"].to_numpy(dtype=float),
            "IG_RMM": rmm_ig_best["value"].to_numpy(dtype=float),
            "MI/IG_surp": ratio_df["mi_ig_ratio_surp"].to_numpy(dtype=float),
            "MI/IG_RMM": ratio_df["mi_ig_ratio_rmm"].to_numpy(dtype=float),
            "lag_MI_surp": surp_mi_best["best_lag"].to_numpy(dtype=int),
            "lag_MI_RMM": rmm_mi_best["best_lag"].to_numpy(dtype=int),
            "lag_IG_surp": surp_ig_best["best_lag"].to_numpy(dtype=int),
            "lag_IG_RMM": rmm_ig_best["best_lag"].to_numpy(dtype=int),
            "q_MI_surp": surp_mi_best["q_phase"].to_numpy(dtype=float),
            "q_MI_RMM": rmm_mi_best["q_phase"].to_numpy(dtype=float),
            "q_IG_surp": surp_ig_best["q_phase"].to_numpy(dtype=float),
            "q_IG_RMM": rmm_ig_best["q_phase"].to_numpy(dtype=float),
        }
    )
    comparison.to_csv(results_dir / "surprisal_vs_rmm_comparison_table.csv", index=False)

    ds = xr.open_dataset(args.receiver_nc)
    time = pd.to_datetime(ds["time"].values)
    eval_mask = (time >= pd.Timestamp("2011-01-01")) & (time <= pd.Timestamp("2020-12-31"))
    phase_idx = ds["driver_phase_index"].values.astype(int)
    phase_raw = ds["driver_rmm_phase_raw"].values.astype(float) if "driver_rmm_phase_raw" in ds else np.where(phase_idx >= 0, phase_idx + 1, np.nan)
    active_mask = phase_idx >= 0
    occ_rows = []
    for phase in range(1, 9):
        idx = phase - 1
        occ_rows.append(
            {
                "phase": phase,
                "n_raw_phase_days": int(np.sum(eval_mask & (phase_raw == float(phase)))),
                "occupancy_raw_phase_days": float(np.mean(phase_raw[eval_mask] == float(phase))),
                "n_active_days": int(np.sum(eval_mask & active_mask & (phase_idx == idx))),
                "occupancy_active_days_within_active_subset": float(np.mean((phase_idx[eval_mask & active_mask] == idx)))
                if np.any(eval_mask & active_mask)
                else np.nan,
                "occupancy_active_days_all_eval_days": float(np.mean(phase_idx[eval_mask] == idx)),
            }
        )
    pd.DataFrame(occ_rows).to_csv(results_dir / "rmm_phase_occupancy.csv", index=False)

    mi_obs = pd.read_csv(rmm_mi_obs)
    ig_obs = pd.read_csv(rmm_ig_obs)
    sample_rows = []
    for phase in range(1, 9):
        lag_mi = int(rmm_mi_best.loc[rmm_mi_best["phase"] == phase, "best_lag"].iloc[0])
        lag_ig = int(rmm_ig_best.loc[rmm_ig_best["phase"] == phase, "best_lag"].iloc[0])
        row_mi = mi_obs[(mi_obs["phase"] == phase) & (mi_obs["lag"] == lag_mi)].iloc[0]
        row_ig = ig_obs[(ig_obs["phase"] == phase) & (ig_obs["lag"] == lag_ig)].iloc[0]
        sample_rows.append(
            {
                "phase": phase,
                "lag_mi": lag_mi,
                "n_total_mi": int(row_mi["n_total"]),
                "lag_ig": lag_ig,
                "n_on_ig": int(row_ig["n_on"]),
                "n_off_ig": int(row_ig["n_off"]),
                "n_total_ig": int(row_ig["n_total"]),
                "underpowered_mi": int(row_mi["underpowered"]),
                "underpowered_ig": int(row_ig["underpowered"]),
            }
        )
    pd.DataFrame(sample_rows).to_csv(results_dir / "rmm_sample_support_by_phase.csv", index=False)

    plot_null_histograms(
        ig_bp_npz=rmm_ig_bp,
        ig_cs_npz=rmm_ig_cs,
        mi_bp_npz=rmm_mi_bp,
        mi_cs_npz=rmm_mi_cs,
        out_path=figures_dir / "rmm_phase5_null_histograms.png",
        phase=5,
    )

    save_metric_arrays(rmm_mi_obs, "MI_bits", results_dir / "rmm_mi_arrays.npz", rmm_mi_bp, rmm_mi_cs)
    save_metric_arrays(rmm_ig_obs, "IG_bits", results_dir / "rmm_ig_arrays.npz", rmm_ig_bp, rmm_ig_cs)


if __name__ == "__main__":
    main()
