#!/usr/bin/env python3
"""
compute_phasebank_telecom_metrics.py

Derive telecom-style summary metrics from existing phasebank outputs:
  - Spectral-efficiency proxy: per-phase MI_max (bits/symbol) and realized
    throughput proxy (bits/day) using phase occupancy.
  - Coding efficiency: IG / H_L(G|k) at the IG-best lag (H_L is the
    conditional binary entropy of the burst gate on the same lag-paired
    source samples used for that IG estimate).
  - Outage probability at operating point: 1 - Pd@Pfa for AUC-style detector,
    and a low-tail failure fraction (q=0.2) at the IG-best lag.

Outputs CSVs into manuscript_results/ and a compact LaTeX table summarizing
per-phase metrics for quick inclusion in the manuscript.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr


def binary_entropy(p: float) -> float:
    if not np.isfinite(p) or p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def load_best_lag(df_summary: pd.DataFrame, key_lag: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for _, r in df_summary.dropna(subset=[key_lag]).iterrows():
        out[int(r["phase"])] = int(r[key_lag])
    return out


def compute_capacity_and_throughput(receiver_nc: Path, results_dir: Path) -> pd.DataFrame:
    df_mi_obs = pd.read_csv(results_dir / "info_gain_cont_gatealigned_observed.csv")
    df_mi_sum = pd.read_csv(results_dir / "info_gain_cont_gatealigned_nullsummary.csv")
    bestlag = load_best_lag(df_mi_sum, "best_lag_obs")

    # Phase occupancy over the test window used in MI run
    test_window = str(df_mi_obs.iloc[0]["test_window"]) if "test_window" in df_mi_obs.columns else ""
    ds = xr.open_dataset(receiver_nc)
    t = pd.to_datetime(ds["time"].values)
    if "," in test_window:
        start, end = [pd.to_datetime(x.strip()) for x in test_window.split(",")]
        mask = (t >= start) & (t <= end)
    else:
        mask = np.ones(t.size, dtype=bool)
    # The receiver stores zero-based phase indices (0..7, with -1 invalid), while
    # manuscript tables use MJO phase labels 1..8.
    phase_labels = ds["driver_phase_index"].values.astype(float) + 1.0
    occ: Dict[int, float] = {}
    for ph in range(1, 9):
        if np.any(mask):
            occ[ph] = float(np.mean(phase_labels[mask] == ph))
        else:
            occ[ph] = np.nan

    rows = []
    for ph in range(1, 9):
        if ph not in bestlag:
            rows.append({
                "phase": ph,
                "best_lag_mi": np.nan,
                "mi_max_bits": np.nan,
                "spectral_eff_bits_per_symbol": np.nan,
                "phase_occupancy": occ.get(ph, np.nan),
                "throughput_bits_per_day": np.nan,
            })
            continue
        L = bestlag[ph]
        rec = df_mi_obs[(df_mi_obs["phase"] == ph) & (df_mi_obs["lag"] == L)]
        mi_bits = float(rec.iloc[0]["MI_bits"]) if not rec.empty else np.nan
        rows.append({
            "phase": ph,
            "best_lag_mi": L,
            "mi_max_bits": mi_bits,
            "spectral_eff_bits_per_symbol": mi_bits,
            "phase_occupancy": occ.get(ph, np.nan),
            "throughput_bits_per_day": (occ.get(ph, np.nan) * mi_bits) if np.isfinite(mi_bits) else np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(results_dir / "telecom_capacity_throughput.csv", index=False)
    return out


def compute_coding_efficiency(results_dir: Path) -> pd.DataFrame:
    df_ig_obs = pd.read_csv(results_dir / "info_gain_p85_observed.csv")
    df_ig_sum = pd.read_csv(results_dir / "info_gain_p85_nullsummary.csv")
    bestlag = load_best_lag(df_ig_sum, "best_lag_obs")
    rows = []
    for ph in range(1, 9):
        if ph not in bestlag:
            rows.append({
                "phase": ph,
                "best_lag_ig": np.nan,
                "ig_best_bits": np.nan,
                "p_on": np.nan,
                "H_G_bits": np.nan,
                "coding_efficiency": np.nan,
            })
            continue
        L = bestlag[ph]
        rec = df_ig_obs[(df_ig_obs["phase"] == ph) & (df_ig_obs["lag"] == L)]
        if rec.empty:
            ig_bits = np.nan
            p_on = np.nan
        else:
            ig_bits = float(rec.iloc[0]["IG_bits"])
            p_on = float(rec.iloc[0]["p_on"])
        h_g = binary_entropy(p_on)
        rows.append({
            "phase": ph,
            "best_lag_ig": L,
            "ig_best_bits": ig_bits,
            "p_on": p_on,
            "H_G_bits": h_g,
            "coding_efficiency": (ig_bits / h_g) if (np.isfinite(ig_bits) and h_g > 0) else np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(results_dir / "telecom_coding_efficiency.csv", index=False)
    return out


def compute_outage_tables(results_dir: Path) -> pd.DataFrame:
    df_out = pd.read_csv(results_dir / "phasebank_skill_outage_q20_observed.csv")
    df_lcr = pd.read_csv(results_dir / "phasebank_skill_lcr_q20_observed.csv")
    df_afd = pd.read_csv(results_dir / "phasebank_skill_afd_q20_observed.csv")
    df_pd = pd.read_csv(results_dir / "phasebank_skill_pd_p10_observed.csv")
    df_ig_sum = pd.read_csv(results_dir / "info_gain_p85_nullsummary.csv")
    bestlag = load_best_lag(df_ig_sum, "best_lag_obs")

    rows = []
    for ph in range(1, 9):
        L = bestlag.get(ph, None)
        if L is None:
            rows.append({"phase": ph, "best_lag_ig": np.nan, "outage_q20": np.nan, "lcr_q20": np.nan, "afd_q20": np.nan, "pd_at_pfa": np.nan, "outage_at_pfa": np.nan})
            continue
        r_out = df_out[(df_out["phase"] == ph) & (df_out["lag"] == L)]
        r_lcr = df_lcr[(df_lcr["phase"] == ph) & (df_lcr["lag"] == L)]
        r_afd = df_afd[(df_afd["phase"] == ph) & (df_afd["lag"] == L)]
        r_pd = df_pd[(df_pd["phase"] == ph) & (df_pd["lag"] == L)]
        outage_q20 = float(r_out.iloc[0]["outage"]) if not r_out.empty else np.nan
        lcr_q20 = float(r_lcr.iloc[0]["lcr"]) if not r_lcr.empty else np.nan
        afd_q20 = float(r_afd.iloc[0]["afd"]) if not r_afd.empty else np.nan
        pd_at_pfa = float(r_pd.iloc[0]["pd_at_pfa"]) if not r_pd.empty else np.nan
        rows.append({
            "phase": ph,
            "best_lag_ig": L,
            "outage_q20": outage_q20,
            "lcr_q20": lcr_q20,
            "afd_q20": afd_q20,
            "pd_at_pfa": pd_at_pfa,
            "outage_at_pfa": (1.0 - pd_at_pfa) if np.isfinite(pd_at_pfa) else np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(results_dir / "telecom_outage_table.csv", index=False)
    return out


def write_latex_table(cap_df: pd.DataFrame, eff_df: pd.DataFrame, outg_df: pd.DataFrame, out_tex: Path) -> None:
    # Avoid duplicate columns on merge
    outg_df = outg_df.drop(columns=[c for c in ["best_lag_ig"] if c in outg_df.columns])
    # Merge by phase
    df = cap_df.merge(eff_df, on="phase", how="outer").merge(outg_df, on="phase", how="outer")
    cols = [
        "phase",
        "mi_max_bits",
        "best_lag_mi",
        "spectral_eff_bits_per_symbol",
        "phase_occupancy",
        "throughput_bits_per_day",
        "ig_best_bits",
        "best_lag_ig",
        "H_G_bits",
        "coding_efficiency",
        "outage_at_pfa",
        "outage_q20",
    ]
    df = df[cols]
    lines = []
    lines.append("% Auto-generated by compute_phasebank_telecom_metrics.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\footnotesize")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append(
        "\\begin{tabular}{r r r r r r r r r r r r}"
    )
    lines.append(
        "Phase & MI$_{\\max}$ (bits) & MI lag & Spectral eff. & Occupancy & Throughput & IG$_{\\max}$ (bits) & IG lag & H$_L$(G$\\mid$k) (bits) & Coding eff. & Outage ($P_{\\rm fa}=0.1$) & Low-tail ($q=0.2$) \\\\" 
    )
    lines.append("\\hline")
    for _, r in df.sort_values("phase").iterrows():
        fmt = (
            f"{int(r['phase'])} & "
            f"{r['mi_max_bits']:.3f} & {int(r['best_lag_mi']) if np.isfinite(r['best_lag_mi']) else -1} & "
            f"{r['spectral_eff_bits_per_symbol']:.3f} & {r['phase_occupancy']:.3f} & {r['throughput_bits_per_day']:.3f} & "
            f"{r['ig_best_bits']:.3f} & {int(r['best_lag_ig']) if np.isfinite(r['best_lag_ig']) else -1} & "
            f"{r['H_G_bits']:.3f} & {r['coding_efficiency']:.2f} & "
            f"{r['outage_at_pfa']:.3f} & {r['outage_q20']:.3f} \\\\" 
        )
        lines.append(fmt)
    lines.append("\\end{tabular}}")
    lines.append("\\caption{Communication-inspired metrics by MJO phase: MI-based spectral-efficiency proxy and realized-throughput proxy (occupancy-weighted), coding efficiency (IG/$H_L(G\\mid k)$), fixed-operating-point outage ($1-P_d$ at $P_{\\rm fa}=0.1$), and a separate low-tail failure fraction at $q=0.2$. $H_L(G\\mid k)$ is computed from the same lag-paired source samples used at the IG-optimal lag, which can differ slightly from the all-window duty-cycle entropy in Table~\\ref{tab:duty_cycle} because the last $L$ source days have no receiver partner.}")
    lines.append("\\label{tab:telecom_metrics}")
    lines.append("\\end{table}")
    out_tex.write_text("\n".join(lines))


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--receiver_nc", default="z500_receiver_phasebank_Bint_domain120_300_20_80_train1991_2010.nc")
    a.add_argument("--results_dir", default="manuscript_results")
    args = a.parse_args()

    results_dir = Path(args.results_dir)
    receiver_nc = Path(args.receiver_nc)

    cap = compute_capacity_and_throughput(receiver_nc, results_dir)
    eff = compute_coding_efficiency(results_dir)
    outg = compute_outage_tables(results_dir)
    write_latex_table(cap, eff, outg, results_dir / "telecom_metrics_table.tex")
    print("Wrote:")
    for p in [
        results_dir / "telecom_capacity_throughput.csv",
        results_dir / "telecom_coding_efficiency.csv",
        results_dir / "telecom_outage_table.csv",
        results_dir / "telecom_metrics_table.tex",
    ]:
        print("  ", p)


if __name__ == "__main__":
    main()
