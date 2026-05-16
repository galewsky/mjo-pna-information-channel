#!/usr/bin/env python3
"""
Temporal validation of the surprisal-based MJO index against ROMI and RMM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from build_rmm_phasebank_receiver import align_rmm_to_time, load_rmm_text
from surprisal_temporal_utils import (
    DEFAULT_EVAL_WINDOW,
    DEFAULT_GATE_PERCENTILE,
    DEFAULT_GATE_SMOOTH_DAYS,
    DEFAULT_GATE_TRAIN,
    DEFAULT_MJO_TIMESERIES,
    extract_events,
    load_surprisal_temporal_fields,
)
from validate_surprisal_vs_olr_composites import load_romi


DEFAULT_RESULTS_DIR = Path("results/surprisal_olr_validation")
DEFAULT_ROMI = Path("romi.cpcolr.1x.txt")
DEFAULT_RMM = Path("rmm.74toRealtime.txt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mjo", type=Path, default=DEFAULT_MJO_TIMESERIES)
    p.add_argument("--romi", type=Path, default=DEFAULT_ROMI)
    p.add_argument("--rmm", type=Path, default=DEFAULT_RMM)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--eval-window", default="1991-01-01,2020-12-31")
    p.add_argument("--gate-train", default="1991-01-01,2010-12-31")
    p.add_argument("--gate-percentile", type=float, default=DEFAULT_GATE_PERCENTILE)
    p.add_argument("--gate-smooth-days", type=int, default=DEFAULT_GATE_SMOOTH_DAYS)
    p.add_argument("--romi-smooth-days", type=int, default=7)
    p.add_argument("--romi-amp-threshold", type=float, default=1.0)
    p.add_argument("--rmm-amp-threshold", type=float, default=1.0)
    p.add_argument("--lag-max", type=int, default=15)
    return p.parse_args()


def parse_window(expr: str) -> Tuple[str, str]:
    parts = [p.strip() for p in expr.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Window spec must be 'YYYY-MM-DD,YYYY-MM-DD'")
    return parts[0], parts[1]


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return float("nan")
    return float(num / den)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def round_nested(obj: Any, digits: int = 4) -> Any:
    if isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return round(obj, digits)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if not np.isfinite(val):
            return None
        return round(val, digits)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: round_nested(v, digits=digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_nested(v, digits=digits) for v in obj]
    return obj


def contingency_metrics(pred_mask: np.ndarray, ref_active: np.ndarray, valid_mask: np.ndarray) -> Dict[str, float | int]:
    pred = np.asarray(pred_mask, dtype=bool)
    ref = np.asarray(ref_active, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)

    pred = pred[valid]
    ref = ref[valid]

    a = int(np.sum(pred & ref))
    b = int(np.sum(pred & ~ref))
    c = int(np.sum(~pred & ref))
    d = int(np.sum(~pred & ~ref))

    pod = safe_div(a, a + c)
    far = safe_div(b, a + b)
    freq_bias = safe_div(a + b, a + c)
    hss_num = 2.0 * (a * d - b * c)
    hss_den = (a + c) * (c + d) + (a + b) * (b + d)
    hss = safe_div(hss_num, hss_den)
    csi = safe_div(a, a + b + c)

    return {
        "valid_days": int(valid.sum()),
        "reference_active_days": int(np.sum(ref)),
        "pred_active_days": int(np.sum(pred)),
        "hit_days": a,
        "false_alarm_days": b,
        "miss_days": c,
        "correct_null_days": d,
        "pod": pod,
        "far": far,
        "frequency_bias": freq_bias,
        "hss": hss,
        "csi": csi,
    }


def shifted_views(
    x: np.ndarray,
    y: np.ndarray,
    valid_mask: np.ndarray,
    x_active: np.ndarray,
    y_active: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if lag > 0:
        x_view = x[:-lag]
        y_view = y[lag:]
        valid_view = valid_mask[:-lag] & valid_mask[lag:]
        active_view = x_active[:-lag] | y_active[lag:]
    elif lag < 0:
        offset = -lag
        x_view = x[offset:]
        y_view = y[:-offset]
        valid_view = valid_mask[offset:] & valid_mask[:-offset]
        active_view = x_active[offset:] | y_active[:-offset]
    else:
        x_view = x
        y_view = y
        valid_view = valid_mask
        active_view = x_active | y_active
    return x_view, y_view, valid_view, active_view


def lagged_correlation_series(
    source_amp: np.ndarray,
    ref_amp: np.ndarray,
    valid_mask: np.ndarray,
    source_active: np.ndarray,
    ref_active: np.ndarray,
    lag_max: int,
) -> tuple[pd.DataFrame, Dict[str, float | int]]:
    rows = []
    for lag in range(-lag_max, lag_max + 1):
        x_view, y_view, valid_view, active_view = shifted_views(
            source_amp,
            ref_amp,
            valid_mask,
            source_active,
            ref_active,
            lag,
        )
        all_mask = valid_view
        active_mask = valid_view & active_view
        rows.append(
            {
                "lag_days": lag,
                "r_all_days": pearson_r(x_view[all_mask], y_view[all_mask]),
                "n_all_days": int(np.sum(all_mask)),
                "r_active_any": pearson_r(x_view[active_mask], y_view[active_mask]),
                "n_active_any": int(np.sum(active_mask)),
            }
        )

    df = pd.DataFrame(rows)
    peak_all = df.loc[df["r_all_days"].idxmax()] if df["r_all_days"].notna().any() else None
    peak_active = df.loc[df["r_active_any"].idxmax()] if df["r_active_any"].notna().any() else None
    summary = {
        "pearson_r_all_days": float(df.loc[df["lag_days"] == 0, "r_all_days"].iloc[0]),
        "pearson_r_active_any": float(df.loc[df["lag_days"] == 0, "r_active_any"].iloc[0]),
        "peak_lag_all_days": int(peak_all["lag_days"]) if peak_all is not None else None,
        "peak_r_all_days": float(peak_all["r_all_days"]) if peak_all is not None else float("nan"),
        "peak_lag_active_any": int(peak_active["lag_days"]) if peak_active is not None else None,
        "peak_r_active_any": float(peak_active["r_active_any"]) if peak_active is not None else float("nan"),
    }
    return df, summary


def match_events(
    reference_name: str,
    reference_events: pd.DataFrame,
    burst_events: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, float | int]]:
    overlapped_burst_ids: set[int] = set()
    rows = []
    onset_offsets = []

    for ref in reference_events.itertuples(index=False):
        overlaps = burst_events[
            (burst_events["start_index"] <= ref.end_index) & (burst_events["end_index"] >= ref.start_index)
        ].copy()
        if overlaps.empty:
            rows.append(
                {
                    "reference": reference_name,
                    "row_type": "reference_event",
                    "reference_event_id": int(ref.event_id),
                    "reference_onset_date": ref.onset_date.date().isoformat(),
                    "reference_end_date": ref.end_date.date().isoformat(),
                    "reference_duration_days": int(ref.duration_days),
                    "burst_event_id": pd.NA,
                    "burst_onset_date": pd.NA,
                    "burst_end_date": pd.NA,
                    "burst_duration_days": pd.NA,
                    "overlap_flag": False,
                    "overlap_days": 0,
                    "match_status": "miss",
                    "onset_offset_days": pd.NA,
                }
            )
            continue

        overlaps["overlap_days"] = (
            np.minimum(overlaps["end_index"], ref.end_index) - np.maximum(overlaps["start_index"], ref.start_index) + 1
        )
        overlaps["abs_onset_offset"] = np.abs(overlaps["start_index"] - ref.start_index)
        overlaps = overlaps.sort_values(["overlap_days", "abs_onset_offset", "start_index"], ascending=[False, True, True])
        best = overlaps.iloc[0]
        overlapped_burst_ids.update(int(v) for v in overlaps["event_id"].tolist())

        onset_offset = int(best["start_index"] - ref.start_index)
        onset_offsets.append(onset_offset)
        rows.append(
            {
                "reference": reference_name,
                "row_type": "reference_event",
                "reference_event_id": int(ref.event_id),
                "reference_onset_date": ref.onset_date.date().isoformat(),
                "reference_end_date": ref.end_date.date().isoformat(),
                "reference_duration_days": int(ref.duration_days),
                "burst_event_id": int(best["event_id"]),
                "burst_onset_date": pd.Timestamp(best["onset_date"]).date().isoformat(),
                "burst_end_date": pd.Timestamp(best["end_date"]).date().isoformat(),
                "burst_duration_days": int(best["duration_days"]),
                "overlap_flag": True,
                "overlap_days": int(best["overlap_days"]),
                "match_status": "hit",
                "onset_offset_days": onset_offset,
            }
        )

    for burst in burst_events.itertuples(index=False):
        if int(burst.event_id) in overlapped_burst_ids:
            continue
        rows.append(
            {
                "reference": reference_name,
                "row_type": "burst_event",
                "reference_event_id": pd.NA,
                "reference_onset_date": pd.NA,
                "reference_end_date": pd.NA,
                "reference_duration_days": pd.NA,
                "burst_event_id": int(burst.event_id),
                "burst_onset_date": burst.onset_date.date().isoformat(),
                "burst_end_date": burst.end_date.date().isoformat(),
                "burst_duration_days": int(burst.duration_days),
                "overlap_flag": False,
                "overlap_days": 0,
                "match_status": "false_alarm",
                "onset_offset_days": pd.NA,
            }
        )

    events_df = pd.DataFrame(rows)
    n_hits = int(np.sum(events_df["match_status"].eq("hit")))
    n_misses = int(np.sum(events_df["match_status"].eq("miss")))
    n_false_alarms = int(np.sum(events_df["match_status"].eq("false_alarm")))
    summary = {
        "reference_events": int(len(reference_events)),
        "burst_events": int(len(burst_events)),
        "hits": n_hits,
        "misses": n_misses,
        "false_alarms": n_false_alarms,
        "hit_rate": safe_div(n_hits, len(reference_events)),
        "false_alarm_rate": safe_div(n_false_alarms, len(burst_events)),
        "mean_onset_offset_days": float(np.mean(onset_offsets)) if onset_offsets else float("nan"),
        "median_onset_offset_days": float(np.median(onset_offsets)) if onset_offsets else float("nan"),
    }
    return events_df, summary


def build_status_markdown(
    output_dir: Path,
    surprisal_meta: Dict[str, Any],
    overlap_row: pd.Series,
    contingency_df: pd.DataFrame,
    amplitude_summary: Dict[str, Dict[str, float | int]],
    event_summary: Dict[str, Dict[str, float | int]],
    included_refs: Iterable[str],
) -> str:
    lines = [
        "# Temporal Validation Status",
        "",
        "Status: complete",
        "",
        f"Outputs directory: `{output_dir}`",
        "",
        "Provenance:",
        f"- Surprisal timeseries: `{surprisal_meta['source_file']}`",
        f"- Evaluation window: {surprisal_meta['eval_start']} to {surprisal_meta['eval_end']}",
        (
            f"- Gate definition: 10-day centered mean of `mjo_amp`, "
            f"q{int(float(surprisal_meta['gate_percentile']))}_train over "
            f"{surprisal_meta['gate_train_start']} to {surprisal_meta['gate_train_end']}"
        ),
        f"- Gate threshold: {float(surprisal_meta['gate_threshold']):.4f}",
        "- Active-only amplitude correlation uses days when either the surprisal burst mask or the reference active mask is on.",
        "- Lag sign convention: positive lag means the reference amplitude lags the surprisal amplitude.",
        "",
        "Gate/burst overlap:",
        f"- Gate-on days: {int(overlap_row['gate_on_days'])}",
        f"- Burst days: {int(overlap_row['burst_days'])}",
        f"- Burst days also gate-on: {float(overlap_row['fraction_burst_days_also_gate_on']):.4f}",
        f"- Gate-on days also in burst: {float(overlap_row['fraction_gate_days_also_in_burst']):.4f}",
        (
            f"- Burst events: {int(overlap_row['burst_event_count'])}, "
            f"mean duration {float(overlap_row['mean_burst_duration_days']):.4f} d, "
            f"median duration {float(overlap_row['median_burst_duration_days']):.4f} d"
        ),
        "",
        "Reference comparisons:",
    ]

    for ref in included_refs:
        burst_row = contingency_df[(contingency_df["reference"] == ref) & (contingency_df["state"] == "burst")].iloc[0]
        gate_row = contingency_df[(contingency_df["reference"] == ref) & (contingency_df["state"] == "gate")].iloc[0]
        amp = amplitude_summary[ref]
        evt = event_summary[ref]
        lines.extend(
            [
                f"- {ref} burst-vs-active: HSS={burst_row['hss']:.4f}, CSI={burst_row['csi']:.4f}, POD={burst_row['pod']:.4f}, FAR={burst_row['far']:.4f}",
                f"- {ref} gate-vs-active: HSS={gate_row['hss']:.4f}, CSI={gate_row['csi']:.4f}, POD={gate_row['pod']:.4f}, FAR={gate_row['far']:.4f}",
                (
                    f"- {ref} amplitude correlation: r_all={float(amp['pearson_r_all_days']):.4f}, "
                    f"r_active_any={float(amp['pearson_r_active_any']):.4f}, "
                    f"peak lag={int(amp['peak_lag_all_days']) if amp['peak_lag_all_days'] is not None else 'NA'} d "
                    "(positive means reference lags surprisal)"
                ),
                (
                    f"- {ref} event-level: hit rate={float(evt['hit_rate']):.4f}, "
                    f"false alarm rate={float(evt['false_alarm_rate']):.4f}, "
                    f"mean onset offset={float(evt['mean_onset_offset_days']):.4f} d"
                ),
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    eval_window = parse_window(args.eval_window)
    gate_train = parse_window(args.gate_train)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    surp_df, surp_meta = load_surprisal_temporal_fields(
        path=args.mjo,
        eval_window=eval_window,
        gate_train_window=gate_train,
        gate_percentile=args.gate_percentile,
        gate_smooth_days=args.gate_smooth_days,
    )
    time_index = surp_df.index

    burst_mask = surp_df["burst_mask"].to_numpy(dtype=bool)
    gate_mask = surp_df["gate_mask"].to_numpy(dtype=bool)
    surp_amp = surp_df["surprisal_amp"].to_numpy(dtype=np.float64)

    burst_events, _ = extract_events(burst_mask, time_index, min_duration=1, event_prefix="burst")
    burst_durations = burst_events["duration_days"].to_numpy(dtype=np.float64) if not burst_events.empty else np.array([], dtype=np.float64)
    overlap_days = int(np.sum(burst_mask & gate_mask))
    overlap_stats = pd.DataFrame(
        [
            {
                "period_start": eval_window[0],
                "period_end": eval_window[1],
                "gate_on_days": int(np.sum(gate_mask)),
                "burst_days": int(np.sum(burst_mask)),
                "overlap_days": overlap_days,
                "fraction_burst_days_also_gate_on": safe_div(overlap_days, int(np.sum(burst_mask))),
                "fraction_gate_days_also_in_burst": safe_div(overlap_days, int(np.sum(gate_mask))),
                "burst_event_count": int(len(burst_events)),
                "mean_burst_duration_days": float(np.mean(burst_durations)) if burst_durations.size else float("nan"),
                "median_burst_duration_days": float(np.median(burst_durations)) if burst_durations.size else float("nan"),
            }
        ]
    )

    references: Dict[str, Dict[str, Any]] = {}

    romi = load_romi(args.romi, args.romi_smooth_days, args.romi_amp_threshold).reindex(time_index)
    references["ROMI"] = {
        "active": romi["active"].fillna(False).to_numpy(dtype=bool),
        "amp": romi["amp"].to_numpy(dtype=np.float64),
        "valid": romi["amp"].notna().to_numpy(dtype=bool),
        "meta": {"source_file": str(args.romi)},
    }

    if args.rmm.exists():
        rmm_df = load_rmm_text(args.rmm)
        rmm_ds = align_rmm_to_time(
            rmm_df=rmm_df,
            source_file=args.rmm,
            time_index=time_index,
            amp_active_min=args.rmm_amp_threshold,
            smooth_days=args.gate_smooth_days,
            train_window=gate_train,
            gate_percentile=args.gate_percentile,
        )
        references["RMM"] = {
            "active": rmm_ds["rmm_active_mask"].values.astype(bool),
            "amp": rmm_ds["rmm_amplitude"].values.astype(np.float64),
            "valid": rmm_ds["rmm_qc_valid"].values.astype(bool),
            "meta": {
                "source_file": str(args.rmm),
                "invalid_days": int((~rmm_ds["rmm_qc_valid"].values.astype(bool)).sum()),
            },
        }

    contingency_rows = []
    amplitude_rows = []
    amplitude_summary: Dict[str, Dict[str, float | int]] = {}
    event_frames = []
    event_summary: Dict[str, Dict[str, float | int]] = {}

    for ref_name, ref in references.items():
        valid_mask = ref["valid"] & np.isfinite(surp_amp) & np.isfinite(ref["amp"])
        ref_active = ref["active"] & valid_mask

        for state_name, pred_mask in (("burst", burst_mask), ("gate", gate_mask)):
            row = contingency_metrics(pred_mask, ref_active, valid_mask)
            row["reference"] = ref_name
            row["state"] = state_name
            contingency_rows.append(row)

        lag_df, lag_summary = lagged_correlation_series(
            source_amp=surp_amp,
            ref_amp=ref["amp"],
            valid_mask=valid_mask,
            source_active=burst_mask,
            ref_active=ref_active,
            lag_max=args.lag_max,
        )
        lag_df.insert(0, "reference", ref_name)
        amplitude_rows.append(lag_df)
        amplitude_summary[ref_name] = lag_summary

        ref_events, _ = extract_events(
            ref_active,
            time_index,
            min_duration=7,
            valid_mask=valid_mask,
            event_prefix=f"{ref_name.lower()}_event",
        )
        matched_df, matched_summary = match_events(ref_name, ref_events, burst_events)
        event_frames.append(matched_df)
        event_summary[ref_name] = matched_summary

    contingency_df = pd.DataFrame(contingency_rows)[
        [
            "reference",
            "state",
            "valid_days",
            "reference_active_days",
            "pred_active_days",
            "hit_days",
            "false_alarm_days",
            "miss_days",
            "correct_null_days",
            "pod",
            "far",
            "frequency_bias",
            "hss",
            "csi",
        ]
    ]
    amplitude_df = pd.concat(amplitude_rows, ignore_index=True)
    events_df = pd.concat(event_frames, ignore_index=True)

    contingency_path = args.output_dir / "temporal_validation_contingency.csv"
    events_path = args.output_dir / "temporal_validation_events.csv"
    amplitude_path = args.output_dir / "temporal_validation_amplitude_correlation.csv"
    overlap_path = args.output_dir / "temporal_validation_gate_burst_overlap.csv"
    summary_path = args.output_dir / "temporal_validation_summary.json"
    status_path = args.output_dir / "status_temporal_validation.md"

    contingency_df.to_csv(contingency_path, index=False, float_format="%.4f")
    events_df.to_csv(events_path, index=False, float_format="%.4f")
    amplitude_df.to_csv(amplitude_path, index=False, float_format="%.4f")
    overlap_stats.to_csv(overlap_path, index=False, float_format="%.4f")

    summary = {
        "surprisal": {
            **surp_meta,
            "gate_burst_overlap": overlap_stats.iloc[0].to_dict(),
        },
        "references": {},
        "active_any_definition": "Days when either the surprisal burst mask or the reference active mask is true.",
        "lag_sign_convention": "Correlation is computed between surprisal A(t) and reference amplitude at t+lag; positive lag means the reference lags surprisal.",
    }
    for ref_name in references:
        day_rows = contingency_df[contingency_df["reference"] == ref_name]
        summary["references"][ref_name] = {
            "day_level": {
                row["state"]: row.drop(labels=["reference", "state"]).to_dict()
                for _, row in day_rows.iterrows()
            },
            "amplitude_correlation": amplitude_summary[ref_name],
            "event_level": event_summary[ref_name],
            "meta": references[ref_name]["meta"],
        }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(round_nested(summary, digits=4), f, indent=2)
        f.write("\n")

    status_md = build_status_markdown(
        output_dir=args.output_dir.resolve(),
        surprisal_meta=surp_meta,
        overlap_row=overlap_stats.iloc[0],
        contingency_df=contingency_df,
        amplitude_summary=amplitude_summary,
        event_summary=event_summary,
        included_refs=references.keys(),
    )
    status_path.write_text(status_md, encoding="utf-8")

    event_summary_df = pd.DataFrame(
        [
            {
                "reference": ref_name,
                **event_summary[ref_name],
            }
            for ref_name in references
        ]
    )
    amplitude_peak_df = pd.DataFrame(
        [
            {
                "reference": ref_name,
                **amplitude_summary[ref_name],
            }
            for ref_name in references
        ]
    )

    print("Day-level contingency summary:")
    print(contingency_df[["reference", "state", "pod", "far", "frequency_bias", "hss", "csi"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nEvent-level summary:")
    print(event_summary_df[["reference", "reference_events", "burst_events", "hit_rate", "false_alarm_rate", "mean_onset_offset_days"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nAmplitude-correlation summary:")
    print(amplitude_peak_df[["reference", "pearson_r_all_days", "pearson_r_active_any", "peak_lag_all_days", "peak_r_all_days", "peak_lag_active_any", "peak_r_active_any"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nSaved {contingency_path}")
    print(f"Saved {events_path}")
    print(f"Saved {amplitude_path}")
    print(f"Saved {overlap_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
