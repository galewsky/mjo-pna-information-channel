#!/usr/bin/env python3
"""
Shared utilities for temporal validation of the surprisal-based MJO index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from phasebank_information_gain import compute_gate


DEFAULT_MJO_TIMESERIES = Path("mjo_index_from_surprisal_timeseries.nc")
DEFAULT_GATE_TRAIN = ("1991-01-01", "2010-12-31")
DEFAULT_GATE_PERCENTILE = 90.0
DEFAULT_GATE_SMOOTH_DAYS = 10
DEFAULT_EVAL_WINDOW = ("1991-01-01", "2020-12-31")


def load_surprisal_temporal_fields(
    path: Path = DEFAULT_MJO_TIMESERIES,
    *,
    eval_window: Tuple[str, str] = DEFAULT_EVAL_WINDOW,
    gate_train_window: Tuple[str, str] = DEFAULT_GATE_TRAIN,
    gate_percentile: float = DEFAULT_GATE_PERCENTILE,
    gate_smooth_days: int = DEFAULT_GATE_SMOOTH_DAYS,
) -> tuple[pd.DataFrame, Dict[str, float | str | int]]:
    ds = xr.open_dataset(path)
    full_time_index = pd.DatetimeIndex(ds["time"].values)
    amp = ds["mjo_amp"].values.astype(np.float64, copy=False)
    burst_mask = ds["burst_mask"].values.astype(bool, copy=False)
    burst_onset = ds["burst_onset"].values.astype(bool, copy=False)
    burst_id = ds["burst_id"].values.astype(np.int32, copy=False)

    gate_vals, gate_threshold, intensity_smoothed = compute_gate(
        intensity=amp,
        times=full_time_index.values.astype("datetime64[ns]"),
        train_window=gate_train_window,
        percentile=gate_percentile,
        smooth_days=gate_smooth_days,
        smooth_centered=True,
        assume_smoothed=False,
    )
    gate_mask = gate_vals > 0.5

    df = pd.DataFrame(
        {
            "surprisal_amp": amp.astype(np.float32),
            "surprisal_intensity_smoothed": intensity_smoothed.astype(np.float32),
            "burst_mask": burst_mask,
            "burst_onset": burst_onset,
            "burst_id": burst_id,
            "gate_mask": gate_mask,
        },
        index=full_time_index,
    )
    df.index.name = "time"
    df = df.loc[eval_window[0] : eval_window[1]].copy()

    meta: Dict[str, float | str | int] = {
        "source_file": str(path),
        "gate_train_start": gate_train_window[0],
        "gate_train_end": gate_train_window[1],
        "gate_percentile": float(gate_percentile),
        "gate_smooth_days": int(gate_smooth_days),
        "gate_threshold": float(gate_threshold),
        "eval_start": eval_window[0],
        "eval_end": eval_window[1],
    }
    return df, meta


def extract_events(
    mask: np.ndarray,
    times: pd.DatetimeIndex,
    *,
    min_duration: int = 1,
    valid_mask: np.ndarray | None = None,
    event_prefix: str = "event",
) -> tuple[pd.DataFrame, np.ndarray]:
    active = np.asarray(mask, dtype=bool)
    if valid_mask is not None:
        active = active & np.asarray(valid_mask, dtype=bool)

    event_ids = np.zeros(active.size, dtype=np.int32)
    rows = []
    event_id = 0
    i = 0
    while i < active.size:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < active.size and active[j]:
            j += 1
        duration = j - i
        if duration >= int(min_duration):
            event_id += 1
            event_ids[i:j] = event_id
            rows.append(
                {
                    "event_id": event_id,
                    "event_label": f"{event_prefix}_{event_id:03d}",
                    "start_index": i,
                    "end_index": j - 1,
                    "onset_date": pd.Timestamp(times[i]),
                    "end_date": pd.Timestamp(times[j - 1]),
                    "duration_days": duration,
                }
            )
        i = j

    events = pd.DataFrame(rows)
    return events, event_ids
