#!/usr/bin/env python3
"""
Compute mean OLR surprisal composites for ENSO phases and MJO phases.

Uses daily surprisal fields and classifies ENSO states from the Nino3.4
anomaly time series. MJO phases are derived from ROMI (romi1/romi2) and
optionally filtered to strong MJO days.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_SURPRISAL_PATH = "olr_tropics_pointwise_info.nc"
DEFAULT_OUTPUT = "olr_surprisal_phase_composites.nc"
DEFAULT_NINO_PATH = Path("nino34_daily_anomalies.nc")
DEFAULT_ROMI_PATH = Path("romi.cpcolr.1x.txt")

ENSO_SMOOTH_DAYS = 30
ENSO_NEUTRAL_BAND = 0.4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surprisal", default=DEFAULT_SURPRISAL_PATH, help="daily surprisal NetCDF")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="target NetCDF file")
    parser.add_argument("--overwrite", action="store_true", help="replace output if it exists")
    parser.add_argument("--nino-path", type=Path, default=DEFAULT_NINO_PATH, help="Niño 3.4 anomaly NetCDF")
    parser.add_argument("--romi-path", type=Path, default=DEFAULT_ROMI_PATH, help="ROMI phase text file")
    parser.add_argument("--enso-threshold", type=float, default=0.6, help="SST anomaly threshold (°C)")
    parser.add_argument(
        "--enso-persistence-days",
        type=int,
        default=56,
        help="Minimum consecutive days over threshold to flag ENSO state",
    )
    parser.add_argument("--romi-amp-threshold", type=float, default=1.0, help="Amplitude threshold for active MJO")
    parser.add_argument("--romi-smooth-days", type=int, default=7, help="Smoothing window for ROMI components")
    parser.add_argument("--chunks", default="auto", help="Dask chunks for time dimension")
    return parser.parse_args()


def _ensure_output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"Output {path} exists; pass --overwrite to replace it")
    if path.exists():
        path.unlink()


def _drop_invalid_times(da: xr.DataArray) -> xr.DataArray:
    time_values = da["time"].values
    valid = ~np.isnat(time_values)
    if not valid.all():
        da = da.isel(time=np.where(valid)[0])
    return da


def _find_periods(mask: np.ndarray, min_duration: int) -> List[Tuple[int, int]]:
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    transitions = np.diff(padded.astype(int))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    spans: List[Tuple[int, int]] = []
    for start, end in zip(starts, ends):
        if end - start >= min_duration:
            spans.append((start, end))
    return spans


def _classify_enso_states(
    path: Path,
    threshold: float,
    persistence_days: int,
    smooth_days: int = ENSO_SMOOTH_DAYS,
    neutral_band: float = ENSO_NEUTRAL_BAND,
) -> pd.Series:
    if not path.exists():
        raise SystemExit(f"Niño 3.4 file not found: {path}")
    nino = xr.open_dataset(path)["nino34_anom"].astype("float32")
    time_index = pd.to_datetime(nino["time"].values)
    series = pd.Series(nino.values, index=time_index, name="nino34_anom")
    smooth = series.rolling(smooth_days, center=True, min_periods=1).mean()

    def _mark(mask: pd.Series) -> np.ndarray:
        mask_vals = mask.fillna(False).to_numpy()
        flagged = np.zeros_like(mask_vals, dtype=bool)
        for start, end in _find_periods(mask_vals, persistence_days):
            flagged[start:end] = True
        return flagged

    el_days = _mark(smooth >= threshold)
    la_days = _mark(smooth <= -threshold)
    neu_days = _mark((smooth > -neutral_band) & (smooth < neutral_band))

    flags = np.full(len(series), "UNK", dtype="U3")
    flags[neu_days] = "NEU"
    flags[la_days] = "LN"
    flags[el_days] = "EN"
    return pd.Series(flags, index=series.index, name="enso_flag")


def _load_romi_index(path: Path, smooth_days: int, amp_threshold: float) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"ROMI file not found: {path}")
    romi = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        header=None,
        names=["year", "month", "day", "flag", "romi1", "romi2", "amp"],
    )
    romi["time"] = pd.to_datetime(dict(year=romi.year, month=romi.month, day=romi.day))
    romi = romi.set_index("time").sort_index()
    r1s = romi["romi1"].rolling(smooth_days, center=True, min_periods=1).mean()
    r2s = romi["romi2"].rolling(smooth_days, center=True, min_periods=1).mean()
    amp = np.hypot(r1s, r2s)
    phase_deg = (np.degrees(np.arctan2(r2s, r1s)) % 360.0)
    phase = (np.floor(phase_deg / 45.0) + 1).astype(int).astype("int8")
    active = (amp >= amp_threshold).astype(bool)
    return pd.DataFrame({"phase": phase, "active": active, "amp": amp.astype("float32")})


def _align_composite_inputs(
    surprisal: xr.DataArray,
    enso_flags: pd.Series,
    romi: pd.DataFrame,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    surprisal_time = pd.to_datetime(surprisal["time"].values)
    common = pd.Index(surprisal_time).intersection(enso_flags.index).intersection(romi.index)
    if common.empty:
        raise SystemExit("No overlapping days between surprisal, ENSO, and ROMI datasets")
    common_np = common.to_numpy(dtype="datetime64[ns]")
    surprisal_sel = surprisal.sel(time=common_np)
    time_coord = surprisal_sel["time"]
    enso_da = xr.DataArray(enso_flags.loc[common].values.astype("U3"), coords={"time": time_coord}, dims="time")
    romi_phase = xr.DataArray(romi.loc[common, "phase"].to_numpy(dtype="int8"), coords={"time": time_coord}, dims="time")
    romi_active = xr.DataArray(romi.loc[common, "active"].to_numpy(dtype=bool), coords={"time": time_coord}, dims="time")
    return surprisal_sel, enso_da, romi_phase, romi_active


def _composite_mean(field: xr.DataArray, mask: xr.DataArray) -> Tuple[xr.DataArray | None, int]:
    mask = mask.astype(bool)
    count = int(mask.sum().item())
    if count == 0:
        return None, 0
    comp = field.where(mask).mean("time", skipna=True).astype("float32")
    return comp, count


def _compute_enso_composites(
    field: xr.DataArray,
    enso_flags: xr.DataArray,
    threshold: float,
    persistence_days: int,
) -> Tuple[Dict[str, xr.DataArray], Dict[str, xr.DataArray]]:
    composites: Dict[str, xr.DataArray] = {}
    counts: Dict[str, xr.DataArray] = {}
    labels = {
        "EN": ("elnino", f"Niño 3.4 ≥ +{threshold}°C for ≥{persistence_days} days"),
        "LN": ("lanina", f"Niño 3.4 ≤ -{threshold}°C for ≥{persistence_days} days"),
        "NEU": ("neutral", f"|Niño 3.4| < {ENSO_NEUTRAL_BAND}°C for ≥{persistence_days} days"),
    }
    for code, (prefix, desc) in labels.items():
        comp, n = _composite_mean(field, enso_flags == code)
        if comp is None:
            continue
        name = f"{prefix}_mean_surprisal"
        comp.name = name
        comp.attrs.update(
            {
                "long_name": f"Mean surprisal during {prefix.replace('lanina', 'La Niña').replace('elnino', 'El Niño').replace('neutral', 'neutral')} periods",
                "units": "bits",
                "description": desc,
            }
        )
        composites[name] = comp
        count_name = f"{prefix}_day_count"
        counts[count_name] = xr.DataArray(
            np.int32(n),
            name=count_name,
            attrs={"long_name": f"Number of days in {prefix} composite", "units": "days"},
        )
    return composites, counts


def _compute_mjo_phase_composites(
    field: xr.DataArray,
    romi_phase: xr.DataArray,
    romi_active: xr.DataArray,
    amp_threshold: float,
) -> Tuple[Dict[str, xr.DataArray], Dict[str, xr.DataArray]]:
    composites: Dict[str, xr.DataArray] = {}
    counts: Dict[str, xr.DataArray] = {}
    for phase in range(1, 9):
        mask = romi_active & (romi_phase == phase)
        comp, n = _composite_mean(field, mask)
        if comp is None:
            continue
        name = f"mjo_phase{phase:02d}_mean_surprisal"
        comp.name = name
        comp.attrs.update(
            {
                "long_name": f"Mean surprisal, ROMI phase {phase}",
                "units": "bits",
                "description": f"Days with ROMI amplitude ≥ {amp_threshold} and phase {phase}",
            }
        )
        composites[name] = comp
        count_name = f"mjo_phase{phase:02d}_day_count"
        counts[count_name] = xr.DataArray(
            np.int32(n),
            name=count_name,
            attrs={"long_name": f"Number of active ROMI days in phase {phase}", "units": "days"},
        )
    return composites, counts


def main() -> int:
    args = _parse_args()
    surprisal_path = Path(args.surprisal)
    output_path = Path(args.output)
    if not surprisal_path.exists():
        raise SystemExit(f"Surprisal file not found: {surprisal_path}")
    _ensure_output_path(output_path, args.overwrite)

    if args.chunks == "auto":
        chunks = "auto"
    else:
        chunks = {"time": int(args.chunks)}

    print(f"Opening {surprisal_path}")
    ds = xr.open_dataset(surprisal_path, chunks=chunks)
    if "olr_surprisal" not in ds:
        raise SystemExit("Expected variable 'olr_surprisal' not found")

    surprisal = ds["olr_surprisal"].astype("float32")
    surprisal = _drop_invalid_times(surprisal)

    print("Classifying ENSO states from", args.nino_path)
    enso_flags = _classify_enso_states(args.nino_path, args.enso_threshold, args.enso_persistence_days)

    print("Loading ROMI index from", args.romi_path)
    romi = _load_romi_index(args.romi_path, args.romi_smooth_days, args.romi_amp_threshold)

    print("Aligning datasets")
    surprisal_common, enso_da, romi_phase, romi_active = _align_composite_inputs(surprisal, enso_flags, romi)

    print("Computing ENSO composites")
    enso_fields, enso_counts = _compute_enso_composites(
        surprisal_common, enso_da, args.enso_threshold, args.enso_persistence_days
    )

    print("Computing MJO phase composites")
    mjo_fields, mjo_counts = _compute_mjo_phase_composites(
        surprisal_common, romi_phase, romi_active, args.romi_amp_threshold
    )

    out = xr.Dataset({**enso_fields, **mjo_fields, **enso_counts, **mjo_counts})

    composite_time = pd.to_datetime(surprisal_common["time"].values)
    out.attrs.update(
        {
            "history": f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} computed ENSO/MJO surprisal composites",
            "source": str(surprisal_path),
            "enso_threshold_degC": args.enso_threshold,
            "enso_persistence_days": args.enso_persistence_days,
            "enso_source": str(args.nino_path),
            "romi_source": str(args.romi_path),
            "romi_active_amplitude_threshold": args.romi_amp_threshold,
            "romi_component_smoothing_days": args.romi_smooth_days,
            "composite_time_start": str(composite_time[0]) if len(composite_time) else "NaT",
            "composite_time_end": str(composite_time[-1]) if len(composite_time) else "NaT",
        }
    )

    compression_targets: Iterable[str] = list(enso_fields.keys()) + list(mjo_fields.keys())
    print(f"Writing {output_path}")
    encoding = {name: {"zlib": True, "complevel": 4} for name in compression_targets}
    out.to_netcdf(output_path, encoding=encoding)
    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
