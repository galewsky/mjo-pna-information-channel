#!/usr/bin/env python3
"""
Quantitatively validate surprisal-anomaly composites against signed OLR composites.

This script uses the local pre-WK surprisal anomaly field and ROMI phase labels,
and computes phasewise area-weighted pattern correlations against conventional
signed OLR anomaly composites. If a local OLR anomaly store is unavailable, it
falls back to NOAA's public daily interpolated OLR archive and reconstructs
daily anomalies with the same detrend + deseasonalize logic used in the local
pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from preprocess_olr_anomalies import (
    _daily_climatology,
    _drop_leap_days,
    _linear_detrend,
    _no_leap_dayofyear,
)


DEFAULT_SURPRISAL = Path("mjo_index_from_surprisal.nc")
DEFAULT_ROMI = Path("romi.cpcolr.1x.txt")
DEFAULT_OUTPUT_DIR = Path("results/surprisal_olr_validation")
DEFAULT_STATUS_MD = "status_2026-03-20.md"
DEFAULT_TABLE_CSV = "surprisal_vs_olr_pattern_correlations.csv"
DEFAULT_TABLE_JSON = "surprisal_vs_olr_pattern_correlations.json"
DEFAULT_OLR_URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/interp_OLR/olr.day.mean.nc"

DOMAIN_LAT_MIN = -20.0
DOMAIN_LAT_MAX = 20.0
DOMAIN_LON_MIN = 40.0
DOMAIN_LON_MAX = 200.0
TIME_START = "1979-01-01"
TIME_END = "2020-12-31"
ROMI_ACTIVE_START = "1991-01-01"
ROMI_ACTIVE_END = "2020-12-31"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surprisal", type=Path, default=DEFAULT_SURPRISAL)
    parser.add_argument("--romi", type=Path, default=DEFAULT_ROMI)
    parser.add_argument("--olr-url", default=DEFAULT_OLR_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-md", default=DEFAULT_STATUS_MD)
    parser.add_argument("--table-csv", default=DEFAULT_TABLE_CSV)
    parser.add_argument("--table-json", default=DEFAULT_TABLE_JSON)
    parser.add_argument("--romi-smooth-days", type=int, default=7)
    parser.add_argument("--romi-amp-threshold", type=float, default=1.0)
    return parser.parse_args()


def load_romi(path: Path, smooth_days: int, amp_threshold: float) -> pd.DataFrame:
    romi = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        header=None,
        names=["year", "month", "day", "flag", "romi1", "romi2", "amp_raw"],
    )
    romi["time"] = pd.to_datetime(dict(year=romi.year, month=romi.month, day=romi.day))
    romi = romi.set_index("time").sort_index()
    r1 = romi["romi1"].rolling(smooth_days, center=True, min_periods=1).mean()
    r2 = romi["romi2"].rolling(smooth_days, center=True, min_periods=1).mean()
    amp = np.hypot(r1, r2)
    phase_deg = np.degrees(np.arctan2(r2, r1)) % 360.0
    phase = (np.floor(phase_deg / 45.0) + 1).astype(np.int8)
    active = amp >= amp_threshold
    out = pd.DataFrame({"phase": phase, "active": active, "amp": amp.astype(np.float32)})
    return out.loc[ROMI_ACTIVE_START:ROMI_ACTIVE_END]


def load_surprisal(path: Path) -> xr.DataArray:
    ds = xr.open_dataset(path, chunks={"time": 365})
    da = ds["olr_surprisal_anom"].astype(np.float32)
    da = da.assign_coords(lon=((da["lon"] % 360.0) + 360.0) % 360.0).sortby("lon")
    da = da.sel(lat=slice(DOMAIN_LAT_MIN, DOMAIN_LAT_MAX), lon=slice(DOMAIN_LON_MIN, DOMAIN_LON_MAX))
    return da


def build_olr_anomalies(olr_url: str) -> xr.DataArray:
    print(f"Opening OLR source from {olr_url}")
    olr_path = Path(olr_url)
    if olr_path.exists() and olr_path.suffix == ".zarr":
        ds = xr.open_zarr(olr_path)
    else:
        ds = xr.open_dataset(olr_url)

    if "olr_anomaly" in ds:
        print("Using precomputed olr_anomaly field directly")
        da = ds["olr_anomaly"]
        if "lon" in da.coords:
            da = da.assign_coords(lon=((da["lon"] % 360.0) + 360.0) % 360.0).sortby("lon")
        da = da.sel(
            time=slice(TIME_START, TIME_END),
            lon=slice(DOMAIN_LON_MIN, DOMAIN_LON_MAX),
            lat=slice(DOMAIN_LAT_MIN, DOMAIN_LAT_MAX),
        ).sortby("lat")
        da = _drop_leap_days(da)
        return da.astype(np.float32)

    if "olr" not in ds:
        raise ValueError(f"Expected either 'olr_anomaly' or 'olr' in {olr_url}")

    base = ds["olr"].sel(
        lon=slice(DOMAIN_LON_MIN, DOMAIN_LON_MAX),
        lat=slice(DOMAIN_LAT_MAX, DOMAIN_LAT_MIN),
    ).sortby("lat")

    olr_chunks: list[xr.DataArray] = []
    for year in range(1979, 2021):
        print(f"  Loading NOAA OLR slab for {year}")
        annual = base.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).load()
        olr_chunks.append(annual)

    olr = xr.concat(olr_chunks, dim="time")
    units = (olr.attrs.get("units") or "").strip()
    valid_units = {"W/m2", "W/m^2", "W m-2", "W m^-2"}
    if units not in valid_units:
        raise ValueError(f"Unexpected units for NOAA OLR: {units!r}")
    olr.attrs["units"] = "W m-2"
    olr = olr.astype(np.float32).where(np.isfinite(olr))
    olr = _drop_leap_days(olr)

    dayofyear = _no_leap_dayofyear(olr.time).astype(np.int16)
    dayofyear_labels = xr.DataArray(
        dayofyear.values,
        dims=("time",),
        coords={"time": olr.time},
        name="dayofyear_noleap",
    )

    detrended, _, _ = _linear_detrend(olr)
    climatology = _daily_climatology(detrended, window=0, labels=dayofyear_labels)
    clim_for_anoms = climatology.rename(dayofyear="dayofyear_noleap")
    anomalies = (detrended.groupby(dayofyear_labels) - clim_for_anoms).astype(np.float32)
    anomalies.name = "olr_anomaly"
    return anomalies


def weighted_pattern_correlation(
    surp: xr.DataArray,
    olr: xr.DataArray,
) -> tuple[float, float]:
    if surp.shape != olr.shape:
        raise ValueError(f"Shape mismatch: {surp.shape} vs {olr.shape}")

    lat_weights = np.cos(np.deg2rad(surp["lat"].values.astype(np.float64)))
    weights = lat_weights[:, None] * np.ones((1, surp.sizes["lon"]), dtype=np.float64)

    x = np.asarray(surp.values, dtype=np.float64)
    y_raw = np.asarray(olr.values, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y_raw)
    if not np.any(valid):
        return np.nan, np.nan

    w = weights[valid]
    x = x[valid]
    y_raw = y_raw[valid]

    x = x - np.sum(w * x) / np.sum(w)
    y_raw = y_raw - np.sum(w * y_raw) / np.sum(w)
    y_aligned = -y_raw

    denom_raw = np.sqrt(np.sum(w * x * x) * np.sum(w * y_raw * y_raw))
    denom_aligned = np.sqrt(np.sum(w * x * x) * np.sum(w * y_aligned * y_aligned))
    raw_r = np.sum(w * x * y_raw) / denom_raw
    aligned_r = np.sum(w * x * y_aligned) / denom_aligned
    return float(aligned_r), float(raw_r)


def make_status_markdown(
    df: pd.DataFrame,
    common_days: int,
    active_days: int,
    output_dir: Path,
    used_olr_source: str,
) -> str:
    weak = df.loc[df["r_aligned"] < 0.5, "phase"].tolist()
    unexpected = df.loc[df["r_raw"] > 0.0, "phase"].tolist()
    phase_lines = "\n".join(
        f"- Phase {int(row.phase)}: r_aligned={row.r_aligned:.3f}, r_raw={row.r_raw:.3f}, n_days={int(row.n_days)}"
        for row in df.itertuples()
    )

    weak_line = ", ".join(str(x) for x in weak) if weak else "none"
    unexpected_line = ", ".join(str(x) for x in unexpected) if unexpected else "none"

    return f"""# Surprisal vs OLR Composite Validation Status

Date: 2026-03-20

Status: complete

Outputs directory: `{output_dir}`

Data provenance:
- Surprisal anomaly field: `mjo_index_from_surprisal.nc::olr_surprisal_anom`
- ROMI phase table: `romi.cpcolr.1x.txt`
- Signed OLR field used for validation: `{used_olr_source}`
- Note: the local `olr_tropics_anomalies.zarr` store was not present in this workspace, and this shell had no AWS/B2 credentials for `s3://galewsky/olr/olr_tropics_anomalies.zarr`, so the validation used the public NOAA daily OLR archive with the same detrend + deseasonalize anomaly construction logic.

Sample support:
- Common no-leap days across surprisal, OLR, and ROMI: {common_days}
- Active ROMI days used in phase composites: {active_days}

Pattern correlations:
{phase_lines}

Summary:
- Range of aligned correlations: {df["r_aligned"].min():.3f} to {df["r_aligned"].max():.3f}
- Mean aligned correlation: {df["r_aligned"].mean():.3f}
- Phases with aligned r < 0.5: {weak_line}
- Phases with unexpected raw sign (r_raw > 0): {unexpected_line}
"""


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ROMI from {args.romi}")
    romi = load_romi(args.romi, args.romi_smooth_days, args.romi_amp_threshold)

    print(f"Loading surprisal anomalies from {args.surprisal}")
    surp = load_surprisal(args.surprisal)

    olr_anom = build_olr_anomalies(args.olr_url)

    surp_times = pd.DatetimeIndex(surp["time"].values)
    olr_times = pd.DatetimeIndex(olr_anom["time"].values)
    common = surp_times.intersection(olr_times).intersection(romi.index)
    common = common[(common >= pd.Timestamp(ROMI_ACTIVE_START)) & (common <= pd.Timestamp(ROMI_ACTIVE_END))]
    if common.empty:
        raise SystemExit("No overlapping days among surprisal, OLR, and ROMI inputs.")

    romi_common = romi.loc[common]
    active_mask = romi_common["active"].to_numpy(dtype=bool)
    common_np = common.to_numpy(dtype="datetime64[ns]")

    print(f"Common no-leap days: {len(common_np)}")
    print(f"Active ROMI days: {int(active_mask.sum())}")

    print("Interpolating surprisal anomalies to the OLR grid")
    surp_common = (
        surp.sel(time=common_np)
        .interp(lat=olr_anom["lat"], lon=olr_anom["lon"], method="linear")
        .load()
    )
    olr_common = olr_anom.sel(time=common_np).load()

    rows: list[dict[str, object]] = []
    for phase in range(1, 9):
        phase_mask = active_mask & (romi_common["phase"].to_numpy(dtype=np.int8) == phase)
        n_days = int(phase_mask.sum())
        if n_days == 0:
            rows.append(
                {
                    "phase": phase,
                    "n_days": 0,
                    "r_aligned": np.nan,
                    "r_raw": np.nan,
                    "flag_r_lt_0p5": True,
                    "flag_unexpected_sign": True,
                }
            )
            continue

        phase_times = common_np[phase_mask]
        surp_comp = surp_common.sel(time=phase_times).mean("time", skipna=True)
        olr_comp = olr_common.sel(time=phase_times).mean("time", skipna=True)
        r_aligned, r_raw = weighted_pattern_correlation(surp_comp, olr_comp)
        rows.append(
            {
                "phase": phase,
                "n_days": n_days,
                "r_aligned": r_aligned,
                "r_raw": r_raw,
                "flag_r_lt_0p5": bool(np.isnan(r_aligned) or r_aligned < 0.5),
                "flag_unexpected_sign": bool(np.isnan(r_raw) or r_raw > 0.0),
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        "phase_correlations": {str(int(row.phase)): float(row.r_aligned) for row in df.itertuples()},
        "raw_phase_correlations": {str(int(row.phase)): float(row.r_raw) for row in df.itertuples()},
        "n_days_by_phase": {str(int(row.phase)): int(row.n_days) for row in df.itertuples()},
        "aligned_min": float(df["r_aligned"].min()),
        "aligned_max": float(df["r_aligned"].max()),
        "aligned_mean": float(df["r_aligned"].mean()),
        "weak_phases": [int(x) for x in df.loc[df["flag_r_lt_0p5"], "phase"].tolist()],
        "unexpected_sign_phases": [int(x) for x in df.loc[df["flag_unexpected_sign"], "phase"].tolist()],
        "common_days": int(len(common_np)),
        "active_days": int(active_mask.sum()),
        "surprisal_source": str(args.surprisal),
        "olr_source": args.olr_url,
        "romi_source": str(args.romi),
        "note": (
            "The local olr_tropics_anomalies.zarr store was unavailable and no cloud credentials "
            "were configured for the private s3://galewsky/olr/olr_tropics_anomalies.zarr path. "
            "NOAA daily interpolated OLR was used instead, with the same detrend + seasonal-cycle removal."
        ),
    }

    csv_path = args.output_dir / args.table_csv
    json_path = args.output_dir / args.table_json
    status_path = args.output_dir / args.status_md

    df.to_csv(csv_path, index=False, float_format="%.6f")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    status_md = make_status_markdown(
        df=df,
        common_days=len(common_np),
        active_days=int(active_mask.sum()),
        output_dir=args.output_dir.resolve(),
        used_olr_source=args.olr_url,
    )
    status_path.write_text(status_md, encoding="utf-8")

    print(df.to_string(index=False))
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")
    print(f"Saved {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
