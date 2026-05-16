#!/usr/bin/env python3
"""
Build a daily ENSO state mask from Nino-3.4 anomalies.

Outputs:
  - nino34_smooth: smoothed anomaly (degC)
  - enso_state:    categorical code (-1 La Nina, 0 Neutral, 1 El Nino)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import xarray as xr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="nino34_daily_anomalies.nc", help="Input NetCDF with nino34_anom(time)")
    p.add_argument("--var", default="nino34_anom", help="Variable name in input file")
    p.add_argument("--output", default="enso_state_daily.nc", help="Output NetCDF path")
    p.add_argument("--smooth_days", type=int, default=30, help="Centered running-mean window in days")
    p.add_argument("--threshold", type=float, default=0.5, help="Absolute threshold for EN/LN classification (degC)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ds = xr.open_dataset(args.input)
    if args.var not in ds:
        raise KeyError(f"{args.var} not found in {args.input}")

    da = ds[args.var].astype(np.float64)
    ser = pd.Series(da.values, index=pd.to_datetime(da["time"].values))
    smooth = ser.rolling(args.smooth_days, center=True, min_periods=1).mean().to_numpy()

    state = np.zeros_like(smooth, dtype=np.int8)
    state[smooth >= args.threshold] = 1
    state[smooth <= -args.threshold] = -1

    out = xr.Dataset(
        {
            "nino34_smooth": (("time",), smooth.astype(np.float32)),
            "enso_state": (("time",), state),
        },
        coords={"time": da["time"].values},
    )
    out["nino34_smooth"].attrs["units"] = "degC"
    out["nino34_smooth"].attrs["long_name"] = "Smoothed Nino-3.4 anomaly"
    out["enso_state"].attrs["long_name"] = "ENSO state code (-1 La Nina, 0 Neutral, 1 El Nino)"
    out.attrs["source_input"] = args.input
    out.attrs["source_var"] = args.var
    out.attrs["smooth_days"] = int(args.smooth_days)
    out.attrs["threshold_degC"] = float(args.threshold)

    out.to_netcdf(args.output)
    counts = {
        "ElNino_days": int(np.sum(state == 1)),
        "Neutral_days": int(np.sum(state == 0)),
        "LaNina_days": int(np.sum(state == -1)),
    }
    print(f"Wrote {args.output}")
    print(counts)


if __name__ == "__main__":
    main()
