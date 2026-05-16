#!/usr/bin/env python3
"""
mjo_index_from_surprisal.py

Build an MJO index from an OLR surprisal field by:
  1) Removing seasonal cycle (day-of-year climatology)
  2) Applying WK99-style eastward (k=1..5, 30-96 d) filter in (time,lon)
  3) Computing EOF1/EOF2 of the filtered field over a canonical Indo-Pacific belt
  4) Defining amplitude A(t) and phase phi(t)
  5) Defining "bursts" as contiguous intervals where A(t) exceeds a threshold

Input file header (given):
  float olr_surprisal(time, lat, lon)

Assumptions:
  - time is daily and regular (checked)
  - lon is uniform and global (checked-ish; warns if not)
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import xarray as xr
from scipy.fft import fft2, ifft2, fftfreq

# -------------------------------
# Utilities
# -------------------------------
def _check_regular_daily_time(time: xr.DataArray, allow_leap_gaps: bool = True) -> None:
    """Fail fast if time spacing is not ~1 day and regular."""
    tvals = time.values.astype("datetime64[ns]")
    dt = np.diff(tvals) / np.timedelta64(1, "D")
    if dt.size == 0:
        raise ValueError("time dimension is empty.")
    if np.all(np.isfinite(dt)) and np.allclose(dt, 1.0, atol=1e-6, rtol=0.0):
        return
    if not allow_leap_gaps:
        raise ValueError(
            "Time coordinate is not regular daily spacing. "
            "This WK filter assumes uniform 1-day sampling."
        )
    gap_idx = np.where(~np.isclose(dt, 1.0, atol=1e-6, rtol=0.0))[0]
    if gap_idx.size == 0:
        return
    if not np.all(np.isclose(dt[gap_idx], 2.0, atol=1e-6, rtol=0.0)):
        raise ValueError(
            "Time coordinate has irregular spacing beyond missing leap days. "
            "This WK filter assumes uniform 1-day sampling."
        )
    dates = pd.to_datetime(tvals)
    ok = True
    for i in gap_idx:
        d0 = dates[i]
        d1 = dates[i + 1]
        if not (d0.month == 2 and d0.day == 28 and d1.month == 3 and d1.day == 1):
            ok = False
            break
    if not ok:
        raise ValueError(
            "Time coordinate has irregular spacing beyond missing leap days. "
            "This WK filter assumes uniform 1-day sampling."
        )
    print(f"WARNING: Found {gap_idx.size} missing leap days (Feb 29). Proceeding anyway.")

def _check_lon_uniform_global(lon: xr.DataArray) -> None:
    """Warn if lon not uniform. The filter assumes a uniform periodic grid."""
    lv = lon.values.astype("float64")
    if lv.size < 2:
        raise ValueError("lon dimension too small.")
    dlon = np.diff(lv)
    if not np.allclose(dlon, dlon[0], atol=1e-6, rtol=0.0):
        print("WARNING: lon is not uniformly spaced. WK k interpretation may be off.")
    # global-ish sanity: total span near 360
    span = lv[-1] - lv[0] + dlon[0]
    if not (350.0 <= span <= 370.0):
        print(f"WARNING: lon span ~{span:.2f} deg. Not ~360. Check periodicity.")


# -------------------------------
# WK filter kernel
# -------------------------------
def wk_mjo_filter_time_lon_kernel(data_2d: np.ndarray) -> np.ndarray:
    """
    Apply a simple WK99 band-pass in (time, lon) to a 2D numpy array.

    Input shape: (time, lon)
    Output shape: (time, lon)

    Keeps:
      - periods 30 to 96 days
      - zonal wavenumbers |k|=1..5
      - eastward propagation
    """
    # NaNs: safest simple handling is fill with 0 ONLY if sparse.
    # Note: Upstream handling in main now ensures data_2d shouldn't have NaNs,
    # but we keep this check for robustness.
    if np.isnan(data_2d).any():
        data_2d = np.nan_to_num(data_2d, nan=0.0)

    nt, nx = data_2d.shape

    # 2D FFT (unshifted)
    F = fft2(data_2d)

    # frequency grids (unshifted)
    f = fftfreq(nt, d=1.0)          # cycles/day (assumes daily)
    k = fftfreq(nx, d=1.0 / nx)     # integer zonal wavenumbers on uniform global grid
    k_grid, f_grid = np.meshgrid(k, f)  # (timefreq, lonwvn)

    # MJO band limits
    f_low, f_high = 1.0/96.0, 1.0/30.0
    k_low, k_high = 1, 5

    # Eastward definition (clean):
    # Keep quadrant: f > 0 and k > 0, plus its Hermitian partner for real inverse.
    mask_quad = (
        (f_grid > 0.0) &
        (k_grid > 0.0) &
        (f_grid >= f_low) & (f_grid <= f_high) &
        (np.abs(k_grid) >= k_low) & (np.abs(k_grid) <= k_high)
    )

    # Mirror to preserve reality: partner of (f,k) is (-f,-k)
    mask = mask_quad | np.flipud(np.fliplr(mask_quad))

    F_filt = F * mask
    recon = ifft2(F_filt).real
    return recon.astype(np.float32)


# -------------------------------
# EOF / PC computation
# -------------------------------
def compute_eof12_from_field(
    da: xr.DataArray,
    lat_name: str = "lat",
    lon_name: str = "lon",
    time_name: str = "time",
    weight_lat: bool = True,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Compute EOF1/EOF2 (spatial patterns) and PC1/PC2 (time series) from da(time,lat,lon).

    Implementation notes:
      - Computes on a fully in-memory matrix [time, space].
      - Uses SVD on anomalies already (mean removed).
      - Optionally uses sqrt(cos(lat)) weighting (common for global fields).
    """
    # Ensure finite mask consistent across time (drop points that are NaN any time)
    da2 = da
    valid = np.isfinite(da2).all(time_name)
    da2 = da2.where(valid, drop=True)

    # weights
    if weight_lat:
        lat = da2[lat_name].values.astype("float64")
        w_lat = np.sqrt(np.cos(np.deg2rad(lat)))
        w = xr.DataArray(w_lat, coords={lat_name: da2[lat_name]}, dims=(lat_name,))
        da2w = da2 * w
    else:
        da2w = da2

    # stack space
    X = da2w.stack(space=(lat_name, lon_name)).transpose(time_name, "space")

    # load to numpy
    Xv = X.values.astype("float64")

    # remove temporal mean of each space point (should be ~0 already)
    Xv = Xv - np.nanmean(Xv, axis=0, keepdims=True)

    # SVD: X = U S Vt
    # PCs are U*S, EOFs are V (columns) reshaped to space
    U, s, Vt = np.linalg.svd(Xv, full_matrices=False)

    pc1 = U[:, 0] * s[0]
    pc2 = U[:, 1] * s[1]

    eof1 = Vt[0, :]
    eof2 = Vt[1, :]

    # unstack EOF patterns back to lat/lon
    eof1_da = xr.DataArray(eof1, coords={"space": X["space"]}, dims=("space",)).unstack("space")
    eof2_da = xr.DataArray(eof2, coords={"space": X["space"]}, dims=("space",)).unstack("space")

    # If weighting was applied, undo it in EOF patterns for interpretability
    if weight_lat:
        eof1_da = eof1_da / w
        eof2_da = eof2_da / w

    pc1_da = xr.DataArray(pc1.astype(np.float32), coords={time_name: da2[time_name]}, dims=(time_name,), name="PC1")
    pc2_da = xr.DataArray(pc2.astype(np.float32), coords={time_name: da2[time_name]}, dims=(time_name,), name="PC2")

    eof1_da = eof1_da.astype(np.float32)
    eof2_da = eof2_da.astype(np.float32)
    eof1_da.name = "EOF1"
    eof2_da.name = "EOF2"

    return eof1_da, eof2_da, pc1_da, pc2_da


def bursts_from_amplitude(
    amp: xr.DataArray,
    thresh_kind: str = "sigma",
    thresh_value: float = 1.0,
    min_duration_days: int = 7,
    min_separation_days: int = 7,
) -> xr.Dataset:
    """
    Define burst intervals from amplitude A(t).

    thresh_kind:
      - "sigma": A > mean + thresh_value*std (mean ~0)
      - "percentile": A > percentile(thresh_value)

    Returns dataset with:
      - burst_mask(time): bool
      - burst_onset(time): bool
      - burst_id(time): int (0 means not in burst)
    """
    a = amp

    if thresh_kind == "sigma":
        mu = float(a.mean().values)
        sd = float(a.std().values)
        thr = mu + thresh_value * sd
    elif thresh_kind == "percentile":
        thr = float(np.nanpercentile(a.values, thresh_value))
    else:
        raise ValueError("thresh_kind must be 'sigma' or 'percentile'")

    active = (a > thr).astype(bool).values

    # Identify contiguous True runs
    burst_id = np.zeros(active.size, dtype=np.int32)
    onset = np.zeros(active.size, dtype=bool)

    bid = 0
    i = 0
    while i < active.size:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < active.size and active[j]:
            j += 1
        # run is [i, j)
        dur = j - i
        if dur >= min_duration_days:
            bid += 1
            burst_id[i:j] = bid
            onset[i] = True
            i = j
        else:
            # discard short run
            i = j

    # Declustering: enforce min separation between onsets
    if min_separation_days > 0 and bid > 1:
        onset_idx = np.where(onset)[0]
        keep = []
        last = -10**9
        for idx in onset_idx:
            if idx - last >= min_separation_days:
                keep.append(idx)
                last = idx
        # rebuild burst_id with only kept onsets by dropping bursts whose onset removed
        keep = np.array(keep, dtype=int)
        keep_bids = set(burst_id[keep].tolist())
        burst_id2 = burst_id.copy()
        for k_bid in range(1, bid + 1):
            if k_bid not in keep_bids:
                burst_id2[burst_id2 == k_bid] = 0
        # relabel consecutively
        uniq = [u for u in np.unique(burst_id2) if u != 0]
        relabel = {u: i+1 for i, u in enumerate(uniq)}
        burst_id = np.array([relabel.get(x, 0) for x in burst_id2], dtype=np.int32)
        onset[:] = False
        for u in np.unique(burst_id):
            if u == 0:
                continue
            onset[np.where(burst_id == u)[0][0]] = True

    ds = xr.Dataset(
        {
            "burst_mask": (("time",), burst_id > 0),
            "burst_onset": (("time",), onset),
            "burst_id": (("time",), burst_id),
        },
        coords={"time": amp["time"]},
    )
    ds["burst_mask"].attrs["description"] = "True when MJO amplitude exceeds threshold (after duration/separation rules)."
    ds["burst_onset"].attrs["description"] = "True at first day of each burst interval."
    ds["burst_id"].attrs["description"] = "Integer burst identifier, 0 means not in burst."
    ds.attrs["burst_threshold_kind"] = thresh_kind
    ds.attrs["burst_threshold_value"] = thresh_value
    ds.attrs["burst_min_duration_days"] = min_duration_days
    ds.attrs["burst_min_separation_days"] = min_separation_days
    return ds


# -------------------------------
# Main
# -------------------------------
def main():
    p = argparse.ArgumentParser(description="Build MJO index from OLR surprisal via WK filtering + EOFs.")
    p.add_argument("--input", default="olr_tropics_pointwise_info.nc")
    p.add_argument("--var", default="olr_surprisal")
    p.add_argument("--output", default="mjo_index_from_surprisal.nc")

    # domain for EOF/index
    p.add_argument("--lat0", type=float, default=-15.0)
    p.add_argument("--lat1", type=float, default=15.0)
    p.add_argument("--lon0", type=float, default=60.0)
    p.add_argument("--lon1", type=float, default=180.0)

    # coarsen for speed (optional)
    p.add_argument("--coarsen", type=int, default=2, help="coarsen factor for lat/lon before filtering (1 disables)")

    # burst parameters
    p.add_argument("--burst-kind", choices=["sigma", "percentile"], default="sigma")
    p.add_argument("--burst-value", type=float, default=1.0, help="sigma multiplier or percentile (0-100)")
    p.add_argument("--burst-min-dur", type=int, default=7)
    p.add_argument("--burst-min-sep", type=int, default=7)

    args = p.parse_args()

    print(f"Opening {args.input} ...", flush=True)
    ds = xr.open_dataset(args.input, chunks={"lat": 10, "time": -1, "lon": -1})
    da = ds[args.var]

    # Basic checks
    _check_regular_daily_time(ds["time"])
    _check_lon_uniform_global(ds["lon"])

    # 1) Sanitize Input: Remove Inf values which destroy mean calculations
    # Surprisal can be Inf if probability is 0. We treat these as missing (NaN) first.
    if np.any(np.isinf(da)):
        print("WARNING: Input contains Infinite values (likely probability=0). Converting to NaN.", flush=True)
        da = da.where(np.isfinite(da))

    # Optional coarsen (saves a lot of time on 1440 lon grid)
    if args.coarsen and args.coarsen > 1:
        print(f"Coarsening lat/lon by factor {args.coarsen} ...", flush=True)
        da = da.coarsen(lat=args.coarsen, lon=args.coarsen, boundary="trim").mean()

    # Seasonal cycle removal (dayofyear)
    print("Removing seasonal cycle (dayofyear climatology) ...", flush=True)
    clim = da.groupby("time.dayofyear").mean("time")
    anom = da.groupby("time.dayofyear") - clim
    if "dayofyear" in anom.coords:
        anom = anom.drop_vars("dayofyear")

    # 2) Explicitly fill NaNs in the anomaly field
    # We fill with 0.0 (the mean) so that:
    #   a) The saved anomaly field is valid float32
    #   b) The filter receives consistent data
    print("Filling NaNs in anomaly field with 0.0 (mean) ...", flush=True)
    anom = anom.fillna(0.0)

    # Ensure time/lon contiguous for FFT kernel
    print("Rechunking for FFT kernel ...", flush=True)
    anom = anom.chunk({"time": -1, "lon": -1, "lat": 10})

    # Apply WK filter (vectorized across lat)
    print("Applying WK MJO filter (k=1..5, 30-96d, eastward) ...", flush=True)
    mjo_field = xr.apply_ufunc(
        wk_mjo_filter_time_lon_kernel,
        anom,
        input_core_dims=[["time", "lon"]],
        output_core_dims=[["time", "lon"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
    )
    mjo_field.name = "olr_surprisal_mjo"
    mjo_field.attrs["description"] = "WK filtered MJO component of OLR surprisal anomalies (eastward, k=1..5, 30-96d)."

    # Variance fraction map (time variance)
    print("Computing variance fractions ...", flush=True)
    var_total = anom.var("time")
    var_mjo = mjo_field.var("time")
    # Avoid divide by zero if a point has 0 variance (e.g. constant 0s)
    frac = (var_mjo / var_total).where(var_total > 1e-12, other=0.0)
    frac.name = "mjo_variance_fraction"
    frac.attrs["description"] = "var(MJO filtered surprisal)/var(total surprisal anomalies) at each grid point."

    # Build EOF-based index on canonical belt
    print("Subsetting belt for EOF index ...", flush=True)
    belt = mjo_field.sel(lat=slice(args.lat0, args.lat1), lon=slice(args.lon0, args.lon1))

    # Load belt into memory for EOFs (this is the heavy step)
    print("Loading belt into memory for EOF1/EOF2 ...", flush=True)
    belt = belt.compute()

    print("Computing EOF1/EOF2 and PCs ...", flush=True)
    eof1, eof2, pc1, pc2 = compute_eof12_from_field(belt, weight_lat=True)

    # Define amplitude + phase
    amp = np.sqrt(pc1**2 + pc2**2)
    amp.name = "mjo_amp"
    amp.attrs["description"] = "MJO amplitude from PCs of WK-filtered surprisal field."

    phase = xr.apply_ufunc(np.arctan2, pc2, pc1).astype(np.float32)
    phase.name = "mjo_phase_rad"
    phase.attrs["description"] = "MJO phase angle (radians) from PCs (atan2(PC2, PC1))."

    # Bursts
    print("Defining bursts from amplitude ...", flush=True)
    burst_ds = bursts_from_amplitude(
        amp,
        thresh_kind=args.burst_kind,
        thresh_value=args.burst_value,
        min_duration_days=args.burst_min_dur,
        min_separation_days=args.burst_min_sep,
    )

    # Package output
    ds_out = xr.Dataset(
        {
            "olr_surprisal_anom": anom.astype(np.float32),
            "olr_surprisal_mjo": mjo_field,
            "mjo_variance_fraction": frac.astype(np.float32),
            "EOF1": eof1,
            "EOF2": eof2,
            "PC1": pc1,
            "PC2": pc2,
            "mjo_amp": amp.astype(np.float32),
            "mjo_phase_rad": phase,
            "burst_mask": burst_ds["burst_mask"],
            "burst_onset": burst_ds["burst_onset"],
            "burst_id": burst_ds["burst_id"],
        }
    )

    ds_out.attrs["index_domain_lat"] = f"{args.lat0} to {args.lat1}"
    ds_out.attrs["index_domain_lon"] = f"{args.lon0} to {args.lon1}"
    ds_out.attrs["wk_filter"] = "eastward, k=1..5, 30-96 days"
    ds_out.attrs.update(burst_ds.attrs)

    print(f"Writing {args.output} ...", flush=True)
    enc = {v: {"zlib": True, "complevel": 5} for v in ds_out.data_vars}
    ds_out.to_netcdf(args.output, encoding=enc)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
