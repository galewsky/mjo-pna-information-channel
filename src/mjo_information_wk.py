import argparse
import xarray as xr
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq

def filter_mjo_numpy_kernel(data_2d):
    """
    Core numpy function to be applied in parallel via xarray.apply_ufunc.
    Input: 2D numpy array (time, lon)
    """
    # 1. Handle NaNs 
    if np.isnan(data_2d).any():
        data_2d = np.nan_to_num(data_2d, nan=0.0)
    
    nt, nx = data_2d.shape
    
    # 2. Perform 2D FFT (Unshifted)
    # We maintain the standard numpy order: 
    # [0, 1, ... N/2-1, -N/2, ... -1]
    f_coef = fft2(data_2d)
    
    # 3. Create Frequency/Wavenumber Grids (Unshifted)
    # d=1.0 means units are cycles/index. 
    # For time: cycles/day (assuming daily data).
    # For space: zonal wavenumbers (assuming global domain).
    f = fftfreq(nt, d=1.0)       
    k = fftfreq(nx, d=1.0/nx)    
    k_grid, f_grid = np.meshgrid(k, f)
    
    # 4. Define MJO Band (WK99 Standard)
    # Period: 30 to 96 days
    f_low, f_high = 1/96.0, 1/30.0
    # Wavenumber: 1 to 5
    k_low, k_high = 1, 5
    
    # 5. Create Mask (Eastward + Conjugate Symmetry)
    # We specifically define the "Eastward" quadrant as f > 0 and k > 0.
    # Note: In standard Numpy FFT conventions (exp(-2pi i ...)), 
    # (f>0, k>0) technically corresponds to Westward phase velocity. 
    # However, in WK99 diagram conventions, this quadrant is often 
    # labeled Eastward. We follow the user's explicit definition here.
    
    # A. Positive Quadrant (f > 0, k > 0)
    mask_pos = (
        (f_grid > 0) & (k_grid > 0) &
        (np.abs(k_grid) >= k_low) & (np.abs(k_grid) <= k_high) &
        (np.abs(f_grid) >= f_low) & (np.abs(f_grid) <= f_high)
    )
    
    # B. Negative Quadrant (f < 0, k < 0)
    # This is the Hermitian symmetric partner of A. 
    # Essential for the Inverse FFT to produce Real-valued output.
    # We use explicit condition rather than flip to avoid index 0 issues.
    mask_neg = (
        (f_grid < 0) & (k_grid < 0) &
        (np.abs(k_grid) >= k_low) & (np.abs(k_grid) <= k_high) &
        (np.abs(f_grid) >= f_low) & (np.abs(f_grid) <= f_high)
    )
    
    # Combine
    mask = mask_pos | mask_neg
    
    # 6. Apply Filter & Reconstruct
    f_coef_filtered = f_coef * mask
    
    # No ifftshift needed since we stayed in the unshifted domain
    reconstructed = ifft2(f_coef_filtered).real
    
    return reconstructed

def main():
    # --- Configuration ---
    parser = argparse.ArgumentParser(description="WK99 MJO filter on info-production field.")
    parser.add_argument("--input", default="olr_tropics_pointwise_info_production.nc")
    parser.add_argument("--output", default="mjo_info_injection_efficient.nc")
    parser.add_argument("--var", default="info_production_bits_per_second")
    parser.add_argument("--lat-min", type=float, default=None, help="optional latitude min")
    parser.add_argument("--lat-max", type=float, default=None, help="optional latitude max")
    parser.add_argument("--coarsen", type=int, default=1, help="coarsen factor for lat/lon")
    args = parser.parse_args()
    
    # 1. LOAD DATA
    print(f"Opening {args.input}...")
    ds = xr.open_dataset(args.input, chunks={'lat': 10, 'time': -1, 'lon': -1})
    da = ds[args.var]

    # --- SANITY CHECK: Longitude Orientation ---
    # Ensure longitude is increasing (e.g., 0, 2.5, 5.0...) 
    # If decreasing, k>0 implies Westward, effectively flipping the filter.
    if da.lon.size > 1 and (da.lon[1] < da.lon[0]):
        print("\n" + "!"*60)
        print("WARNING: Longitude coordinate appears to be DECREASING.")
        print("The current filter definition (k>0) assumes INCREASING longitude.")
        print("This may filter for Westward waves instead of Eastward MJO.")
        print("!"*60 + "\n")

    if args.lat_min is not None or args.lat_max is not None:
        lat_min = -90.0 if args.lat_min is None else args.lat_min
        lat_max = 90.0 if args.lat_max is None else args.lat_max
        da = da.sel(lat=slice(lat_min, lat_max))
        print(f"Subsetting lat to {lat_min}..{lat_max}", flush=True)

    if args.coarsen > 1:
        print(f"Coarsening lat/lon by factor {args.coarsen}...", flush=True)
        da = da.coarsen(lat=args.coarsen, lon=args.coarsen, boundary="trim").mean()
    
    # 2. CALCULATE ANOMALIES (Lazy)
    print("Calculating anomalies (removing seasonal cycle)...")
    climatology = da.groupby('time.dayofyear').mean('time')
    anomalies = da.groupby('time.dayofyear') - climatology
    
    if 'dayofyear' in anomalies.coords:
        anomalies = anomalies.drop_vars('dayofyear')

    # Re-chunking for FFT
    print("Re-chunking data for spectral analysis...")
    anomalies = anomalies.chunk({'time': -1, 'lon': -1, 'lat': 10})

    # 3. APPLY SPECTRAL FILTER (Parallel)
    print("Applying WK99 MJO filter via apply_ufunc (Unshifted grid)...")
    mjo_extracted = xr.apply_ufunc(
        filter_mjo_numpy_kernel,
        anomalies,
        input_core_dims=[['time', 'lon']], 
        output_core_dims=[['time', 'lon']], 
        dask='parallelized',                
        vectorize=True,                     
        output_dtypes=[np.float32]          
    )
    
    mjo_extracted.name = 'mjo_info_production'
    mjo_extracted.attrs['units'] = 'bits/s'
    mjo_extracted.attrs['description'] = 'MJO Filtered (k=1-5, T=30-96d, Eastward)'

    # 4. COMPUTE VARIANCE STATS
    print("Defining variance statistics...")
    var_total = anomalies.var(dim='time')
    var_mjo = mjo_extracted.var(dim='time')
    
    mjo_fraction = var_mjo / var_total
    mjo_fraction.name = 'mjo_variance_fraction'
    mjo_fraction.attrs['long_name'] = "Fraction of Info Production Variance Explained by MJO"

    # 5. EXECUTE AND SAVE
    print(f"Streaming results to {args.output}...")
    ds_out = xr.Dataset({
        'mjo_info_production': mjo_extracted,
        'mjo_fraction': mjo_fraction,
    })
    
    encoding = {v: {'zlib': True, 'complevel': 5} for v in ds_out.data_vars}
    ds_out.to_netcdf(args.output, encoding=encoding)
    print("Done! Computation complete.")

if __name__ == "__main__":
    main()
