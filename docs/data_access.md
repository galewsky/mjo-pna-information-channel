# Data Access and Excluded Products

This repository does not store raw climate datasets or large derived arrays.

## External Inputs

- **CLARA-A3 daily OLR**: daily top-of-atmosphere outgoing longwave radiation from the CM SAF CLARA-A3 climate data record.
- **ERA5 Z500**: daily 500-hPa geopotential-height fields from ERA5 through ECMWF/Copernicus Climate Data Store.
- **ROMI index**: local input file used in the analysis as `romi.cpcolr.1x.txt`.
- **RMM index**: local validation input file used as `rmm.74toRealtime.txt`.
- **Nino3.4 daily anomalies**: local input file used for ENSO-conditioned sensitivity analysis as `nino34_daily_anomalies.nc`.

## Large Products Excluded from GitHub

The following products are intentionally excluded and should be regenerated or distributed separately if bitwise reproduction is required:

- `olr_tropics.zarr`
- `olr_tropics_anomalies.zarr`
- `olr_tropics_surprisal_knn.zarr`
- `olr_tropics_pointwise_info.nc`
- `mjo_index_from_surprisal.nc`
- `mjo_index_from_surprisal_timeseries.nc`
- `z500_global_raw.nc`
- `z500_receiver_phasebank_Bint_domain120_300_20_80_train1991_2010.nc`
- `manuscript_results/phasebank_codebook_weighted.nc`
- `manuscript_results/phasebank_codebook_gated.nc`
- null-distribution archives such as `*_nulldist.npz` and `*_null_dist.npz`

The lightweight CSV/TeX/Markdown products in `outputs/` are included to document the manuscript's numerical tables and validation summaries.

