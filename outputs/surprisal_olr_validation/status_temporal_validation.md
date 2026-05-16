# Temporal Validation Status

Status: complete

Outputs directory: `/home/galewsky/Dropbox/research/era5/data/olr/results/surprisal_olr_validation`

Provenance:
- Surprisal timeseries: `mjo_index_from_surprisal_timeseries.nc`
- Evaluation window: 1991-01-01 to 2020-12-31
- Gate definition: 10-day centered mean of `mjo_amp`, q90_train over 1991-01-01 to 2010-12-31
- Gate threshold: 14.6811
- Active-only amplitude correlation uses days when either the surprisal burst mask or the reference active mask is on.
- Lag sign convention: positive lag means the reference amplitude lags the surprisal amplitude.

Gate/burst overlap:
- Gate-on days: 898
- Burst days: 1667
- Burst days also gate-on: 0.5387
- Gate-on days also in burst: 1.0000
- Burst events: 67, mean duration 24.8806 d, median duration 14.0000 d

Reference comparisons:
- ROMI burst-vs-active: HSS=0.0973, CSI=0.1828, POD=0.2006, FAR=0.3269
- ROMI gate-vs-active: HSS=0.0607, CSI=0.1071, POD=0.1123, FAR=0.3007
- ROMI amplitude correlation: r_all=0.2459, r_active_any=-0.0239, peak lag=-2 d (positive means reference lags surprisal)
- ROMI event-level: hit rate=0.2639, false alarm rate=0.1791, mean onset offset=11.2281 d
- RMM burst-vs-active: HSS=0.0309, CSI=0.1544, POD=0.1668, FAR=0.3245
- RMM gate-vs-active: HSS=0.0181, CSI=0.0871, POD=0.0908, FAR=0.3174
- RMM amplitude correlation: r_all=0.1356, r_active_any=-0.0686, peak lag=-1 d (positive means reference lags surprisal)
- RMM event-level: hit rate=0.2580, false alarm rate=0.1343, mean onset offset=0.9726 d
