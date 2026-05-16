Phasebank Information Gain Pipeline
===================================

This note records the subset of OLR-processing scripts that feed into
`phasebank_information_gain.py`, ordered from the rawest inputs forward
through phasebank evaluation.

1. `build_olr_zarr.py`
   - Subsets the CM SAF CLARA-A3 daily OLR archive (`OLRdm*.nc`) to a deep-tropics belt.
   - Writes `olr_tropics.zarr` with consistent chunking/metadata for later steps.

2. `preprocess_olr_anomalies.py`
   - Starts from `olr_tropics.zarr`.
   - Applies QC (missing-data mask), removes Feb-29 (optional), detrends, and removes the
     day-of-year climatology.
   - Outputs `olr_tropics_anomalies.zarr` plus ancillary masks/trend diagnostics.

3. `compute_olr_surprisal_knn.py`
   - Reads `olr_tropics_anomalies.zarr`.
   - Performs season-conditioned 1-D kNN density estimates at every grid point to convert
     anomalies into surprisal (bits).
   - Writes `olr_tropics_surprisal_knn.zarr`.

4. `make_tropics_pointwise_info_production.py`
   - Extracts the desired latitude belt from the surprisal Zarr and writes a portable
     NetCDF (`olr_tropics_pointwise_info.nc`) containing `olr_surprisal`.

5. `mjo_index_from_surprisal.py`
   - Ingests `olr_tropics_pointwise_info.nc` and constructs the WK-filtered MJO field.
   - Computes EOF1/EOF2, PC1/PC2, amplitude/phase, and burst metadata, saving everything to
     `mjo_index_from_surprisal.nc`.
   - A trimmed time-series-only file (`mjo_index_from_surprisal_timeseries.nc`) is created
     from this product and provides the PCs/amp/phase/burst series used downstream.

6. `build_z500_phasebank_receiver.py`
   - Combines `z500_global_raw.nc` with `mjo_index_from_surprisal_timeseries.nc` and the
     ROMI phase table (`romi.cpcolr.1x.txt`).
   - Regresses Z500 anomalies onto phase-conditioned drivers for each lag to produce
     lag×phase templates, amplitudes, `z500_score_weighted`, gate/phase metadata, etc.
   - Outputs the phasebank receiver NetCDF (e.g., `z500_receiver_phasebank_*.nc`).

7. `phasebank_information_gain.py`
   - Final evaluation script. Opens the receiver NetCDF, selects the requested test window,
     and computes phase-resolved information gain / MI metrics, optionally saving null
     distributions and FDR-adjusted statistics.


2026-02 Analysis/Manuscript Update
==================================

This branch extends the original phasebank pipeline to produce a complete
manuscript-ready analysis package.

Key script updates/additions
----------------------------

1. `phasebank_information_gain.py`
   - Benjamini-Hochberg FDR calculation corrected.
   - Added optional sample conditioning via:
     - `--condition_file`
     - `--condition_var`
     - `--condition_values`
   - This supports ENSO-conditioned phase-resolved analysis without breaking lag semantics.

2. `evaluate_phasebank_skill.py`
   - Benjamini-Hochberg FDR calculation corrected.

3. `make_enso_state_mask.py`
   - Builds daily ENSO state codes from `nino34_daily_anomalies.nc`:
     - `1` = El Nino
     - `0` = Neutral
     - `-1` = La Nina

4. `make_phasebank_manuscript_figures.py`
   - Generates reproducible manuscript figures and summary tables in
     `manuscript_results/figures/`.

2026-05 technical-revision notes
--------------------------------

- `phasebank_information_gain.py` now estimates MI/IG from one smoothed joint
  receiver/source count table. Marginals and conditionals are derived from that
  table, avoiding negative information values caused by subtracting separately
  smoothed entropy estimates. Continuous-intensity bins can include the p85 gate
  edge with `--include_gate_edge` for a cleaner MI/IG comparison.
- `build_z500_phasebank_receiver.py` and the auxiliary receiver builders use
  lag-aware phase matching:
  `score(t,L) = z500_amp(t,L, phase_index(t-L))`. This matches the information
  analysis pairing of source day `tau` with receiver day `tau+L`.
- `evaluate_phasebank_skill.py` uses the same lag-aware source/receiver pairing
  and can recompute a p85 gate from the training window with
  `--gate_source recompute --gate_percentile 85 --gate_train ...`.
- `compute_phasebank_telecom_metrics.py` and
  `make_phasebank_manuscript_figures.py` report coding efficiency as
  `IG/H_L(G|k)`, the conditional binary entropy within phase computed on the
  same lag-paired samples used for the IG estimate, not the smaller entropy of
  the joint phase-and-gate event.


Reproducible command sequence
-----------------------------

Base receiver (already present in this workspace):

`z500_receiver_phasebank_Bint_domain120_300_20_80_train1991_2010.nc`

1. Main phase-resolved information analysis:

```bash
python phasebank_information_gain.py \
  --receiver_nc z500_receiver_phasebank_Bint_domain120_300_20_80_train1991_2010.nc \
  --test 2011-01-01,2020-12-31 \
  --out_prefix manuscript_results/info_gain_p85 \
  --target amp_phase \
  --gate_source recompute \
  --gate_percentile 85 \
  --gate_train 1991-01-01,2010-12-31 \
  --assume_intensity_already_smoothed \
  --window_mode binary_gate \
  --null_mode blockperm \
  --block_len 60 \
  --n_null 2000 \
  --underpowered_policy flag_only \
  --global_test --save_null_dist
```

2. Continuous-intensity MI run:

```bash
python phasebank_information_gain.py \
  --receiver_nc z500_receiver_phasebank_Bint_domain120_300_20_80_train1991_2010.nc \
  --test 2011-01-01,2020-12-31 \
  --out_prefix manuscript_results/info_gain_cont_gatealigned \
  --target amp_phase \
  --gate_source recompute \
  --gate_percentile 85 \
  --gate_train 1991-01-01,2010-12-31 \
  --assume_intensity_already_smoothed \
  --window_mode continuous_intensity \
  --include_gate_edge \
  --null_mode blockperm \
  --block_len 60 \
  --n_null 2000 \
  --w_bins 5 \
  --underpowered_policy flag_only \
  --global_test --save_null_dist
```

Red-noise robustness (exact spectrum-preserving null via circular shift):

```bash
python phasebank_information_gain.py \
  --receiver_nc z500_receiver_phasebank_Bint_domain120_300_20_80_train1991_2010.nc \
  --test 2011-01-01,2020-12-31 \
  --out_prefix manuscript_results/info_gain_p85_cshift \
  --target amp_phase \
  --gate_source recompute --gate_percentile 85 --gate_train 1991-01-01,2010-12-31 \
  --assume_intensity_already_smoothed \
  --window_mode binary_gate \
  --null_mode circshift \
  --n_null 2000 \
  --global_test --save_null_dist

python phasebank_information_gain.py \
  --receiver_nc z500_receiver_phasebank_Bint_domain120_300_20_80_train1991_2010.nc \
  --test 2011-01-01,2020-12-31 \
  --out_prefix manuscript_results/info_gain_cont_cshift \
  --target amp_phase \
  --gate_source recompute \
  --gate_percentile 85 \
  --gate_train 1991-01-01,2010-12-31 \
  --assume_intensity_already_smoothed \
  --window_mode continuous_intensity \
  --include_gate_edge \
  --null_mode circshift \
  --n_null 2000 \
  --global_test --save_null_dist
```

3. Independent receiver skill checks:

```bash
python evaluate_phasebank_skill.py \
  --receiver_nc z500_receiver_phasebank_Bint_domain120_300_20_80_train1991_2010.nc \
  --test 2011-01-01,2020-12-31 \
  --null blockperm --block_len 60 --n_null 2000 \
  --stat snr \
  --output_prefix manuscript_results/phasebank_skill_snr \
  --gate_source recompute \
  --gate_percentile 85 \
  --gate_train 1991-01-01,2010-12-31 \
  --assume_intensity_already_smoothed \
  --global_test --save_null_dist

python evaluate_phasebank_skill.py \
  --receiver_nc z500_receiver_phasebank_Bint_domain120_300_20_80_train1991_2010.nc \
  --test 2011-01-01,2020-12-31 \
  --null blockperm --block_len 60 --n_null 2000 \
  --stat auc --two_sided \
  --output_prefix manuscript_results/phasebank_skill_auc \
  --gate_source recompute \
  --gate_percentile 85 \
  --gate_train 1991-01-01,2010-12-31 \
  --assume_intensity_already_smoothed \
  --global_test
```

4. ENSO conditioning:

```bash
python make_enso_state_mask.py \
  --input nino34_daily_anomalies.nc \
  --output manuscript_results/enso_state_daily.nc \
  --smooth_days 30 \
  --threshold 0.5
```

Example (El Nino, continuous MI):

```bash
python phasebank_information_gain.py \
  --receiver_nc z500_receiver_phasebank_Bint_domain120_300_20_80_train1991_2010.nc \
  --test 2011-01-01,2020-12-31 \
  --out_prefix manuscript_results/info_gain_cont_elnino_mt60 \
  --target amp_phase \
  --window_mode continuous_intensity \
  --gate_source from_file \
  --condition_file manuscript_results/enso_state_daily.nc \
  --condition_var enso_state \
  --condition_values 1 \
  --min_total 60 \
  --null_mode blockperm --block_len 60 --n_null 1000
```

Repeat with `--condition_values 0` and `--condition_values -1` for Neutral and La Nina.

5. Figure and table generation:

```bash
python make_phasebank_manuscript_figures.py \
  --results_dir manuscript_results \
  --output_dir manuscript_results/figures
```

This stage also exports both template codebooks for inspection:

- `manuscript_results/phasebank_codebook_weighted.nc`
- `manuscript_results/phasebank_codebook_gated.nc`

6. Manuscript compilation:

```bash
latexmk -pdf templateV6.1.tex
```
