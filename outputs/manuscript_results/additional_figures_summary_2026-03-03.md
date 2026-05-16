# Additional Reviewer Figures Summary (Generated 2026-03-03)

## Figure Files

- `wk_spectrum`: `manuscript_results/figures/fig13_wk_surprisal_spectrum.png`
- `eof_patterns`: `manuscript_results/figures/fig14_eof_patterns_variance.png`
- `phase5_lag_evolution`: `manuscript_results/figures/fig15_phase5_codebook_lag_evolution.png`
- `phase6_companion`: `manuscript_results/figures/fig16_phase6_companion_snapshot.png`
- `receiver_pdfs`: `manuscript_results/figures/fig17_receiver_pdf_gate_on_off.png`
- `four_metric_scorecard`: `manuscript_results/figures/fig18_four_metric_scorecard.png`
- `roc_curves`: `manuscript_results/figures/fig19_roc_curves_phase_contrast.png`

## Quantitative Diagnostics

### 1) WK Spectrum
- Peak enhanced power in retained band: `k=2.00`, `f=0.0104` cpd (period `96.0` days).
- Mean power/background ratio inside retained band: `1.215`.
- Welch segments used: `493` (96 d windows; 65 d overlap).

### 2) EOF Structure
- Variance explained: EOF1 `12.68%`, EOF2 `11.86%`, EOF1+EOF2 `24.54%`.
- PC orthogonality check: corr(PC1,PC2) at lag 0 = `0.000`; max abs lead-lag corr = `0.854` at -11 days.

### 3) Phase-Template Diagnostics
- Phase 5 MI peak: `0.232` bits at lag `25` days.
- MI at plotted lags (bits): lag12:0.167, lag16:0.114, lag20:0.147, lag24:0.230.
- 2×2 snapshot scaling: MI global peak `0.232` bits; Phase 5 weighted `0.232` (lag 25), Phase 5 gated `0.087` (lag 13); Phase 6 weighted `0.218` (lag 8), Phase 6 gated `0.131` (lag 8).

### 4) Receiver PDFs and ROC
- PDF contrast (signal phase 5, lag 13): AUC `0.701`, Cohen's d `0.59`.
- PDF contrast (null phase 1, lag 15): AUC `0.524`, Cohen's d `0.02`.
- ROC phase 1: AUC `0.524` (lag 15, n_on 23, n_off 460).
- ROC phase 5: AUC `0.701` (lag 13, n_on 21, n_off 405).
- ROC phase 6: AUC `0.489` (lag 8, n_on 25, n_off 384).

### 5) Four-Metric Scorecard
- Top spectral-efficiency phase: `5`; top throughput phase: `1`.
- Top coding-efficiency phase: `6`; lowest outage phase: `8`.

## Notes

- WK spectrum follows WK99: sym/anti decomposition (15S–15N), 96-day overlapping segments (step 31), 10% Tukey time taper, no temporal subsampling; background = (sym+anti)/2 smoothed by repeated 1–2–1 filters (freq×40, wav×20).
- EOF scree computed on a coarsened 2-degree grid over 60-180E, 15S-15N; map panels and scree are internally consistent.
- ROC/PDF diagnostics follow the manuscript convention: within-phase gate-on/off classification using `burst_gate` and phase-matched receiver amplitudes.
