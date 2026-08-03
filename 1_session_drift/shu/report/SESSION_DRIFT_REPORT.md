# Session Drift Report — shu_session_drift_v1

- Session pairs analyzed: **250** across **25** subjects (status=ok only).
- Bands: mu = [8, 13] Hz, beta = [13, 30] Hz; fs = 250 Hz.

## Metric summary (mean / median / std)

| metric | mean | median | std | what it means |
|---|---|---|---|---|
| `mmd` | 0.3556 | 0.3437 | 0.1489 | overall distribution distance (higher = more drift) |
| `coral` | 0.0000 | 0.0000 | 0.0000 | channel-covariance distance (higher = more drift) |
| `mu_power_shift` | 0.0372 | 0.0290 | 0.5073 | mu power change j-i (0 = stable) |
| `beta_power_shift` | -0.0044 | -0.0015 | 0.4996 | beta power change j-i (0 = stable) |
| `mu_ks_stat` | 0.2784 | 0.2515 | 0.1760 | mu-power distribution shift (0 = identical) |
| `erd_mu_corr` | 0.5267 | 0.5564 | 0.2455 | mu ERD/ERS spatial stability (1 = identical pattern) |
| `erd_beta_corr` | 0.5318 | 0.5871 | 0.2647 | beta ERD/ERS spatial stability (1 = identical) |
| `csp_similarity` | 0.3438 | 0.3341 | 0.1130 | spatial-filter stability (1 = identical) |
| `rms_ratio_median` | 1.3157 | 1.0323 | 1.1775 | amplitude ratio j/i (1 = no change) |
| `fisher_i` | 0.0135 | 0.0109 | 0.0092 | MI separability in session i |
| `fisher_j` | 0.0124 | 0.0093 | 0.0088 | MI separability in session j |
| `fisher_shift` | -0.0012 | -0.0005 | 0.0120 | separability change j-i (sign = direction) |

## How to read it

- **High MMD/CORAL, low CSP/ERD-ERS correlation** ⇒ spatial pattern drifts → favor alignment (Euclidean Alignment / CORAL).
- **Large `rms_ratio` deviation from 1 / mu-beta power shift** ⇒ amplitude/spectral drift → favor BN adaptation / re-normalization / filter-bank alignment.
- **Positive `fisher_shift` on later sessions** ⇒ MI separability improves (learning effect) → favor online/test-time adaptation.
- These point to which adaptation mechanism the cross-session model should use.

## Figures

- `figures/distribution_distance_hist.png`
- `figures/band_power_shift_hist.png`
- `figures/erd_ers_correlation_hist.png`
- `figures/csp_similarity_hist.png`
- `figures/fisher_ratio_scatter.png`
- `figures/rms_ratio_hist.png`
- `figures/metric_correlation_matrix.png`
- `figures/session_pair_comparison.png`
