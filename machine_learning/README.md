# MAVEN Electron Spectra Machine Learning

This folder is an independent workspace for machine-learning work on MAVEN SWE electron spectra.
By default, downloaded data is shared with the project-level `data/maven` tree, while analysis products stay inside this folder.

## Directory Map

| Path | What it does |
| --- | --- |
| `README.md` | This guide. It explains the scripts, common workflows, options, and output files in `machine_learning/`. |
| `download_electron_spectra.py` | Downloads MAVEN SWE `svypad` electron spectra into the shared `data/maven` tree. It calls the project-level `download_maven_data.py` helper and writes download manifests under `outputs/downloads/`. |
| `analyze_electron_spectra_ml.py` | Baseline spectral-shape clustering. It loads SWE `svypad` CDF files, splits each timestamp into parallel and anti-parallel pitch-angle spectra, preprocesses flux with log/normalization options, then clusters the feature vectors with NumPy k-means. |
| `analyze_electron_spectra_gmm.py` | GMM version of the baseline spectral analysis. It uses the same paired parallel/anti-parallel spectral features as `analyze_electron_spectra_ml.py`, then fits a diagonal-covariance Gaussian Mixture Model with NumPy EM and can choose component count by BIC. |
| `analyze_electron_spectra_pca_gmm.py` | Large-sample PCA + sampled GMM for ordinary flux spectra. It builds the same normalized spectral vectors, reduces them with PCA, scans/trains GMM on a random subset, then predicts labels and probabilities for all samples. |
| `analyze_electron_spectra_derivative_ml.py` | Baseline derivative-spectrum clustering. It optionally uses LPW `mrgscpot` spacecraft potential to convert measured energy to plasma energy, turns spectra into `dFlux/dE`, normalizes derivative features, then clusters with k-means. |
| `analyze_electron_spectra_derivative_gmm.py` | GMM version of derivative-spectrum analysis. It reuses derivative preprocessing from `analyze_electron_spectra_derivative_ml.py`, then fits a diagonal-covariance GMM and can choose component count by BIC. |
| `analyze_electron_spectra_derivative_pca_gmm.py` | Large-sample PCA + sampled GMM for derivative spectra. It converts samples to paired `dFlux/dE` spectra, applies derivative-feature normalization by default, reduces with PCA, and runs sampled GMM. |
| `analyze_electron_spectra_derivative_spatial_pca_gmm.py` | Spatially split derivative PCA-GMM. It first assigns samples to 12 altitude/SZA bins using MAG `ss1s` positions, then runs derivative PCA-GMM separately inside each bin. It forces derivative-feature normalization to `none`. |
| `plot_cluster_spatial_distributions.py` | Post-processing for cluster assignments. It reads a `full_assignments.csv`, attaches MAG position metadata, and plots per-cluster SZA, altitude, and longitude-latitude distributions. |
| `outputs/` | Analysis products. Each script writes timestamped run folders here; this directory can become large and is output data rather than source code. |

## Script Families

- Ordinary flux spectra:
  - k-means: `analyze_electron_spectra_ml.py`
  - GMM: `analyze_electron_spectra_gmm.py`
  - PCA + sampled GMM: `analyze_electron_spectra_pca_gmm.py`
- Energy-derivative spectra:
  - k-means: `analyze_electron_spectra_derivative_ml.py`
  - GMM: `analyze_electron_spectra_derivative_gmm.py`
  - PCA + sampled GMM: `analyze_electron_spectra_derivative_pca_gmm.py`
  - altitude/SZA split + PCA + sampled GMM: `analyze_electron_spectra_derivative_spatial_pca_gmm.py`
- Spatial post-processing:
  - assignment maps and histograms: `plot_cluster_spatial_distributions.py`

## Download Data

Before downloading, test LASP connectivity:

```bash
python machine_learning/download_electron_spectra.py --check-connection --start-date 2024-11-07 --end-date 2024-11-07
```

```bash
python machine_learning/download_electron_spectra.py --year 2024
```

Output:

- `data/maven/swe/l2/svypad/...`
- `machine_learning/outputs/downloads/download_manifest.json`
- `machine_learning/outputs/downloads/download_manifest.csv`

For a shorter test interval:

```bash
python machine_learning/download_electron_spectra.py --start-date 2024-11-07 --end-date 2024-11-07
```

Only SWE `svypad` electron spectra are downloaded. The manifest records each file path plus the actual UTC time range and number of time samples inside the CDF.

## Run ML Analysis

```bash
python machine_learning/analyze_electron_spectra_ml.py --start 2024-11-07T00:00:00 --end 2024-11-08T00:00:00 --clusters 4
```

To let the script choose the number of clusters:

```bash
python machine_learning/analyze_electron_spectra_ml.py --start 2024-11-07T00:00:00 --end 2024-11-08T00:00:00 --auto-clusters --min-clusters 2 --max-clusters 10
```

To run the GMM version:

```bash
python machine_learning/analyze_electron_spectra_gmm.py --start 2024-11-07T00:00:00 --end 2024-11-08T00:00:00 --auto-clusters --min-clusters 2 --max-clusters 10
```

To run the large-sample PCA + sampled GMM version:

```bash
python machine_learning/analyze_electron_spectra_pca_gmm.py --start 2024-11-01T00:00:00 --end 2024-11-08T00:00:00 --pca-components 32 --sample-size 100000 --min-clusters 6 --max-clusters 15
```

To cluster energy-derivative spectra:

```bash
python machine_learning/analyze_electron_spectra_derivative_ml.py --start 2024-11-07T00:00:00 --end 2024-11-08T00:00:00 --auto-clusters --min-clusters 2 --max-clusters 10
```

To run the GMM version on energy-derivative spectra:

```bash
python machine_learning/analyze_electron_spectra_derivative_gmm.py --start 2024-11-07T00:00:00 --end 2024-11-08T00:00:00 --auto-clusters --min-clusters 2 --max-clusters 10
```

To run the large-sample PCA + sampled GMM version on energy-derivative spectra:

```bash
python machine_learning/analyze_electron_spectra_derivative_pca_gmm.py --start 2024-11-01T00:00:00 --end 2024-11-08T00:00:00 --pca-components 32 --sample-size 100000 --min-clusters 6 --max-clusters 15
```

To split derivative spectra by altitude/SZA first, then run PCA + sampled GMM separately in each spatial bin:

```bash
python machine_learning/analyze_electron_spectra_derivative_spatial_pca_gmm.py --start 2024-11-01T00:00:00 --end 2024-11-08T00:00:00 --pca-components 32 --sample-size 100000 --min-clusters 2 --max-clusters 6
```

The spatial split script uses MAG `ss1s` positions to assign each derivative sample to one altitude bin and one SZA bin:

| Code | Definition |
| --- | --- |
| `A1` | low altitude, `alt < 800 km` |
| `A2` | transition altitude, `800 <= alt < 2000 km` |
| `A3` | middle/high altitude, `2000 <= alt < 4500 km` |
| `A4` | very high altitude, `alt >= 4500 km` |
| `S1` | dayside, `SZA < 90 deg` |
| `S2` | near-dayside/terminator, `90 <= SZA < 110 deg` |
| `S3` | deep nightside, `SZA >= 110 deg` |

It writes one PCA-GMM result folder for each populated `A#_S#` bin. Each bin scans `k=2..6` by default and only writes the selected best-k result, not per-k candidate result folders. Derivative features are not normalized in this script; internally it calls derivative loading with `normalization=none`.

To plot SZA, altitude, and longitude-latitude distributions for the resulting clusters:

```bash
python machine_learning/plot_cluster_spatial_distributions.py machine_learning/outputs/derivative_pca_gmm_analysis/<run_name>/full_assignments.csv
```

Before analysis, the script checks whether `data/maven` contains every daily SWE `svypad` file needed by `--start` and `--end`. Missing days are downloaded automatically. To only check what is available or missing:

```bash
python machine_learning/analyze_electron_spectra_ml.py --start 2024-11-01T00:00:00 --end 2024-11-08T00:00:00 --check-data-only
```

Useful options:

- `--clusters 4`: number of characteristic spectral groups.
- `--auto-clusters`: try a cluster-count range and choose the lowest Davies-Bouldin score.
- `--min-clusters 2` / `--max-clusters 10`: range used by `--auto-clusters`.
- `--min-cluster-fraction 0.01`: reject automatic choices with very tiny clusters.
- `--no-save-candidates`: skip saving per-k candidate result folders when using `--auto-clusters`; the spatial split derivative PCA-GMM script already skips per-k candidate folders by design.
- `--pca-components 32`: for the PCA-GMM script, reduce the feature vectors to this many PCA dimensions.
- `--sample-size 100000`: for the PCA-GMM script, train and scan GMM on this many randomly selected samples.
- `--random-seed 0`: make the PCA-GMM training subset reproducible.
- `--direction both|parallel|anti_parallel`: `both` keeps one timestamp as one sample by pairing parallel and anti-parallel spectra; the other options use one direction only.
- `--parallel-pitch-max 30`: upper pitch-angle bound for parallel spectra.
- `--anti-parallel-pitch-min 150`: lower pitch-angle bound for anti-parallel spectra.
- `--min-direction-valid-fraction 0.1`: discard timestamps where a used direction has too few positive finite energy bins.
- `--normalization log|global_zscore|zscore|minmax|l2`: preprocessing method. Invalid, infinite, and negative flux values are set to `0`; `log` then maps zero values to a fixed small floor before `log10`.
- `--normalization zscore|global_zscore|minmax|l2|none`: derivative-feature preprocessing method used by derivative scripts. `analyze_electron_spectra_derivative_spatial_pca_gmm.py` intentionally does not expose this option and always uses `none`.
- `--stride 10`: use every 10th timestamp for a faster first pass.
- `--swe-file path/to/file.cdf`: analyze explicit CDF files instead of searching `data/maven`.
- `--no-auto-download`: fail if required SWE files are missing instead of downloading them.
- `--check-data-only`: check/download data coverage and exit before ML analysis.

Outputs are written to descriptive run folders under `machine_learning/outputs/analysis/`, for example:

```text
machine_learning/outputs/analysis/20241101T000000_20241108T000000_both_auto-k2-10_log/
```

GMM outputs are written under `machine_learning/outputs/gmm_analysis/`.
PCA-GMM outputs are written under `machine_learning/outputs/pca_gmm_analysis/`.
Derivative-spectrum outputs are written under `machine_learning/outputs/derivative_analysis/`.
Derivative GMM outputs are written under `machine_learning/outputs/derivative_gmm_analysis/`.
Derivative PCA-GMM outputs are written under `machine_learning/outputs/derivative_pca_gmm_analysis/`.
Spatially split derivative PCA-GMM outputs are written under `machine_learning/outputs/derivative_spatial_pca_gmm_analysis/`.

- `ml_summary.json`: settings, energy grid, PCA summary, representative times.
- `gmm_summary.json`: GMM settings, BIC/log-likelihood, weights, PCA summary, representative times.
- `pca_gmm_summary.json`: PCA-GMM settings, PCA explained variance, sampled k scan, full-data GMM scores, and representative times.
- `k_scan_metrics.csv`: BIC, AIC, and cluster-size diagnostics for every scanned k in PCA-GMM.
- `full_assignments.csv`: full-data PCA-GMM label and maximum probability for every sample.
- `predict_proba_full.npz`: compressed full-data labels and GMM probabilities.
- `spatial_distributions/`: per-cluster SZA histograms, altitude histograms, longitude-latitude maps, and enriched cluster metadata.
- `derivative_ml_summary.json`: derivative-spectrum settings and representative times.
- `derivative_gmm_summary.json`: derivative-spectrum GMM settings, BIC/log-likelihood, weights, and representative times.
- `derivative_pca_gmm_summary.json`: derivative-spectrum PCA-GMM settings, PCA explained variance, sampled k scan, full-data GMM scores, and representative times.
- `derivative_spatial_pca_gmm_summary.json`: top-level summary for the 12 altitude/SZA bin workflow, including bin sample counts and each bin's status.
- `spatial_sample_index.csv`: per-sample altitude/SZA bin assignment before bin-wise PCA-GMM.
- `spatial_assignments.csv`: per-bin cluster assignments with altitude, SZA, MAG time, source file, and maximum GMM probability.
- `representative_times.csv`: the timestamp closest to each characteristic paired spectrum.
- `characteristic_spectra.png`: median spectrum and nearest real spectrum for each cluster.
- `characteristic_derivative_spectra.png`: median derivative spectrum and nearest real derivative spectrum for each cluster.
- `pca_clusters.png`: PCA view of the normalized spectra colored by cluster.

When `--auto-clusters` is used, the best result stays in the run folder root. Results for every tried cluster count are also written under:

```text
candidate_clusters/k2/
candidate_clusters/k3/
...
```

Each candidate folder contains its own `characteristic_spectra.png`, `pca_clusters.png`, `representative_times.csv`, and `cluster_summary.json`.
