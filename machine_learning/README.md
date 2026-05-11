# MAVEN Electron Spectra Machine Learning

This folder is an independent workspace for machine-learning work on MAVEN SWE electron spectra.
By default, downloaded data is shared with the project-level `data/maven` tree, while analysis products stay inside this folder.

## Files

- `download_electron_spectra.py`
  - Controls downloading of SWE `svypad` electron spectra.
  - Calls the project-level `download_maven_data.py` helper instead of duplicating download logic.
- `analyze_electron_spectra_ml.py`
  - Loads local SWE `svypad` CDF files.
  - Splits each timestamp into `parallel` and `anti_parallel` spectra using pitch-angle bins, following `process_maven_spectra.py`.
  - Applies `log10` scaling and per-spectrum normalization.
  - Uses NumPy k-means clustering to find characteristic spectral distributions.
  - Finds the real timestamp nearest to each characteristic cluster spectrum.
- `analyze_electron_spectra_gmm.py`
  - Uses the same data checking and paired spectral features.
  - Fits a diagonal-covariance Gaussian Mixture Model with NumPy EM.
  - Uses BIC for automatic component-count selection.
- `analyze_electron_spectra_pca_gmm.py`
  - Uses the same paired spectral features as the GMM script.
  - Reduces the normalized 128-D spectra to PCA scores before GMM.
  - Trains/scans GMM on a reproducible random subset, then predicts labels and probabilities for all samples.
  - Writes BIC/AIC/cluster-size diagnostics for every scanned k.
- `analyze_electron_spectra_derivative_pca_gmm.py`
  - Converts each sample to paired `dFlux/dE` spectra first.
  - Then applies PCA, sampled GMM k scan, and full-data probability prediction.
- `plot_cluster_spatial_distributions.py`
  - Reads a PCA-GMM `full_assignments.csv`.
  - Uses MAG positions to plot per-cluster SZA, altitude, and longitude-latitude distributions.
  - Longitude-latitude maps can use the same crustal-field background style as the orbit map.
- `analyze_electron_spectra_derivative_ml.py`
  - Uses the same inputs and paired spectral sampling.
  - Converts each spectrum to `dFlux/dE` before normalization.
  - Clusters the resulting `(flux-change-rate, energy)` spectra with k-means.
- `analyze_electron_spectra_derivative_gmm.py`
  - Uses the same derivative preprocessing.
  - Fits a diagonal-covariance GMM to the `(flux-change-rate, energy)` spectra.
  - Uses BIC for automatic component-count selection.

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
- `--no-save-candidates`: skip saving per-k candidate result folders when using `--auto-clusters`.
- `--pca-components 32`: for the PCA-GMM script, reduce the feature vectors to this many PCA dimensions.
- `--sample-size 100000`: for the PCA-GMM script, train and scan GMM on this many randomly selected samples.
- `--random-seed 0`: make the PCA-GMM training subset reproducible.
- `--direction both|parallel|anti_parallel`: `both` keeps one timestamp as one sample by pairing parallel and anti-parallel spectra; the other options use one direction only.
- `--parallel-pitch-max 30`: upper pitch-angle bound for parallel spectra.
- `--anti-parallel-pitch-min 150`: lower pitch-angle bound for anti-parallel spectra.
- `--min-direction-valid-fraction 0.1`: discard timestamps where a used direction has too few positive finite energy bins.
- `--normalization log|global_zscore|zscore|minmax|l2`: preprocessing method. Invalid, infinite, and negative flux values are set to `0`; `log` then maps zero values to a fixed small floor before `log10`.
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
- `representative_times.csv`: the timestamp closest to each characteristic paired spectrum.
- `characteristic_spectra.png`: median spectrum and nearest real spectrum for each cluster.
- `pca_clusters.png`: PCA view of the normalized spectra colored by cluster.

When `--auto-clusters` is used, the best result stays in the run folder root. Results for every tried cluster count are also written under:

```text
candidate_clusters/k2/
candidate_clusters/k3/
...
```

Each candidate folder contains its own `characteristic_spectra.png`, `pca_clusters.png`, `representative_times.csv`, and `cluster_summary.json`.
