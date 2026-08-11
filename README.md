# Alert-Budget-Aware Assessment of Unsupervised IDS for Smart-Grid Traffic

This repository contains the reproducibility artifact for **“Alert-Budget-Aware Assessment Workflow for Evaluating Unsupervised IDS in IEC 61850 Smart-Grid Communication.”** It preserves the window-level SGSim-derived dataset and accepted evaluation notebook and adds the repeated-seed evaluation used for the reported uncertainty and low-FPR analysis.

## Scope

The empirical unit is a 100 ms communication window represented by 61 numeric features. The dataset contains 500,565 windows: 433,598 normal and 66,967 attack windows. Normal traffic is partitioned at capture-file level into training, validation, and test sets. Thresholds are calibrated only from normal validation scores. DoS and FDI labels are used for test-set metric computation and attack-family recall.

The artifact evaluates five representative unsupervised anomaly-scoring mechanisms:

- Dense Autoencoder
- LSTM Autoencoder
- Hybrid Autoencoder
- Isolation Forest
- One-Class SVM

## Repository structure

```text
Autoencoder_v0.5.1_operational_IDS_eval.ipynb   accepted notebook analysis
all.parquet                                     processed SGSim-derived feature dataset
multiseed_evaluation.py                         repeated-seed and low-FPR evaluation
make_manuscript_figures.py                      regenerates the two manuscript figures
environment_freeze.txt                          exact Python package freeze for the repeated-seed campaign
requirements_accepted_artifact.txt              dependency snapshot retained from the accepted artifact
results/
  manifest.json
  per_seed_reference.csv
  per_seed_budgets.csv
  summary_reference.csv
  summary_budgets.csv
  timings.csv
figures/
  multiseed_budget_f1.pdf/.png
  raw_false_triggers_per_day.pdf/.png
archive/
  camera_ready_multiseed_final.zip              locked result archive
PROVENANCE.md
SHA256SUMS.txt
```

## Reproducing the repeated-seed evaluation

The reported experiment uses Python 3.11.9 and seeds 67--76. The exact package set is recorded in `environment_freeze.txt`. The original accepted notebook dependency snapshot is retained separately as `requirements_accepted_artifact.txt`.

From a Python environment containing the packages in `environment_freeze.txt`, run:

```bash
python multiseed_evaluation.py --data all.parquet --out rerun_results
```

The script evaluates nominal validation FPR budgets of 0.01%, 0.02%, 0.05%, 0.1%, 0.2%, 0.5%, 1%, 2%, 5%, and 10%. The principal manuscript analysis uses 0.1%--10%; 0.01%--0.05% are retained as extreme-tail sensitivity points because the finite normal-validation set limits empirical percentile resolution in that region.

To regenerate the two manuscript figures from either the locked or rerun summary results:

```bash
python make_manuscript_figures.py --results results --out figures_regenerated
```

## Interpretation of false-positive burden

`raw_false_triggers_per_day` is derived from observed normal-test FPR under continuous 100 ms scoring (864,000 windows/day). It therefore counts **raw threshold-positive normal windows before temporal aggregation, correlation, suppression, or incident-level alert formation**. It should not be interpreted as a count of independent analyst-facing incidents.

## Specification-aware detection boundary

The processed artifact does not contain the SCD file or raw packet captures needed for a faithful SCD-conformance or frame-level specification-based baseline. Aggregate GOOSE/SV counter statistics in the feature table are not treated as a substitute for that information. The manuscript therefore evaluates statistical anomaly scoring and identifies specification-aware or layered detection as a complementary direction rather than reporting an approximate SCD baseline.

## Provenance

The SHA-256 of the processed dataset used for the repeated-seed evaluation is:

```text
ad033be782decb00a043d7a59bd6801e4c66a446bc95b088d68eabc7237cc9dc  all.parquet
```

Additional hashes and execution metadata are recorded in `PROVENANCE.md` and `SHA256SUMS.txt`.
