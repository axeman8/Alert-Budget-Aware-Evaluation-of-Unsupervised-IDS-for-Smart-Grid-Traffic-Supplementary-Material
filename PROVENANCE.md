# Provenance and analysis lock

## Dataset identity

- File: `all.parquet`
- SHA-256: `ad033be782decb00a043d7a59bd6801e4c66a446bc95b088d68eabc7237cc9dc`
- Rows: 500,565
- Features: 61
- Normal training windows: 296,727
- Normal validation windows: 47,447
- Normal test windows: 89,424

The dataset hash above matches the dataset used in the repeated-seed campaign.

## Repeated-seed campaign

- Seeds: 67, 68, 69, 70, 71, 72, 73, 74, 75, 76
- Nominal validation FPR budgets (%): 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0
- Reference percentile: 95.0
- Window duration: 0.1 s
- Assumed continuous windows/day: 864,000
- Empirical normal-validation FPR resolution: 0.00210761%
- Python: 3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]
- Platform: Windows-10-10.0.26200-SP0
- NumPy: 2.4.3
- pandas: 3.0.1
- SciPy: 1.17.0
- scikit-learn: 1.8.0
- TensorFlow: 2.20.0
- TensorFlow devices: /physical_device:CPU:0

The 95% intervals in the summary tables are t intervals across the ten stochastic seeds. They quantify seed-to-seed variation under the fixed data partition.

## Locked result archive

- `archive/camera_ready_multiseed_final.zip` SHA-256: `362c87110569bdb4ea8c851741be24a8ae40b71eadd5fed1e2b5192ee15a161e`

## Key file hashes

- `multiseed_evaluation.py`: `1d77d06449b5c61098f08aae207eb8cf358bc4806e1a2504301093bb9ba336c2`
- `results/per_seed_budgets.csv`: `79672aaeb458e2f01d11f261298aa68619b9e0d9539ee5a883e4b03cd5e48437`
- `results/per_seed_reference.csv`: `377d41153a687e12bc0d567af9c7b54722f67a65726d28c1d37040fd1f7796ec`
- `results/summary_budgets.csv`: `e37de821711f9eaeda08a66cea78fbc50bfabebaf1944145392d80d517093b30`
- `results/summary_reference.csv`: `97655148516fc86c387b9c5837c490528c68ced8fd09e70626bce5d4b94e6c8f`
- `results/timings.csv`: `e892948060035259291491e5e2bf19972d32488ba2863d281fe2f6a3458842fa`
- `results/manifest.json`: `3cefd2666da305efa37f60e37a658e7fba9d6e80a208b2aac5bec06e2e89d6db`

## Scope boundary

The artifact does not include SCD configuration files or raw packet captures. It therefore does not support a faithful SCD-conformance or frame-level specification-based baseline. The manuscript treats specification-aware detection as complementary future work rather than deriving an approximate baseline from aggregate window features.
