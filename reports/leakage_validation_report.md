# Leakage Validation Report

- PASS - target / date columns are NOT among the 21 predictors
- PASS - every input day <= forecast origin < target date
- PASS - trend features use only the 3-day input window
- PASS - lags date-aligned (independent re-check: 0 mismatches)
- PASS - calendar features use the target date only
- PASS - no shuffling: each split sorted by target date
- PASS - imputer fitted on training data only
- PASS - no feature selection yet (all 21 candidates kept)
- PASS - no synthetic rows (target == real Tmax of target date)
- PASS - no duplicate (origin, horizon) inside / across splits
- PASS - train targets <= 2022-12-31
- PASS - validation targets within 2023-2024
- PASS - test targets within 2025
- PASS - target-date ranges disjoint (train < val < test)