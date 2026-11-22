# Data Quality Report - Delhi Daily Weather (raw)
Generated: 2026-11-22 19:18

## Coverage
- Rows in raw file: 4018 (expected 4018)
- Missing dates: 0 | duplicate dates: 0
- Chronological: True | unexpected gaps: 0

## Missing values
          missing  missing_%
tmax            0        0.0
humidity        0        0.0
wind            0        0.0
solar           0        0.0

## Duplicates
- Exact duplicate rows: 0 | conflicting duplicate dates: 0

## Physical validity
          hard-rule violations  warning-range flags   min    max
variable                                                        
tmax                         0                    0  10.3  46.50
humidity                     0                    0  21.0  99.00
wind                         0                    0   5.5  32.30
solar                        0                    0   1.2  29.21

## Outlier flags (investigation only)
          iqr  zscore  seasonal_anomaly  day_jump  vs_neighbours
tmax        3       1                18         8              3
humidity   21       0                21         3              3
wind       75      43                31         0              0
solar       9       9                73       116             68
- Tmax days flagged by >=2 methods: 6

## Recommended cleaning actions
1. Standardise dtypes/order; remove exact duplicates (stop on conflicting duplicates).
2. Reindex to the full 4,018-day calendar so gaps stay explicit.
3. Convert hard-rule violations to NaN (audited). No imputation here.
4. Keep all genuine hot extremes; no smoothing.
