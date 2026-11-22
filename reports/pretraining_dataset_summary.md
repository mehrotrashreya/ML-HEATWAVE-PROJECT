# Pre-Training Dataset Summary - Delhi Daily Tmax (3-day forecast)
Generated: 2026-11-22 17:13

## Dataset
- Source: delhi_daily_clean.csv (4018 days, 2015-01-01 -> 2025-12-31), md5 a6862fcb32...
- Supervised examples: 12042 | per horizon: h=1: 4015, h=2: 4014, h=3: 4013
- Input features: 21 (all numeric) | target: target_tmax (°C, continuous)

## Chronological split (by target date)
                       target period first target last target first origin last origin  rows  share_%   h=1   h=2   h=3
split                                                                                                                  
train       2015-01-01 -> 2022-12-31   2015-01-04  2022-12-31   2015-01-03  2022-12-30  8754    72.70  2919  2918  2917
validation  2023-01-01 -> 2024-12-31   2023-01-01  2024-12-31   2022-12-29  2024-12-30  2193    18.21   731   731   731
test        2025-01-01 -> 2025-12-31   2025-01-01  2025-12-31   2024-12-29  2025-12-30  1095     9.09   365   365   365

## Target distribution by split and horizon (°C)
                             count   mean   std   min  median   max
split      forecast_horizon                                        
train      1                  2919  31.20  6.81  10.3    32.5  45.8
           2                  2918  31.21  6.80  10.3    32.5  45.8
           3                  2917  31.21  6.80  10.3    32.5  45.8
validation 1                   731  30.94  7.01  12.6    32.3  46.5
           2                   731  30.94  7.01  12.6    32.3  46.5
           3                   731  30.94  7.01  12.6    32.3  46.5
test       1                   365  30.71  6.31  14.7    31.8  43.7
           2                   365  30.71  6.31  14.7    31.8  43.7
           3                   365  30.71  6.31  14.7    31.8  43.7

## Missing-value status
              rows  rows with missing predictor  max column missing %
train       8754.0                          0.0                   0.0
validation  2193.0                          0.0                   0.0
test        1095.0                          0.0                   0.0
- Imputer: median, fitted on train only (no-op when nothing is missing). Scaling: none.

## No-training baseline metrics (validation + test)
                                                   n  MAE (°C)  MSE (°C²)  RMSE (°C)     R²  MedAE (°C)  Max error (°C)  Bias (°C)
split      baseline                  horizon                                                                                      
validation Persistence (tmax_d0)     h=1       731.0     1.198      2.739      1.655  0.944       0.900           7.400      0.007
                                     h=2       731.0     1.728      5.212      2.283  0.894       1.300           7.900      0.015
                                     h=3       731.0     2.079      7.310      2.704  0.851       1.700           9.600      0.023
                                     all      2193.0     1.669      5.087      2.255  0.896       1.300           9.600      0.015
           3-day mean (tmax_mean_3d) h=1       731.0     1.488      3.898      1.974  0.921       1.133           7.333      0.015
                                     h=2       731.0     1.880      5.956      2.441  0.879       1.533           8.500      0.023
                                     h=3       731.0     2.145      7.639      2.764  0.844       1.733           9.000      0.029
                                     all      2193.0     1.838      5.831      2.415  0.881       1.433           9.000      0.023
test       Persistence (tmax_d0)     h=1       365.0     1.330      3.479      1.865  0.912       1.000           7.400     -0.012
                                     h=2       365.0     1.843      5.852      2.419  0.852       1.500           8.100     -0.021
                                     h=3       365.0     2.158      7.484      2.736  0.811       1.800           8.900     -0.027
                                     all      1095.0     1.777      5.605      2.368  0.859       1.400           8.900     -0.020
           3-day mean (tmax_mean_3d) h=1       365.0     1.559      4.178      2.044  0.895       1.267           6.567     -0.020
                                     h=2       365.0     1.912      6.077      2.465  0.847       1.600           7.633     -0.027
                                     h=3       365.0     2.103      7.420      2.724  0.813       1.767           8.267     -0.036
                                     all      1095.0     1.858      5.891      2.427  0.851       1.467           8.267     -0.028

## Notes
- Data are model-derived (ERA5 archive) estimates, not station observations.
- The heatwave rule (45 / 47 °C) is a simplified project rule, not the full IMD definition.
- Heatwave classes are rare (support: {'validation': {0: 2163, 1: 30, 2: 0}, 'test': {0: 1095, 1: 0, 2: 0}}) -> accuracy alone is misleading.
- No XGBoost model was imported, trained, tuned or evaluated.
