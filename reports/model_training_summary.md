# Model Training Summary - Delhi 3-day Tmax (XGBoost Regressor)
Generated: 2026-11-22 19:22

## Setup
- One pooled XGBoost regressor, direct multi-horizon (horizon 1, 2, 3 via the forecast_horizon feature).
- Target: target_tmax (°C). Train 8754 rows (2015-2022 targets), validation 2193 (2023-2024), test 1095 (2025).

## Tuning (validation only)
- Initial model validation MAE: 1.580 °C | best of 40 random configurations: 1.555 °C
- Parameters used: {'max_depth': 3, 'learning_rate': 0.08, 'min_child_weight': 5, 'subsample': 0.7, 'colsample_bytree': 0.5, 'reg_lambda': 10.0, 'reg_alpha': 0.1, 'gamma': 0.5}

## Feature selection (train + validation only)
                                feature set  n_features  best_iteration  val_MAE  val_RMSE
                            all 21 features          21              79   1.5554    2.0437
       positive permutation importance (19)          19              83   1.5647    2.0546
top 12 by permutation importance (+horizon)          13              82   1.5578    2.0508
- Chosen: top 12 by permutation importance (+horizon) (13 features). Dropped: ['humidity_d1', 'humidity_d2', 'wind_d0', 'wind_d1', 'wind_d2', 'solar_d1', 'month_sin', 'tmax_trend_3d']

## Final model
- Trained on train + validation (10947 rows), 103 trees, saved to models/xgb_tmax_regressor.json.

## Validation metrics (train-only model)
          n  MAE (°C)  MSE (°C²)  RMSE (°C)     R²  MedAE (°C)  Max error (°C)  Bias (°C)
h=1   731.0     1.357      3.168      1.780  0.935       1.059           7.802      0.078
h=2   731.0     1.575      4.202      2.050  0.914       1.237           7.579      0.076
h=3   731.0     1.742      5.247      2.291  0.893       1.334           8.150      0.074
all  2193.0     1.558      4.206      2.051  0.914       1.202           8.150      0.076

## Test metrics (final model, evaluated once)
          n  MAE (°C)  MSE (°C²)  RMSE (°C)     R²  MedAE (°C)  Max error (°C)  Bias (°C)
h=1   365.0     1.272      3.059      1.749  0.923       0.915           6.715      0.087
h=2   365.0     1.566      4.269      2.066  0.892       1.227           7.121      0.072
h=3   365.0     1.733      5.113      2.261  0.871       1.366           7.200      0.057
all  1095.0     1.524      4.147      2.036  0.895       1.143           7.200      0.072

## Persistence baseline on test
          n  MAE (°C)  MSE (°C²)  RMSE (°C)     R²  MedAE (°C)  Max error (°C)  Bias (°C)
h=1   365.0     1.330      3.479      1.865  0.912         1.0             7.4     -0.012
h=2   365.0     1.843      5.852      2.419  0.852         1.5             8.1     -0.021
h=3   365.0     2.158      7.484      2.736  0.811         1.8             8.9     -0.027
all  1095.0     1.777      5.605      2.368  0.859         1.4             8.9     -0.020

## Heatwave rule + hot-day metrics
     split  accuracy  macro precision  macro recall  macro F1  weighted F1  total support
validation    0.9863           0.4932           0.5    0.4966       0.9795           2193
      test    1.0000           1.0000           1.0    1.0000       1.0000           1095

     split  cut (°C)  accuracy  precision  recall    f1  support (actual hot days)  TP  FP  FN   TN
validation        40     0.961      0.820   0.714 0.763                        192 137  30  55 1971
validation        42     0.967      0.968   0.294 0.451                        102  30   1  72 2090
      test        40     0.952      0.738   0.571 0.644                         84  48  17  36  994
      test        42     0.970      0.714   0.139 0.233                         36   5   2  31 1057

## Caveats
- The 45 / 47 °C classes are very rare (no test examples) so accuracy there is trivial; use MAE/RMSE as main metrics.
- Archive (ERA5) training data vs operational forecast live data may differ - validate before relying on live output.
- The heatwave rule is simplified, not the full IMD definition.
