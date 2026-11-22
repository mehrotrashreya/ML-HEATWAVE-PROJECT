# Feature Engineering Report

| # | Feature | Definition |
|---|---|---|
| 1 | `tmax_d0` | tmax on day T |
| 2 | `tmax_d1` | tmax on day T-1 |
| 3 | `tmax_d2` | tmax on day T-2 |
| 4 | `humidity_d0` | humidity on day T |
| 5 | `humidity_d1` | humidity on day T-1 |
| 6 | `humidity_d2` | humidity on day T-2 |
| 7 | `wind_d0` | wind on day T |
| 8 | `wind_d1` | wind on day T-1 |
| 9 | `wind_d2` | wind on day T-2 |
| 10 | `solar_d0` | solar on day T |
| 11 | `solar_d1` | solar on day T-1 |
| 12 | `solar_d2` | solar on day T-2 |
| 13 | `month_sin` | sin(2*pi*(month-1)/12) of TARGET date |
| 14 | `month_cos` | cos(2*pi*(month-1)/12) of TARGET date |
| 15 | `day_sin` | sin(2*pi*(day_of_year-1)/days_in_year) of TARGET date (365/366, leap-aware) |
| 16 | `day_cos` | cos(2*pi*(day_of_year-1)/days_in_year) of TARGET date (365/366, leap-aware) |
| 17 | `tmax_mean_3d` | mean(tmax_d0, tmax_d1, tmax_d2) |
| 18 | `tmax_trend_3d` | tmax_d0 - tmax_d2 |
| 19 | `humidity_mean_3d` | mean of the 3 humidity values |
| 20 | `solar_sum_3d` | sum of the 3 solar totals |
| 21 | `forecast_horizon` | 1, 2 or 3 = days ahead of the forecast origin |