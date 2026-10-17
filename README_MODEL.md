# Model and planner

The model combines two point forecasts with equal weights: direct gradient boosting and a model based on minutes.
Inputs include form, starts, expected goals (xG), expected assists (xA), defensive actions, price, opponent strength and fixtures.

The minutes model estimates three outcomes: no appearance, 1–59 minutes and at least 60 minutes.
Calibration uses later matches than training. Each player has a separate minutes forecast.
FPL status and news adjust availability for each match date, including expected injury returns.

Each forecast covers up to six gameweeks with data available at the calculation time.
Double gameweeks sum both matches. Blank gameweeks contribute zero points.

## Planner

HiGHS selects transfers, the starting lineup, captain and chips across the forecast period.
It checks the budget, sale prices, club limits, formation, free transfers and transfer hits.
Wildcard changes the permanent squad. Free Hit restores the previous squad after the gameweek.
Bench Boost includes bench points. Triple Captain triples captain points.
A separate step improves the lineup, bench order and C/VC choices with expected automatic substitutions.

The planner holds current prices constant across the forecast period.
If a chip remains available beyond this period, its use incurs a reserve cost: WC 25, FH 20, BB 18, TC 12 points.
The reserve is zero when the chip expires within the forecast period.

## Forecast results

Root mean square error (RMSE) of points across all players. Lower values indicate smaller errors.

| Period | Direct model | Combined model |
|---|---:|---:|
| 2025/26, 38 gameweeks | 1.949 | 1.930 |
| 2026/27, GW1–GW6 | 2.203 | 2.194 |

The evaluation reconstructs successive decisions from historical data. The 2025/26 season also informed model development.
See [`evaluation/`](evaluation/) for parameters, other metrics and minutes results.
