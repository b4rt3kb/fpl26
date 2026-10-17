# Model evaluation

- `forecast-2025.json`: successive gameweek forecasts for the 2025/26 season.
- `forecast-2026.json`: forecasts for GW1–GW6 of the 2026/27 season.

The files contain point errors, minutes errors, evaluation parameters and data checksums.
Each gameweek uses history available before its start.

```bash
python evaluate_strategy_models.py --season 2025-26 --output state/evaluation
```

`strategy_backtest.py` simulates transfers, budgets, chips and automatic substitutions across a season.
