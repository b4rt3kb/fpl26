# Model evaluation

- `forecast-2025.json`: successive gameweek forecasts for the 2025/26 season.
- `forecast-2026.json`: forecasts for GW1–GW6 of the 2026/27 season.
- `strategy-2025.json`: full-season transfer and chip results.
- `strategy-2025-evidence.zip`: weekly decisions, scores, source code and input checksums.

## Strategy results

Development replay for 2025/26. All runs start with the same squad.
Scores include transfer hits and lineup refinement.

| Strategy | Net points | Hit points |
|---|---:|---:|
| Direct model, no transfers | 1477 | 0 |
| Direct model, transfers without chips | 2080 | 84 |
| Combined model, transfers without chips | 2062 | 56 |
| Combined model, transfers and chips | 1985 | 76 |

Each solver stage has a 10-second limit. The chip policy runs two stages per decision.
Live reports allow 60 seconds for the joint search, plus up to 15 seconds for its baseline.
The combined model scored 77 fewer points with chips in this replay.

## Run

```bash
python evaluate_strategy_models.py --season 2025-26 --output state/evaluation
```

`strategy_backtest.py` simulates transfers, budgets, chips and automatic substitutions across a season.
