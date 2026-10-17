# Forecast outputs

`gw2.csv` through `gw7.csv` and the two GW1 files are preserved historical outputs of the old implementation. They are not current recommendations: later forecasts used stale current-season data.

The repaired pipeline writes `all_players_gwN.csv` and an accompanying JSON metadata file after validating official FPL data. The CSV covers every player and gameweek in the requested horizon. It is an input to `plan_transfers.py`, not a personal transfer instruction.

Refresh data and predictions before planning; the full procedure and model limitations are in the root README.
