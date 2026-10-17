# fpl26

An FPL model that selects the lineup, captain, bench, transfers and chips for each gameweek.
Open the next gameweek folder in [`gameweeks/`](gameweeks/) to read the decisions.

## How it works

1. Fetch player data, fixtures and team history from the FPL API.
2. Reconstruct the squad, bank, sale prices, free transfers (FT) and used chips.
3. Predict minutes, availability and points for up to six gameweeks.
4. Plan transfers, lineups and chips. Include double gameweeks, banked transfers and transfer hits.
5. Save the report as Markdown, HTML, JSON and CSV.

GitHub Actions checks the deadline every hour.
It generates a report within 24 hours of the deadline and refreshes it within the final two hours.
The schedule runs Python on GitHub. It requires no AI API or running local computer.

The last published squad is the starting point. Confirm transfers and lineup changes in FPL.

## Run

Use Python 3.11. Set the repository variable `FPL_ENTRY_ID` for the scheduled workflow.

```bash
pip install -r requirements.txt
python fetch_fpl.py --workers 6
python predict_gw.py --horizon 6 --no-squad
python weekly_decision.py --entry 8863206 --publish-dir gameweeks
```

To refresh manually, select **Actions → weekly-gameweek-decisions → Run workflow** on GitHub.

## Files

| Folder | Contents |
|---|---|
| [`gameweeks/`](gameweeks/) | Decisions for each gameweek |
| [`predictions/`](predictions/) | Player forecasts and archived inputs |
| [`evaluation/`](evaluation/) | Model evaluation results |
| [`tests/`](tests/) | Tests for data, forecasts and FPL rules |
| [`data/`](data/) | Description of downloaded data |

Read [`README_MODEL.md`](README_MODEL.md) for the model and planner details.
