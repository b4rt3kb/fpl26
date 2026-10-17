# fpl26 — forecasts and a stateful FPL planner

Python tools to fetch validated FPL data, forecast players, reconstruct a real team, and plan transfers, lineups and chips across several gameweeks. Recommendations are written to files for review; no transfers or chip activations are sent to FPL.

The October 2026 audit found stale current-season archive data, duplicate match records, Assistant Manager records in player training, and information leakage within double gameweeks. **The old README's backtest scores and claimed +288-point advantage are withdrawn.** A corrected internal comparison is in `README_MODEL.md`; a more complete optimizer is not evidence of predictive superiority.

## Run

Python 3.11+ is recommended. Optimization uses HiGHS (`highspy` for the stateful planner, SciPy for the fresh-squad benchmark) and works without the legacy CBC executable.

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python fetch_fpl.py --workers 6
python predict_gw.py --horizon 6 --no-squad
```

The forecast starts at the next official event deadline. Full-player forecasts include zero-point rows for blank gameweeks and are saved to `predictions/all_players_gwN.csv`, with metadata alongside them. All forecast gameweeks use the history available at the initial decision time. Refresh before making decisions.

Current data comes directly from the official FPL `bootstrap-static`, `fixtures`, and `element-summary` endpoints. Prior seasons come from the vaastav archive. The fetcher validates IDs, fields, player totals and coverage of completed fixtures; removes identical historical duplicates and Assistant Manager rows; and publishes a checksummed manifest last. Expired, missing or damaged current snapshots stop prediction rather than falling back silently.

## Import your actual team

```bash
python import_team.py --entry YOUR_ENTRY_ID --through-gw 6 \
  --free-transfers 3 --bank 0.4
python plan_transfers.py --state state/team_YOUR_ENTRY_ID.json \
  --forecasts predictions/all_players_gw7.csv \
  --output state/transfer_plan.json
```

The importer replays public transfers against every historical squad, checks bank and points, and records chip use. Money is stored as integer tenths of a million. Purchases made after GW1 use actual transfer prices. Initial GW1 purchase prices are reconstructed from season-start prices and labelled as inferred; late entrants need explicit purchase prices. Check these against the FPL Transfers screen before acting.

Public picks describe the team after a historical deadline, including automatic substitutions. They do not reveal unpublished changes for the upcoming gameweek. Free transfers and any changed bank must be confirmed by the user. Account files and caches stay under ignored `state/` and `data/` paths; never commit cookies or private team state.

## What the planner handles

- Your existing 15 players, bank, purchase/selling prices, and available free transfers.
- Squad/club limits, legal XI, captain, proposed vice-captain and bench order.
- Multi-gameweek permanent transfers, banked transfers up to the current limit, and four-point hits.
- Wildcard as a permanent rebuild; Free Hit as a temporary squad with reversion.
- Bench Boost and Triple Captain scoring, chip windows, one chip per event, and Free Hit cooldown.
- Retention of banked free transfers when using Wildcard or Free Hit.
- Solver status and an independent legality/budget check before returning a plan.

The solver starts with a legal plan. Joint chip search is seeded by a separately optimized no-chip plan and cannot replace it with a worse result under the same decision objective. Up to 15 seconds are used for that baseline in addition to the requested joint-search limit. It may return a validated feasible plan when time expires; the JSON reports whether optimality was established within tolerance. `--seconds`, `--discount`, `--ft-value`, and `--chip-reserves` expose the planning assumptions. `--no-chips` provides a comparison without using chips.

Exact same-club/position dominance reduces the candidate pool while retaining every owned player. Reduction uses every forecast gameweek, not an arbitrary top-N shortlist, and records its assumptions and witnesses in the plan.

## Limits that matter

The planner uses expected points and holds current prices fixed across the horizon. It does not yet simulate uncertainty, automatic substitutions, captain non-appearance, future price changes or future injury return dates. The vice-captain and bench order are recommendations; their contingency value is not included in the objective. A tiny bench term breaks scoring ties.

An imported squad with a grandfathered club-limit overflow can be inspected, but the planner currently requires a legal initial club count. This does not affect the imported GW6 squad used for the current repair.

To avoid consuming every chip at the end of a short horizon, default reserve values charge 25 expected points for Wildcard, 20 for Free Hit, 18 for Bench Boost and 12 for Triple Captain when the chip remains usable beyond the horizon. These are explicit, configurable **heuristics**, not learned values. Chips expiring inside the horizon have no terminal reserve. Compare sensitivity before relying on chip timing.

Forecasts remain a gradient-boosting baseline with historical scoring-regime differences and limited early-season evidence. Historical feature cutoffs use the first kickoff of each event where archived deadline snapshots are unavailable; this removes within-event results but does not recreate historical injury reports, prices or all information available at the real deadline. The evaluation's fresh-squad benchmark is not a full-season transfer-policy backtest. See `README_MODEL.md`.

## Files

- `fpl_api.py`, `fetch_fpl.py`: resilient reads, caches, validation and snapshot manifest.
- `fpl_features.py`, `fpl_model.py`, `predict_gw.py`: cleaned, time-safe features and forecasts.
- `train_v03.py`: corrected model comparison and explicitly limited fresh-squad benchmark.
- `team_state.py`, `import_team.py`: private, auditable account reconstruction.
- `plan_transfers.py`, `planner_candidates.py`: finite-horizon transfer/chip MILP and exact candidate reduction.
- `track_record.md`: actual account scores, separate from model results.
- `tests/`: data, model, account and FPL-rule regressions.
- `select_team.py`, `train_ml.py`: legacy GW1 experiments, retained for reference.

GitHub Actions runs offline tests, refreshes data and publishes general player forecasts. It does not infer a user's unpublished changes or execute a personal plan.
