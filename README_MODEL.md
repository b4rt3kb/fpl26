# Forecast and evaluation methodology

## Audit and status

The original v0.3 evaluation cannot substantiate the previously published 2188-point result or +288-point advantage. Its inputs contained identical player-fixture duplicates and Assistant Manager records, and rolling features for a second match in a double gameweek could use the result of the first match. The current-season archive also stopped at GW1 while automation continued to publish later forecasts.

Historical results from the old implementation are not acceptance criteria for this repaired version. Recompute metrics after cleaning; establish a separate untouched test period before claiming an improvement.

## Data boundaries

Current-season players, prices, availability, events, fixtures and player match histories come from the official FPL API. Completed prior-season archives supply training examples and player ID/code maps. Stable player codes link seasons; names are display fields where mappings are available. Exact duplicates are removed before aggregates or rolling calculations, conflicting duplicates fail validation, and only GK/DEF/MID/FWD records are retained.

Current predictions require a fresh, intact, checksummed data manifest. Finished-match histories must cover finished fixtures and agree with official player totals outside a live match. Future fixtures never update observed player or team form.

## Features and models

The baseline uses previous-season aggregates, shifted recent points/minutes/xG/xA/starts, team goal form, price, position, home/away and gameweek. The incompatible static team-strength fields are excluded. Model variants include gradient boosting and a coherent two-stage appearance probability times expected points conditional on an appearance.

Every fixture in an event sees one frozen historical state. Current forecasts use the official event deadline together with the current data boundary. For past seasons without archived deadlines, the earliest kickoff in the event is a documented fallback. No match from the same event contributes to that event's form features.

For a forecast horizon, future event rows all use the initial observed history. Fixture-specific predictions are summed to player/gameweek totals; blank gameweeks are explicit zeros. Current availability remains an imperfect estimate of future minutes, especially over multiple events.

## Evaluation

Walk-forward model fitting trains on earlier events. Ranking metrics are computed after aggregating to player/gameweek. MAE, rank correlation and top-10 accuracy measure different properties and should be inspected together.

The fresh-squad benchmark still constructs a new squad every gameweek. It is an internal forecast comparison only: it does not reproduce transfer continuity, chip decisions, selling prices or automatic substitutions. A realistic policy benchmark must replay the same state transitions as the new planner using archived information available before every deadline. That benchmark has not yet established predictive or strategic superiority.

Changes to FPL scoring between seasons are another source of distribution shift. Simply combining historical total points does not reconstruct current-rule targets; full rescoring requires adequate raw statistics, including bonus dependencies. Missing historical information must not be invented.

Recomputed on 17 October 2026 against the cleaned 2025/26 season:

| Model | MAE per player/GW | Spearman | Precision@10 | Fresh-squad points |
|---|---:|---:|---:|---:|
| Rolling naive | 1.061 | 0.714 | 0.089 | 1841 |
| Ridge | 1.066 | 0.695 | 0.132 | — |
| Gradient boosting | 1.011 | 0.709 | 0.121 | 2111 |
| Two-stage appearance model | 1.008 | 0.710 | 0.113 | 2068 |

These are internal comparisons under the limitations above. The same season has informed development and model choice, so it is not a new untouched final test. The 2111-point figure is not a simulated score for a real account following legal season-long transfers. Run `python train_v03.py` after fetching the default three historical seasons to reproduce the table.

## Planner assumptions

The separate planner optimizes discounted expected points over the supplied horizon. It tracks permanent ownership and original acquisition prices so that selling and rebuying a player cannot repeatedly realize the original profit. Free Hit squads are temporary and priced relative to the retained squad's selling budget. Transfer-bank transitions, hits and chip effects are constrained explicitly.

Terminal free-transfer and chip values are configurable assumptions. They should be calibrated on a realistic historical policy simulation and stress-tested against forecast changes. The current planner does not model stochastic automatic substitutions, vice-captain substitution, price uncertainty or return dates. Its output is a reviewable plan, not an instruction that FPL has executed.
