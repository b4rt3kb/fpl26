import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fpl_features import FEATS, build_season, clean_rows, pick_squad, season_agg, validate_squad
from fpl_model import PointsModel, player_gameweeks, ranking_metrics
from predict_gw import complete_pool, historical_seasons


def match_rows(future=False):
    rows = []
    for fixture in range(1, 8):
        unknown = future and fixture > 5
        for element, team, opponent in [(1, "A", 2), (2, "B", 1)]:
            rows.append({"element": element, "code": 100+element,
                         "name": team, "position": "MID", "team": team,
                         "opponent_team": opponent, "was_home": element == 1,
                         "round": fixture if fixture <= 5 else 6, "fixture": fixture,
                         "value": 50, "kickoff_time": f"2026-01-{fixture*2:02}T12:00:00Z",
                         "minutes": np.nan if unknown else 90,
                         "total_points": np.nan if unknown else fixture,
                         "team_h_score": np.nan if unknown else fixture,
                         "team_a_score": np.nan if unknown else 0,
                         "expected_goals": np.nan if unknown else .1,
                         "expected_assists": np.nan if unknown else .1,
                         "starts": np.nan if unknown else 1})
    return pd.DataFrame(rows)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.teams = self.folder / "teams.csv"
        pd.DataFrame({"id": [1, 2], "name": ["A", "B"]}).to_csv(self.teams, index=False)

    def build(self, rows, **kwargs):
        path = self.folder / "gw_2025-26.csv"
        rows.to_csv(path, index=False)
        return build_season(path, self.teams, **kwargs)

    def test_same_gw_results_never_change_dgw_features(self):
        original = match_rows()
        changed = original.copy()
        changed.loc[changed.fixture.eq(6), ["total_points", "team_h_score"]] += 100
        a = self.build(original)
        b = self.build(changed)
        fields = ["pts_r3", "pts_r5", "mins_r3", "own_att_r5", "opp_att_r5"]
        pd.testing.assert_frame_equal(a[a.gw.eq(6)][fields], b[b.gw.eq(6)][fields])
        self.assertEqual(a[(a.element == 1) & (a.fixture == 7)].pts_r5.iloc[0], 3)

    def test_future_targets_do_not_advance_rolling_windows(self):
        rows = match_rows(True)
        features = self.build(rows[rows.fixture.le(5)], extra_rows=rows[rows.fixture.gt(5)])
        future = features[features._forecast & features.element.eq(1)]
        self.assertEqual(future.pts_r5.tolist(), [3, 3])
        self.assertEqual(future.own_att_r5.tolist(), [3, 3])

    def test_horizon_freezes_history_before_initial_cutoff(self):
        rows = match_rows()
        features = self.build(rows, as_of="2026-01-11T00:00:00Z")
        self.assertEqual(features[(features.element == 1) & (features.fixture == 7)].pts_r5.iloc[0], 3)

    def test_exact_duplicates_removed_conflicts_rejected(self):
        rows = match_rows()
        cleaned = clean_rows(pd.concat([rows, rows.iloc[[0]]]))
        self.assertEqual(len(cleaned), len(rows))
        duplicate = rows.iloc[[0]].copy()
        duplicate["total_points"] = 99
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            clean_rows(pd.concat([rows, duplicate]))

    def test_managers_are_excluded_before_training(self):
        rows = match_rows()
        manager = rows.iloc[[0]].copy()
        manager["position"], manager["element"], manager["total_points"] = "AM", 99, 19
        features = self.build(pd.concat([rows, manager]))
        self.assertNotIn(99, features.element.tolist())

    def test_stable_code_join_survives_name_and_element_change(self):
        previous = match_rows().iloc[:1].copy()
        path = self.folder / "previous.csv"
        previous.to_csv(path, index=False)
        prior = season_agg(path)
        current = previous.copy()
        current["name"], current["element"] = "New display name", 999
        features = self.build(current, prev_agg=prior)
        self.assertEqual(features.prev_pts_total.iloc[0], 1)

    def test_historical_mapping_supplies_stable_code(self):
        rows = match_rows().drop(columns="code")
        mapping = self.folder / "players.csv"
        pd.DataFrame({"id": [1, 2], "code": [501, 502]}).to_csv(mapping, index=False)
        self.assertEqual(clean_rows(rows, mapping).key.iloc[0], "code:501")

    def test_unknown_aggregate_statistics_stay_unknown(self):
        rows = match_rows().drop(columns=["expected_goals", "expected_assists", "starts"])
        path = self.folder / "previous.csv"
        rows.to_csv(path, index=False)
        self.assertTrue(season_agg(path).prev_xg90.isna().all())

    def test_deadline_excludes_late_previous_round_match(self):
        rows = match_rows()
        # A prior-round match starts only one hour before the target deadline.
        rows.loc[rows.fixture.eq(5), "kickoff_time"] = "2026-01-12T10:00:00Z"
        features = self.build(rows, deadlines={6: "2026-01-12T11:00:00Z"})
        self.assertEqual(features[(features.element == 1) & (features.fixture == 7)].pts_r5.iloc[0], 2.5)


class ModelTests(unittest.TestCase):
    def test_only_manifest_covered_historical_seasons_are_consumed(self):
        seasons = ["2024-25", "2025-26"]
        manifest = {"historical_seasons": seasons,
                    "files": {f"{prefix}_{season}.csv": {} for season in seasons for prefix in ("gw", "teams", "players")}}
        self.assertEqual(historical_seasons(manifest, 2026), seasons)
        manifest["historical_seasons"] = ["2023-24", "2025-26"]
        with self.assertRaisesRegex(ValueError, "contiguous"):
            historical_seasons(manifest, 2026)
        manifest["historical_seasons"] = seasons
        del manifest["files"]["players_2024-25.csv"]
        with self.assertRaisesRegex(ValueError, "not covered"):
            historical_seasons(manifest, 2026)

    def test_two_stage_conditions_both_parts_on_any_appearance(self):
        rows = pd.DataFrame(0.0, index=range(3), columns=FEATS)
        rows["minutes"], rows["total_points"] = [0, 30, 90], [0, 1, 8]
        with patch("fpl_model.hgb") as reg, patch("fpl_model.HistGradientBoostingClassifier") as clf:
            PointsModel("two").fit(rows)
            self.assertEqual(clf.return_value.fit.call_args.args[1].tolist(), [False, True, True])
            self.assertEqual(reg.return_value.fit.call_args.args[1].tolist(), [1, 8])

    def test_rankings_sum_player_double_fixtures(self):
        rows = pd.DataFrame({"gw": [1, 1, 1], "key": ["A", "A", "B"],
                             "p": [4, 5, 8], "total_points": [6, 7, 10]})
        grouped = player_gameweeks(rows, "p").set_index("key")
        self.assertEqual(grouped.loc["A", "actual"], 13)
        self.assertEqual(grouped.loc["A", "pred"], 9)

    def test_nonfinite_model_output_cannot_be_reported_as_metrics(self):
        rows = pd.DataFrame({"gw": [1], "key": ["A"], "p": [np.inf], "total_points": [2]})
        with self.assertRaisesRegex(ValueError, "finite predictions"):
            ranking_metrics(rows, "p")

    def test_blank_gw_present_for_every_player(self):
        players = pd.DataFrame({"id": [1, 2], "code": [101, 102], "web_name": ["A", "B"],
                                "team": [1, 2], "element_type": [3, 3], "now_cost": [50, 50],
                                "status": ["a", "d"], "chance_of_playing_next_round": [100, 50],
                                "can_select": [True, True]})
        teams = pd.DataFrame({"id": [1, 2], "short_name": ["AAA", "BBB"]})
        rows = pd.DataFrame({"element": [1, 1], "gw": [7, 7], "fixture": [10, 11], "pred": [3, 4]})
        pool = complete_pool(players, teams, rows, [7, 8], pd.Timestamp("2026-01-01", tz="UTC"))
        self.assertEqual(len(pool), 4)
        self.assertEqual(pool[pool.gw.eq(8)].pred.sum(), 0)
        self.assertEqual(pool[(pool.element == 1) & (pool.gw == 7)].pred.iloc[0], 7)


class OptimizerTests(unittest.TestCase):
    def candidates(self):
        positions = ["GK"]*6 + ["DEF"]*12 + ["MID"]*12 + ["FWD"]*9
        return pd.DataFrame({"element": range(len(positions)), "pos": positions,
                             "price": [5 + i % 5 for i in range(len(positions))],
                             "team_key": [i % 10 for i in range(len(positions))],
                             "pred": [2 + (i*7) % 11 for i in range(len(positions))]})

    def test_solver_returns_a_legal_squad(self):
        squad = pick_squad(self.candidates())
        validate_squad(squad)
        self.assertLessEqual(squad.price.sum(), 100)
        self.assertEqual(squad[squad.is_cap.eq(1)].pred.iloc[0], squad[squad.in_xi.eq(1)].pred.max())

    def test_infeasible_squad_fails_instead_of_returning_illegal_selection(self):
        candidates = self.candidates()
        candidates["team_key"] = 1
        with self.assertRaisesRegex(RuntimeError, "did not reach an optimum"):
            pick_squad(candidates)


if __name__ == "__main__":
    unittest.main()
