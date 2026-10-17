from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from fpl_features import (build_season, MINUTES_FEATURES, MINUTES_HISTORY,
                          MINUTES_SCHEDULE, EXTENDED_FEATURES, FEATS)
from minutes_model import MinutesModel, PROBABILITY_COLUMNS, _ProbabilityCalibrator


def training_rows(events=15, players=45):
    rng = np.random.default_rng(732)
    rows = []
    for event in range(events):
        cutoff = pd.Timestamp("2024-08-01", tz="UTC") + pd.Timedelta(days=7*event)
        for player in range(players):
            role = player % 3
            state = rng.choice(3, p=[.7, .2, .1] if role == 0 else ([.2, .55, .25] if role == 1 else [.1, .15, .75]))
            minutes = [0, 25, 85][state]
            row = {name: np.nan for name in MINUTES_FEATURES}
            row.update({"feature_cutoff": cutoff, "kickoff_time": cutoff+pd.Timedelta(hours=3),
                        "minutes": minutes, "starts": int(minutes >= 60),
                        "mins_r5": [5, 25, 75][role], "mins_r3": [5, 25, 75][role],
                        "pos_code": player % 4, "price": 4+role, "forecast_lead_days": .125,
                        "gw": event+1, "fixture": event, "element": player, "finished": True})
            rows.append(row)
    return pd.DataFrame(rows)


class MinutesModelTests(unittest.TestCase):
    def model(self, **kwargs):
        return MinutesModel(min_train_rows=200, min_calibration_rows=100,
                            min_class_count=10, max_iter=12, **kwargs)

    def test_probabilities_minutes_and_start_are_coherent(self):
        rows = training_rows()
        model = self.model().fit(rows)
        pred = model.predict(rows.iloc[:60])
        self.assertTrue(model.diagnostics_["calibration_applied"])
        np.testing.assert_allclose(pred[PROBABILITY_COLUMNS].sum(axis=1), 1, atol=1e-12)
        self.assertTrue(pred[PROBABILITY_COLUMNS].ge(0).all().all())
        self.assertTrue(pred.expected_minutes.between(0, 90).all())
        self.assertTrue(pred.p_start.le(1-pred.p_no_appearance+1e-12).all())
        self.assertTrue(pred.expected_minutes.ge(pred.p_1_59+60*pred.p_60_plus-1e-12).all())
        self.assertTrue(pred.expected_minutes.le(59*pred.p_1_59+90*pred.p_60_plus+1e-12).all())

    def test_calibration_uses_later_complete_blocks_and_purges_postponed_targets(self):
        rows = training_rows()
        boundary = sorted(rows.feature_cutoff.unique())[-4]
        rows.loc[0, "kickoff_time"] = boundary + pd.Timedelta(days=2)
        model = self.model().fit(rows)
        self.assertNotIn(0, model.base_indices_)
        self.assertNotIn(0, model.calibration_indices_)
        self.assertEqual(model.diagnostics_["purged_overlap_rows"], 1)
        self.assertTrue((rows.loc[model.base_indices_, "kickoff_time"]+pd.Timedelta(hours=2) <= boundary).all())
        self.assertTrue(rows.loc[model.calibration_indices_, "feature_cutoff"].ge(boundary).all())
        self.assertFalse(set(model.base_indices_) & set(model.calibration_indices_))

    def test_insufficient_data_has_explicit_uncalibrated_fallback(self):
        rows = training_rows(events=1, players=20)
        rows["minutes"], rows["starts"] = 0, 0
        model = self.model().fit(rows)
        self.assertFalse(model.diagnostics_["calibration_applied"])
        self.assertEqual(model.diagnostics_["base_rows"], len(rows))
        self.assertTrue(np.isfinite(model.predict(rows)[PROBABILITY_COLUMNS].to_numpy()).all())
        self.assertTrue(model.predict(rows).p_start.isna().all())

    def test_start_is_unknown_when_target_history_is_missing(self):
        rows = training_rows(events=2).drop(columns="starts")
        model = self.model().fit(rows)
        self.assertFalse(model.diagnostics_["start_supported"])
        self.assertTrue(model.predict(rows).p_start.isna().all())

    def test_extrapolated_lead_is_flagged(self):
        rows = training_rows(events=2)
        model = self.model().fit(rows)
        targets = rows.iloc[:2].copy()
        targets.forecast_lead_days = [.125, 35]
        self.assertEqual(model.predict(targets).lead_out_of_support.tolist(), [False, True])

    def test_calibration_corrects_overconfident_probabilities(self):
        labels = np.tile(np.r_[np.zeros(6, dtype=int), np.ones(3, dtype=int), [2]], 100)
        raw = np.tile([.97, .02, .01], (len(labels), 1))
        calibrated = _ProbabilityCalibrator().fit(raw, labels).predict(raw)
        truth = np.eye(3)[labels]
        self.assertLess(np.square(calibrated-truth).sum(axis=1).mean(),
                        np.square(raw-truth).sum(axis=1).mean()-.1)

    def test_unknown_historic_finished_allowed_but_explicit_unfinished_rejected(self):
        rows = training_rows(events=2)
        rows["finished"] = pd.Series([None]*len(rows), dtype=object)
        self.model().fit(rows)
        rows.loc[0, "finished"] = False
        with self.assertRaisesRegex(ValueError, "Unfinished"):
            self.model().fit(rows)

    def test_invalid_targets_and_temporal_rows_fail(self):
        rows = training_rows(events=2)
        for bad in [-1, 91, np.nan]:
            invalid = rows.copy(); invalid.loc[0, "minutes"] = bad
            with self.assertRaisesRegex(ValueError, "Minutes targets"):
                self.model().fit(invalid)
        with self.assertRaisesRegex(ValueError, "Chronological"):
            self.model().fit(rows.drop(columns="feature_cutoff"))
        rows.loc[0, "feature_cutoff"] = rows.loc[0, "kickoff_time"]+pd.Timedelta(minutes=1)
        with self.assertRaisesRegex(ValueError, "chronology"):
            self.model().fit(rows)


class MinutesFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.path = self.folder / "gw_2025-26.csv"
        self.teams = self.folder / "teams.csv"
        pd.DataFrame({"id": [1, 2], "name": ["A", "B"]}).to_csv(self.teams, index=False)

    def rows(self):
        rows = []
        for fixture, day in enumerate([1, 8, 15, 22, 29, 32, 36], 1):
            for player, team, other in [(1, "A", 2), (2, "B", 1)]:
                rows.append({"element": player, "code": 100+player, "name": team,
                    "position": "MID", "team": team, "opponent_team": other,
                    "was_home": player == 1, "round": min(fixture, 5), "fixture": fixture,
                    "value": 50, "kickoff_time": pd.Timestamp("2025-08-01", tz="UTC")+pd.Timedelta(days=day),
                    "minutes": [0, 30, 90, 60, 85, 20, 90][fixture-1], "total_points": fixture,
                    "team_h_score": 1, "team_a_score": 0, "expected_goals": .2,
                    "expected_assists": .1, "starts": int(fixture > 2), "defensive_contribution": 12})
        return pd.DataFrame(rows)

    def build(self, rows, **kwargs):
        rows.to_csv(self.path, index=False)
        return build_season(self.path, self.teams, **kwargs)

    def test_future_results_do_not_change_minutes_or_rate_inputs(self):
        rows = self.rows()
        cutoff = rows.loc[rows.fixture.eq(5), "kickoff_time"].min()-pd.Timedelta(hours=1)
        original = self.build(rows, as_of=cutoff)
        changed = rows.copy()
        changed.loc[changed.fixture.ge(5), ["minutes", "starts", "expected_goals", "expected_assists", "defensive_contribution"]] = 999
        modified = self.build(changed, as_of=cutoff)
        fields = list(dict.fromkeys(MINUTES_FEATURES+EXTENDED_FEATURES))
        pd.testing.assert_frame_equal(original.loc[original.fixture.ge(5), fields],
                                      modified.loc[modified.fixture.ge(5), fields])
        self.assertEqual(original.loc[(original.fixture == 7)&(original.element == 1), "mins_last"].iloc[0], 60)

    def test_schedule_captures_double_gameweek_without_observing_results(self):
        features = self.build(self.rows())
        dgw = features[(features.gw == 5)&(features.element == 1)]
        self.assertEqual(dgw.fixtures_in_gw.tolist(), [3, 3, 3])
        self.assertEqual(dgw.fixture_order_in_gw.tolist(), [1, 2, 3])
        self.assertEqual(dgw.team_rest_days.tolist(), [7, 3, 4])
        self.assertEqual(dgw.team_matches_prev7.tolist(), [1, 1, 2])

    def test_full_calendar_counts_match_beyond_exported_horizon(self):
        rows = self.rows()
        calendar = rows[rows.element.eq(1)][["fixture", "round", "kickoff_time"]].rename(columns={"fixture": "id", "round": "event"})
        calendar["team_h"], calendar["team_a"] = 1, 2
        short = rows[rows.fixture.le(6)]
        features = self.build(short, fixture_calendar=calendar)
        self.assertEqual(features.loc[(features.fixture == 6)&(features.element == 1), "team_matches_next7"].iloc[0], 1)

    def test_missing_rates_stay_unknown_and_baseline_features_are_unchanged(self):
        original = self.rows()
        missing = original.drop(columns=["defensive_contribution", "expected_goals", "expected_assists"])
        features = self.build(missing)
        self.assertTrue(features[["xg90_r5", "xa90_r5", "defcon_r5", "defcon_threshold_share_r5"]].isna().all().all())
        self.assertFalse(set(MINUTES_HISTORY+MINUTES_SCHEDULE+EXTENDED_FEATURES) & set(FEATS))

    def test_per90_is_ratio_of_totals_not_average_of_match_rates(self):
        rows = self.rows()
        features = self.build(rows)
        target = features[(features.fixture == 5)&(features.element == 1)].iloc[0]
        self.assertAlmostEqual(target.xg90_r5, 90*.8/180)
        self.assertAlmostEqual(target.defcon_threshold_share_r5, 1)


if __name__ == "__main__":
    unittest.main()
