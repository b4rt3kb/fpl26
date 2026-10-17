"""Integration invariants for minutes -> points -> dated availability -> GW pool."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from availability_model import apply_fixture_availability
from evaluate_strategy_models import origin_forecasts
from fpl_features import build_season, FEATS, MINUTES_FEATURES, EXTENDED_FEATURES
from fpl_model import PointsModel
from predict_gw import complete_pool


def fitted_rows():
    fields = list(dict.fromkeys(FEATS+MINUTES_FEATURES+EXTENDED_FEATURES))
    rows = pd.DataFrame(0., index=range(180), columns=fields)
    rows["minutes"] = np.tile([0, 30, 90], 60)
    rows["total_points"] = np.tile([0, 1, 6], 60)
    rows["starts"] = rows.minutes.ge(60).astype(int)
    rows["feature_cutoff"] = pd.Timestamp("2025-08-01T10:00:00Z")
    rows["kickoff_time"] = rows.feature_cutoff+pd.Timedelta(hours=3)
    return rows


class MixtureIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = fitted_rows()
        cls.model = PointsModel("ensemble").fit(cls.training)

    def test_known_band_returns_and_future_outcomes_do_not_route_the_mixture(self):
        future = self.training.iloc[:3].copy()
        future[["minutes", "total_points", "starts"]] = [[0, 100, 0], [30, -50, 1], [90, 500, 1]]
        output = self.model.predict_components(future)
        # Identical features, different unavailable future labels: same forecast.
        np.testing.assert_allclose(output.pred, output.pred.iloc[0])
        np.testing.assert_allclose(output.points_given_p_1_59, 1, atol=1e-6)
        np.testing.assert_allclose(output.points_given_p_60_plus, 6, atol=1e-6)
        expected = output.p_1_59 + 6*output.p_60_plus
        np.testing.assert_allclose(output.component_pred, expected, atol=1e-6)
        np.testing.assert_allclose(output.pred, .5*output.direct_pred+.5*output.component_pred, atol=1e-12)
        future[["minutes", "starts", "total_points"]] = np.nan
        np.testing.assert_allclose(self.model.predict(future), output.pred)

    def test_frozen_origin_pipeline_survives_mutated_future_outcomes_and_prices(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            teams = folder/"teams_2025-26.csv"
            pd.DataFrame({"id": [1, 2], "name": ["A", "B"]}).to_csv(teams, index=False)
            pd.DataFrame({"id": [1, 2], "code": [101, 102]}).to_csv(folder/"players_2025-26.csv", index=False)
            records = []
            for gw in range(1, 9):
                for player, club, opponent in [(1, "A", 2), (2, "B", 1)]:
                    records.append({"element": player, "code": 100+player, "name": club,
                        "position": "MID", "team": club, "opponent_team": opponent,
                        "was_home": player == 1, "round": gw, "fixture": gw, "value": 50,
                        "kickoff_time": pd.Timestamp("2025-08-01", tz="UTC")+pd.Timedelta(days=gw*7),
                        "minutes": 90, "starts": 1, "total_points": gw,
                        "expected_goals": .2, "expected_assists": .1,
                        "defensive_contribution": 12, "team_h_score": 2, "team_a_score": 1})
            raw = pd.DataFrame(records); path=folder/"gw_2025-26.csv"
            raw.to_csv(path, index=False)
            current = build_season(path, teams)
            future, pool, _ = origin_forecasts(folder, "2025-26", None, current, 3, 6)
            raw.loc[raw["round"].ge(3), ["minutes", "starts", "total_points", "expected_goals", "expected_assists", "defensive_contribution", "team_h_score", "team_a_score"]] = 999
            raw.loc[raw["round"].gt(3), "value"] = 1000
            raw.to_csv(path, index=False)
            changed = build_season(path, teams)
            alternative, second_pool, _ = origin_forecasts(folder, "2025-26", None, changed, 3, 6)
            fields = list(dict.fromkeys(FEATS+MINUTES_FEATURES+EXTENDED_FEATURES))
            pd.testing.assert_frame_equal(future[fields], alternative[fields])
            pd.testing.assert_series_equal(pool.price, second_pool.price)
            np.testing.assert_allclose(self.model.predict(future), self.model.predict(alternative), atol=1e-12)
            self.assertGreater(future.forecast_lead_days.max(), 28)


class PoolAndAvailabilityIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.players = pd.DataFrame({"id": [1, 2], "code": [101, 102], "web_name": ["A", "B"],
            "team": [1, 2], "element_type": [3, 3], "now_cost": [50, 50], "status": ["d", "a"],
            "chance_of_playing_next_round": [50, 100], "news": ["", ""], "can_select": [True, True]})
        self.teams = pd.DataFrame({"id": [1, 2], "short_name": ["AAA", "BBB"]})
        self.cutoff = pd.Timestamp("2026-10-17T08:00:00Z")
        self.fixtures = pd.DataFrame({"element": [1, 1], "gw": [7, 7], "fixture": [10, 11],
            "kickoff_time": ["2026-10-18T12:00:00Z", "2026-10-21T12:00:00Z"],
            "pred": [4., 4.], "p_no_appearance": [.2, .2], "p_1_59": [.2, .2], "p_60_plus": [.6, .6],
            "expected_minutes": [60., 60.], "p_start": [.6, .6], "lead_out_of_support": [False, False]})

    def test_dgw_points_minutes_and_no_show_use_adjusted_fixtures_exactly_once(self):
        adjusted = apply_fixture_availability(self.fixtures, self.players, as_of=self.cutoff, next_event=7)
        pool = complete_pool(self.players, self.teams, adjusted, [7, 8], self.cutoff)
        dgw = pool[(pool.element == 1)&(pool.gw == 7)].iloc[0]
        # News caps Pplay from .8 to .5; it is not multiplied again in GW aggregation.
        self.assertAlmostEqual(dgw.pred, 5)
        self.assertAlmostEqual(dgw.expected_minutes, 75)
        self.assertAlmostEqual(dgw.p_no_appearance, .25)
        self.assertAlmostEqual(dgw.expected_starts, .75)
        self.assertAlmostEqual(dgw.p_60_at_least_one_fixture, 1-.625**2)
        self.assertEqual(dgw.fixture_count, 2)
        blanks = pool[pool.fixture_count.eq(0)]
        self.assertTrue(blanks.pred.eq(0).all())
        self.assertTrue(blanks.expected_minutes.eq(0).all())
        self.assertTrue(blanks.p_no_appearance.eq(1).all())

    def test_unknown_start_probability_is_not_invented_as_zero_for_scheduled_matches(self):
        rows = self.fixtures.copy(); rows["p_start"] = np.nan
        adjusted = apply_fixture_availability(rows, self.players, as_of=self.cutoff, next_event=7)
        pool = complete_pool(self.players, self.teams, adjusted, [7, 8], self.cutoff)
        self.assertTrue(pool.loc[pool.fixture_count.gt(0), "expected_starts"].isna().all())
        self.assertTrue(pool.loc[pool.fixture_count.eq(0), "expected_starts"].eq(0).all())


if __name__ == "__main__":
    unittest.main()
