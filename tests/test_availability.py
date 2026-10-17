"""Offline checks for dated availability and non-duplicated appearance risk."""
import unittest

import numpy as np
import pandas as pd

from availability_model import (apply_fixture_availability, availability_for_fixture,
                                parse_return_news, validate_overrides)


AS_OF = "2026-10-17T00:00:00Z"


def player(status="a", **fields):
    return dict(id=1, status=status, news=fields.pop("news", ""),
                chance_of_playing_next_round=fields.pop("chance", None), **fields)


def probability(who, kickoff="2026-10-18T14:00:00Z", gw=7, **kwargs):
    return availability_for_fixture(who, kickoff_time=kickoff, gw=gw, as_of=AS_OF,
                                    next_event=7, fixture=10, **kwargs)


def forecasts(with_bands=True):
    frame = pd.DataFrame({"element": [1, 1], "fixture": [10, 11], "gw": [7, 7],
                          "kickoff_time": ["2026-10-18T14:00:00Z", "2026-10-22T19:00:00Z"],
                          "pred": [4.0, 4.0], "expected_minutes": [60.0, 60.0],
                          "p_start": [0.6, 0.6]})
    if with_bands:
        frame["p_no_appearance"] = .2
        frame["p_1_59"] = .2
        frame["p_60_plus"] = .6
    return frame


class ReturnDateTests(unittest.TestCase):
    def test_recognised_phrase_only_and_explicit_year(self):
        parsed = parse_return_news("Calf injury - Expected back 3rd January 2027", as_of=AS_OF)
        self.assertEqual(parsed["date"].isoformat(), "2027-01-03")
        self.assertEqual(parsed["date_assumption"], "year_explicit_in_official_news")
        self.assertIsNone(parse_return_news("Transferred on 3 Jan 2027", as_of=AS_OF)["date"])

    def test_december_news_rolls_january_into_next_year(self):
        parsed = parse_return_news("Suspended until 2 Jan", as_of="2026-12-30",
                                   news_added="2026-12-28T12:00:00Z")
        self.assertEqual(parsed["date"].isoformat(), "2027-01-02")

    def test_past_date_is_not_moved_to_following_year(self):
        parsed = parse_return_news("Expected back 15 Oct", as_of=AS_OF,
                                   news_added="2026-10-16T12:00:00Z")
        self.assertEqual(parsed["date"].isoformat(), "2026-10-15")
        self.assertIn("has_passed", parsed["warning"])

    def test_missing_news_timestamp_uses_football_season(self):
        parsed = parse_return_news("Expected back 30 Dec", as_of="2027-01-05")
        self.assertEqual(parsed["date"].isoformat(), "2026-12-30")
        self.assertIn("timestamp_missing", parsed["date_assumption"])

    def test_bad_or_ambiguous_date_is_unknown(self):
        for news in ("Expected back 31 Feb", "Expected back 1 Octopus",
                     "Expected back 1 Nov or Expected back 8 Nov", "Unknown return date"):
            with self.subTest(news=news):
                self.assertIsNone(parse_return_news(news, as_of=AS_OF)["date"])

    def test_future_news_rejected(self):
        with self.assertRaisesRegex(ValueError, "after as_of"):
            probability(player("d", chance=75, news_added="2026-10-18T00:00:00Z"))


class AvailabilityTests(unittest.TestCase):
    def test_active_means_no_penalty_not_certain_appearance(self):
        info = probability(player("a", chance=100))
        self.assertEqual(info["availability_probability"], 1)
        self.assertIn("not_guaranteed", info["availability_assumption"])
        rows = forecasts()
        output = apply_fixture_availability(rows, pd.DataFrame([player()]), as_of=AS_OF, next_event=7)
        np.testing.assert_allclose(output.p_no_appearance, [.2, .2])
        self.assertEqual(output.pred.tolist(), [4, 4])

    def test_doubt_uses_official_chance_now_future_range_is_assumption(self):
        who = player("d", chance=75)
        current = probability(who)
        future = probability(who, "2026-11-01T12:00:00Z", 9)
        self.assertEqual(current["availability_probability_low"], .75)
        self.assertEqual(future["availability_probability_low"], .50)
        self.assertEqual(future["availability_probability_base"], .75)
        self.assertEqual(future["availability_probability_high"], 1)
        self.assertIn("carried_forward", future["availability_assumption"])

    def test_missing_doubt_chance_explicit_half_fallback(self):
        info = probability(player("d"))
        self.assertEqual(info["availability_probability"], .5)
        self.assertIn("missing_chance", info["availability_source"])

    def test_injury_return_across_gameweeks(self):
        who = player("i", chance=0, news="Hamstring injury - Expected back 25 Oct",
                     news_added="2026-10-10T12:00:00Z")
        self.assertEqual(probability(who)["availability_probability"], 0)
        self.assertEqual(probability(who, "2026-10-25T12:00:00Z", 8)["availability_probability"], .5)
        self.assertEqual(probability(who, "2026-11-01T12:00:00Z", 9)["availability_probability"], .75)
        self.assertEqual(probability(who, "2026-11-08T12:00:00Z", 10)["availability_probability"], .9)
        self.assertEqual(probability(who, "2026-10-25T12:00:00Z", 8, policy="low")["availability_probability"], .25)

    def test_dgw_uses_each_fixture_date_and_labels_coarse_chance_override(self):
        who = player("s", chance=0, news="Suspended until 20 Oct", news_added="2026-10-10T12:00:00Z")
        output = apply_fixture_availability(forecasts(), pd.DataFrame([who]), as_of=AS_OF, next_event=7)
        self.assertEqual(output.availability_factor.tolist(), [0, .95])
        self.assertEqual(output.pred.tolist(), [0, 4])
        self.assertIn("overrides_coarse", output.availability_warning.iloc[1])
        self.assertAlmostEqual(output.pred.sum(), 4)

    def test_return_day_compared_in_uk_timezone(self):
        who = player("i", news="Expected back 19 Oct", news_added="2026-10-10T00:00:00Z")
        self.assertEqual(probability(who, "2026-10-18T23:30:00Z")["availability_probability"], .5)

    def test_unknown_return_not_claimed_known_long_absence(self):
        for status in ("i", "s"):
            info = probability(player(status, chance=0, news="Unknown return date"), "2026-11-28T15:00:00Z", 12)
            self.assertEqual(info["availability_probability"], 0)
            self.assertIn("not_known_absence", info["availability_assumption"])
            self.assertIn("override", info["availability_warning"])

    def test_unavailable_player_not_revived_by_return_date(self):
        for status in ("u", "n"):
            self.assertEqual(probability(player(status, news="Expected back 18 Oct"))["availability_probability"], 0)


class FusionTests(unittest.TestCase):
    def test_cap_instead_of_double_counting_preserves_band_ratios(self):
        rows = forecasts()
        output = apply_fixture_availability(rows, pd.DataFrame([player("d", chance=50)]), as_of=AS_OF, next_event=7)
        self.assertAlmostEqual(output.availability_multiplier.iloc[0], .5 / .8)
        self.assertAlmostEqual(output.pred.iloc[0], 2.5)
        self.assertAlmostEqual(output.p_no_appearance.iloc[0], .5)
        self.assertAlmostEqual(output.p_1_59.iloc[0], .125)
        self.assertAlmostEqual(output.p_60_plus.iloc[0], .375)
        self.assertAlmostEqual(output.expected_minutes.iloc[0], 37.5)
        self.assertAlmostEqual(output.p_start.iloc[0], .375)
        self.assertEqual(output.raw_pred.tolist(), [4, 4])
        self.assertNotIn("raw_pred", rows)

    def test_lower_learned_appearance_probability_unchanged(self):
        rows = forecasts()
        rows[["p_no_appearance", "p_1_59", "p_60_plus", "p_start"]] = [.8, .05, .15, .15]
        output = apply_fixture_availability(rows, pd.DataFrame([player("d", chance=50)]), as_of=AS_OF, next_event=7)
        self.assertEqual(output.pred.tolist(), [4, 4])
        self.assertEqual(output.availability_multiplier.tolist(), [1, 1])

    def test_zero_appearance_and_missing_start_supported(self):
        rows = forecasts()
        rows[["p_no_appearance", "p_1_59", "p_60_plus", "p_start", "expected_minutes", "pred"]] = [1, 0, 0, np.nan, 0, 0]
        output = apply_fixture_availability(rows, pd.DataFrame([player()]), as_of=AS_OF, next_event=7)
        self.assertTrue(output.p_start.isna().all())
        self.assertEqual(output.p_no_appearance.tolist(), [1, 1])
        self.assertEqual(output.pred.tolist(), [0, 0])

    def test_legacy_model_multiplies_and_labels_method(self):
        output = apply_fixture_availability(forecasts(False), pd.DataFrame([player("d", chance=50)]), as_of=AS_OF, next_event=7)
        self.assertEqual(output.pred.tolist(), [2, 2])
        self.assertIn("legacy_raw", output.availability_fusion.iloc[0])

    def test_reapplying_or_changing_policy_reuses_original_raw_values(self):
        who = pd.DataFrame([player("i", news="Expected back 18 Oct")])
        once = apply_fixture_availability(forecasts(), who, as_of=AS_OF, next_event=7)
        twice = apply_fixture_availability(once, who, as_of=AS_OF, next_event=7)
        pd.testing.assert_frame_equal(once, twice)
        higher = apply_fixture_availability(once, who, as_of=AS_OF, next_event=7, policy="high")
        direct = apply_fixture_availability(forecasts(), who, as_of=AS_OF, next_event=7, policy="high")
        pd.testing.assert_frame_equal(higher, direct)

    def test_bad_bands_and_nonfinite_values_rejected(self):
        for column, value in (("p_no_appearance", .5), ("p_60_plus", np.nan),
                              ("expected_minutes", 91), ("p_start", .9), ("pred", np.inf)):
            rows = forecasts()
            rows[column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                apply_fixture_availability(rows, pd.DataFrame([player()]), as_of=AS_OF, next_event=7)


class OverrideTests(unittest.TestCase):
    def test_specific_override_wins_and_records_reason_source(self):
        overrides = [{"element": 1, "probability": .2, "reason": "general"},
                     {"element": 1, "gw": 7, "probability": .4, "reason": "GW"},
                     {"element": 1, "fixture": 10, "probability": .7,
                      "probability_low": .5, "probability_high": .8,
                      "reason": "club update", "source_url": "https://club.example/update"}]
        info = probability(player("i"), overrides=overrides)
        self.assertEqual(info["availability_probability"], .7)
        self.assertEqual(info["availability_probability_low"], .5)
        self.assertEqual(info["availability_source"], "manual_override")
        self.assertIn("club update", info["availability_assumption"])
        self.assertEqual(info["availability_source_url"], "https://club.example/update")

    def test_override_date_window_is_inclusive(self):
        overrides = [{"element": 1, "from_date": "2026-10-18", "until_date": "2026-10-18",
                      "probability": .8, "reason": "specific day"}]
        self.assertEqual(probability(player("i"), overrides=overrides)["availability_probability"], .8)
        self.assertEqual(probability(player("i"), "2026-10-19T12:00:00Z", overrides=overrides)["availability_probability"], 0)

    def test_equal_priority_overlap_fails(self):
        override = {"element": 1, "probability": .7, "reason": "x"}
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            probability(player(), overrides=[override, override])

    def test_invalid_override_fields_values_and_unknown_players_fail(self):
        for value in (-1, 1.1, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_overrides([{"element": 1, "probability": value, "reason": "x"}])
        with self.assertRaisesRegex(ValueError, "requires reason"):
            validate_overrides([{"element": 1, "probability": .5}])
        with self.assertRaisesRegex(ValueError, "Unknown player"):
            apply_fixture_availability(forecasts(), pd.DataFrame([player()]), as_of=AS_OF, next_event=7,
                                       overrides=[{"element": 999, "probability": .5, "reason": "typo"}])


if __name__ == "__main__":
    unittest.main()
