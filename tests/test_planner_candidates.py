import unittest

import pandas as pd

from planner_candidates import prune_dominated


def candidates(predictions=None):
    predictions = predictions or {i: [2, 2] for i in range(1, 6)}
    return pd.DataFrame([{"element": pid, "gw": 7 + offset, "team_key": 1,
                          "pos": "MID", "price": 5.0, "pred": points, "can_select": True}
                         for pid, values in predictions.items() for offset, points in enumerate(values)])


class ExactPruningTests(unittest.TestCase):
    def test_equal_candidate_cycle_keeps_three_with_stable_id_tiebreak(self):
        original = candidates()
        before = original.copy(deep=True)
        filtered, audit = prune_dominated(original, [])
        self.assertEqual(sorted(filtered.element.unique()), [1, 2, 3])
        self.assertEqual(audit["removed_dominated_count"], 2)
        self.assertFalse(audit["heuristic"])
        pd.testing.assert_frame_equal(original, before)

    def test_owned_players_always_survive_even_when_unselectable(self):
        data = candidates()
        data.loc[data.element == 5, ["can_select", "pred"]] = [False, 0]
        filtered, audit = prune_dominated(data, [5])
        self.assertIn(5, filtered.element.unique())
        self.assertEqual(audit["owned_players_retained"], 1)

    def test_owned_players_cannot_be_dominance_witnesses(self):
        data = candidates({1: [10, 10], 2: [9, 9], 3: [8, 8], 4: [1, 1]})
        filtered, _ = prune_dominated(data, [1])
        self.assertIn(4, filtered.element.unique())

    def test_every_gameweek_must_be_dominated_not_only_the_average(self):
        data = candidates({1: [3, 3], 2: [4, 4], 3: [9, 1], 4: [2, 2]})
        filtered, _ = prune_dominated(data, [])
        self.assertIn(4, filtered.element.unique())

    def test_expensive_or_other_club_or_other_position_is_not_a_witness(self):
        for field, value in [("price", 5.1), ("team_key", 2), ("pos", "DEF")]:
            data = candidates({1: [4, 4], 2: [5, 5], 3: [6, 6], 4: [1, 1]})
            data.loc[data.element == 3, field] = value
            with self.subTest(field=field):
                filtered, _ = prune_dominated(data, [])
                self.assertIn(4, filtered.element.unique())

    def test_three_no_more_expensive_dominators_allow_removal(self):
        data = candidates({1: [4, 4], 2: [5, 5], 3: [6, 6], 4: [1, 1]})
        data.loc[data.element.isin([1, 2, 3]), "price"] = 4.5
        filtered, audit = prune_dominated(data, [])
        self.assertNotIn(4, filtered.element.unique())
        self.assertEqual(audit["dominated_removals"], [{"element": 4, "dominated_by_at_removal": [1, 2, 3]}])

    def test_unselectable_unowned_players_are_removed_before_dominance(self):
        data = candidates({1: [4, 4], 2: [5, 5], 3: [6, 6], 4: [1, 1]})
        data.loc[data.element == 3, "can_select"] = False
        filtered, audit = prune_dominated(data, [])
        self.assertNotIn(3, filtered.element.unique())
        self.assertIn(4, filtered.element.unique())
        self.assertEqual(audit["removed_unselectable_count"], 1)

    def test_variable_prices_or_selectability_rejected(self):
        for field, value in [("price", 4.9), ("can_select", False)]:
            data = candidates()
            data.loc[(data.element == 1) & (data.gw == 8), field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "constant"):
                prune_dominated(data, [])

    def test_partial_horizon_and_missing_owned_player_rejected(self):
        data = candidates()
        with self.assertRaisesRegex(ValueError, "every GW"):
            prune_dominated(data.iloc[1:], [])
        with self.assertRaisesRegex(ValueError, "omits owned"):
            prune_dominated(data, [100])

    def test_witnesses_were_still_retained_when_each_removal_happened(self):
        data = candidates({i: [i, i] for i in range(1, 12)})
        _, audit = prune_dominated(data, [])
        retained = set(data.element)
        for removal in audit["dominated_removals"]:
            self.assertTrue(set(removal["dominated_by_at_removal"]) <= retained)
            retained.remove(removal["element"])
        self.assertGreaterEqual(len(retained), 3)


if __name__ == "__main__":
    unittest.main()
