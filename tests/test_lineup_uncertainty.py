import copy
from itertools import product
import unittest

import numpy as np
import pandas as pd

from lineup_uncertainty import expected_lineup_points, refine_lineup
from strategy_backtest import realized_score


def scenario():
    positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    frame = pd.DataFrame({"element": range(1, 16), "pos": positions,
                          "gw": [7] * 15, "pred": [2.0] * 15,
                          "p_no_appearance": [0.0] * 15})
    step = {"gw": 7, "squad": list(range(1, 16)), "starting_xi": [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15],
            "bench": [2, 12, 6, 7], "captain": 13, "vice_captain": 8,
            "chip": None, "hit_cost": 4, "predicted_points": 24, "net_predicted_points": 20}
    return step, frame


def update(frame, pid, **values):
    for field, value in values.items():
        frame.loc[frame.element.eq(pid), field] = value


class ExpectationTests(unittest.TestCase):
    def test_all_play_equals_standard_points(self):
        step, frame = scenario()
        scored = expected_lineup_points(step, frame)
        self.assertEqual(scored["total"], 24)
        self.assertEqual(scored["goalkeeper_autosub_points"], 0)
        self.assertEqual(scored["outfield_autosub_points"], 0)
        self.assertEqual(scored["vice_captain_fallback_bonus"], 0)

    def test_absent_goalkeeper_uses_unconditional_bench_xp_once(self):
        step, frame = scenario()
        update(frame, 1, p_no_appearance=1, pred=0)
        update(frame, 2, p_no_appearance=.5, pred=3)
        scored = expected_lineup_points(step, frame)
        self.assertEqual(scored["goalkeeper_autosub_points"], 3)
        self.assertEqual(scored["total"], 25)

    def test_minimum_defenders_blocks_first_midfielder(self):
        step, frame = scenario()
        update(frame, 3, p_no_appearance=1, pred=0)
        update(frame, 12, pred=100)
        update(frame, 6, pred=7)
        update(frame, 7, pred=50)
        scored = expected_lineup_points(step, frame)
        # Bench priority selects DEF6, not MID12 or higher-scoring DEF7.
        self.assertEqual(scored["outfield_autosub_points"], 7)

    def test_captain_fallback_and_triple_captain(self):
        step, frame = scenario()
        update(frame, 13, pred=3, p_no_appearance=.5)
        update(frame, 8, pred=4, p_no_appearance=.2)
        scored = expected_lineup_points(step, frame)
        self.assertEqual(scored["captain_primary_bonus"], 3)
        self.assertEqual(scored["vice_captain_fallback_bonus"], 2)
        step["chip"] = "3xc"
        triple = expected_lineup_points(step, frame)
        self.assertEqual(triple["captain_primary_bonus"], 6)
        self.assertEqual(triple["vice_captain_fallback_bonus"], 4)
        self.assertEqual(triple["total"] - scored["total"], 5)

    def test_bench_boost_counts_each_player_once(self):
        step, frame = scenario()
        step["chip"] = "bboost"
        update(frame, 3, pred=0, p_no_appearance=1)
        scored = expected_lineup_points(step, frame)
        self.assertEqual(scored["total"], frame.pred.sum() + 2)
        self.assertEqual(scored["bench_boost_points"], 8)
        self.assertEqual(scored["outfield_autosub_points"], 0)

    def test_dgw_any_appearance_prevents_substitution(self):
        step, frame = scenario()
        frame["fixture"] = 101
        update(frame, 1, pred=2, p_no_appearance=.5)
        second = frame[frame.element.eq(1)].copy()
        second["fixture"], second["pred"], second["p_no_appearance"] = 102, 3, .4
        dgw = pd.concat([frame, second], ignore_index=True)
        scored = expected_lineup_points(step, dgw)
        self.assertAlmostEqual(scored["goalkeeper_autosub_points"], .5 * .4 * 2)
        aggregate = frame.copy()
        update(aggregate, 1, pred=5, p_no_appearance=.2)
        self.assertAlmostEqual(expected_lineup_points(step, aggregate)["total"], scored["total"])
        refined = refine_lineup(step, dgw, improve_xi=False)
        self.assertTrue(any("DGW inputs aggregated" in a for a in refined["lineup_refinement"]["assumptions"]))

    def test_exact_enumeration_matches_realized_scorer_for_random_sparse_uncertainty(self):
        step, frame = scenario()
        uncertain = [1, 3, 4, 8, 13, 12, 6, 7]
        for index, pid in enumerate(uncertain):
            update(frame, pid, p_no_appearance=.15 + .1 * (index % 4), pred=1 + (index % 5))
        meta = {int(row.element): {"pos": row.pos} for row in frame.itertuples()}
        rows = frame.set_index("element").to_dict("index")
        expected = 0.0
        for played in product((False, True), repeat=len(uncertain)):
            weight = 1.0
            actual = {pid: {"minutes": 90, "total_points": values["pred"]} for pid, values in rows.items()}
            for pid, appears in zip(uncertain, played):
                p_no = rows[pid]["p_no_appearance"]
                weight *= 1 - p_no if appears else p_no
                actual[pid] = {"minutes": 90 if appears else 0,
                               "total_points": rows[pid]["pred"] / (1 - p_no) if appears else 0}
            expected += weight * realized_score(step, meta, actual)["gross_points"]
        self.assertAlmostEqual(expected_lineup_points(step, frame)["total"], expected, places=10)


class RefinementTests(unittest.TestCase):
    def test_refinement_optimises_bench_priority_fixed_xi(self):
        step, frame = scenario()
        update(frame, 3, p_no_appearance=1, pred=0)
        update(frame, 6, pred=3)
        update(frame, 7, pred=8)
        original = copy.deepcopy(step)
        output = refine_lineup(step, frame, improve_xi=False)
        self.assertEqual(set(output["starting_xi"]), set(step["starting_xi"]))
        defenders = [i for i in output["bench"] if i in (6, 7)]
        self.assertEqual(defenders, [7, 6])
        self.assertEqual(step, original)
        self.assertAlmostEqual(output["lineup_refinement"]["expected_improvement"], 5)

    def test_captain_pair_optimised_with_fallback(self):
        step, frame = scenario()
        update(frame, 13, pred=5, p_no_appearance=.1)
        update(frame, 8, pred=4, p_no_appearance=.75)
        output = refine_lineup(step, frame, improve_xi=False)
        self.assertEqual((output["captain"], output["vice_captain"]), (8, 13))
        self.assertAlmostEqual(output["expected_points_components"]["captain_primary_bonus"] +
                               output["expected_points_components"]["vice_captain_fallback_bonus"], 7.75)

    def test_single_swap_promotes_strong_reserve_but_preserves_original_milp_values(self):
        step, frame = scenario()
        update(frame, 12, pred=10)
        output = refine_lineup(step, frame)
        self.assertIn(12, output["starting_xi"])
        self.assertEqual(output["predicted_points"], step["predicted_points"])
        self.assertEqual(output["net_predicted_points"], step["net_predicted_points"])
        self.assertGreater(output["expected_points_with_contingencies"], 24)
        self.assertEqual(output["expected_net_points_with_contingencies"], output["expected_points_with_contingencies"] - 4)
        self.assertTrue(output["lineup_refinement"]["transfer_objective_unchanged"])

    def test_bb_can_promote_bench_captain_without_adding_bench_points_twice(self):
        step, frame = scenario()
        step["chip"] = "bboost"
        update(frame, 12, pred=10)
        output = refine_lineup(step, frame)
        self.assertEqual(output["captain"], 12)
        self.assertEqual(output["expected_points_with_contingencies"], frame.pred.sum() + 10)

    def test_original_included_and_never_worse_in_random_scenarios(self):
        rng = np.random.RandomState(41)
        for chip in (None, "3xc", "bboost"):
            step, frame = scenario()
            step["chip"] = chip
            frame["pred"] = rng.uniform(-.2, 8, len(frame))
            frame["p_no_appearance"] = rng.uniform(0, .8, len(frame))
            output = refine_lineup(step, frame)
            self.assertGreaterEqual(output["expected_points_with_contingencies"] + 1e-9,
                                    expected_lineup_points(step, frame)["total"])
            self.assertGreaterEqual(output["lineup_refinement"]["candidate_xi_count"], 1)
            self.assertEqual(len(set(output["starting_xi"])), 11)
            self.assertEqual(set(output["bench"]), set(step["squad"]) - set(output["starting_xi"]))

    def test_no_probabilities_do_not_invent_appearance_model(self):
        step, frame = scenario()
        with self.assertRaisesRegex(ValueError, "requires"):
            refine_lineup(step, frame.drop(columns="p_no_appearance"))

    def test_invalid_input_rejected(self):
        step, frame = scenario()
        for field, value in (("p_no_appearance", -1), ("p_no_appearance", np.nan),
                             ("p_no_appearance", np.inf), ("pred", np.inf)):
            bad = frame.copy()
            bad.loc[0, field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                refine_lineup(step, bad)
        bad = frame.copy()
        update(bad, 1, p_no_appearance=1, pred=2)
        with self.assertRaisesRegex(ValueError, "zero expected points"):
            refine_lineup(step, bad)
        with self.assertRaisesRegex(ValueError, "distinct fixture"):
            refine_lineup(step, pd.concat([frame, frame.iloc[[0]]]))


if __name__ == "__main__":
    unittest.main()
