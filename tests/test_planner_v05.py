"""Financial/FT decisions survive uncertainty-aware post-selection lineups."""
from collections import Counter
import copy
import unittest

import pandas as pd

from plan_transfers import plan, refine_plan_lineups, _scoring_chip_seed


def scenario():
    positions = ["GK"]*2+["DEF"]*5+["MID"]*5+["FWD"]*3
    state = {"next_event": 7, "free_transfers": 2, "bank": 10,
        "rules": {"max_free_transfers": 5, "free_transfer_overrides": {"8": 5}},
        "chips": [{"id": 1, "name": "wildcard", "start_event": 1, "stop_event": 19,
                    "status": "available", "used_event": None}], "chips_used": [],
        "players": [{"id": i, "position": pos, "team": i, "now_cost": 50,
                     "purchase_price": 50, "selling_price": 50} for i, pos in enumerate(positions, 1)]}
    positions += ["MID", "DEF", "FWD"]
    forecast = pd.DataFrame([{"element": i, "web_name": f"P{i}", "team_key": i,
        "pos": pos, "price": 5.0, "gw": gw, "pred": 6.0 if i > 15 else 1.+i%4,
        "p_no_appearance": .65 if i in (8, 16) else .1, "can_select": True}
        for gw in (7, 8) for i, pos in enumerate(positions, 1)])
    return state, forecast


class PlannerRefinementIntegrationTests(unittest.TestCase):
    def test_zero_time_joint_search_uses_profitable_expiring_chip(self):
        state, forecast = scenario()
        forecast = forecast[forecast.gw.eq(7)]
        state["chips"] = [{"id": 1, "name": "3xc", "start_event": 1,
                           "stop_event": 7, "status": "available"}]
        result = plan(state, forecast, seconds=0)
        step = result["steps"][0]
        self.assertEqual(step["chip"], "3xc")
        self.assertEqual(step["transfer_in"], [])
        self.assertEqual(step["bank"], state["bank"])
        points = forecast.set_index("element").pred
        self.assertAlmostEqual(step["predicted_points"],
            points.loc[step["starting_xi"]].sum() + 2 * points.loc[step["captain"]])

    def test_chip_seed_joint_assignment_and_reserve(self):
        steps = [{"gw": 18, "captain": 1, "bench": [2], "chip": None},
                 {"gw": 19, "captain": 1, "bench": [2], "chip": None}]
        xp = {(1,18): 10., (1,19): 9., (2,18): 9., (2,19): 1.}
        chips = [{"name": n, "start_event": 1, "stop_event": 19}
                 for n in ("3xc", "bboost")]
        reserves = {"3xc": 12., "bboost": 18.}
        # Greedily using TC in GW18 loses the better combined BB18+TC19.
        self.assertEqual(set(_scoring_chip_seed(steps, chips, xp, reserves, 1.)), {(1,0),(0,1)})
        for chip in chips:
            chip["stop_event"] = 38
        self.assertEqual(_scoring_chip_seed(steps, chips, xp, reserves, 1.), ())

    def test_forced_wildcard_then_dated_ft_override_survives_refinement(self):
        state, forecast = scenario()
        original = plan(state, forecast, forced_chip=("wildcard", 7), seconds=5)
        saved = copy.deepcopy(original)
        refined = refine_plan_lineups(original, forecast)
        self.assertEqual(original, saved, "Refinement must not mutate the original solver result")
        a, b = refined["steps"]
        self.assertEqual(a["chip"], "wildcard")
        self.assertEqual(a["free_transfers_before"], 2)
        self.assertEqual(a["free_transfers_next"], 5)
        self.assertEqual(b["free_transfers_before"], 5)
        self.assertEqual(a["hit_cost"], 0)
        self.assertTrue(set([16, 17, 18]).issubset(a["permanent_squad"]))
        unchanged = ["gw", "chip", "transfer_in", "transfer_out", "temporary_in", "temporary_out",
                     "bank", "hit_cost", "free_transfers_before", "free_transfers_next", "permanent_squad", "squad"]
        for before, after in zip(original["steps"], refined["steps"]):
            for name in unchanged:
                self.assertEqual(before[name], after[name], name)
            self.assertGreaterEqual(after["bank"], 0)
            self.assertEqual(len(set(after["starting_xi"])), 11)
            self.assertEqual(set(after["starting_xi"]) | set(after["bench"]), set(after["squad"]))
            self.assertEqual(len(set(after["starting_xi"]) & set(after["bench"])), 0)
            positions = forecast[forecast.gw.eq(after["gw"])].set_index("element").pos
            counts = Counter(positions.loc[after["starting_xi"]])
            self.assertEqual(counts["GK"], 1)
            self.assertGreaterEqual(counts["DEF"], 3)
            self.assertGreaterEqual(counts["MID"], 2)
            self.assertGreaterEqual(counts["FWD"], 1)
            points = forecast[forecast.gw.eq(after["gw"])].set_index("element").pred
            expected = points.loc[after["starting_xi"]].sum()+points.loc[after["captain"]]
            self.assertAlmostEqual(after["predicted_points"], expected, places=3)
            self.assertAlmostEqual(after["net_predicted_points"], expected-after["hit_cost"], places=3)
            self.assertFalse(after["lineup_refinement"]["legacy_predicted_points_are_original_planner_values"])
        discount = refined["assumptions"]["discount"]
        self.assertAlmostEqual(refined["discounted_net_points"],
            round(sum(discount**t*step["net_predicted_points"] for t, step in enumerate(refined["steps"])), 3))
        self.assertAlmostEqual(refined["discounted_points_with_contingencies"],
            round(sum(discount**t*step["expected_net_points_with_contingencies"] for t, step in enumerate(refined["steps"])), 3))
        self.assertEqual(refined["milp_original_discounted_net_points"], original["discounted_net_points"])
        self.assertEqual(refined["solver"], original["solver"])

    def test_starting_deadline_override_is_used_by_the_transition_graph(self):
        state, forecast = scenario()
        state["rules"]["free_transfer_overrides"] = {"7": 4, "8": 5}
        result = plan(state, forecast, forced_chip=("wildcard", 7), seconds=0)
        self.assertEqual(result["steps"][0]["free_transfers_before"], 4)
        self.assertEqual(result["steps"][0]["free_transfers_next"], 5)
        self.assertEqual(result["steps"][1]["free_transfers_before"], 5)


if __name__ == "__main__":
    unittest.main()
