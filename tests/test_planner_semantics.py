"""Independent regressions for transfer ownership and chip boundary semantics."""
import unittest
from collections import Counter
from itertools import combinations
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from plan_transfers import Program, plan


def scenario(horizon=2):
    positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    players = [{"id": i, "position": pos, "team": i, "now_cost": 50,
                "purchase_price": 50, "selling_price": 50} for i, pos in enumerate(positions, 1)]
    state = {"next_event": 7, "players": players, "bank": 0, "free_transfers": 3,
             "chips": [], "chips_used": []}
    positions += ["MID", "DEF"]
    rows = [{"element": i, "web_name": f"P{i}", "gw": 7 + offset, "team_key": i,
             "pos": pos, "price": 5.0, "pred": 2.0, "can_select": True}
            for offset in range(horizon) for i, pos in enumerate(positions, 1)]
    return state, pd.DataFrame(rows)


def freehit(status="available"):
    return {"id": 3, "name": "freehit", "start_event": 2, "stop_event": 19,
            "status": status, "used_event": None}


class PlannerSemantics(unittest.TestCase):
    def test_invalid_time_limits_rejected_before_solver_configuration(self):
        state, df = scenario(horizon=1)
        for invalid in (-1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(seconds=invalid):
                with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                    plan(state, df, seconds=invalid)
                with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                    Program().solve(invalid)

    def test_invalid_discount_ft_value_and_reserves_rejected(self):
        state, df = scenario(horizon=1)
        for option in ("discount", "ft_value"):
            for invalid in (-1, float("nan"), float("inf"), -float("inf")):
                with self.subTest(option=option, value=invalid), self.assertRaisesRegex(ValueError, "discount or terminal FT"):
                    plan(state, df, **{option: invalid})
        for invalid in (-1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(reserve=invalid), self.assertRaisesRegex(ValueError, "chip reserve"):
                plan(state, df, chip_reserves={"wildcard": invalid})

    def test_both_baseline_fallbacks_keep_original_pruning_audit(self):
        state, df = scenario(horizon=1)
        state["chips"] = [{"id": 5, "name": "3xc", "start_event": 1,
                           "stop_event": 19, "status": "available", "used_event": None}]
        df.loc[df.element == 17, "can_select"] = False
        actual_solve = Program.solve
        for failure in ("no_incumbent", "worse_incumbent"):
            calls = 0

            def controlled_solve(program, seconds, seed=None):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return actual_solve(program, 0, seed)
                if failure == "no_incumbent":
                    return SimpleNamespace(x=None, status=2, message="Synthetic timeout", mip_gap=None)
                # The only chip variable is first. Forcing TC into the seed is
                # feasible but its 2-point gain is below the 12-point reserve.
                inferior_seed = dict(seed)
                inferior_seed[0] = 1
                return actual_solve(program, 0, inferior_seed)

            with self.subTest(failure=failure), patch.object(Program, "solve", controlled_solve):
                result = plan(state, df, seconds=0)
            self.assertFalse(result["chip_search"]["accepted"])
            self.assertEqual(result["candidate_pruning"]["input_players"], 17)
            self.assertEqual(result["candidate_pruning"]["retained_players"], 16)
            self.assertEqual(result["candidate_pruning"]["removed_unselectable_count"], 1)
            self.assertIsNone(result["steps"][0]["chip"])

    def test_zero_seconds_returns_legal_plan_at_least_as_good_as_holding(self):
        state, df = scenario(horizon=3)
        # Strong transfer candidates make it tempting for an unseeded timeout
        # path to return no recommendation. The legal hold solution must survive.
        df.loc[df.element > 15, "pred"] = 100
        for gw in (7, 8, 9):
            df.loc[(df.element <= 15) & df.gw.eq(gw), "pred"] = [
                float((pid * gw) % 7) + 0.25 for pid in range(1, 16)]
        state["chips"] = [{"id": i, "name": name, "start_event": 2, "stop_event": 19,
                           "status": "available", "used_event": None}
                          for i, name in enumerate(["wildcard", "freehit", "bboost", "3xc"], 1)]
        owned_ids = [p["id"] for p in state["players"]]
        positions = {p["id"]: p["position"] for p in state["players"]}
        hold = 0
        for offset, gw in enumerate((7, 8, 9)):
            points = df[df.gw.eq(gw)].set_index("element").pred.to_dict()
            possible = []
            for starters in combinations(owned_ids, 11):
                counts = Counter(positions[i] for i in starters)
                if counts["GK"] == 1 and 3 <= counts["DEF"] <= 5 and 2 <= counts["MID"] <= 5 and 1 <= counts["FWD"] <= 3:
                    possible.append(sum(points[i] for i in starters) + max(points[i] for i in starters))
            hold += 0.9 ** offset * max(possible)
        for use_chips in (False, True):
            with self.subTest(allow_chips=use_chips):
                result = plan(state, df, allow_chips=use_chips, seconds=0)
                self.assertGreaterEqual(result["discounted_net_points"] + 0.002, hold)
                self.assertEqual(result["solver"]["backend"], "highspy")
                self.assertFalse(result["solver"]["optimal_within_tolerance"])
                for step in result["steps"]:
                    self.assertEqual(len(set(step["squad"])), 15)
                    self.assertEqual(len(set(step["starting_xi"])), 11)
                    self.assertTrue(set(step["starting_xi"]) <= set(step["squad"]))
                    self.assertIn(step["captain"], step["starting_xi"])
                    self.assertNotEqual(step["captain"], step["vice_captain"])
                    self.assertGreaterEqual(step["bank"], 0)
                    self.assertTrue(1 <= step["free_transfers_next"] <= 5)
                    self.assertEqual(step["hit_cost"], 0)

    def test_future_freehit_retains_original_price_spread(self):
        state, df = scenario()
        state["players"][7].update(now_cost=60, purchase_price=50, selling_price=55)
        state["chips"] = [freehit()]
        df.loc[df.element == 8, "price"] = 6
        df.loc[df.element == 16, ["price", "pred"]] = [6, 200]
        df = df[df.element != 17]
        for gw in (7, 8):
            with self.subTest(gw=gw):
                result = plan(state, df, forced_chip=("freehit", gw), seconds=10)
                step = next(s for s in result["steps"] if s["gw"] == gw)
                self.assertIn(8, step["squad"])
                self.assertNotIn(16, step["squad"], "Selling an appreciated player must not fund a market-value replacement")
                self.assertEqual(step["bank"], 0)

    def test_freehit_cooldown_blocks_next_week_and_allows_following_week(self):
        state, df = scenario()
        state["chips"] = [freehit("blocked")]
        state["chips_used"] = [{"name": "freehit", "event": 6}]
        with self.assertRaisesRegex(RuntimeError, "no feasible"):
            plan(state, df, forced_chip=("freehit", 7), seconds=10)
        result = plan(state, df, forced_chip=("freehit", 8), seconds=10)
        self.assertEqual([s["chip"] for s in result["steps"]], [None, "freehit"])

    def test_unselectable_player_cannot_be_bought_even_for_freehit(self):
        state, df = scenario(horizon=1)
        df.loc[df.element == 16, ["can_select", "pred"]] = [False, 200]
        normal = plan(state, df, seconds=10)["steps"][0]
        self.assertNotIn(16, normal["squad"])
        state["chips"] = [freehit()]
        temporary = plan(state, df, forced_chip=("freehit", 7), seconds=10)["steps"][0]
        self.assertNotIn(16, temporary["squad"])

    def test_sold_and_rebought_player_loses_old_purchase_discount(self):
        state, df = scenario(horizon=3)
        state["players"][7].update(now_cost=60, purchase_price=50, selling_price=55)
        df.loc[df.element == 8, "price"] = 6
        df.loc[df.element == 16, "price"] = 5.5
        df.loc[df.element == 17, "price"] = 4.5
        df.loc[df.element.between(3, 12), "pred"] = 1000
        df.loc[df.element.isin([8, 16, 17]), "pred"] = 0
        df.loc[(df.element == 16) & df.gw.isin([7, 9]), "pred"] = 2000
        df.loc[(df.element == 8) & df.gw.eq(8), "pred"] = 2000
        result = plan(state, df, discount=1, ft_value=0, seconds=10)["steps"]
        self.assertIn(8, result[0]["transfer_out"])
        self.assertIn(8, result[1]["transfer_in"])
        self.assertIn(8, result[2]["transfer_out"])
        # GW8 costs £0.5m to buy P8 back; GW9 sells him at the new £6m
        # acquisition price, so the original defender can be afforded again.
        self.assertNotIn(17, result[2]["permanent_squad"])
        self.assertEqual(result[2]["bank"], 0)


if __name__ == "__main__":
    unittest.main()
