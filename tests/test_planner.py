import unittest

import pandas as pd

from plan_transfers import plan


def fixture(horizon=2, extras=True):
    positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    players = [{"id": i + 1, "name": f"P{i+1}", "position": pos,
                "team": i + 1, "now_cost": 50, "purchase_price": 50,
                "selling_price": 50} for i, pos in enumerate(positions)]
    state = {"entry_id": 1, "next_event": 7, "free_transfers": 3,
             "bank": 4, "players": players, "chips": [], "chips_used": []}
    if extras:
        positions += ["MID", "DEF", "FWD"]
    rows = [{"element": i + 1, "web_name": f"P{i+1}", "gw": 7 + t,
             "team_key": i + 1, "pos": pos, "price": 5.0,
             "pred": 2.0 if i < 15 else 1.0}
            for t in range(horizon) for i, pos in enumerate(positions)]
    return state, pd.DataFrame(rows)


def chip(name, start=7, stop=19, number=1):
    return {"id": number, "name": name, "start_event": start,
            "stop_event": stop, "status": "available", "used_event": None}


class PlannerTests(unittest.TestCase):
    def test_roll_free_transfers_and_cap_at_five(self):
        state, df = fixture(horizon=3, extras=False)
        result = plan(state, df, seconds=10)
        self.assertEqual([s["free_transfers_next"] for s in result["steps"]], [4, 5, 5])
        self.assertTrue(all(not s["transfer_in"] for s in result["steps"]))
        self.assertTrue(all(s["bank"] == 4 for s in result["steps"]))

    def test_use_three_free_transfers_with_no_hit(self):
        state, df = fixture(horizon=1)
        df.loc[df.element > 15, "pred"] = 20
        result = plan(state, df, seconds=10)["steps"][0]
        self.assertEqual(result["transfer_in"], [16, 17, 18])
        self.assertEqual(result["hit_cost"], 0)
        self.assertEqual(result["free_transfers_next"], 1)

    def test_paid_transfers_deduct_four(self):
        state, df = fixture(horizon=1)
        state["free_transfers"] = 1
        df.loc[df.element > 15, "pred"] = 20
        result = plan(state, df, seconds=10)["steps"][0]
        self.assertEqual(result["hit_cost"], 8)
        self.assertAlmostEqual(result["predicted_points"] - result["net_predicted_points"], 8)

    def test_wildcard_is_permanent_and_preserves_ft(self):
        state, df = fixture(horizon=2)
        state["chips"] = [chip("wildcard")]
        df.loc[df.element > 15, "pred"] = 20
        result = plan(state, df, forced_chip=("wildcard", 7), seconds=10)
        a, b = result["steps"]
        self.assertEqual(a["free_transfers_next"], 3)
        self.assertEqual(a["hit_cost"], 0)
        self.assertTrue(set([16, 17, 18]).issubset(b["permanent_squad"]))

    def test_free_hit_reverts_and_preserves_bank_ft(self):
        state, df = fixture(horizon=2)
        state["chips"] = [chip("freehit")]
        df.loc[(df.element > 15) & (df.gw == 7), "pred"] = 20
        result = plan(state, df, forced_chip=("freehit", 7), seconds=10)
        a, b = result["steps"]
        self.assertEqual(a["temporary_in"], [16, 17, 18])
        self.assertEqual(a["permanent_squad"], list(range(1, 16)))
        self.assertEqual(b["squad"], list(range(1, 16)))
        self.assertEqual(a["free_transfers_next"], 3)
        self.assertEqual(a["bank"], 4)

    def test_sale_value_not_market_value_funds_transfer(self):
        state, df = fixture(horizon=1)
        state["bank"] = 0
        state["players"][7].update(now_cost=60, purchase_price=50, selling_price=55)
        df.loc[df.element == 8, "price"] = 6
        df.loc[df.element == 16, ["price", "pred"]] = [6, 20]
        # Selling the appreciated player raises 5.5m, not 6.0m.
        result = plan(state, df, seconds=10)["steps"][0]
        self.assertNotIn(16, result["transfer_in"])
        self.assertGreaterEqual(result["bank"], 0)

    def test_free_hit_can_keep_appreciated_players_without_buying_again(self):
        state, df = fixture(horizon=1, extras=False)
        state["bank"] = 0
        state["players"][7].update(now_cost=60, purchase_price=50, selling_price=55)
        df.loc[df.element == 8, "price"] = 6
        state["chips"] = [chip("freehit")]
        result = plan(state, df, forced_chip=("freehit", 7), seconds=10)["steps"][0]
        self.assertEqual(result["squad"], list(range(1, 16)))
        self.assertEqual(result["bank"], 0)

    def test_triple_captain_and_bench_boost_scoring(self):
        for name, expected in [("3xc", 26), ("bboost", 32)]:
            state, df = fixture(horizon=1, extras=False)
            state["chips"] = [chip(name)]
            result = plan(state, df, forced_chip=(name, 7), seconds=10)["steps"][0]
            self.assertEqual(result["predicted_points"], expected)

    def test_cannot_force_unavailable_chip(self):
        state, df = fixture(horizon=1)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            plan(state, df, forced_chip=("wildcard", 7))

    def test_missing_owned_player_and_duplicate_forecasts_rejected(self):
        state, df = fixture(horizon=1)
        with self.assertRaisesRegex(ValueError, "omits owned"):
            plan(state, df[df.element != 1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            plan(state, pd.concat([df, df.iloc[:1]]))

    def test_chip_expiry_and_one_chip_per_gw(self):
        state, df = fixture(horizon=2, extras=False)
        state["chips"] = [chip("3xc", stop=7, number=1), chip("bboost", stop=8, number=2)]
        result = plan(state, df, seconds=10)["steps"]
        self.assertEqual([s["chip"] for s in result], ["3xc", "bboost"])


if __name__ == "__main__":
    unittest.main()
