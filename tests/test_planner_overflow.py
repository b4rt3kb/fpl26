"""Existing real-club-transfer overflow is legal only until an acquisition."""
from collections import Counter
import unittest

from plan_transfers import plan
from tests.test_planner import fixture, chip


def overflow(horizon=2, extras=True):
    state, frame = fixture(horizon=horizon, extras=extras)
    members = {3, 4, 8, 13}
    frame.loc[frame.element.isin(members), "team_key"] = 100
    for player in state["players"]:
        if player["id"] in members:
            player["team"] = 100
    state["provenance"] = {"allow_club_overflow": True}
    return state, frame


def counts(ids, frame):
    teams = frame.drop_duplicates("element").set_index("element").team_key.to_dict()
    return Counter(teams[i] for i in ids)


class PlannerOverflowTests(unittest.TestCase):
    def test_zero_seconds_hold_with_overflow_is_valid_warm_start(self):
        state, frame = overflow(horizon=3, extras=False)
        result = plan(state, frame, seconds=0)
        for step in result["steps"]:
            self.assertEqual(step["squad"], list(range(1, 16)))
            self.assertEqual(step["transfer_in"], [])
            self.assertEqual(counts(step["squad"], frame)[100], 4)
        self.assertEqual(result["assumptions"]["grandfathered_initial_club_excess"], {100: 1})

    def test_first_transfer_repairs_overflow(self):
        state, frame = overflow(horizon=1)
        frame.loc[frame.element.eq(16), "pred"] = 50
        result = plan(state, frame, seconds=2)["steps"][0]
        self.assertIn(16, result["transfer_in"])
        self.assertLessEqual(max(counts(result["squad"], frame).values()), 3)
        self.assertTrue(set(result["transfer_out"]) & {3, 4, 8, 13})

    def test_buy_from_overflow_club_cannot_keep_or_increase_overflow(self):
        state, frame = overflow(horizon=1)
        frame.loc[frame.element.eq(16), ["team_key", "pred"]] = [100, 100]
        result = plan(state, frame, seconds=2)["steps"][0]
        self.assertIn(16, result["transfer_in"])
        self.assertLessEqual(counts(result["squad"], frame)[100], 3)
        self.assertGreaterEqual(len(set(result["transfer_out"]) & {3, 4, 8, 13}), 2)

    def test_freehit_active_legal_reverts_to_overflow_holdings_even_zero_seconds(self):
        state, frame = overflow(horizon=2)
        state["chips"] = [chip("freehit")]
        result = plan(state, frame, forced_chip=("freehit", 7), seconds=0)
        first, after = result["steps"]
        self.assertEqual(first["permanent_squad"], list(range(1, 16)))
        self.assertEqual(first["transfer_in"], [])
        self.assertLessEqual(max(counts(first["squad"], frame).values()), 3)
        self.assertEqual(after["squad"], list(range(1, 16)))
        self.assertEqual(counts(after["squad"], frame)[100], 4)
        self.assertEqual(first["bank"], state["bank"])
        self.assertEqual(first["free_transfers_next"], 3)

    def test_once_repaired_future_hold_or_purchase_cannot_restore_overflow(self):
        state, frame = overflow(horizon=3)
        frame.loc[frame.element.eq(16), "pred"] = 100
        # A fourth strong owned club player becomes attractive later, but cannot
        # be bought back while keeping the other three from that club.
        frame.loc[frame.element.eq(8) & frame.gw.gt(7), "pred"] = 90
        result = plan(state, frame, seconds=3)
        self.assertTrue(result["steps"][0]["transfer_in"])
        for step in result["steps"]:
            self.assertLessEqual(max(counts(step["permanent_squad"], frame).values()), 3)

    def test_zero_transfer_wildcard_may_hold_grandfathered_squad(self):
        state, frame = overflow(horizon=1, extras=False)
        state["chips"] = [chip("wildcard")]
        result = plan(state, frame, forced_chip=("wildcard", 7), seconds=0)["steps"][0]
        self.assertEqual(result["transfer_in"], [])
        self.assertEqual(counts(result["squad"], frame)[100], 4)
        self.assertEqual(result["free_transfers_next"], 3)

    def test_later_freehit_zero_time_seed_preserves_multiple_price_spreads(self):
        state, frame = overflow(horizon=3)
        state["chips"] = [chip("freehit")]
        for pid in (8, 13):
            state["players"][pid - 1].update(now_cost=60, selling_price=55)
            frame.loc[frame.element.eq(pid), "price"] = 6.0
        result = plan(state, frame, forced_chip=("freehit", 8), seconds=0)
        self.assertEqual([step["bank"] for step in result["steps"]], [4, 4, 4])
        self.assertEqual(result["steps"][2]["squad"], list(range(1, 16)))
        self.assertLessEqual(max(counts(result["steps"][1]["squad"], frame).values()), 3)

    def test_freehit_cannot_use_overflow_when_no_legal_alternative_exists(self):
        state, frame = overflow(horizon=1, extras=False)
        state["chips"] = [chip("freehit")]
        with self.assertRaisesRegex(ValueError, "No legal Free Hit roster"):
            plan(state, frame, forced_chip=("freehit", 7), seconds=0)


if __name__ == "__main__":
    unittest.main()
