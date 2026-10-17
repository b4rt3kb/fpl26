import copy
import unittest

import pandas as pd

from strategy_backtest import (aggregate_outcomes, apply_decision, initial_state_from_forecast,
                               no_transfer_decision, realized_score, replay_strategy,
                               _forecast_metadata)


def fixture(gw=7):
    positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3 + ["MID", "DEF", "FWD"]
    frame = pd.DataFrame([{"element": i, "gw": gw, "pos": pos, "price": 5.0,
                           "pred": 2.0, "team_key": i, "web_name": f"P{i}", "can_select": True}
                          for i, pos in enumerate(positions, 1)])
    players = [{"id": i, "purchase_price": 50, "now_cost": 50, "selling_price": 50,
                "position": positions[i - 1], "team": i} for i in range(1, 16)]
    state = {"next_event": gw, "players": players, "bank": 4, "free_transfers": 3,
             "chips": [], "chips_used": [], "rules": {"max_free_transfers": 5, "transfer_hit": 4, "club_limit": 3}}
    decision = {"gw": gw, "chip": None, "squad": list(range(1, 16)), "permanent_squad": list(range(1, 16)),
                "starting_xi": [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15],
                "bench": [2, 12, 6, 7], "captain": 13, "vice_captain": 8}
    actual = {i: {"minutes": 90, "total_points": 2} for i in range(1, 19)}
    return state, frame, decision, actual


def add_chip(state, name, gw=7, stop=19, number=1):
    state["chips"].append({"id": number, "name": name, "start_event": gw, "stop_event": stop,
                           "status": "available", "used_event": None})


def replace(decision, outgoing, incoming):
    result = copy.deepcopy(decision)
    for key in ("squad", "permanent_squad", "starting_xi", "bench"):
        result[key] = [incoming if i == outgoing else i for i in result[key]]
    for key in ("captain", "vice_captain"):
        if result[key] == outgoing:
            result[key] = incoming
    return result


class RealizedScoringTests(unittest.TestCase):
    def test_no_show_uses_legal_defender_over_higher_priority_midfielder(self):
        _, frame, decision, actual = fixture()
        actual[3] = {"minutes": 0, "total_points": 0}
        actual[12]["total_points"] = 20
        actual[6]["total_points"] = 7
        score = realized_score(decision, _forecast_metadata(frame, 7), actual)
        self.assertEqual(score["automatic_substitutions"], [{"out": 3, "in": 6}])
        self.assertEqual(score["gross_points"], 29)
        self.assertNotIn(12, score["scoring_players"])

    def test_multiple_autosubs_preserve_priority_without_chasing_actual_points(self):
        _, frame, decision, actual = fixture()
        for pid in (3, 8):
            actual[pid] = {"minutes": 0, "total_points": 0}
        actual[12]["total_points"] = -2
        actual[7]["total_points"] = 15
        score = realized_score(decision, _forecast_metadata(frame, 7), actual)
        self.assertEqual(score["automatic_substitutions"], [{"out": 8, "in": 12}, {"out": 3, "in": 6}])
        self.assertNotIn(7, score["scoring_players"])

    def test_keeper_and_captain_fallback_and_triple_captain(self):
        _, frame, decision, actual = fixture()
        actual[1] = {"minutes": 0, "total_points": 0}
        actual[13] = {"minutes": 0, "total_points": 0}
        actual[8]["total_points"] = 7
        decision["chip"] = "3xc"
        score = realized_score(decision, _forecast_metadata(frame, 7), actual)
        self.assertIn({"out": 1, "in": 2}, score["automatic_substitutions"])
        self.assertEqual(score["captain_applied"], 8)
        self.assertEqual(score["captain_bonus"], 14)
        self.assertEqual(score["captain_multiplier"], 3)
        actual[8] = {"minutes": 0, "total_points": 0}
        score = realized_score(decision, _forecast_metadata(frame, 7), actual)
        self.assertIsNone(score["captain_applied"])
        self.assertEqual(score["captain_bonus"], 0)

    def test_dgw_aggregates_before_autosubs_and_blank_scores_zero(self):
        _, frame, decision, _ = fixture()
        rows = [{"element": 13, "gw": 7, "fixture": 1, "minutes": 0, "total_points": 0},
                {"element": 13, "gw": 7, "fixture": 2, "minutes": 1, "total_points": 1},
                {"element": 12, "gw": 7, "fixture": 1, "minutes": 90, "total_points": 9}]
        aggregated = aggregate_outcomes(pd.DataFrame(rows))
        actual = {pid: value for (gw, pid), value in aggregated.items() if gw == 7}
        score = realized_score(decision, _forecast_metadata(frame, 7), actual)
        self.assertEqual(score["captain_applied"], 13)
        self.assertEqual(score["captain_bonus"], 1)
        self.assertEqual(score["gross_points"], 11)
        self.assertFalse(any(s["out"] == 13 for s in score["automatic_substitutions"]))

    def test_bench_boost_counts_bench_once_without_autosubs(self):
        _, frame, decision, actual = fixture()
        decision["chip"] = "bboost"
        actual[13] = {"minutes": 0, "total_points": 0}
        actual[8]["total_points"] = 5
        score = realized_score(decision, _forecast_metadata(frame, 7), actual)
        self.assertEqual(score["gross_points"], 36)
        self.assertEqual(score["automatic_substitutions"], [])
        self.assertEqual(score["bench_points_unused"], 0)

    def test_zero_minute_card_prevents_autosub_but_not_vice_fallback(self):
        _, frame, decision, actual = fixture()
        actual[13] = {"minutes": 0, "total_points": -1, "yellow_cards": 1}
        score = realized_score(decision, _forecast_metadata(frame, 7), actual)
        self.assertEqual(score["automatic_substitutions"], [])
        self.assertEqual(score["captain_applied"], 8)
        self.assertEqual(score["gross_points"], 21)

    def test_duplicate_outcome_rejected(self):
        row = {"element": 1, "gw": 1, "fixture": 1, "minutes": 90, "total_points": 2}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_outcomes(pd.DataFrame([row, row]))
        with self.assertRaisesRegex(ValueError, "fixture IDs"):
            aggregate_outcomes(pd.DataFrame([row, row]).drop(columns="fixture"))

    def test_illegal_xi_and_duplicate_captain_are_rejected(self):
        _, frame, decision, actual = fixture()
        meta = _forecast_metadata(frame, 7)
        decision["vice_captain"] = decision["captain"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            realized_score(decision, meta, actual)
        decision["vice_captain"] = 8
        decision["starting_xi"].remove(3)
        decision["starting_xi"].append(12)
        with self.assertRaisesRegex(ValueError, "formation"):
            realized_score(decision, meta, actual)


class StrategyLedgerTests(unittest.TestCase):
    def test_hits_and_free_transfer_cap(self):
        state, frame, decision, _ = fixture()
        state["free_transfers"] = 1
        changed = replace(replace(replace(decision, 8, 16), 3, 17), 13, 18)
        next_state, ledger = apply_decision(state, frame, changed)
        self.assertEqual(ledger["hit_cost"], 8)
        self.assertEqual(next_state["free_transfers"], 1)
        state["free_transfers"] = 5
        next_state, _ = apply_decision(state, frame, decision)
        self.assertEqual(next_state["free_transfers"], 5)

    def test_half_profit_sale_and_buyback_reset_purchase_price(self):
        state, frame, decision, _ = fixture()
        state["bank"] = 0
        frame.loc[frame.element.eq(8), "price"] = 6.1
        frame.loc[frame.element.eq(16), "price"] = 5.5
        changed = replace(decision, 8, 16)
        next_state, ledger = apply_decision(state, frame, changed)
        self.assertEqual(ledger["sale_proceeds"], {8: 55})
        self.assertEqual(next_state["bank"], 0)
        frame["gw"] = 8
        frame.loc[frame.element.eq(8), "price"] = 5.5
        changed["gw"] = 8
        back = replace(changed, 16, 8)
        final, _ = apply_decision(next_state, frame, back)
        self.assertEqual(next(p for p in final["players"] if p["id"] == 8)["purchase_price"], 55)

    def test_losses_are_fully_borne_and_unaffordable_purchase_rejected(self):
        state, frame, decision, _ = fixture()
        state["bank"] = 0
        frame.loc[frame.element.eq(8), "price"] = 4.5
        with self.assertRaisesRegex(ValueError, "budget"):
            apply_decision(state, frame, replace(decision, 8, 16))
        frame.loc[frame.element.eq(16), "price"] = 4.4
        _, ledger = apply_decision(state, frame, replace(decision, 8, 16))
        self.assertEqual(ledger["sale_proceeds"], {8: 45})
        self.assertEqual(ledger["bank_after"], 1)

    def test_freehit_reverts_holdings_purchase_prices_bank_and_ft(self):
        state, frame, decision, _ = fixture()
        add_chip(state, "freehit")
        frame.loc[frame.element.eq(8), "price"] = 6.0
        changed = replace(decision, 8, 16)
        changed["chip"] = "freehit"
        changed["permanent_squad"] = decision["permanent_squad"]
        next_state, ledger = apply_decision(state, frame, changed)
        self.assertEqual({p["id"] for p in next_state["players"]}, set(range(1, 16)))
        self.assertEqual(next(p for p in next_state["players"] if p["id"] == 8)["purchase_price"], 50)
        self.assertEqual(next_state["bank"], 4)
        self.assertEqual(next_state["free_transfers"], 3)
        self.assertEqual(ledger["temporary_bank"], 9)
        self.assertEqual(ledger["hit_cost"], 0)
        self.assertEqual(next_state["chips"][0]["used_event"], 7)

    def test_freehit_budget_uses_sale_proceeds_and_retained_holdings_not_rebuy(self):
        state, frame, decision, _ = fixture()
        add_chip(state, "freehit")
        state["bank"] = 0
        frame.loc[frame.element.eq(8), "price"] = 6.0
        decision["chip"] = "freehit"
        _, ledger = apply_decision(state, frame, decision)
        self.assertEqual(ledger["temporary_bank"], 0)
        frame.loc[frame.element.eq(16), "price"] = 6.0
        changed = replace(decision, 8, 16)
        changed["permanent_squad"] = decision["permanent_squad"]
        with self.assertRaisesRegex(ValueError, "Free Hit exceeds"):
            apply_decision(state, frame, changed)

    def test_wildcard_permanent_and_preserves_free_transfers(self):
        state, frame, decision, _ = fixture()
        add_chip(state, "wildcard")
        changed = replace(decision, 8, 16)
        changed["chip"] = "wildcard"
        next_state, ledger = apply_decision(state, frame, changed)
        self.assertIn(16, [p["id"] for p in next_state["players"]])
        self.assertEqual(next_state["free_transfers"], 3)
        self.assertEqual(ledger["hit_cost"], 0)

    def test_chip_window_expiry_reuse_and_consecutive_freehit_rejected(self):
        state, frame, decision, _ = fixture()
        add_chip(state, "3xc", stop=6)
        decision["chip"] = "3xc"
        with self.assertRaisesRegex(ValueError, "unavailable"):
            apply_decision(state, frame, decision)
        state["chips"][0].update(stop_event=7, status="used", used_event=6)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            apply_decision(state, frame, decision)
        add_chip(state, "freehit", number=2)
        state["chips_used"] = [{"name": "freehit", "event": 6}]
        decision["chip"] = "freehit"
        with self.assertRaisesRegex(ValueError, "Consecutive"):
            apply_decision(state, frame, decision)

    def test_acquisition_selectability_and_club_limits(self):
        state, frame, decision, _ = fixture()
        frame.loc[frame.element.eq(16), "can_select"] = False
        with self.assertRaisesRegex(ValueError, "unselectable"):
            apply_decision(state, frame, replace(decision, 8, 16))
        frame["can_select"] = True
        frame.loc[frame.element.isin([3, 4, 5, 16]), "team_key"] = 1
        with self.assertRaisesRegex(ValueError, "Club limit"):
            apply_decision(state, frame, replace(decision, 8, 16))


class EndToEndReplayTests(unittest.TestCase):
    def test_full_38_gw_no_transfer_baseline_and_gw1_ft(self):
        state, frame, _, _ = fixture(gw=1)
        state["free_transfers"] = 1
        forecasts = {gw: frame.assign(gw=gw) for gw in range(1, 39)}
        outcomes = pd.DataFrame([{"element": pid, "gw": gw, "minutes": 90, "total_points": 2}
                                 for gw in range(1, 39) for pid in range(1, 19)])
        result = replay_strategy(state, forecasts, outcomes, policy="no_transfer")
        self.assertTrue(result["full_season"])
        self.assertEqual(result["net_points"], 38 * 24)
        self.assertEqual(result["weeks"][0]["ledger"]["free_transfers_next"], 1)
        self.assertEqual(result["weeks"][1]["ledger"]["free_transfers_next"], 2)
        self.assertEqual(result["final_state"]["free_transfers"], 5)
        self.assertEqual(result["final_state"]["bank"], state["bank"])
        self.assertEqual(state["next_event"], 1)

    def test_replay_decisions_independent_of_target_outcomes(self):
        state, frame, _, _ = fixture()
        forecasts = {7: frame, 8: frame.assign(gw=8)}
        outcomes = pd.DataFrame([{"element": i, "gw": gw, "minutes": 90, "total_points": 2}
                                 for i in range(1, 19) for gw in (7, 8)])
        a = replay_strategy(state, forecasts, outcomes, policy="no_transfer")
        outcomes["total_points"] = 12345
        outcomes["minutes"] = 0
        b = replay_strategy(state, forecasts, outcomes, policy="no_transfer")
        self.assertEqual([r["decision"] for r in a["weeks"]], [r["decision"] for r in b["weeks"]])

    def test_policy_never_receives_target_columns_in_forecast_csv(self):
        state, frame, _, _ = fixture()
        frame = frame.assign(actual=999, actual_minutes=90, total_points=999, minutes=90)
        observed = []
        def policy(s, f):
            observed.extend(f.columns)
            return no_transfer_decision(s, f)
        empty = pd.DataFrame(columns=["element", "gw", "minutes", "total_points"])
        replay_strategy(state, {7: frame}, empty, decision_fn=policy)
        self.assertFalse(set(observed) & {"actual", "actual_minutes", "total_points", "minutes"})

    def test_explicit_seasonal_free_transfer_override(self):
        state, frame, _, _ = fixture(gw=16)
        state["free_transfers"] = 1
        state["rules"]["free_transfer_overrides"] = {"16": 5}
        outcome = pd.DataFrame(columns=["element", "gw", "minutes", "total_points"])
        result = replay_strategy(state, {16: frame}, outcome, policy="no_transfer")
        self.assertEqual(result["weeks"][0]["ledger"]["free_transfers_before"], 5)
        state["next_event"] = 15
        frame["gw"] = 15
        result = replay_strategy(state, {15: frame, 16: frame.assign(gw=16)}, outcome, policy="no_transfer")
        self.assertEqual(result["weeks"][0]["ledger"]["free_transfers_next"], 5)
        self.assertEqual(result["weeks"][1]["ledger"]["free_transfers_before"], 5)

    def test_planner_integration_executes_first_step_only(self):
        state, frame, _, _ = fixture()
        frame.loc[frame.element.gt(15), "pred"] = 20
        outcomes = pd.DataFrame([{"element": i, "gw": 7, "minutes": 90, "total_points": 2} for i in range(1, 19)])
        result = replay_strategy(state, {7: frame}, outcomes, policy="no_chips", planner_options={"seconds": 2})
        self.assertEqual(result["permanent_transfers"], 3)
        self.assertEqual(result["hit_points"], 0)
        self.assertEqual(result["net_points"], 24)
        self.assertIsNotNone(result["weeks"][0]["solver"])

    def test_initial_squad_uses_forecast_only_and_respects_budget(self):
        _, frame, _, _ = fixture(gw=1)
        state = initial_state_from_forecast(frame, budget=800)
        self.assertEqual(len(state["players"]), 15)
        self.assertEqual(state["bank"], 50)
        self.assertEqual(state["next_event"], 1)

    def test_malformed_origin_and_nonfrozen_price_rejected(self):
        state, frame, _, _ = fixture()
        empty = pd.DataFrame(columns=["element", "gw", "minutes", "total_points"])
        with self.assertRaisesRegex(ValueError, "consecutive"):
            replay_strategy(state, {7: frame, 9: frame.assign(gw=9)}, empty, policy="no_transfer")
        future = frame.assign(gw=8, price=5.1)
        with self.assertRaisesRegex(ValueError, "frozen"):
            no_transfer_decision(state, pd.concat([frame, future]))


if __name__ == "__main__":
    unittest.main()
