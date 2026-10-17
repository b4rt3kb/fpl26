"""Public-account reconciliation and game-rule regressions, using synthetic data."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import tempfile
import unittest

from import_team import money_tenths
from team_state import (StateValidationError, atomic_write_json, build_team_state,
                        chip_windows, free_transfers_next, replay_transfers,
                        selling_price, validate_state)


def sample_account():
    positions = [1] * 2 + [2] * 5 + [3] * 5 + [4] * 3
    costs = [45, 45] + [50] * 5 + [80] * 5 + [85, 85, 90]
    elements = [{"id": i, "web_name": f"Player {i}", "element_type": position,
                 "team": (i - 1) % 5 + 1, "now_cost": cost, "cost_change_start": 0}
                for i, (position, cost) in enumerate(zip(positions, costs), 1)]
    elements.append({"id": 16, "web_name": "Replacement", "element_type": 2,
                     "team": 3, "now_cost": 48, "cost_change_start": 0})
    chips = [{"id": n * 2 + half + 1, "name": name, "start_event": start, "stop_event": stop}
             for n, name in enumerate(["wildcard", "freehit", "bboost", "3xc"])
             for half, start, stop in [(0, 1 if name in ("bboost", "3xc") else 2, 19), (1, 20, 38)]]
    bootstrap = {"elements": elements, "teams": [{"id": t, "short_name": f"T{t}"} for t in range(1, 6)],
                 "chips": chips, "game_settings": {"squad_total_spend": 1000, "max_extra_free_transfers": 4},
                 "game_config": {"settings": {"static_content_url": "https://example.test/2026_27/"}}}
    row = {"event": 1, "points": 10, "total_points": 10, "bank": 0, "value": 1000,
           "event_transfers": 0, "event_transfers_cost": 0, "points_on_bench": 2}
    ordered = [1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 2, 7, 12, 15]
    picks = [{"element": pid, "position": n, "multiplier": 2 if pid == 13 else int(n <= 11),
              "is_captain": pid == 13, "is_vice_captain": pid == 14,
              "element_type": elements[pid - 1]["element_type"]} for n, pid in enumerate(ordered, 1)]
    entry = {"id": 123, "name": "Test FC", "started_event": 1,
             "player_first_name": "PRIVATE", "player_last_name": "PRIVATE", "leagues": {"private": "PRIVATE"}}
    history = {"current": [row], "chips": []}
    payloads = {1: {"active_chip": None, "automatic_subs": [], "entry_history": deepcopy(row), "picks": picks}}
    return bootstrap, entry, history, [], payloads


def add_gw(data, *, chip=None, transfer=False, hit=0):
    bootstrap, entry, history, transfers, payloads = data
    gw = len(history["current"]) + 1
    previous = history["current"][-1]
    row = {**previous, "event": gw, "points": 10, "total_points": previous["total_points"] + 10 - hit,
           "event_transfers": int(transfer), "event_transfers_cost": hit}
    picks = deepcopy(payloads[gw - 1]["picks"])
    if transfer:
        row["bank"] += 2
        transfers.append({"event": gw, "time": f"2026-09-{gw:02}T10:00:00Z",
                          "element_out": 3, "element_in": 16, "element_out_cost": 50, "element_in_cost": 48})
        for pick in picks:
            if pick["element"] == 3:
                pick["element"] = 16
    if chip:
        history["chips"].append({"name": chip, "event": gw, "time": "unused"})
    history["current"].append(row)
    payloads[gw] = {"active_chip": chip, "automatic_subs": [], "entry_history": deepcopy(row), "picks": picks}
    return data


class SaleAndTransferRules(unittest.TestCase):
    def test_price_gains_round_down_and_losses_are_full(self):
        for buy, current, expected in [(75, 78, 76), (75, 76, 75), (75, 75, 75), (75, 72, 72), (40, 45, 42)]:
            with self.subTest(buy=buy, current=current):
                self.assertEqual(selling_price(buy, current), expected)

    def test_money_never_accepts_float_prices(self):
        with self.assertRaises(StateValidationError):
            selling_price(7.5, 80)
        self.assertEqual(money_tenths("0.4"), 4)
        for value in ("0.45", "NaN", "-0.1", "Infinity"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                money_tenths(value)

    def test_chips_preserve_ft_and_ordinary_gws_roll_and_cap(self):
        self.assertEqual(free_transfers_next(3, 12, "wildcard"), 3)
        self.assertEqual(free_transfers_next(3, 8, "freehit"), 3)
        self.assertEqual(free_transfers_next(3, 0, "bboost"), 4)
        self.assertEqual(free_transfers_next(5, 0), 5)
        self.assertEqual(free_transfers_next(3, 4), 1)

    def test_buyback_uses_new_acquisition_cost_and_input_is_unchanged(self):
        original = {1: {"purchase_price": 50}, 2: {"purchase_price": 60}}
        moves = [{"event": 2, "time": "2026-01-02", "element_out": 3, "element_in": 1, "element_in_cost": 54, "element_out_cost": 60},
                 {"event": 2, "time": "2026-01-01", "element_out": 1, "element_in": 3, "element_in_cost": 60, "element_out_cost": 52}]
        result = replay_transfers(original, moves)
        self.assertEqual(result[1]["purchase_price"], 54)
        self.assertFalse(result[1]["inferred_purchase_price"])
        self.assertEqual(original[1]["purchase_price"], 50)
        with self.assertRaisesRegex(StateValidationError, "unowned"):
            replay_transfers(original, moves[:1])


class AccountReconciliation(unittest.TestCase):
    def test_exact_costs_full_history_and_private_fields_removed(self):
        data = add_gw(sample_account(), transfer=True)
        state = build_team_state(*data, free_transfers=3, bank=2)
        self.assertEqual(state["season"], "2026/27")
        self.assertEqual(state["history"][-1]["total_points"], 20)
        self.assertEqual(state["bank"], 2)
        replacement = next(p for p in state["players"] if p["id"] == 16)
        self.assertEqual(replacement["purchase_price"], 48)
        self.assertFalse(replacement["inferred_purchase_price"])
        self.assertNotIn("PRIVATE", json.dumps(state))

    def test_missing_transfer_fails_instead_of_silently_changing_squad(self):
        data = add_gw(sample_account(), transfer=True)
        data[3].clear()
        with self.assertRaisesRegex(StateValidationError, "replay differs"):
            build_team_state(*data, free_transfers=3)

    def test_reconciles_hits_against_cumulative_total(self):
        data = add_gw(sample_account(), transfer=True, hit=4)
        state = build_team_state(*data, free_transfers=1)
        self.assertEqual(state["history"][-1]["net_points"], 6)
        self.assertEqual(state["history"][-1]["total_points"], 16)
        data[2]["current"][-1]["total_points"] = 20
        data[4][2]["entry_history"]["total_points"] = 20
        with self.assertRaisesRegex(StateValidationError, "net points"):
            build_team_state(*data, free_transfers=1)

    def test_bank_ledger_mismatch_and_history_disagreement_rejected(self):
        data = add_gw(sample_account(), transfer=True)
        data[2]["current"][-1]["bank"] = 3
        with self.assertRaisesRegex(StateValidationError, "history and picks"):
            build_team_state(*data, free_transfers=3)
        data[4][2]["entry_history"]["bank"] = 3
        with self.assertRaisesRegex(StateValidationError, "cash ledger"):
            build_team_state(*data, free_transfers=3)

    def test_freehit_reverts_holdings_prices_and_bank(self):
        data = add_gw(sample_account(), chip="freehit", transfer=True)
        state = build_team_state(*data, free_transfers=3)
        self.assertIn(3, {p["id"] for p in state["players"]})
        self.assertNotIn(16, {p["id"] for p in state["players"]})
        self.assertEqual(state["bank"], 0)
        self.assertEqual(state["lineup"]["source_event"], 1)
        self.assertEqual(state["chips_used"], [{"name": "freehit", "event": 2}])

    def test_freehit_without_public_transfer_records_preserves_permanent_squad(self):
        data = add_gw(sample_account(), chip="freehit", transfer=True)
        data[3].clear()
        state = build_team_state(*data, free_transfers=3)
        self.assertIn(3, {p["id"] for p in state["players"]})
        self.assertEqual(state["history"][-1]["reconciliation"], "free_hit_picks_temporary_no_public_transfer_ledger")

    def test_week_after_freehit_reconciles_to_original_bank(self):
        data = add_gw(sample_account(), chip="freehit", transfer=True)
        add_gw(data)
        data[4][3]["picks"] = deepcopy(data[4][1]["picks"])
        data[2]["current"][-1]["bank"] = 0
        data[4][3]["entry_history"]["bank"] = 0
        state = build_team_state(*data, free_transfers=3)
        self.assertEqual(state["bank"], 0)
        self.assertEqual(state["lineup"]["source_event"], 3)

    def test_wildcard_keeps_new_squad_and_marks_only_first_chip_used(self):
        data = add_gw(sample_account(), chip="wildcard", transfer=True)
        state = build_team_state(*data, free_transfers=3)
        self.assertIn(16, {p["id"] for p in state["players"]})
        windows = [c for c in state["chips"] if c["name"] == "wildcard"]
        self.assertEqual([w["status"] for w in windows], ["used", "upcoming"])

    def test_late_join_requires_explicit_purchase_prices(self):
        data = sample_account()
        data[1]["started_event"] = 2
        data[2]["current"][0]["event"] = 2
        data[4][2] = data[4].pop(1)
        data[4][2]["entry_history"]["event"] = 2
        with self.assertRaisesRegex(StateValidationError, "unknown"):
            build_team_state(*data, free_transfers=1)
        overrides = {p["id"]: p["now_cost"] for p in data[0]["elements"]}
        state = build_team_state(*data, free_transfers=1, purchase_prices=overrides)
        self.assertTrue(all(not p["inferred_purchase_price"] for p in state["players"]))

    def test_bad_initial_costs_and_nonconsecutive_history_are_rejected(self):
        data = sample_account()
        data[0]["elements"][0]["cost_change_start"] = 1
        with self.assertRaisesRegex(StateValidationError, "initial budget"):
            build_team_state(*data, free_transfers=1)
        data = add_gw(sample_account())
        data[2]["current"][-1]["event"] = 3
        with self.assertRaisesRegex(StateValidationError, "consecutive"):
            build_team_state(*data, free_transfers=1)

    def test_user_bank_override_is_explicit_and_does_not_invent_transfers(self):
        state = build_team_state(*sample_account(), free_transfers=3, bank=4)
        self.assertEqual(state["bank"], 4)
        self.assertEqual(state["provenance"]["observed_deadline_bank"], 0)
        self.assertTrue(any("differs" in w for w in state["warnings"]))
        self.assertFalse(state["provenance"]["post_deadline_changes_visible"])

    def test_quota_duplicate_and_bad_sale_price_are_rejected(self):
        valid = build_team_state(*sample_account(), free_transfers=3)
        for mutate in (lambda s: s["players"][0].update(id=s["players"][1]["id"]),
                       lambda s: s["players"][0].update(position="MID"),
                       lambda s: s["players"][0].update(selling_price=99)):
            state = deepcopy(valid)
            mutate(state)
            with self.assertRaises(StateValidationError):
                validate_state(state)

    def test_club_overflow_requires_explicit_exception(self):
        data = sample_account()
        data[0]["elements"][1]["team"] = 1
        with self.assertRaisesRegex(StateValidationError, "exceeds 3"):
            build_team_state(*data, free_transfers=3)
        state = build_team_state(*data, free_transfers=3, allow_club_overflow=True)
        self.assertTrue(state["provenance"]["allow_club_overflow"])

    def test_chip_half_expiration_and_consumption(self):
        bootstrap = sample_account()[0]
        result = chip_windows(bootstrap, [{"name": "wildcard", "event": 5}], 20)
        self.assertEqual([c["status"] for c in result if c["name"] == "wildcard"], ["used", "available"])
        self.assertEqual([c["status"] for c in result if c["name"] == "bboost"], ["expired", "available"])
        with self.assertRaisesRegex(StateValidationError, "twice"):
            chip_windows(bootstrap, [{"name": "wildcard", "event": 5}, {"name": "wildcard", "event": 8}], 20)

    def test_freehit_19_blocks_freehit_20_but_not_21(self):
        bootstrap = sample_account()[0]
        used = [{"name": "freehit", "event": 19}]
        gw20 = chip_windows(bootstrap, used, 20)
        gw21 = chip_windows(bootstrap, used, 21)
        self.assertEqual([c["status"] for c in gw20 if c["name"] == "freehit"], ["used", "blocked"])
        self.assertEqual([c["status"] for c in gw21 if c["name"] == "freehit"], ["used", "available"])

    def test_atomic_output_stays_private_and_failed_write_preserves_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            atomic_write_json(path, {"ok": 1})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(ValueError):
                atomic_write_json(path, {"bad": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"ok": 1})
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
