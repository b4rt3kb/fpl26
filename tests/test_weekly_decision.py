from copy import deepcopy
import json
import hashlib
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from team_state import StateValidationError
from weekly_decision import (reconstructed_free_transfers, fetch_public_state,
                             next_deadline, validate_forecast_for_decision,
                             validate_current_inputs, validate_fixture_schedule,
                             validate_forecast_sources, publish_decision, render_public_decision)
from test_team_state import sample_account


class WeeklyDecisionTests(unittest.TestCase):
    def test_ft_replay_starts_with_one_after_unlimited_gw1_and_preserves_on_chips(self):
        history = {"current": [{"event": gw, "event_transfers": count, "event_transfers_cost": 0}
                   for gw, count in enumerate([20, 0, 2, 0, 10, 0], 1)],
                   "chips": [{"event": 5, "name": "wildcard"}]}
        free, ledger = reconstructed_free_transfers(history, 6)
        self.assertEqual(free, 3)
        self.assertEqual(ledger[0]["free_before"], 1)
        self.assertEqual(ledger[3]["free_next"], 2)
        history["chips"][0]["name"] = "freehit"
        self.assertEqual(reconstructed_free_transfers(history, 6)[0], 3)

    def test_ft_holes_and_inconsistent_paid_transfers_fail(self):
        history = {"current": [{"event": 1, "event_transfers": 0, "event_transfers_cost": 0},
                               {"event": 2, "event_transfers": 3, "event_transfers_cost": 8}], "chips": []}
        self.assertEqual(reconstructed_free_transfers(history, 2)[0], 1)
        history["current"][1]["event_transfers_cost"] = 0
        with self.assertRaisesRegex(StateValidationError, "hit cost"):
            reconstructed_free_transfers(history, 2)
        history["current"][1]["event"] = 3
        with self.assertRaisesRegex(StateValidationError, "complete history"):
            reconstructed_free_transfers(history, 3)

    def test_fresh_public_reconstruction_requires_no_manual_bank_or_ft(self):
        bootstrap, entry, history, transfers, picks = sample_account()
        bootstrap["events"] = [{"id": 1, "deadline_time": "2026-10-10T10:00:00Z", "finished": True, "data_checked": True},
                               {"id": 2, "deadline_time": "2026-10-20T10:00:00Z", "finished": False}]
        payload = {"bootstrap-static/": bootstrap, "entry/123/": entry,
                   "entry/123/history/": history, "entry/123/transfers/": transfers,
                   "entry/123/event/1/picks/": picks[1]}
        class Client:
            records = []
            def get_json(self, path, **kwargs):
                assert kwargs == {"max_age_seconds": 0, "force": True}
                return deepcopy(payload[path])
        state, _, event = fetch_public_state(123, client=Client(), now=pd.Timestamp("2026-10-17T08:00:00Z"),
                                             expected_ft=1, expected_gw=2)
        self.assertEqual(state["free_transfers"], 1)
        self.assertEqual(state["bank"], history["current"][0]["bank"])
        self.assertEqual(event["id"], 2)
        self.assertNotIn("PRIVATE", json.dumps(state))
        self.assertFalse(state["provenance"]["unpublished_edits_known"])
        self.assertNotEqual(state["provenance"]["free_transfers_source"], "user_confirmed")
        with self.assertRaisesRegex(StateValidationError, "anchor failed"):
            fetch_public_state(123, client=Client(), now=pd.Timestamp("2026-10-17T08:00:00Z"), expected_ft=3, expected_gw=2)

    def test_deadline_selection_never_returns_an_expired_week(self):
        bootstrap = {"events": [{"id": 7, "deadline_time": "2026-10-17T10:00:00Z"},
                                {"id": 8, "deadline_time": "2026-10-24T10:00:00Z"}]}
        self.assertEqual(next_deadline(bootstrap, pd.Timestamp("2026-10-17T10:00:00Z"))["id"], 8)

    def test_forecast_must_match_live_event_and_be_before_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"forecast.csv"
            pd.DataFrame({"element": [1], "gw": [7], "pred": [3]}).to_csv(path, index=False)
            meta = {"generated_utc": "2026-10-17T08:00:00Z", "gameweeks": [7], "season": "2026-27"}
            path.with_suffix(".json").write_text(json.dumps(meta))
            state = {"next_event": 7, "season": "2026/27", "players": [{"id": 1}]}
            validate_forecast_for_decision(path, state, "2026-10-17T10:00:00Z", pd.Timestamp("2026-10-17T09:00:00Z"))
            with self.assertRaisesRegex(StateValidationError, "Deadline passed"):
                validate_forecast_for_decision(path, state, "2026-10-17T10:00:00Z", pd.Timestamp("2026-10-17T10:00:00Z"))
            state["next_event"] = 8
            with self.assertRaisesRegex(StateValidationError, "different next"):
                validate_forecast_for_decision(path, state, "2026-10-18T10:00:00Z", pd.Timestamp("2026-10-17T09:00:00Z"))

    def test_changed_candidate_injury_price_or_deadline_rejected_but_counters_ignored(self):
        stored = {"elements": [{"id": 1, "status": "a", "news": "", "news_added": None,
                                "chance_of_playing_next_round": None, "now_cost": 50,
                                "team": 1, "can_select": True, "element_type": 3}],
                  "events": [{"id": 7, "deadline_time": "2026-10-17T10:00:00Z"}]}
        fresh = deepcopy(stored)
        fresh["elements"][0]["transfers_in"] = 5000
        validate_current_inputs(fresh, stored, [7])
        for field, value in (("status", "i"), ("news", "Injury"), ("chance_of_playing_next_round", 25),
                             ("now_cost", 51), ("can_select", False), ("team", 2)):
            changed = deepcopy(fresh)
            changed["elements"][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(StateValidationError, "refresh data"):
                validate_current_inputs(changed, stored, [7])
        fresh["events"][0]["deadline_time"] = "2026-10-17T11:00:00Z"
        with self.assertRaisesRegex(StateValidationError, "deadline changed"):
            validate_current_inputs(fresh, stored, [7])

    def test_schedule_changes_fail_but_scores_do_not_invalidate(self):
        fixtures = [{"id": 1, "event": 7, "team_h": 1, "team_a": 2,
                     "kickoff_time": "2026-10-17T12:30:00Z", "finished": False}]
        fresh = deepcopy(fixtures)
        fresh[0].update(finished=True, team_h_score=2, kickoff_time="2026-10-17T12:30:00+00:00")
        validate_fixture_schedule(fresh, fixtures)
        fresh[0]["event"] = 8
        with self.assertRaisesRegex(StateValidationError, "Fixture schedule changed"):
            validate_fixture_schedule(fresh, fixtures)

    def test_forecast_source_hash_is_verified_before_live_comparison(self):
        bootstrap = {"elements": [{"id": 1, "status": "a"}],
                     "events": [{"id": 7, "deadline_time": "2026-10-17T10:00:00Z"}]}
        fixtures = [{"id": 1, "event": 7, "team_h": 1, "team_a": 2,
                     "kickoff_time": "2026-10-17T12:30:00Z"}]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory/"bootstrap.json").write_text(json.dumps(bootstrap))
            pd.DataFrame(fixtures).to_csv(directory/"fixtures_2627.csv", index=False)
            files = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in directory.iterdir()}
            metadata = {"gameweeks": [7], "data_manifest": {"files": files}}
            validate_forecast_sources(directory, metadata, bootstrap, fixtures)
            (directory/"bootstrap.json").write_text(json.dumps(bootstrap)+" ")
            with self.assertRaisesRegex(StateValidationError, "source hash differs"):
                validate_forecast_sources(directory, metadata, bootstrap, fixtures)

    def test_public_bundle_contains_only_minimal_recommendation_and_gate_fields(self):
        positions = ["GK"]*2+["DEF"]*5+["MID"]*5+["FWD"]*3
        players = [{"id": i, "name": f"Player {i}", "pos": pos, "team": i % 20+1, "price": 55}
                   for i, pos in enumerate(positions, 1)]
        players.append({"id": 16, "name": "Incoming", "pos": "DEF", "team": 20, "price": 54})
        state = {"entry_id": 999, "team_name": "SECRET_PROFILE", "history": ["SECRET_HISTORY"],
                 "players": [{"id": p["id"], "selling_price": 52} for p in players[:15]],
                 "bank": 10, "free_transfers": 3, "as_of_event": 6,
                 "imported_at": "2026-10-17T09:00:00Z"}
        step = {"gw": 7, "chip": None, "transfer_out": [3], "transfer_in": [16],
                "temporary_out": [], "temporary_in": [], "bank": 8, "hit_cost": 0,
                "free_transfers_before": 3, "free_transfers_next": 3,
                "starting_xi": [1, 16, 4, 5, 8, 9, 10, 11, 13, 14, 15], "bench": [2, 6, 7, 12],
                "captain": 13, "vice_captain": 8, "net_predicted_points": 54.0}
        result = {"entry_id": 999, "players": players, "steps": [step],
                  "generated_utc": "2026-10-17T09:01:00Z", "deadline": "2026-10-17T10:00:00Z"}
        metadata = {"season": "2026-27", "gameweeks": [7, 8], "model": "v0.5-ensemble",
                    "generated_utc": "2026-10-17T08:50:00Z"}
        with tempfile.TemporaryDirectory() as tmp:
            target = publish_decision(tmp, result, state, metadata, now=pd.Timestamp("2026-10-17T09:02:00Z"))
            self.assertEqual(target.name, "GW7")
            self.assertEqual({p.name for p in target.iterdir()}, {"decision.json", "decision.html", "lineup.csv", "README.md"})
            report = json.loads((target/"decision.json").read_text())
            self.assertEqual(report["deadline"], result["deadline"])
            self.assertEqual(report["gameweeks"], [7, 8])
            self.assertEqual(report["deadline_local"], "2026-10-17 12:00 CEST")
            self.assertEqual(report["transfers"][0]["out"]["selling_price"], 5.2)
            self.assertEqual(report["transfers"][0]["in"]["price"], 5.4)
            self.assertEqual([p["order"] for p in report["bench"]], [0, 1, 2, 3])
            position_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
            order = [position_order[p["position"]] for p in report["starting_xi"]]
            self.assertEqual(order, sorted(order))
            for path in target.iterdir():
                self.assertNotIn("SECRET_", path.read_text())
                self.assertNotIn("entry_id", path.read_text())
            rows = pd.read_csv(target/"lineup.csv")
            self.assertEqual(len(rows), 15)
            self.assertEqual(rows.element.nunique(), 15)
            self.assertEqual(int(rows.is_captain.sum()), 1)
            self.assertEqual(int(rows.is_vice_captain.sum()), 1)
            self.assertIn("Captain", (target/"README.md").read_text())
            original_data = deepcopy(report)
            render_public_decision(tmp, report)
            self.assertEqual(report, original_data)
            self.assertEqual(json.loads((target/"decision.json").read_text()), original_data)
            self.assertIn('lang="en"', (target/"decision.html").read_text())
            with self.assertRaisesRegex(StateValidationError, "Deadline passed"):
                publish_decision(tmp, result, state, metadata, now=pd.Timestamp("2026-10-17T10:00:00Z"))


if __name__ == "__main__":
    unittest.main()
