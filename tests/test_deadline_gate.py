"""Pure/offline deadline checks; no heavy model imports or live requests."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from deadline_gate import decide, github_outputs, load_decision_metadata, load_forecast_metadata, main


DEADLINE = datetime(2026, 10, 17, 10, tzinfo=timezone.utc)


def bootstrap():
    return {"events": [{"id": gw, "deadline_time": (DEADLINE + timedelta(days=7*(gw-7))).isoformat()}
                       for gw in range(1, 39)]}


def metadata(when, event=7, season="2026-27"):
    return {"generated_utc": when.isoformat(), "gameweeks": list(range(event, min(event+6, 39))), "season": season}


class DeadlineGateTests(unittest.TestCase):
    def test_midweek_skips_even_without_forecast(self):
        result = decide(bootstrap(), now=DEADLINE-timedelta(days=3))
        self.assertFalse(result["due"])
        self.assertEqual(result["reason"], "outside_deadline_windows")

    def test_initial_window_runs_once_and_is_idempotent(self):
        now = DEADLINE-timedelta(hours=23)
        self.assertTrue(decide(bootstrap(), now=now)["due"])
        result = decide(bootstrap(), [metadata(now)], now=now+timedelta(hours=1))
        self.assertFalse(result["due"])
        self.assertEqual(result["reason"], "initial_24h_already_generated")

    def test_final_window_refreshes_initial_once(self):
        initial = metadata(DEADLINE-timedelta(hours=20))
        now = DEADLINE-timedelta(hours=1)
        self.assertEqual(decide(bootstrap(), [initial], now=now)["reason"], "final_2h")
        final = metadata(now)
        self.assertFalse(decide(bootstrap(), [initial, final], now=now+timedelta(minutes=10))["due"])

    def test_exact_threshold_counts_as_completed(self):
        for hours in (24, 2):
            now = DEADLINE-timedelta(hours=hours)
            self.assertTrue(decide(bootstrap(), now=now)["due"])
            self.assertFalse(decide(bootstrap(), [metadata(now)], now=now)["due"])

    def test_late_first_run_does_one_final_generation(self):
        result = decide(bootstrap(), now=DEADLINE-timedelta(minutes=30))
        self.assertEqual(result["reason"], "final_2h")

    def test_manual_dispatch_forces_midweek_but_never_past_season(self):
        now = DEADLINE-timedelta(days=3)
        self.assertTrue(decide(bootstrap(), [metadata(now)], now=now, force=True)["due"])
        end = DEADLINE+timedelta(days=7*31)
        result = decide(bootstrap(), now=end, force=True)
        self.assertEqual(result, {"due": False, "event": 0, "horizon": 0,
                                 "reason": "season_finished", "deadline": None, "latest_generation": None})

    def test_end_of_season_horizon_is_capped(self):
        for event, expected in ((36, 3), (37, 2), (38, 1)):
            now = DEADLINE+timedelta(days=7*(event-7))-timedelta(hours=1)
            result = decide(bootstrap(), now=now)
            self.assertEqual((result["event"], result["horizon"]), (event, expected))

    def test_postponed_deadline_reopens_correct_windows(self):
        data = bootstrap()
        data["events"][6]["deadline_time"] = (DEADLINE+timedelta(days=2)).isoformat()
        old = metadata(DEADLINE-timedelta(hours=1))
        self.assertFalse(decide(data, [old], now=DEADLINE)["due"])
        result = decide(data, [old], now=DEADLINE+timedelta(days=1, hours=1))
        self.assertTrue(result["due"])
        self.assertEqual(result["reason"], "initial_24h")

    def test_past_origin_horizon_does_not_suppress_fresh_next_origin(self):
        now = DEADLINE-timedelta(hours=1)
        self.assertTrue(decide(bootstrap(), [metadata(now, event=6)], now=now)["due"])

    def test_other_season_does_not_suppress_current_forecast(self):
        now = DEADLINE-timedelta(hours=1)
        self.assertTrue(decide(bootstrap(), [metadata(now, season="2025-26")], now=now)["due"])

    def test_latest_of_multiple_generations_is_used(self):
        now = DEADLINE-timedelta(minutes=30)
        records = [metadata(DEADLINE-timedelta(hours=1)), metadata(DEADLINE-timedelta(hours=20))]
        self.assertFalse(decide(bootstrap(), records, now=now)["due"])

    def test_invalid_and_future_metadata_fail_closed(self):
        now = DEADLINE-timedelta(hours=1)
        for invalid in ({}, {**metadata(now), "generated_utc": "invalid"},
                        metadata(now+timedelta(seconds=1)), {**metadata(now), "gameweeks": [7, 9]},
                        {**metadata(now), "gameweeks": [True]}, {**metadata(now), "season": "2026\ninject=true"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                decide(bootstrap(), [invalid], now=now)

    def test_invalid_bootstrap_and_output_injection_are_rejected(self):
        for data in ({}, {"events": []}, {"events": [{"id": "7\ndue=true", "deadline_time": DEADLINE.isoformat()}]}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                decide(data, now=DEADLINE-timedelta(hours=1))
        with self.assertRaises(ValueError):
            github_outputs({"due": "true\nevent=1", "event": 7, "horizon": 6})

    def test_outputs_contain_only_validated_booleans_and_integers(self):
        result = decide(bootstrap(), now=DEADLINE-timedelta(hours=1))
        self.assertEqual(github_outputs(result), "due=true\nevent=7\nhorizon=6\n")

    def test_loader_excludes_personal_files_and_invalid_json_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            record = metadata(DEADLINE-timedelta(hours=1))
            (directory/"all_players_gw7.json").write_text(json.dumps(record))
            (directory/"team_12345.json").write_text("private content")
            self.assertEqual(load_forecast_metadata(directory), [record])
            (directory/"all_players_gw8.json").write_text("bad json")
            with self.assertRaises(ValueError):
                load_forecast_metadata(directory)

    def test_fetch_failure_cannot_fall_back_or_write_due_output(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"github_output"
            with patch("deadline_gate.FPLClient") as client:
                client.return_value.get_json.side_effect = RuntimeError("API unavailable")
                with self.assertRaisesRegex(RuntimeError, "API unavailable"):
                    main(["--metadata-dir", temp, "--github-output", str(output)])
                client.return_value.get_json.assert_called_once_with("bootstrap-static/", force=True)
            self.assertFalse(output.exists())

    def test_missing_decision_not_suppressed_by_recent_general_forecast(self):
        now = DEADLINE-timedelta(hours=1)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory/"all_players_gw7.json").write_text(json.dumps(metadata(now)))
            self.assertTrue(decide(bootstrap(), load_decision_metadata(directory), now=now)["due"])

    def test_complete_decision_bundle_checkpoints_and_incomplete_bundle_retries(self):
        now = DEADLINE-timedelta(hours=1)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)/"GW7"
            directory.mkdir()
            report = {**metadata(now), "deadline": DEADLINE.isoformat()}
            (directory/"decision.json").write_text(json.dumps(report))
            self.assertEqual(load_decision_metadata(temp), [])
            (directory/"index.html").write_text("Public recommendation")
            (directory/"lineup.csv").write_text("element,role\n1,keeper\n")
            records = load_decision_metadata(temp)
            self.assertFalse(decide(bootstrap(), records, now=now)["due"])

    def test_changed_deadline_invalidates_even_recent_decision_checkpoint(self):
        now = DEADLINE-timedelta(hours=1)
        record = {**metadata(now-timedelta(minutes=10)), "deadline": DEADLINE.isoformat()}
        data = bootstrap()
        data["events"][6]["deadline_time"] = (DEADLINE+timedelta(minutes=30)).isoformat()
        self.assertTrue(decide(data, [record], now=now)["due"])

    def test_mismatched_or_expired_decision_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)/"GW7"
            directory.mkdir()
            (directory/"index.html").write_text("Report")
            (directory/"lineup.csv").write_text("element,role\n")
            for report in ({**metadata(DEADLINE), "deadline": DEADLINE.isoformat()},
                           {**metadata(DEADLINE-timedelta(hours=1), event=8), "deadline": DEADLINE.isoformat()}):
                (directory/"decision.json").write_text(json.dumps(report))
                with self.subTest(report=report), self.assertRaises(ValueError):
                    load_decision_metadata(temp)


if __name__ == "__main__":
    unittest.main()
