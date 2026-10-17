"""Offline regression tests for freshness, joins and publication integrity."""
import copy
import hashlib
import io
import json
import tempfile
import unittest
import urllib.error
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from fpl_api import APIError, DataValidationError, FPLClient, utc_now
from fetch_fpl import (build_current_rows, csv_bytes, prepare_historical,
                       season_from_bootstrap, validate_data_manifest)


def sample_data():
    teams = [{"id": i, "name": name, "short_name": name[:3],
              "strength_overall_home": 3, "strength_overall_away": 3}
             for i, name in [(1, "Alpha"), (2, "Bravo"), (3, "Charlie")]]
    # Player 1 transferred to Charlie after playing the Alpha-Bravo fixture.
    players = [{"id": i, "code": 100 + i, "first_name": "Player",
                "second_name": str(i), "web_name": str(i), "team": team,
                "element_type": 1, "now_cost": 50, "status": "a",
                "chance_of_playing_next_round": None, "minutes": 90,
                "total_points": points}
               for i, team, points in [(1, 3, 6), (2, 2, 2)]]
    bootstrap = {"elements": players, "teams": teams, "events": [
        {"id": 1, "finished": True, "deadline_time": "2026-08-21T17:30:00Z"},
        {"id": 2, "finished": False, "is_next": True,
         "deadline_time": "2026-08-28T17:30:00Z"},
    ]}
    fixtures = [{"id": 10, "event": 1, "team_h": 1, "team_a": 2,
                 "kickoff_time": "2026-08-21T19:00:00Z", "finished": True,
                 "started": True, "team_h_score": 1, "team_a_score": 0}]
    summaries = {}
    for player in players:
        i = player["id"]
        summaries[i] = {"history": [{"element": i, "fixture": 10,
            "opponent_team": 2 if i == 1 else 1, "was_home": i == 1,
            "round": 1, "kickoff_time": fixtures[0]["kickoff_time"],
            "minutes": 90, "total_points": player["total_points"], "value": 50,
            "expected_goals": "0.0", "expected_assists": "0.0", "starts": 1}]}
    return bootstrap, fixtures, summaries


class CurrentHistoryTests(unittest.TestCase):
    def test_transferred_player_keeps_match_club_and_stable_code(self):
        bootstrap, fixtures, summaries = sample_data()
        rows, coverage = build_current_rows(bootstrap, fixtures, summaries)
        self.assertEqual(rows[0]["team"], "Alpha")
        self.assertEqual(rows[0]["team_id"], 1)
        self.assertEqual(rows[0]["element"], 1)
        self.assertEqual(rows[0]["code"], 101)
        self.assertEqual(coverage["latest_finished_gw"], 1)
        self.assertEqual(coverage["covered_fixture_sides"], 2)

    def test_missing_player_response_is_rejected(self):
        b, f, s = sample_data()
        del s[2]
        with self.assertRaisesRegex(DataValidationError, "Player-summary IDs"):
            build_current_rows(b, f, s)

    def test_duplicate_player_fixture_is_rejected(self):
        b, f, s = sample_data()
        s[1]["history"].append(copy.deepcopy(s[1]["history"][0]))
        with self.assertRaisesRegex(DataValidationError, "Duplicate current"):
            build_current_rows(b, f, s)

    def test_completed_fixture_side_cannot_disappear(self):
        b, f, s = sample_data()
        s[2]["history"] = []
        with self.assertRaisesRegex(DataValidationError, "completed fixture sides"):
            build_current_rows(b, f, s)

    def test_history_totals_detect_stale_individual_response(self):
        b, f, s = sample_data()
        s[1]["history"][0]["total_points"] = 2
        with self.assertRaisesRegex(DataValidationError, "history total_points"):
            build_current_rows(b, f, s)

    def test_wrong_response_id_is_rejected(self):
        b, f, s = sample_data()
        s[1]["history"][0]["element"] = 2
        with self.assertRaisesRegex(DataValidationError, "Mismatched player"):
            build_current_rows(b, f, s)

    def test_wrong_opponent_is_rejected(self):
        b, f, s = sample_data()
        s[1]["history"][0]["opponent_team"] = 3
        with self.assertRaisesRegex(DataValidationError, "disagrees with fixture"):
            build_current_rows(b, f, s)

    def test_missing_required_feature_is_rejected(self):
        b, f, s = sample_data()
        del s[1]["history"][0]["expected_goals"]
        with self.assertRaisesRegex(DataValidationError, "missing fields"):
            build_current_rows(b, f, s)

    def test_live_match_points_are_not_training_targets(self):
        b, f, s = sample_data()
        live = {**f[0], "id": 20, "event": 2, "finished": False}
        f.append(live)
        for summary in s.values():
            summary["history"].append({**summary["history"][0], "fixture": 20, "round": 2})
        rows, _ = build_current_rows(b, f, s)
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["fixture"] for r in rows}, {10})

    def test_finished_event_with_unfinished_fixture_is_rejected(self):
        b, f, s = sample_data()
        f[0]["finished"] = False
        with self.assertRaisesRegex(DataValidationError, "finished but fixture unfinished"):
            build_current_rows(b, f, s)

    def test_finished_wholly_blank_event_does_not_require_impossible_history(self):
        b, f, s = sample_data()
        b["events"][1]["finished"] = True
        b["events"][1]["is_next"] = False
        rows, coverage = build_current_rows(b, f, s)
        self.assertEqual(len(rows), 2)
        self.assertEqual(coverage["latest_finished_gw"], 2)
        self.assertEqual(coverage["latest_finished_fixture_gw"], 1)
        self.assertEqual(coverage["current_history_max_gw"], 1)

    def test_season_comes_from_first_deadline(self):
        b, _, _ = sample_data()
        self.assertEqual(season_from_bootstrap(b), "2026-27")


class HistoricalTests(unittest.TestCase):
    def historical_data(self):
        b, f, s = sample_data()
        rows, _ = build_current_rows(b, f, s)
        # Archive values are strings, unlike JSON values.
        import csv
        return (list(csv.DictReader(io.StringIO(csv_bytes(rows).decode()))),
                b["elements"], b["teams"])

    def test_assistant_managers_excluded_exact_duplicates_removed(self):
        rows, players, teams = self.historical_data()
        rows.append(copy.deepcopy(rows[0]))
        rows.append({**rows[0], "position": "AM", "element": "999"})
        cleaned, stats = prepare_historical(rows, players, teams, "2025-26")
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned[0]["code"], 101)
        self.assertEqual(stats["exact_duplicates_removed"], 1)
        self.assertEqual(stats["excluded_assistant_manager_rows"], 1)

    def test_conflicting_duplicate_is_rejected(self):
        rows, players, teams = self.historical_data()
        rows.append({**rows[0], "total_points": "10"})
        with self.assertRaisesRegex(DataValidationError, "conflicting duplicate"):
            prepare_historical(rows, players, teams, "2025-26")

    def test_missing_stable_id_mapping_is_rejected(self):
        rows, players, teams = self.historical_data()
        with self.assertRaisesRegex(DataValidationError, "unmatched player"):
            prepare_historical(rows, players[:1], teams, "2025-26")


class Response:
    status = 200

    def __init__(self, payload=b'{"ok": true}'):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self.payload


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.client = FPLClient(self.tmp.name, retries=1)

    def test_valid_cache_avoids_network_and_retains_source_timestamp(self):
        with patch("fpl_api.urllib.request.urlopen", return_value=Response()) as request:
            self.client.get_json("bootstrap-static/")
            self.client.get_json("bootstrap-static/")
        self.assertEqual(request.call_count, 1)
        self.assertFalse(self.client.records[0]["cache_hit"])
        self.assertTrue(self.client.records[1]["cache_hit"])
        self.assertEqual(self.client.records[0]["fetched_at"], self.client.records[1]["fetched_at"])

    def test_stale_cache_is_never_a_fallback_for_network_error(self):
        with patch("fpl_api.urllib.request.urlopen", return_value=Response()):
            self.client.get_json("bootstrap-static/")
        future = utc_now() + timedelta(hours=7)
        with patch("fpl_api.utc_now", return_value=future), patch(
                "fpl_api.urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
            with self.assertRaisesRegex(APIError, "no stale-cache fallback"):
                self.client.get_json("bootstrap-static/")

    def test_corrupt_cache_requires_new_network_response(self):
        with patch("fpl_api.urllib.request.urlopen", return_value=Response()):
            self.client.get_json("bootstrap-static/")
        next(Path(self.tmp.name).glob("*.body")).write_bytes(b"corrupt")
        with patch("fpl_api.urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
            with self.assertRaises(APIError):
                self.client.get_json("bootstrap-static/")

    def test_transient_http_failure_retries(self):
        self.client.retries = 2
        failure = urllib.error.HTTPError("https://example.test", 503, "Unavailable", {}, None)
        with patch("fpl_api.urllib.request.urlopen", side_effect=[failure, Response()]) as request, \
             patch("fpl_api.time.sleep"):
            self.assertEqual(self.client.get_json("bootstrap-static/"), {"ok": True})
        self.assertEqual(request.call_count, 2)

    def test_404_is_not_retried_as_successful_data(self):
        self.client.retries = 3
        failure = urllib.error.HTTPError("https://example.test", 404, "Not found", {}, None)
        with patch("fpl_api.urllib.request.urlopen", side_effect=failure) as request:
            with self.assertRaises(APIError):
                self.client.get_json("bootstrap-static/")
        self.assertEqual(request.call_count, 1)

    def test_error_page_is_rejected(self):
        with patch("fpl_api.urllib.request.urlopen", return_value=Response(b"<html>oops</html>")):
            with self.assertRaises(DataValidationError):
                self.client.get_json("bootstrap-static/")
            with self.assertRaises(DataValidationError):
                self.client.get_csv("https://example.test/data.csv")


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        payload = b"value\n1\n"
        (self.directory / "gw_2026-27.csv").write_bytes(payload)
        self.manifest = {"schema_version": 1, "current_source": "official-fpl", "next_gw": 7,
            "oldest_current_source_at": utc_now().isoformat(),
            "coverage": {"latest_finished_gw": 6, "current_history_max_gw": 6},
            "files": {"gw_2026-27.csv": {"sha256": hashlib.sha256(payload).hexdigest()}}}
        self.write_manifest()

    def write_manifest(self):
        (self.directory / "data_manifest.json").write_text(json.dumps(self.manifest))

    def test_recent_intact_snapshot_passes(self):
        self.assertEqual(validate_data_manifest(self.directory, target_gw=7)["next_gw"], 7)

    def test_partial_refresh_cannot_be_used_with_old_manifest(self):
        (self.directory / "gw_2026-27.csv").write_bytes(b"partial")
        with self.assertRaisesRegex(DataValidationError, "changed or incomplete"):
            validate_data_manifest(self.directory)

    def test_generated_now_does_not_make_old_sources_fresh(self):
        self.manifest["generated_at"] = utc_now().isoformat()
        self.manifest["oldest_current_source_at"] = (utc_now() - timedelta(hours=7)).isoformat()
        self.write_manifest()
        with self.assertRaisesRegex(DataValidationError, "snapshot is"):
            validate_data_manifest(self.directory)

    def test_history_behind_completed_gameweeks_is_rejected(self):
        self.manifest["coverage"]["current_history_max_gw"] = 1
        self.write_manifest()
        with self.assertRaisesRegex(DataValidationError, "incomplete current history"):
            validate_data_manifest(self.directory)

    def test_manifest_accepts_finished_blank_event_with_complete_fixture_history(self):
        self.manifest["coverage"]["current_history_max_gw"] = 5
        self.manifest["coverage"]["latest_finished_fixture_gw"] = 5
        self.write_manifest()
        self.assertEqual(validate_data_manifest(self.directory)["coverage"]["latest_finished_gw"], 6)


if __name__ == "__main__":
    unittest.main()
