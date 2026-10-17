import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import pandas as pd

from forecast_archive import archive_forecast, evaluate_archive, read_archive


class ForecastArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.data = self.folder/"data"; self.data.mkdir()
        self.now = pd.Timestamp("2026-10-17T08:00:00Z")
        self.bootstrap = {"events": [
            {"id": 6, "finished": True, "deadline_time": "2026-10-10T10:00:00Z"},
            {"id": 7, "finished": False, "deadline_time": "2026-10-18T10:00:00Z"},
            {"id": 8, "finished": False, "deadline_time": "2026-10-25T10:00:00Z"}],
            "elements": [{"id": i, "code": 100+i} for i in [1, 2, 3]]}
        self.fixtures = pd.DataFrame([
            {"id": 1, "event": 6, "team_h": 1, "team_a": 2, "finished": True},
            {"id": 10, "event": 7, "team_h": 1, "team_a": 4, "finished": False},
            {"id": 11, "event": 7, "team_h": 1, "team_a": 3, "finished": False},
            {"id": 12, "event": 8, "team_h": 1, "team_a": 2, "finished": False}])
        self.history = pd.DataFrame([{"element": i, "code": 100+i, "fixture": 1,
            "round": 6, "minutes": 90, "total_points": 3} for i in [1, 2, 3]])
        self.manifest = self.save_data()
        self.metadata = {"season": "2026-27", "model": "test-minutes", "gameweeks": [7, 8],
            "history_cutoff": self.now.isoformat(), "generated_utc": self.now.isoformat(), "data_manifest": self.manifest}
        self.forecast = self.folder/"predictions"/"all_players_gw7.csv"
        self.forecast.parent.mkdir()
        pd.DataFrame({"element": [1, 2, 3, 1, 2, 3], "code": [101, 102, 103]*2,
            "gw": [7]*3+[8]*3, "pred": [4, 1, 0, 5, 2, 4], "fixture_count": [2, 2, 0, 1, 1, 1],
            "expected_minutes": [120, 60, 0, 80, 20, 90]}).to_csv(self.forecast, index=False)
        self.fixture_forecast = self.forecast.parent/"fixtures_gw7.csv"
        pd.DataFrame({"element": [1, 1, 2, 2, 1, 2, 3], "fixture": [10, 11, 10, 11, 12, 12, 12],
            "gw": [7]*4+[8]*3, "p_no_appearance": [.1, .1, .5, .5, .1, .5, .1],
            "p_1_59": [.2, .2, .4, .4, .2, .4, .2], "p_60_plus": [.7, .7, .1, .1, .7, .1, .7],
            "expected_minutes": [60, 60, 30, 30, 80, 20, 90]}).to_csv(self.fixture_forecast, index=False)

    def save_data(self):
        (self.data/"bootstrap.json").write_text(json.dumps(self.bootstrap))
        self.fixtures.to_csv(self.data/"fixtures_2627.csv", index=False)
        self.history.to_csv(self.data/"gw_2026-27.csv", index=False)
        files = {}
        for name in ["bootstrap.json", "fixtures_2627.csv", "gw_2026-27.csv"]:
            data = (self.data/name).read_bytes()
            files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        manifest = {"schema_version": 1, "current_source": "official-fpl", "season": "2026-27",
                    "generated_at": self.now.isoformat(), "historical_seasons": [], "files": files}
        (self.data/"data_manifest.json").write_text(json.dumps(manifest))
        return manifest

    def archive(self):
        with patch("forecast_archive._utc_now", return_value=self.now):
            return archive_forecast(self.forecast, self.data, self.metadata, self.fixture_forecast)

    def finish_gameweek(self):
        self.fixtures.loc[self.fixtures.event.eq(7), "finished"] = True
        self.bootstrap["events"][1]["finished"] = True
        labels = pd.DataFrame([
            {"element": 1, "code": 101, "fixture": 10, "round": 7, "minutes": 90, "total_points": 3},
            {"element": 1, "code": 101, "fixture": 11, "round": 7, "minutes": 60, "total_points": 5},
            {"element": 2, "code": 102, "fixture": 10, "round": 7, "minutes": 10, "total_points": 1},
            {"element": 2, "code": 102, "fixture": 11, "round": 7, "minutes": 0, "total_points": 0}])
        self.history = pd.concat([self.history, labels], ignore_index=True)
        self.save_data()

    def test_archive_has_only_allowed_public_inputs_and_is_idempotent(self):
        state = self.data/"state"; state.mkdir()
        (state/"private_account.json").write_text('{"entry_id":123}')
        archive = self.archive()
        original = archive.read_bytes()
        self.assertEqual(self.archive(), archive)
        self.assertEqual(original, archive.read_bytes())
        index, files = read_archive(archive)
        self.assertEqual(files["forecast.csv"], self.forecast.read_bytes())
        self.assertEqual(files["fixture_forecasts.csv"], self.fixture_forecast.read_bytes())
        self.assertFalse(any("account" in name or "state" in name for name in files))
        self.assertTrue(index["forward_deadline_check"])
        self.assertEqual(list(archive.parent.glob("*.tmp")), [])

    def test_same_identity_different_bytes_cannot_overwrite_archive(self):
        archive = self.archive()
        original = archive.read_bytes()
        self.forecast.write_bytes(self.forecast.read_bytes()+b"\n")
        with self.assertRaisesRegex(FileExistsError, "different content"):
            self.archive()
        self.assertEqual(original, archive.read_bytes())

    def test_tampering_detected_without_extracting_members(self):
        archive = self.archive()
        with zipfile.ZipFile(archive) as source:
            members = {name: source.read(name) for name in source.namelist()}
        members["forecast.csv"] += b"\n"
        tampered = archive.parent/"tampered.zip"
        with zipfile.ZipFile(tampered, "w") as target:
            for name, payload in members.items(): target.writestr(name, payload)
        with self.assertRaisesRegex(ValueError, "checksum"):
            read_archive(tampered)

    def test_no_forward_archive_after_deadline_or_with_changed_inputs(self):
        with patch("forecast_archive._utc_now", return_value=pd.Timestamp("2026-10-18T11:00:00Z")):
            with self.assertRaisesRegex(ValueError, "before its first"):
                archive_forecast(self.forecast, self.data, self.metadata)
        with (self.data/"gw_2026-27.csv").open("a") as file: file.write("\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.archive()

    def test_only_completed_gameweeks_scored_and_dgw_states_stay_per_fixture(self):
        archive = self.archive()
        original = archive.read_bytes()
        before = evaluate_archive(archive, self.data)
        self.assertEqual(before["all"]["n"], 0)
        self.finish_gameweek()
        scored = evaluate_archive(archive, self.data)
        self.assertEqual(scored["completed_gameweeks"], [7])
        self.assertEqual(scored["pending_gameweeks"], [8])
        self.assertEqual(scored["all"]["n"], 3)
        self.assertAlmostEqual(scored["all"]["mae"], 4/3)
        self.assertEqual(scored["minutes_fixture"]["all"]["n"], 4)
        self.assertEqual(original, archive.read_bytes())

    def test_active_prior_excludes_two_cameos_and_includes_player_blank_at_origin(self):
        archive = self.archive(); self.finish_gameweek()
        scored = evaluate_archive(archive, self.data)
        self.assertEqual(scored["active_prior"]["n"], 2)  # player1 + blank-now player3
        self.assertEqual(scored["active_target_prior"]["n"], 1)
        self.assertAlmostEqual(scored["active_prior"]["mae"], 2)

    def test_incomplete_event_not_silently_scored_as_zero(self):
        archive = self.archive(); self.finish_gameweek()
        self.fixtures.loc[self.fixtures.id.eq(11), "finished"] = False
        self.save_data()
        scored = evaluate_archive(archive, self.data)
        self.assertEqual(scored["all"]["n"], 0)

    def test_missing_actual_fixture_fails(self):
        archive = self.archive(); self.finish_gameweek()
        self.history = self.history[self.history.fixture.ne(11)]
        self.save_data()
        with self.assertRaisesRegex(ValueError, "missing completed fixtures"):
            evaluate_archive(archive, self.data)

    def test_account_metadata_rejected(self):
        self.metadata["entry_id"] = 123
        with self.assertRaisesRegex(ValueError, "Account state"):
            self.archive()

    def test_cherry_picked_player_pool_cannot_be_forward_tested(self):
        rows = pd.read_csv(self.forecast)
        rows[rows.element.ne(2)].to_csv(self.forecast, index=False)
        with self.assertRaisesRegex(ValueError, "complete origin player pool"):
            self.archive()

    def test_actual_event_mismatch_rejected(self):
        archive = self.archive(); self.finish_gameweek()
        self.history.loc[self.history.fixture.eq(10), "round"] = 8
        self.save_data()
        with self.assertRaisesRegex(ValueError, "disagrees"):
            evaluate_archive(archive, self.data)


if __name__ == "__main__":
    unittest.main()
