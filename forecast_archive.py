"""Immutable public forecast snapshots and later scoring against official actuals.

Archive publication uses an atomic, no-overwrite hard link. Existing identical
payloads are idempotent; conflicting payloads for the same origin fail. ZIPs are
checksummed, not cryptographically signed: commit them before the deadline for
an independently timestamped forward-test record. No account state is included.

python forecast_archive.py evaluate --archive predictions/snapshots/example.zip \
    --data-dir . --output state/forward-evaluation.json
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROBABILITIES = ["p_no_appearance", "p_1_59", "p_60_plus"]
MEMBERS = {"forecast.csv", "fixture_forecasts.csv", "metadata.json", "origin/bootstrap.json",
           "origin/fixtures.csv", "origin/history.csv", "origin/data_manifest.json"}


def _utc_now():
    return datetime.now(timezone.utc)


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()


def _digest(payload):
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def _timestamp(value, label):
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is None:
        raise ValueError(f"{label} must be a timezone-aware timestamp")
    return value.tz_convert("UTC")


def _verified_inputs(directory):
    """Hash-check official inputs without imposing an age limit on final actuals."""
    directory = Path(directory)
    manifest_bytes = (directory/"data_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 1 or manifest.get("current_source") != "official-fpl":
        raise ValueError("Scoring/archive inputs need a validated official-FPL manifest")
    if not re.fullmatch(r"20\d{2}-\d{2}", str(manifest.get("season"))):
        raise ValueError("Invalid manifest season")
    payloads = {}
    for name, expected in manifest.get("files", {}).items():
        if Path(name).name != name or not re.fullmatch(r"(?:bootstrap\.json|(?:gw|players|teams|fixtures|cleaned)_[\d-]+\.csv)", name):
            raise ValueError("Unexpected or unsafe manifest input name")
        payload = (directory/name).read_bytes()
        if _digest(payload) != expected:
            raise ValueError(f"Input checksum mismatch: {name}")
        payloads[name] = payload
    year = int(manifest["season"][:4])
    suffix = f"{year%100:02}{(year+1)%100:02}"
    required = ["bootstrap.json", f"fixtures_{suffix}.csv", f"gw_{manifest['season']}.csv"]
    if not all(name in payloads for name in required):
        raise ValueError("Manifest does not cover the required origin inputs")
    return manifest, {"origin/bootstrap.json": payloads[required[0]],
                      "origin/fixtures.csv": payloads[required[1]],
                      "origin/history.csv": payloads[required[2]],
                      "origin/data_manifest.json": manifest_bytes}


def _forecast_frame(payload):
    rows = pd.read_csv(io.BytesIO(payload))
    if not {"element", "gw", "pred"}.issubset(rows) or rows.empty:
        raise ValueError("Forecast requires element, gw, pred and at least one row")
    if rows.duplicated(["element", "gw"]).any() or not np.isfinite(rows.pred).all():
        raise ValueError("Duplicate or non-finite forecast")
    for name in ("element", "gw"):
        if rows[name].isna().any() or not rows[name].eq(np.floor(rows[name])).all():
            raise ValueError("Invalid forecast identifiers")
    return rows


def read_archive(path):
    """Read verified members without extracting paths onto the filesystem."""
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [item.filename for item in entries]
        if len(names) != len(set(names)) or set(names)-MEMBERS-{"archive_index.json"}:
            raise ValueError("Unexpected or duplicate archive member")
        if sum(item.file_size for item in entries) > 64*1024*1024:
            raise ValueError("Forecast archive exceeds 64 MiB uncompressed")
        if "archive_index.json" not in names:
            raise ValueError("Missing archive index")
        index = json.loads(archive.read("archive_index.json"))
        if index.get("schema_version") != 1 or set(index.get("files", {})) != set(names)-{"archive_index.json"}:
            raise ValueError("Invalid archive index")
        payloads = {name: archive.read(name) for name in index["files"]}
    if not (MEMBERS-{"fixture_forecasts.csv"}).issubset(payloads):
        raise ValueError("Required snapshot member missing")
    for name, payload in payloads.items():
        if _digest(payload) != index["files"][name]:
            raise ValueError(f"Archive checksum mismatch: {name}")
    if index.get("forward_deadline_check") is not True:
        raise ValueError("Archive is not marked as a verified forward snapshot")
    metadata = json.loads(payloads["metadata.json"])
    bootstrap = json.loads(payloads["origin/bootstrap.json"])
    deadlines = {int(event["id"]): _timestamp(event["deadline_time"], "deadline") for event in bootstrap["events"]}
    gameweeks = metadata.get("gameweeks", [])
    if not gameweeks or not set(gameweeks).issubset(deadlines):
        raise ValueError("Archive gameweeks do not have origin deadlines")
    if max(_timestamp(index["archived_at"], "archive time"), _timestamp(metadata["generated_utc"], "generation")) >= min(deadlines[gw] for gw in gameweeks):
        raise ValueError("Archived forecast was not captured before deadline")
    return index, payloads


def archive_forecast(destination, data_dir, metadata, fixture_path=None):
    """Freeze forecast files and essential public origin inputs; return ZIP Path."""
    destination = Path(destination)
    manifest, payloads = _verified_inputs(data_dir)
    if metadata.get("season") != manifest["season"]:
        raise ValueError("Forecast and input season differ")
    if "data_manifest" in metadata and metadata["data_manifest"] != manifest:
        raise ValueError("Inputs changed between forecasting and archive creation")
    forbidden = {"entry_id", "personal_state", "account_state", "picks", "bank", "free_transfers"}
    if forbidden & set(metadata):
        raise ValueError("Account state must not be included in forecast metadata")
    payloads["forecast.csv"] = destination.read_bytes()
    payloads["metadata.json"] = _json_bytes(metadata)
    if fixture_path is not None:
        payloads["fixture_forecasts.csv"] = Path(fixture_path).read_bytes()
    forecast = _forecast_frame(payloads["forecast.csv"])
    gameweeks = sorted(map(int, forecast.gw.unique()))
    if sorted(metadata.get("gameweeks", [])) != gameweeks:
        raise ValueError("Forecast gameweeks do not match metadata")
    generated = _timestamp(metadata["generated_utc"], "generated_utc")
    cutoff = _timestamp(metadata["history_cutoff"], "history_cutoff")
    if cutoff > generated:
        raise ValueError("History cutoff follows forecast generation")
    model = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(metadata.get("model", "unknown")))[:64]
    identity = f"{generated.strftime('%Y%m%dT%H%M%S%fZ')}_gw{gameweeks[0]}-{gameweeks[-1]}_{model}.zip"
    folder = destination.parent/"snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder/identity
    checksums = {name: _digest(payload) for name, payload in payloads.items()}

    def existing_identical():
        index, _ = read_archive(target)
        if index["files"] != checksums:
            raise FileExistsError(f"Immutable forecast identity already has different content: {target}")
        return target

    if target.exists():
        return existing_identical()
    bootstrap = json.loads(payloads["origin/bootstrap.json"])
    origin_players = {int(player["id"]): player for player in bootstrap["elements"]
                      if player.get("element_type", 1) in (1, 2, 3, 4)}
    if any(set(group.element) != set(origin_players) for _, group in forecast.groupby("gw")):
        raise ValueError("Forward test requires the complete origin player pool in every GW")
    if "code" in forecast:
        expected_codes = forecast.element.map({i: player["code"] for i, player in origin_players.items()})
        if forecast.code.isna().any() or not forecast.code.eq(expected_codes).all():
            raise ValueError("Forecast codes do not match the origin player catalogue")
    deadlines = {int(event["id"]): _timestamp(event["deadline_time"], "deadline") for event in bootstrap["events"]}
    if not set(gameweeks).issubset(deadlines):
        raise ValueError("Forecast gameweek missing from origin deadlines")
    archived = _timestamp(_utc_now(), "archive time")
    if generated > archived+pd.Timedelta(minutes=5):
        raise ValueError("Forecast generation timestamp lies in the future")
    if max(generated, archived) >= min(deadlines[gw] for gw in gameweeks):
        raise ValueError("Forward-test snapshot must be archived before its first forecast deadline")
    index = {"schema_version": 1, "archived_at": archived.isoformat(), "files": checksums,
             "season": manifest["season"], "gameweeks": gameweeks,
             "historical_input_hashes": {name: item for name, item in manifest["files"].items()
                 if any(season in name for season in manifest.get("historical_seasons", []))},
             "forward_deadline_check": True}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".snapshot-", suffix=".tmp", dir=folder, delete=False) as stream:
            temporary = Path(stream.name)
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
                for name, payload in sorted({**payloads, "archive_index.json": _json_bytes(index)}.items()):
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, payload)
            stream.flush()
            os.fsync(stream.fileno())
        # link() publishes the complete ZIP atomically and fails if it exists.
        try:
            os.link(temporary, target)
        except FileExistsError:
            return existing_identical()
        os.chmod(target, 0o444)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _calibration(rows):
    bins = pd.cut(rows.pred, [-np.inf, 1, 2, 3, 4, 5, 7, 10, np.inf])
    return [{"bin": str(label), "n": len(group), "mean_pred": float(group.pred.mean()),
             "mean_actual": float(group.actual.mean())} for label, group in rows.groupby(bins, observed=True)]


def _point_metrics(rows):
    if rows.empty:
        return {"n": 0}
    correlations, top20, captain = [], [], []
    for _, group in rows.groupby("gw"):
        if group.pred.nunique() > 1 and group.actual.nunique() > 1:
            correlations.append(float(spearmanr(group.pred, group.actual).statistic))
        top20.append(float(group.nlargest(20, "pred").actual.mean()))
        captain.append(float(group.nlargest(1, "pred").actual.iloc[0]))
    error = rows.pred-rows.actual
    return {"n": len(rows), "mae": float(error.abs().mean()), "rmse": float(np.sqrt(np.square(error).mean())),
            "mean_pred": float(rows.pred.mean()), "mean_actual": float(rows.actual.mean()),
            "bias": float(error.mean()), "spearman_mean_gw": float(np.mean(correlations)) if correlations else None,
            "top20_mean_actual": float(np.mean(top20)), "captain_points_sum": float(np.sum(captain)),
            "calibration": _calibration(rows)}


def _probability_metrics(rows):
    if rows.empty:
        return {"n": 0}
    probability = rows[PROBABILITIES].to_numpy(float)
    if (not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any()
            or not np.allclose(probability.sum(axis=1), 1, atol=2e-5)):
        raise ValueError("Invalid archived minute probabilities")
    states = np.where(rows.actual_minutes == 0, 0, np.where(rows.actual_minutes < 60, 1, 2))
    truth = np.eye(3)[states]
    result = {"n": len(rows), "multiclass_brier": float(np.square(probability-truth).sum(axis=1).mean()),
              "log_loss": float(-np.log(np.clip(probability[np.arange(len(rows)), states], 1e-12, 1)).mean()),
              "calibration": {}}
    for k, name in enumerate(PROBABILITIES):
        frame = pd.DataFrame({"probability": probability[:, k], "actual": truth[:, k]})
        bins = pd.cut(frame.probability, np.linspace(0, 1, 6), include_lowest=True)
        result["calibration"][name] = [{"bin": str(label), "n": len(group),
            "mean_probability": float(group.probability.mean()), "observed_frequency": float(group.actual.mean())}
            for label, group in frame.groupby(bins, observed=True)]
    if "expected_minutes" in rows:
        result["minutes_mae"] = float((rows.expected_minutes-rows.actual_minutes).abs().mean())
    return result


def evaluate_archive(archive_path, data_dir):
    """Score immutable forecasts, only for officially completed gameweeks."""
    index, payloads = read_archive(archive_path)
    metadata = json.loads(payloads["metadata.json"])
    manifest, actual_payloads = _verified_inputs(data_dir)
    if manifest["season"] != metadata["season"]:
        raise ValueError("Actuals and forecast seasons differ")
    bootstrap = json.loads(actual_payloads["origin/bootstrap.json"])
    fixtures = pd.read_csv(io.BytesIO(actual_payloads["origin/fixtures.csv"]))
    history = pd.read_csv(io.BytesIO(actual_payloads["origin/history.csv"]))
    if not {"element", "fixture", "round", "minutes", "total_points"}.issubset(history):
        raise ValueError("Actual history lacks required columns")
    history = history.drop_duplicates()
    if history.duplicated(["element", "fixture"]).any():
        raise ValueError("Conflicting actual player-fixture rows")
    if fixtures.id.duplicated().any() or "finished" not in fixtures:
        raise ValueError("Invalid actual fixture catalogue")
    actual_event = history.fixture.map(fixtures.set_index("id").event)
    if actual_event.isna().any() or not history["round"].eq(actual_event).all():
        raise ValueError("Actual history disagrees with official fixture events")
    finished = fixtures.finished.isin([True, 1, "True"])
    finished_ids = set(fixtures.loc[finished, "id"])
    completed_events = {int(event["id"]) for event in bootstrap["events"] if event.get("finished") is True}
    completed_events -= set(fixtures.loc[~finished & fixtures.event.notna(), "event"].astype(int))
    forecast = _forecast_frame(payloads["forecast.csv"])
    if "code" in forecast and "code" in history:
        codes = history[["element", "code"]].dropna().drop_duplicates()
        if codes.element.duplicated().any():
            raise ValueError("Actual element/code mapping changed within season")
        mapped = forecast.element.map(codes.set_index("element").code)
        if (mapped.notna() & forecast.code.notna() & mapped.ne(forecast.code)).any():
            raise ValueError("Forecast/actual player identity mismatch")
    history = history[history.fixture.isin(finished_ids)].copy()
    if not np.isfinite(history[["minutes", "total_points"]].to_numpy(float)).all():
        raise ValueError("Completed actuals require finite targets")
    # Do not allow an incomplete actual fetch to turn an entire finished match
    # into zero labels. The official fetcher additionally validates both sides.
    expected = set(fixtures.loc[finished & fixtures.event.isin(completed_events), "id"])
    if not expected.issubset(set(history.fixture)):
        raise ValueError("Actual history is missing completed fixtures")
    selected = sorted(set(map(int, forecast.gw)) & completed_events)
    forecast["lead"] = forecast.gw-int(forecast.gw.min())+1
    forecast["active_target_prior"] = False
    active_definition = "unavailable: no saved expected minutes and fixture counts"
    if {"expected_minutes", "fixture_count"}.issubset(forecast):
        per_fixture = forecast.expected_minutes / forecast.fixture_count.where(forecast.fixture_count.gt(0))
        forecast["active_target_prior"] = per_fixture.ge(45)
        active_definition = "fixed origin group: maximum archived expected minutes per fixture across horizon >=45; active_target_prior uses each target GW's ratio"
    active_players = set(forecast.loc[forecast.active_target_prior, "element"])
    forecast["active_prior"] = forecast.element.isin(active_players)
    actual = history.groupby(["element", "round"], as_index=False).agg(
        actual=("total_points", "sum"), actual_minutes=("minutes", "sum")).rename(columns={"round": "gw"})
    scored = forecast[forecast.gw.isin(selected)].merge(actual, on=["element", "gw"], how="left", validate="one_to_one")
    scored[["actual", "actual_minutes"]] = scored[["actual", "actual_minutes"]].fillna(0)
    result = {"schema_version": 1, "archive_sha256": _digest(Path(archive_path).read_bytes())["sha256"],
        "forecast_model": metadata.get("model"), "season": manifest["season"], "origin": metadata["history_cutoff"],
        "archived_at": index["archived_at"], "scored_at": _utc_now().isoformat(),
        "actuals_generated_at": manifest.get("generated_at"), "actual_manifest_sha256": _digest(actual_payloads["origin/data_manifest.json"])["sha256"],
        "completed_gameweeks": selected, "pending_gameweeks": sorted(set(map(int, forecast.gw))-set(selected)),
        "active_prior_definition": active_definition, "all": _point_metrics(scored),
        "active_prior": _point_metrics(scored[scored.active_prior]),
        "active_target_prior": _point_metrics(scored[scored.active_target_prior]),
        "by_lead": {str(int(lead)): {"all": _point_metrics(group), "active_prior": _point_metrics(group[group.active_prior])}
                    for lead, group in scored.groupby("lead")},
        "limitations": ["One archive is one origin, not independent evidence of generalization.",
                        "Finished-GW points use actual final event assignments; rescheduling can change the forecast fixture set.",
                        "Minutes calibration is per fixture; no independence assumption is made to combine DGW states.",
                        "No automatic model selection is performed."]}
    if "fixture_forecasts.csv" in payloads:
        detailed = pd.read_csv(io.BytesIO(payloads["fixture_forecasts.csv"]))
        if {"element", "fixture", "gw"}.issubset(detailed) and set(PROBABILITIES).issubset(detailed):
            if detailed.duplicated(["element", "fixture"]).any():
                raise ValueError("Duplicate archived fixture forecast")
            detail = detailed[detailed.fixture.isin(finished_ids) & detailed.gw.isin(selected)].copy()
            labels = history[["element", "fixture", "minutes"]].rename(columns={"minutes": "actual_minutes"})
            detail = detail.merge(labels, on=["element", "fixture"], how="left", validate="one_to_one")
            detail["actual_minutes"] = detail.actual_minutes.fillna(0)
            detail["active_prior"] = detail.element.isin(active_players)
            detail["lead"] = detail.gw-int(forecast.gw.min())+1
            result["minutes_fixture"] = {"all": _probability_metrics(detail),
                "active_prior": _probability_metrics(detail[detail.active_prior]),
                "by_lead": {str(int(lead)): _probability_metrics(group) for lead, group in detail.groupby("lead")}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--archive", type=Path, required=True)
    evaluate.add_argument("--data-dir", type=Path, default=Path("."))
    evaluate.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_archive(args.archive, args.data_dir)
    payload = _json_bytes(result)
    if args.output:
        if args.output.resolve() == args.archive.resolve():
            raise ValueError("Evaluation output cannot replace its forecast archive")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    print(payload.decode(), end="")


if __name__ == "__main__":
    main()
