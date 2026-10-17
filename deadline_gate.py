"""Cheap public-data deadline check; Python standard library only, no AI APIs.

An hourly Actions run generates once within 24 hours of the next FPL deadline,
then once within its final two hours. The committed gameweek recommendation
report is the checkpoint. A general forecast alone does not suppress decisions.
Failed generation/commit is retried on the next hourly run.
Manual dispatch forces a forecast only when an upcoming deadline exists.
No account fetch, private team state, credentials, or notifications are used by
this gate. The following workflow may publish the user's authorised decisions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re

from fpl_api import FPLClient, parse_timestamp


def _event_id(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 38:
        raise ValueError("Event IDs must be integers from 1 to 38")
    return value


def _utc(value):
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("Timestamp must include a timezone")
        return value.astimezone(timezone.utc)
    return parse_timestamp(value)


def load_forecast_metadata(directory):
    """Read only public forecast JSONs, excluding private decision/state files."""
    records = []
    for path in sorted(Path(directory).glob("all_players_gw*.json")):
        if not re.fullmatch(r"all_players_gw\d+\.json", path.name):
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read forecast metadata: {path.name}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Forecast metadata must be an object: {path.name}")
        records.append(value)
    return records


def load_decision_metadata(directory):
    """Read minimal public report checkpoints; incomplete report bundles retry.

    Folder GW{N} must contain decision.json and at least one HTML and CSV report.
    Account reconstruction and cache files are never read or used as checkpoints.
    """
    records = []
    for path in sorted(Path(directory).glob("GW*/decision.json")):
        match = re.fullmatch(r"GW(\d+)", path.parent.name)
        if match is None:
            continue
        event = _event_id(int(match[1]))
        if not any(path.parent.glob("*.html")) or not any(path.parent.glob("*.csv")):
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read decision checkpoint: {path.parent.name}/{path.name}") from exc
        if not isinstance(report, dict) or not {"generated_utc", "season", "deadline"} <= set(report):
            raise ValueError("Decision checkpoint requires generated_utc, season and deadline")
        gameweeks = report.get("gameweeks")
        if gameweeks is None:
            number = report.get("gameweek", report.get("gw"))
            gameweeks = [_event_id(number)]
        if not isinstance(gameweeks, list) or not gameweeks or gameweeks[0] != event:
            raise ValueError("Decision checkpoint event disagrees with its GW folder")
        if _utc(report["generated_utc"]) >= _utc(report["deadline"]):
            raise ValueError("Decision checkpoint was generated after its deadline")
        records.append({"generated_utc": report["generated_utc"], "season": report["season"],
                        "deadline": report["deadline"], "gameweeks": gameweeks})
    return records


def decide(bootstrap, metadata=(), *, now=None, force=False, horizon=6):
    """Pure decision function; metadata times must be valid and not in the future."""
    now = _utc(now or datetime.now(timezone.utc))
    if not isinstance(force, bool):
        raise ValueError("force must be a boolean")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 38:
        raise ValueError("horizon must be an integer from 1 to 38")
    if not isinstance(bootstrap, dict) or not isinstance(bootstrap.get("events"), list) or not bootstrap["events"]:
        raise ValueError("Official bootstrap requires a nonempty events list")
    events = {}
    for event in bootstrap["events"]:
        if not isinstance(event, dict) or "id" not in event or "deadline_time" not in event:
            raise ValueError("Official event lacks id/deadline_time")
        number = _event_id(event["id"])
        if number in events:
            raise ValueError("Duplicate official event ID")
        events[number] = _utc(event["deadline_time"])
    upcoming = [(deadline, number) for number, deadline in events.items() if deadline > now]
    if not upcoming:
        return {"due": False, "event": 0, "horizon": 0, "reason": "season_finished",
                "deadline": None, "latest_generation": None}
    deadline, number = min(upcoming)
    first_deadline = events[min(events)]
    year = first_deadline.year - int(first_deadline.month < 7)
    season = f"{year}-{(year + 1) % 100:02}"
    generated = []
    for record in metadata:
        if not isinstance(record, dict) or not {"generated_utc", "gameweeks", "season"} <= set(record):
            raise ValueError("Forecast metadata requires generated_utc, gameweeks and season")
        time = _utc(record["generated_utc"])
        if time > now:
            raise ValueError("Forecast generation timestamp is in the future")
        gameweeks = record["gameweeks"]
        if not isinstance(gameweeks, list) or not gameweeks:
            raise ValueError("Forecast metadata needs nonempty gameweeks")
        for gw in gameweeks:
            _event_id(gw)
        if gameweeks != list(range(gameweeks[0], gameweeks[0] + len(gameweeks))):
            raise ValueError("Forecast gameweeks must be ordered and consecutive")
        if not isinstance(record["season"], str) or not re.fullmatch(r"20\d{2}-\d{2}", record["season"]):
            raise ValueError("Invalid forecast season")
        # An older origin's six-GW horizon is not a new forecast for this origin.
        matching_deadline = "deadline" not in record or _utc(record["deadline"]) == deadline
        if gameweeks[0] == number and record["season"] == season and matching_deadline:
            generated.append(time)
    latest = max(generated) if generated else None
    result = {"due": False, "event": number,
              "horizon": min(horizon, max(events) - number + 1),
              "deadline": deadline.isoformat(),
              "latest_generation": latest.isoformat() if latest else None}
    if force:
        return {**result, "due": True, "reason": "manual_dispatch"}
    initial, final = deadline - timedelta(hours=24), deadline - timedelta(hours=2)
    if now < initial:
        return {**result, "reason": "outside_deadline_windows"}
    threshold, stage = (final, "final_2h") if now >= final else (initial, "initial_24h")
    due = latest is None or latest < threshold
    return {**result, "due": due, "reason": stage if due else f"{stage}_already_generated"}


def github_outputs(result):
    """Only validated control values enter GITHUB_OUTPUT, never API text."""
    due, event, horizon = result["due"], result["event"], result["horizon"]
    if not isinstance(due, bool):
        raise ValueError("due output must be boolean")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (event, horizon)):
        raise ValueError("event and horizon outputs must be integers")
    if event == 0 and horizon == 0 and not due:
        pass
    else:
        _event_id(event)
        if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 39 - event:
            raise ValueError("Invalid output horizon")
    return f"due={'true' if due else 'false'}\nevent={event}\nhorizon={horizon}\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-dir", type=Path, default=Path("gameweeks"),
                        help="Published GW directories used as completion checkpoints")
    parser.add_argument("--metadata-dir", type=Path,
                        help="Explicit forecast-only checkpoint mode; omitting this checks decision reports")
    parser.add_argument("--event-name", choices=("schedule", "workflow_dispatch"), default="schedule")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--github-output", type=Path,
                        default=Path(os.environ["GITHUB_OUTPUT"]) if os.environ.get("GITHUB_OUTPUT") else None)
    args = parser.parse_args(argv)
    cache = Path(os.environ.get("RUNNER_TEMP", "data/cache")) / "fpl-deadline-cache"
    # Fresh public GET every run. Outages/invalid data fail, never use stale data.
    bootstrap = FPLClient(cache_dir=cache, timeout=20, retries=2).get_json("bootstrap-static/", force=True)
    records = (load_forecast_metadata(args.metadata_dir) if args.metadata_dir is not None
               else load_decision_metadata(args.decision_dir))
    result = decide(bootstrap, records,
                    force=args.event_name == "workflow_dispatch", horizon=args.horizon)
    output = github_outputs(result)
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write(output)
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
