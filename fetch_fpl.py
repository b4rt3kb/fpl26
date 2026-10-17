"""Official current-season data plus validated historical archives.

A refresh publishes a checksummed manifest last. Predictors must validate it
so that stale data or an interrupted refresh cannot be used silently.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
from pathlib import Path

from fpl_api import (APIError, DataValidationError, FPLClient, atomic_write,
                     json_bytes, parse_timestamp, utc_now)

ARCHIVE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
PLAYER_REQUIRED = {"id", "code", "first_name", "second_name", "web_name", "team",
                   "element_type", "now_cost", "status", "chance_of_playing_next_round",
                   "minutes", "total_points"}
TEAM_REQUIRED = {"id", "name", "short_name", "strength_overall_home", "strength_overall_away"}
FIXTURE_REQUIRED = {"id", "event", "team_h", "team_a", "kickoff_time", "finished",
                    "team_h_score", "team_a_score"}
HISTORY_REQUIRED = {"element", "fixture", "opponent_team", "was_home", "round",
                    "kickoff_time", "minutes", "total_points", "value",
                    "expected_goals", "expected_assists", "starts"}
GW_REQUIRED = {"name", "position", "team", "element", "fixture", "opponent_team",
               "was_home", "round", "kickoff_time", "minutes", "total_points", "value",
               "team_h_score", "team_a_score"}


def require_fields(row, required, label):
    if not isinstance(row, dict) or required - row.keys():
        missing = required - row.keys() if isinstance(row, dict) else required
        raise DataValidationError(f"{label}: missing fields {sorted(missing)}")


def positive_int(value, label):
    if isinstance(value, bool):
        raise DataValidationError(f"{label}: invalid integer {value!r}")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{label}: invalid integer {value!r}") from exc
    if number <= 0 or str(number) != str(value):
        raise DataValidationError(f"{label}: invalid positive integer {value!r}")
    return number


def index_unique(rows, key, label, required=()):
    result = {}
    for row in rows:
        require_fields(row, set(required) | {key}, label)
        identifier = positive_int(row[key], f"{label}.{key}")
        if identifier in result:
            raise DataValidationError(f"{label}: duplicate {key}={identifier}")
        result[identifier] = row
    if not result:
        raise DataValidationError(f"{label}: empty data")
    return result


def season_from_bootstrap(bootstrap):
    events = bootstrap.get("events", [])
    if not events:
        raise DataValidationError("Bootstrap has no events")
    first = min(parse_timestamp(event["deadline_time"]) for event in events)
    start = first.year if first.month >= 7 else first.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def season_suffix(season):
    start, end = season.split("-")
    return start[-2:] + end


def validate_catalogue(bootstrap, fixtures):
    require_fields(bootstrap, {"elements", "teams", "events"}, "bootstrap")
    players = index_unique(bootstrap["elements"], "id", "players", PLAYER_REQUIRED)
    index_unique(bootstrap["elements"], "code", "players", PLAYER_REQUIRED)
    teams = index_unique(bootstrap["teams"], "id", "teams", TEAM_REQUIRED)
    events = index_unique(bootstrap["events"], "id", "events", {"deadline_time", "finished"})
    matches = index_unique(fixtures, "id", "fixtures", FIXTURE_REQUIRED)
    for player in players.values():
        if player["team"] not in teams or player["element_type"] not in POSITIONS:
            raise DataValidationError(f"Invalid player team/position: {player['id']}")
    for match in matches.values():
        if match["team_h"] not in teams or match["team_a"] not in teams:
            raise DataValidationError(f"Unknown fixture club: {match['id']}")
        if match["team_h"] == match["team_a"]:
            raise DataValidationError(f"Fixture has same club on both sides: {match['id']}")
        if match["event"] is not None and match["event"] not in events:
            raise DataValidationError(f"Unknown fixture event: {match['id']}")
        if match["finished"] and (match["team_h_score"] is None or match["team_a_score"] is None):
            raise DataValidationError(f"Finished fixture lacks score: {match['id']}")
    for event in events.values():
        if event["finished"]:
            event_matches = [m for m in matches.values() if m["event"] == event["id"]]
            if any(not m["finished"] for m in event_matches):
                raise DataValidationError(f"Event GW{event['id']} finished but fixture unfinished")
    return players, teams, matches


def build_current_rows(bootstrap, fixtures, summaries):
    """Join match-side club IDs, never a transferred player's present club."""
    players, teams, matches = validate_catalogue(bootstrap, fixtures)
    if set(summaries) != set(players):
        raise DataValidationError("Player-summary IDs do not match the bootstrap player catalogue")
    rows, seen, fixture_sides = [], set(), set()
    for element, player in players.items():
        summary = summaries[element]
        require_fields(summary, {"history"}, f"element-summary/{element}")
        if not isinstance(summary["history"], list):
            raise DataValidationError(f"Invalid history for player {element}")
        for source in summary["history"]:
            require_fields(source, HISTORY_REQUIRED, f"player {element} history")
            if source["element"] != element or source["fixture"] not in matches:
                raise DataValidationError(f"Mismatched player/fixture in history for {element}")
            match = matches[source["fixture"]]
            if not isinstance(source["was_home"], bool):
                raise DataValidationError(f"Invalid was_home in player {element} history")
            home = source["was_home"]
            own = match["team_h"] if home else match["team_a"]
            opponent = match["team_a"] if home else match["team_h"]
            if source["opponent_team"] != opponent or source["round"] != match["event"]:
                raise DataValidationError(f"History disagrees with fixture {match['id']}")
            key = (element, source["fixture"])
            if key in seen:
                raise DataValidationError(f"Duplicate current player-fixture row: {key}")
            seen.add(key)
            if not match["finished"]:
                continue  # Do not train on live, partial match points.
            if source["minutes"] is None or source["total_points"] is None:
                raise DataValidationError(f"Finished player-fixture lacks target: {key}")
            parse_timestamp(match["kickoff_time"])
            row = {**source, "name": player["first_name"] + " " + player["second_name"],
                   "position": POSITIONS[player["element_type"]], "team": teams[own]["name"],
                   "team_id": own, "code": player["code"], "GW": source["round"],
                   "team_h_score": match["team_h_score"], "team_a_score": match["team_a_score"],
                   "kickoff_time": match["kickoff_time"]}
            rows.append(row)
            fixture_sides.add((match["id"], own))
    expected_sides = {(m["id"], t) for m in matches.values() if m["finished"]
                      for t in (m["team_h"], m["team_a"])}
    missing = expected_sides - fixture_sides
    if missing:
        raise DataValidationError(f"History missing completed fixture sides: {sorted(missing)}")
    finished_events = {e["id"] for e in bootstrap["events"] if e["finished"]}
    latest_finished = max(finished_events, default=0)
    # A finished event may have no fixtures after postponements. It cannot
    # contribute history rows; require coverage through the latest played one.
    latest_finished_fixture = max((m["event"] for m in matches.values()
                                   if m["finished"] and m["event"] in finished_events), default=0)
    history_max = max((r["round"] for r in rows), default=0)
    if history_max < latest_finished_fixture:
        raise DataValidationError(f"History through GW{history_max}, but GW{latest_finished_fixture} has finished fixtures")
    # Totals detect stale player histories even when all fixtures have some rows.
    in_progress = any(m.get("started") and not m["finished"] for m in matches.values())
    if not in_progress:
        totals = {element: {"minutes": 0, "total_points": 0} for element in players}
        for row in rows:
            for stat in totals[row["element"]]:
                totals[row["element"]][stat] += row[stat]
        for element, sums in totals.items():
            for stat, value in sums.items():
                if value != players[element][stat]:
                    raise DataValidationError(f"Player {element}: history {stat}={value}, "
                                              f"bootstrap={players[element][stat]}; refresh caches")
    return rows, {"latest_finished_gw": latest_finished,
                  "latest_finished_fixture_gw": latest_finished_fixture,
                  "current_history_max_gw": history_max,
                  "completed_fixture_count": len(expected_sides) // 2,
                  "covered_fixture_sides": len(fixture_sides), "history_rows": len(rows)}


def prepare_historical(gw_rows, player_rows, team_rows, season):
    """Keep footballers only, deduplicate exact records, attach stable codes."""
    players = index_unique(player_rows, "id", f"{season} players", {"code", "element_type"})
    teams = index_unique(team_rows, "id", f"{season} teams", TEAM_REQUIRED)
    names = {team["name"] for team in teams.values()}
    seen, rows, excluded, duplicates = {}, [], 0, 0
    for source in gw_rows:
        require_fields(source, GW_REQUIRED, f"{season} history")
        if source["position"] == "AM":
            excluded += 1
            continue
        if source["position"] not in POSITIONS.values():
            raise DataValidationError(f"{season}: unknown position {source['position']!r}")
        element = positive_int(source["element"], "historical element")
        if element not in players or source["team"] not in names:
            raise DataValidationError(f"{season}: unmatched player or team for {element}")
        key = (element, positive_int(source["fixture"], "historical fixture"))
        if key in seen:
            if source != seen[key]:
                raise DataValidationError(f"{season}: conflicting duplicate player-fixture {key}")
            duplicates += 1
            continue
        seen[key] = source
        rows.append({**source, "code": positive_int(players[element]["code"], "player code")})
    if not rows:
        raise DataValidationError(f"{season}: no historical footballer rows")
    return rows, {"excluded_assistant_manager_rows": excluded, "exact_duplicates_removed": duplicates}


def csv_bytes(rows, columns=None):
    stream = io.StringIO(newline="")
    columns = columns or list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    return stream.getvalue().encode("utf-8")


def validate_data_manifest(directory=".", max_age_hours=6, target_gw=None):
    """Require an intact, recent, complete official snapshot before predicting."""
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "data_manifest.json").read_text())
        if manifest["schema_version"] != 1 or manifest["current_source"] != "official-fpl":
            raise DataValidationError("Unsupported or non-official data manifest")
        age = (utc_now() - parse_timestamp(manifest["oldest_current_source_at"])).total_seconds()
        if age < 0 or age > max_age_hours * 3600:
            raise DataValidationError(f"Current snapshot is {age / 3600:.1f}h old; run fetch_fpl.py")
        coverage = manifest["coverage"]
        required_history_gw = coverage.get("latest_finished_fixture_gw", coverage["latest_finished_gw"])
        if coverage["current_history_max_gw"] < required_history_gw:
            raise DataValidationError("Manifest reports incomplete current history")
        for filename, metadata in manifest["files"].items():
            if Path(filename).name != filename:
                raise DataValidationError("Unsafe data path in manifest")
            payload = (directory / filename).read_bytes()
            if hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
                raise DataValidationError(f"Snapshot file changed or incomplete: {filename}")
        if target_gw is not None and manifest.get("next_gw") is not None:
            if target_gw > manifest["next_gw"]:
                raise DataValidationError("Initial forecast target is beyond the next official deadline")
        return manifest
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DataValidationError("Missing or invalid data_manifest.json; run fetch_fpl.py") from exc


def fetch_snapshot(directory=".", *, cache_dir="data/cache", workers=4,
                   max_age_hours=6, historical_seasons=3, force=False):
    if not 1 <= workers <= 6:
        raise ValueError("workers must be between 1 and 6")
    if max_age_hours <= 0 or historical_seasons < 1:
        raise ValueError("max age and number of historical seasons must be positive")
    directory = Path(directory)
    client = FPLClient(cache_dir=cache_dir)
    age = max_age_hours * 3600
    bootstrap = client.get_json("bootstrap-static/", max_age_seconds=age, force=force)
    fixtures = client.get_json("fixtures/", max_age_seconds=age, force=force)
    players, _, _ = validate_catalogue(bootstrap, fixtures)
    season = season_from_bootstrap(bootstrap)
    suffix = season_suffix(season)
    print(f"Fetching official {season}: {len(players)} player histories, {workers} workers", flush=True)

    def get_summary(element):
        return element, client.get_json(f"element-summary/{element}/", max_age_seconds=age, force=force)

    summaries = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (element, summary) in enumerate(pool.map(get_summary, players), 1):
            summaries[element] = summary
            if i % 100 == 0 or i == len(players):
                print(f"Player histories: {i}/{len(players)}", flush=True)
    current_rows, coverage = build_current_rows(bootstrap, fixtures, summaries)
    current_sources = list(client.records)
    payloads = {
        f"players_{suffix}.csv": csv_bytes(bootstrap["elements"]),
        f"teams_{suffix}.csv": csv_bytes(bootstrap["teams"]),
        f"fixtures_{suffix}.csv": csv_bytes(fixtures),
        f"gw_{season}.csv": csv_bytes(current_rows, columns=None if current_rows else sorted(
            GW_REQUIRED | {"code", "team_id", "GW", "expected_goals", "expected_assists", "starts"})),
        "bootstrap.json": json_bytes(bootstrap),
    }
    start = int(season[:4])
    historical, cleaning = [], {}
    for year in range(start - historical_seasons, start):
        old = f"{year}-{str(year + 1)[-2:]}"
        historical.append(old)
        base = f"{ARCHIVE}/{old}"
        gw_rows = client.get_csv(base + "/gws/merged_gw.csv", force=force)
        player_rows = client.get_csv(base + "/players_raw.csv", force=force)
        team_rows = client.get_csv(base + "/teams.csv", force=force)
        cleaned, stats = prepare_historical(gw_rows, player_rows, team_rows, old)
        cleaning[old] = stats
        payloads[f"gw_{old}.csv"] = csv_bytes(cleaned)
        payloads[f"players_{old}.csv"] = csv_bytes(player_rows)
        payloads[f"teams_{old}.csv"] = csv_bytes(team_rows)
    # Compatibility for the old heuristic; never used as current data.
    previous = historical[-1]
    payloads[f"cleaned_{season_suffix(previous)}.csv"] = csv_bytes(
        client.get_csv(f"{ARCHIVE}/{previous}/cleaned_players.csv", force=force))
    next_events = [e["id"] for e in bootstrap["events"] if e.get("is_next")]
    manifest = {"schema_version": 1, "generated_at": utc_now().isoformat(),
                "season": season, "historical_seasons": historical,
                "current_source": "official-fpl", "next_gw": min(next_events, default=None),
                "oldest_current_source_at": min(r["fetched_at"] for r in current_sources),
                "coverage": coverage, "historical_cleaning": cleaning,
                "sources": client.records,
                "files": {name: {"sha256": hashlib.sha256(payload).hexdigest(),
                                  "bytes": len(payload)} for name, payload in payloads.items()}}
    # Manifest last: interrupted file replacements fail the old checksums.
    for name, payload in payloads.items():
        atomic_write(directory / name, payload)
    atomic_write(directory / "data_manifest.json", json_bytes(manifest))
    validate_data_manifest(directory, max_age_hours=max_age_hours)
    print(f"Validated {season}: {coverage['completed_fixture_count']} completed fixtures, "
          f"history through GW{coverage['current_history_max_gw']}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default=".")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-age-hours", type=float, default=6)
    parser.add_argument("--historical-seasons", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        fetch_snapshot(args.directory, cache_dir=args.cache_dir, workers=args.workers,
                       max_age_hours=args.max_age_hours, historical_seasons=args.historical_seasons,
                       force=args.force)
    except (APIError, DataValidationError, ValueError) as exc:
        parser.exit(1, f"Data refresh failed: {exc}\n")


if __name__ == "__main__":
    main()
