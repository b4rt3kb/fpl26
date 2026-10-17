"""Weekly recommendations from complete public FPL account history.

No login or manual bank/FT state is required. Public data cannot reveal edits
since the last deadline: recommendations are conditional on that published
squad. This script never executes transfers, changes lineups or activates chips.
Full account state stays local. --publish-dir exports only the recommendation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html import escape
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import pandas as pd

from fpl_api import FPLClient, atomic_write, json_bytes, parse_timestamp
from team_state import build_team_state, free_transfers_next, atomic_write_json, StateValidationError
from plan_transfers import plan, refine_plan_lineups


def next_deadline(bootstrap, now=None):
    now = now or datetime.now(timezone.utc)
    events = [event for event in bootstrap["events"] if parse_timestamp(event["deadline_time"]) > now]
    if not events:
        raise StateValidationError("No future gameweek deadline remains")
    return min(events, key=lambda event: parse_timestamp(event["deadline_time"]))


def reconstructed_free_transfers(history, through_event, *, maximum=5, overrides=None):
    """GW1 is unlimited; GW2 starts with one FT. Replay all subsequent weeks."""
    rows = sorted((r for r in history["current"] if r["event"] <= through_event), key=lambda r: r["event"])
    if [r["event"] for r in rows] != list(range(1, through_event+1)):
        raise StateValidationError("Automatic FT reconstruction requires complete history from GW1")
    chips = {}
    for chip in history.get("chips", []):
        if chip["event"] in chips:
            raise StateValidationError("More than one chip in a gameweek")
        chips[chip["event"]] = chip["name"]
    overrides = {int(gw): int(value) for gw, value in (overrides or {}).items()}
    if any(not 1 <= gw <= 38 or not 1 <= value <= maximum for gw, value in overrides.items()):
        raise StateValidationError("Invalid dated free-transfer override")
    free, ledger = overrides.get(2, 1), []
    for row in rows[1:]:
        gw = row["event"]
        free = overrides.get(gw, free)
        count = row["event_transfers"]
        chip = chips.get(gw)
        expected_hit = 0 if chip in ("wildcard", "freehit") else 4*max(0, count-free)
        if row["event_transfers_cost"] != expected_hit:
            raise StateValidationError(f"GW{gw} published hit cost disagrees with reconstructed FT")
        following = overrides.get(gw+1, free_transfers_next(free, count, chip, maximum))
        ledger.append({"gw": gw, "free_before": free, "transfers": count, "chip": chip,
                       "hit_cost": expected_hit, "free_next": following})
        free = following
    return free, ledger


def fetch_public_state(entry_id, *, data_dir=Path("."), client=None, now=None,
                       expected_ft=None, expected_gw=None):
    if type(entry_id) is not int or entry_id <= 0:
        raise StateValidationError("Entry ID must be positive")
    if (expected_ft is None) != (expected_gw is None):
        raise StateValidationError("A one-off FT anchor needs both expected_ft and expected_gw")
    now = now or datetime.now(timezone.utc)
    client = client or FPLClient(cache_dir=Path(data_dir)/"data/private-cache")
    def get(endpoint):
        return client.get_json(endpoint, max_age_seconds=0, force=True)
    bootstrap = get("bootstrap-static/")
    target = next_deadline(bootstrap, now)
    through = int(target["id"])-1
    if through < 1:
        raise StateValidationError("No published starting squad exists before GW1")
    entry = get(f"entry/{entry_id}/")
    history = get(f"entry/{entry_id}/history/")
    transfers = get(f"entry/{entry_id}/transfers/")
    if entry.get("id") != entry_id or entry.get("started_event", 1) != 1:
        raise StateValidationError("Entry identity differs or entry did not start in GW1")
    if max((r["event"] for r in history.get("current", [])), default=0) != through:
        raise StateValidationError("Public account history is not current through the previous deadline")
    if any(t["event"] > through for t in transfers):
        raise StateValidationError("Public pending transfers require reconciliation before a recommendation")
    with ThreadPoolExecutor(max_workers=4) as workers:
        payloads = list(workers.map(get, [f"entry/{entry_id}/event/{gw}/picks/" for gw in range(1, through+1)]))
    picks = dict(zip(range(1, through+1), payloads))
    free, ledger = reconstructed_free_transfers(history, through,
        maximum=bootstrap.get("game_settings", {}).get("max_extra_free_transfers", 4)+1)
    if target["id"] == expected_gw and free != expected_ft:
        raise StateValidationError(f"GW{expected_gw} FT anchor failed: reconstructed {free}, expected {expected_ft}")
    state = build_team_state(bootstrap, entry, history, transfers, picks,
                            free_transfers=free, through_event=through, bank=None,
                            imported_at=now.isoformat(), source="fresh_public_fpl_api")
    state["provenance"].update(free_transfers_source="complete_public_history_replay_GW2_starts_at_one",
                               free_transfers_ledger=ledger, fetches=client.records,
                               target_deadline=target["deadline_time"], unpublished_edits_known=False)
    return state, bootstrap, target


def validate_forecast_for_decision(path, state, deadline, now=None):
    now = now or datetime.now(timezone.utc)
    if now >= parse_timestamp(deadline):
        raise StateValidationError("Deadline passed; refusing to publish an expired decision")
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    generated = parse_timestamp(metadata["generated_utc"])
    if generated > now or (now-generated).total_seconds() > 6*3600:
        raise StateValidationError("Forecast is stale or has a future generation time")
    if generated >= parse_timestamp(deadline):
        raise StateValidationError("Forecast was generated after its target deadline")
    if metadata["gameweeks"][0] != state["next_event"]:
        raise StateValidationError("Forecast targets a different next gameweek")
    if state.get("season") and metadata["season"] != state["season"].replace("/", "-"):
        raise StateValidationError("Forecast and account season differ")
    frame = pd.read_csv(path)
    if sorted(frame.gw.unique()) != metadata["gameweeks"]:
        raise StateValidationError("Forecast rows and metadata disagree")
    if not {p["id"] for p in state["players"]}.issubset(set(frame.element)):
        raise StateValidationError("Forecast omits a currently owned player")
    return frame, metadata


def validate_current_inputs(fresh, stored, gameweeks=None):
    """Reject changes that affect purchases, availability or target deadlines."""
    fields = ("status", "news", "news_added", "chance_of_playing_next_round",
              "now_cost", "team", "can_select", "element_type")
    old = {int(p["id"]): p for p in stored["elements"]}
    new = {int(p["id"]): p for p in fresh["elements"]}
    if set(old) != set(new):
        raise StateValidationError("Player catalogue changed; refresh data and regenerate forecast")
    for ident in old:
        changed = [field for field in fields if old[ident].get(field) != new[ident].get(field)]
        if changed:
            raise StateValidationError(f"Player {ident} changed ({', '.join(changed)}); refresh data and regenerate forecast")
    old_events = {int(e["id"]): e["deadline_time"] for e in stored["events"]}
    new_events = {int(e["id"]): e["deadline_time"] for e in fresh["events"]}
    for gw in gameweeks or old_events:
        if old_events.get(gw) != new_events.get(gw):
            raise StateValidationError(f"GW{gw} deadline changed; refresh data and regenerate forecast")


def validate_fixture_schedule(fresh, stored):
    fields = ("id", "event", "team_h", "team_a", "kickoff_time")
    def schedule(rows):
        result = {}
        for row in rows:
            values = []
            for field in fields:
                value = row.get(field)
                if value is None or pd.isna(value):
                    value = None
                elif field == "kickoff_time":
                    value = parse_timestamp(value).isoformat()
                else:
                    value = int(value)
                values.append(value)
            if values[0] in result:
                raise StateValidationError("Duplicate fixture in schedule")
            result[values[0]] = tuple(values)
        return result
    if schedule(fresh) != schedule(stored):
        raise StateValidationError("Fixture schedule changed; refresh data and regenerate forecast")


def validate_forecast_sources(data_dir, metadata, fresh_bootstrap, fresh_fixtures):
    """Compare live inputs with the exact source bytes used by the forecast."""
    data_dir = Path(data_dir)
    files = metadata.get("data_manifest", {}).get("files", {})
    fixture_names = [name for name in files if name.startswith("fixtures_") and name.endswith(".csv")]
    if "bootstrap.json" not in files or len(fixture_names) != 1:
        raise StateValidationError("Forecast lacks verified source files; refresh data and regenerate forecast")
    def verified(name):
        if Path(name).name != name:
            raise StateValidationError("Invalid source filename in forecast metadata")
        payload = (data_dir/name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != files[name]["sha256"]:
            raise StateValidationError(f"Forecast source hash differs for {name}; regenerate forecast")
        return payload
    stored_bootstrap = json.loads(verified("bootstrap.json"))
    fixture_name = fixture_names[0]
    verified(fixture_name)
    stored_fixtures = pd.read_csv(data_dir/fixture_name).to_dict("records")
    validate_current_inputs(fresh_bootstrap, stored_bootstrap, metadata["gameweeks"])
    validate_fixture_schedule(fresh_fixtures, stored_fixtures)


def decision_html(result, state, deadline):
    names = {p["id"]: p.get("name", str(p["id"])) for p in result["players"]}
    def labels(ids):
        return ", ".join(escape(names.get(i, str(i))) for i in ids) or "none"
    step = result["steps"][0]
    lineup = "".join(f"<li>{escape(names[i])}{' (C)' if i == step['captain'] else ' (VC)' if i == step['vice_captain'] else ''}</li>" for i in step["starting_xi"])
    horizon = "".join(f"<tr><td>GW{s['gw']}</td><td>{escape(s['chip'] or 'save chips')}</td><td>{labels(s['transfer_out'])}</td><td>{labels(s['transfer_in'])}</td><td>{s['hit_cost']}</td></tr>" for s in result["steps"])
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FPL decision GW{step['gw']}</title>
<style>body{{font:17px/1.6 system-ui;max-width:940px;margin:40px auto;padding:0 24px;color:#172238}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #ddd}}.note{{background:#fff4d6;padding:16px}}small{{color:#526074}}</style>
<h1>GW{step['gw']}: decision before the deadline</h1><p>Deadline: <strong>{escape(deadline)}</strong> · available FT: {state['free_transfers']} · bank: £{state['bank']/10:.1f}m</p>
<p class="note">The plan uses the latest public squad and complete transfer history. Changes after the last deadline are not public. This report makes no account changes.</p>
<h2>Transfers and chip</h2><p><strong>Sell:</strong> {labels(step['transfer_out'])}<br><strong>Buy:</strong> {labels(step['transfer_in'])}<br><strong>Chip:</strong> {escape(step['chip'] or 'save')} · cost: {step['hit_cost']} points · bank after transfers: £{step['bank']/10:.1f}m</p>
<h2>Starting XI, captain and bench</h2><ul>{lineup}</ul><p><strong>Bench order:</strong> {labels(step['bench'])}</p>
<p>Net points forecast with modelled substitutions: <strong>{step.get('expected_net_points_with_contingencies', step['net_predicted_points']):.1f}</strong>.</p>
<h2>Future plan, recalculated each week</h2><table><tr><th>Gameweek</th><th>Chip</th><th>Sell</th><th>Buy</th><th>Hit</th></tr>{horizon}</table>
<p><small>Public squad after GW{state['as_of_event']}. Fetched at {escape(state['imported_at'])}. Player appearances are modelled independently.</small></p></html>"""


def public_recommendation(result, state, metadata):
    """Explicit allowlist: no profile, entry ID, history or full player pool."""
    step = result["steps"][0]
    players = {p["id"]: p for p in result["players"]}
    owned = {p["id"]: p for p in state["players"]}
    def player(ident):
        row = players[ident]
        return {"id": ident, "name": row["name"], "position": row["pos"],
                "price": row["price"]/10}
    outgoing = step["temporary_out"] if step["chip"] == "freehit" else step["transfer_out"]
    incoming = step["temporary_in"] if step["chip"] == "freehit" else step["transfer_in"]
    transfers = []
    for position in ("GK", "DEF", "MID", "FWD"):
        old = sorted(i for i in outgoing if players[i]["pos"] == position)
        new = sorted(i for i in incoming if players[i]["pos"] == position)
        if len(old) != len(new):
            raise StateValidationError("Transfer positions do not balance")
        for before, after in zip(old, new):
            sale = player(before)
            sale["selling_price"] = owned[before]["selling_price"]/10
            transfers.append({"out": sale, "in": player(after)})
    deadline = result["deadline"]
    position_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    starting_xi = sorted((player(i) for i in step["starting_xi"]),
                         key=lambda row: (position_order[row["position"]], row["name"]))
    bench, outfield_order = [], 0
    for ident in step["bench"]:
        row = player(ident)
        if row["position"] == "GK":
            row["order"] = 0
        else:
            outfield_order += 1
            row["order"] = outfield_order
        bench.append(row)
    return {"schema_version": 1, "generated_utc": result["generated_utc"],
            "season": metadata["season"], "gameweek": step["gw"],
            "gameweeks": metadata["gameweeks"], "deadline": deadline,
            "deadline_local": parse_timestamp(deadline).astimezone(ZoneInfo("Europe/Warsaw")).strftime("%Y-%m-%d %H:%M %Z"),
            "model": metadata["model"], "forecast_generated_utc": metadata["generated_utc"],
            "public_state_event": state["as_of_event"], "public_state_fetched_utc": state["imported_at"],
            "transfers": transfers, "chip": step["chip"], "hit_cost": step["hit_cost"],
            "bank_before": state["bank"]/10, "bank_after": step["bank"]/10,
            "free_transfers_before": step["free_transfers_before"],
            "free_transfers_next": step["free_transfers_next"],
            "starting_xi": starting_xi,
            "captain": player(step["captain"]), "vice_captain": player(step["vice_captain"]),
            "bench": bench,
            "expected_net_points": round(step.get("expected_net_points_with_contingencies", step["net_predicted_points"]), 2)}


def publish_decision(destination, result, state, metadata, now=None):
    now = now or datetime.now(timezone.utc)
    if now >= parse_timestamp(result["deadline"]):
        raise StateValidationError("Deadline passed; refusing to publish an expired decision")
    report = public_recommendation(result, state, metadata)
    return render_public_decision(destination, report)


def render_public_decision(destination, report):
    """Render an existing public decision without changing its data or timestamps."""
    target = Path(destination)/f"GW{report['gameweek']}"
    def md(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    def name(row):
        return md(row["name"])
    moves = [f"{name(t['out'])} (£{t['out']['selling_price']:.1f}m) → {name(t['in'])} (£{t['in']['price']:.1f}m)" for t in report["transfers"]]
    labels = {"wildcard": "Wildcard", "freehit": "Free Hit", "bboost": "Bench Boost", "3xc": "Triple Captain"}
    chip = labels.get(report["chip"], report["chip"] or "none")
    lines = [f"# GW{report['gameweek']}", "", f"**Deadline: {report['deadline_local']}.**", "",
             "**Transfers:**", ""]
    lines += [f"- {move}" for move in moves] if moves else ["Keep the squad."]
    lines += ["",
             f"**Chip:** {chip}. **Cost:** {report['hit_cost']} points.", "",
             f"Bank: £{report['bank_before']:.1f}m → £{report['bank_after']:.1f}m. "
             f"FT: {report['free_transfers_before']} now → {report['free_transfers_next']} next gameweek.", "",
             f"**Captain:** {name(report['captain'])}. **Vice-captain:** {name(report['vice_captain'])}.", "",
             "| Position | Starting XI |", "|---|---|"]
    for row in report["starting_xi"]:
        suffix = " (C)" if row["id"] == report["captain"]["id"] else " (VC)" if row["id"] == report["vice_captain"]["id"] else ""
        lines.append(f"| {row['position']} | {name(row)}{suffix} |")
    reserve_keeper = next(p for p in report["bench"] if p["position"] == "GK")
    lines += ["", "**Bench:** " + ", ".join(f"{p['order']}. {name(p)}" for p in report["bench"] if p["position"] != "GK") + ".",
              f"**Reserve goalkeeper:** {name(reserve_keeper)}.", "",
              f"Net points forecast: **{report['expected_net_points']:.1f}**. Model: `{report['model']}`.", "",
              f"Input: public squad after GW{report['public_state_event']}. Updated: {report['generated_utc']}.", "",
              "[HTML report](decision.html) · [Lineup CSV](lineup.csv) · [Decision data](decision.json)", ""]
    markdown = "\n".join(lines)
    lineup_rows = []
    for role, rows in (("XI", report["starting_xi"]), ("bench", report["bench"])):
        for row in rows:
            lineup_rows.append({"element": row["id"], "name": row["name"], "position": row["position"],
                                "role": role, "bench_order": row.get("order", ""),
                                "is_captain": row["id"] == report["captain"]["id"],
                                "is_vice_captain": row["id"] == report["vice_captain"]["id"], "price": row["price"]})
    table = "".join(f"<tr><td>{escape(r['position'])}</td><td>{escape(r['name'])}{' (C)' if r['is_captain'] else ' (VC)' if r['is_vice_captain'] else ''}</td><td>{'XI' if r['role'] == 'XI' else 'Reserve goalkeeper' if r['position'] == 'GK' else 'Bench '+str(r['bench_order'])}</td></tr>" for r in lineup_rows)
    moves_html = "<ul>" + "".join(f"<li>{escape(move)}</li>" for move in moves) + "</ul>" if moves else "<p>Keep the squad.</p>"
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FPL GW{report['gameweek']}</title>
<style>body{{font:17px/1.6 system-ui;max-width:820px;margin:36px auto;padding:0 24px;color:#172238}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:8px;border-bottom:1px solid #ddd}}</style>
<h1>GW{report['gameweek']}</h1><p><strong>Deadline: {escape(report['deadline_local'])}</strong></p>
<h2>Transfers</h2>{moves_html}<p><strong>Chip:</strong> {escape(chip)} · cost: {report['hit_cost']} points.</p>
<p>Bank: £{report['bank_before']:.1f}m → £{report['bank_after']:.1f}m. FT: {report['free_transfers_before']} now → {report['free_transfers_next']} next gameweek.</p>
<p><strong>Captain:</strong> {escape(report['captain']['name'])}. <strong>Vice-captain:</strong> {escape(report['vice_captain']['name'])}.</p>
<table><tr><th>Position</th><th>Player</th><th>Lineup</th></tr>{table}</table><p>Net points forecast: <strong>{report['expected_net_points']:.1f}</strong>. Model: {escape(report['model'])}.</p>
<p>Input: public squad after GW{report['public_state_event']}. Updated: {escape(report['generated_utc'])}.</p></html>"""
    # Publish the checkpoint last, after every companion file is complete.
    atomic_write(target/"decision.html", html.encode())
    atomic_write(target/"lineup.csv", pd.DataFrame(lineup_rows).to_csv(index=False).encode())
    atomic_write(target/"README.md", markdown.encode())
    atomic_write(target/"decision.json", json_bytes(report))
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--forecast", type=Path)
    parser.add_argument("--refresh-forecast", action="store_true")
    parser.add_argument("--expected-ft", type=int, help="Optional one-off audit anchor, paired with --expected-gw")
    parser.add_argument("--expected-gw", type=int)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--publish-dir", type=Path, help="Publish a minimal report under GW<number>; keep full state private")
    args = parser.parse_args(argv)
    data = args.data_dir.resolve()
    state, bootstrap, event = fetch_public_state(args.entry, data_dir=data,
        expected_ft=args.expected_ft, expected_gw=args.expected_gw)
    output = data/"state"/"weekly"
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output/"team_state.json", state)
    forecast = args.forecast or data/"predictions"/f"all_players_gw{event['id']}.csv"
    if args.refresh_forecast:
        subprocess.run([sys.executable, str(Path(__file__).with_name("predict_gw.py")),
                        "--data-dir", str(data), "--gw", str(event["id"]), "--horizon", str(min(6, 39-event["id"])),
                        "--no-squad", "--output", str(forecast)], check=True)
    frame, metadata = validate_forecast_for_decision(forecast, state, event["deadline_time"])
    client = FPLClient(cache_dir=data/"data/private-cache")
    fresh_bootstrap = client.get_json("bootstrap-static/", max_age_seconds=0, force=True)
    fresh_fixtures = client.get_json("fixtures/", max_age_seconds=0, force=True)
    validate_current_inputs(fresh_bootstrap, bootstrap, metadata["gameweeks"])
    validate_forecast_sources(data, metadata, fresh_bootstrap, fresh_fixtures)
    # Explicit, documented seasonal overrides may be added to state.rules;
    # there is no invented recurring AFCON reset for a different season.
    result = refine_plan_lineups(plan(state, frame, seconds=args.seconds), frame)
    if datetime.now(timezone.utc) >= parse_timestamp(event["deadline_time"]):
        raise StateValidationError("Deadline passed during planning; no decision will be published")
    result.update(generated_utc=datetime.now(timezone.utc).isoformat(), deadline=event["deadline_time"],
                  forecast_generated_utc=metadata["generated_utc"], public_state_event=state["as_of_event"],
                  unpublished_edits_known=False, transfers_executed=False)
    atomic_write_json(output/"decision.json", result)
    atomic_write(output/"decision.html", decision_html(result, state, event["deadline_time"]).encode())
    if args.publish_dir:
        destination = args.publish_dir if args.publish_dir.is_absolute() else data/args.publish_dir
        published = publish_decision(destination, result, state, metadata)
        print(f"Recommendation saved for GW{event['id']} in {published}; no account changes executed.")
    else:
        print(f"Private decision saved for GW{event['id']} in state/weekly; no account changes executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
