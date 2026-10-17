"""Validated, privacy-minimal FPL account snapshots. All money is integer £0.1m.

Public picks describe a deadline, not a manager's unpublished next-GW edits.
Initial purchase prices are inferred only for GW1, when prices were fixed.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile

SCHEMA_VERSION = 1
POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
QUOTAS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
API_BASE = "https://fantasy.premierleague.com/api/"


class StateValidationError(ValueError):
    """An account cannot be reconstructed consistently from its evidence."""


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise StateValidationError(f"{label} must be an integer >= {minimum}: {value!r}")
    return value


def selling_price(purchase_price: int, current_price: int) -> int:
    """FPL keeps half of price gains, rounded down; losses are borne in full."""
    integer(purchase_price, "purchase_price", 1)
    integer(current_price, "current_price", 1)
    return min(current_price, purchase_price + max(0, current_price - purchase_price) // 2)


def free_transfers_next(free_transfers: int, transfers: int, chip=None, maximum=5) -> int:
    integer(free_transfers, "free_transfers")
    integer(transfers, "transfers")
    if not 1 <= free_transfers <= maximum:
        raise StateValidationError("free_transfers must be between 1 and the seasonal maximum")
    if chip in ("wildcard", "freehit"):
        return free_transfers
    return min(maximum, max(0, free_transfers - transfers) + 1)


def replay_transfers(holdings: dict, transfers: list) -> dict:
    """Return new holdings, preserving exact acquisition prices through buybacks."""
    result = deepcopy(holdings)
    for transfer in sorted(transfers, key=lambda t: (t["event"], t["time"])):
        outgoing = integer(transfer["element_out"], "element_out", 1)
        incoming = integer(transfer["element_in"], "element_in", 1)
        price = integer(transfer["element_in_cost"], "element_in_cost", 1)
        integer(transfer["element_out_cost"], "element_out_cost", 1)
        if outgoing not in result:
            raise StateValidationError(f"Transfer sells unowned player {outgoing}")
        if incoming in result:
            raise StateValidationError(f"Transfer buys already-owned player {incoming}")
        del result[outgoing]
        result[incoming] = {
            "purchase_price": price,
            "purchase_price_source": "public_transfer.element_in_cost",
            "purchase_event": transfer["event"],
            "purchase_time": transfer["time"],
            "inferred_purchase_price": False,
        }
    return result


def chip_windows(bootstrap: dict, used_chips: list, next_event: int) -> list:
    result = []
    for chip in bootstrap.get("chips", []):
        used = [c for c in used_chips if c["name"] == chip["name"]
                and chip["start_event"] <= c["event"] <= chip["stop_event"]]
        if len(used) > 1:
            raise StateValidationError(f"Chip {chip['name']} used twice in one availability window")
        status = ("used" if used else "expired" if next_event > chip["stop_event"]
                  else "upcoming" if next_event < chip["start_event"] else "available")
        consecutive_freehit = (chip["name"] == "freehit" and
                               any(c["name"] == "freehit" and c["event"] == next_event - 1 for c in used_chips))
        if status == "available" and consecutive_freehit:
            status = "blocked"
        result.append({
            "id": chip["id"], "name": chip["name"], "start_event": chip["start_event"],
            "stop_event": chip["stop_event"], "status": status,
            "used_event": used[0]["event"] if used else None,
            "unavailable_reason": "freehit_cannot_be_used_in_consecutive_gameweeks" if status == "blocked" else None,
        })
    for used in used_chips:
        if not any(c["name"] == used["name"] and c["used_event"] == used["event"] for c in result):
            raise StateValidationError(f"Used chip has no matching seasonal window: {used}")
    return result


def _safe_picks(payload: dict, elements: dict) -> list:
    picks = []
    for raw in payload["picks"]:
        pid = integer(raw["element"], "pick.element", 1)
        if pid not in elements:
            raise StateValidationError(f"Player {pid} missing from bootstrap")
        picks.append({
            "element": pid, "name": elements[pid]["web_name"],
            "position": integer(raw["position"], "pick.position", 1),
            "multiplier": integer(raw["multiplier"], "pick.multiplier"),
            "is_captain": bool(raw["is_captain"]),
            "is_vice_captain": bool(raw["is_vice_captain"]),
            "element_type": raw.get("element_type", elements[pid]["element_type"]),
        })
    if len(picks) != 15 or len({p["element"] for p in picks}) != 15:
        raise StateValidationError("Each gameweek must contain 15 distinct picks")
    if {p["position"] for p in picks} != set(range(1, 16)):
        raise StateValidationError("Pick positions must cover 1..15 exactly")
    return sorted(picks, key=lambda p: p["position"])


def _lineup(picks, event):
    captain = [p["element"] for p in picks if p["is_captain"]]
    vice = [p["element"] for p in picks if p["is_vice_captain"]]
    if len(captain) != 1 or len(vice) != 1 or captain == vice:
        raise StateValidationError("Exactly one distinct captain and vice-captain are required")
    return {"source_event": event, "starting_xi": [p["element"] for p in picks[:11]],
            "bench": [p["element"] for p in picks[11:]],
            "captain": captain[0], "vice_captain": vice[0]}


def validate_state(state: dict, *, allow_club_overflow=False) -> None:
    if state.get("schema_version") != SCHEMA_VERSION:
        raise StateValidationError("Unsupported team state schema_version")
    players = state.get("players", [])
    if len(players) != 15 or len({p["id"] for p in players}) != 15:
        raise StateValidationError("Squad must contain 15 distinct players")
    if Counter(p["position"] for p in players) != Counter(QUOTAS):
        raise StateValidationError("Squad must contain 2 GKP, 5 DEF, 5 MID and 3 FWD")
    for player in players:
        integer(player["id"], "player.id", 1)
        integer(player["team"], "player.team", 1)
        integer(player["selling_price"], "player.selling_price", 1)
        if selling_price(player["purchase_price"], player["now_cost"]) != player["selling_price"]:
            raise StateValidationError(f"Incorrect selling price for player {player['id']}")
    counts = Counter(p["team"] for p in players)
    if max(counts.values()) > 3 and not allow_club_overflow:
        raise StateValidationError("Squad exceeds 3 players from one club; use explicit club-overflow override only for a real club transfer")
    integer(state["bank"], "bank")
    integer(state["free_transfers"], "free_transfers", 1)
    if state["free_transfers"] > state["rules"]["max_free_transfers"]:
        raise StateValidationError("Free transfers exceed the seasonal maximum")
    lineup = state["lineup"]
    selected = lineup["starting_xi"] + lineup["bench"]
    if len(lineup["starting_xi"]) != 11 or len(lineup["bench"]) != 4 or len(set(selected)) != 15:
        raise StateValidationError("Lineup must contain 11 starters and four ordered substitutes")
    if set(selected) != {p["id"] for p in players}:
        raise StateValidationError("Lineup and squad do not match")
    if (lineup["captain"] == lineup["vice_captain"] or
            not {lineup["captain"], lineup["vice_captain"]} <= set(lineup["starting_xi"])):
        raise StateValidationError("Captain and vice-captain must be distinct starters")
    roles = {p["id"]: p["position"] for p in players}
    formation = Counter(roles[pid] for pid in lineup["starting_xi"])
    if formation["GKP"] != 1 or formation["DEF"] < 3 or formation["MID"] < 2 or formation["FWD"] < 1:
        raise StateValidationError("Starting formation is invalid")
    if roles[lineup["bench"][0]] != "GKP":
        raise StateValidationError("First substitute slot must be the reserve goalkeeper")


def build_team_state(bootstrap: dict, entry: dict, history: dict, transfers: list,
                     picks_by_event: dict, *, free_transfers: int, bank: int | None = None,
                     through_event: int | None = None, purchase_prices: dict | None = None,
                     allow_club_overflow=False, source="public_fpl_api", imported_at=None) -> dict:
    """Reconstruct permanent holdings, validate every GW, and retain public history.

    free_transfers and an optional bank are explicit user-confirmed inputs. The
    public API cannot establish unpublished transfers/chip activations. A Free
    Hit's picks are temporary; permanent acquisition prices survive it.
    """
    elements = {p["id"]: p for p in bootstrap["elements"]}
    teams = {t["id"]: t for t in bootstrap["teams"]}
    rows = sorted(history["current"], key=lambda r: r["event"])
    if not rows:
        raise StateValidationError("No public gameweek history is available")
    through_event = through_event if through_event is not None else rows[-1]["event"]
    integer(through_event, "through_event", 1)
    rows = [r for r in rows if r["event"] <= through_event]
    if not rows or rows[-1]["event"] != through_event:
        raise StateValidationError(f"History does not contain GW{through_event}")
    first = rows[0]["event"]
    if first != entry.get("started_event", first) or [r["event"] for r in rows] != list(range(first, through_event + 1)):
        raise StateValidationError("Complete consecutive history from the entry's first GW is required")
    picks_by_event = {int(k): v for k, v in picks_by_event.items()}
    if any(r["event"] not in picks_by_event for r in rows):
        raise StateValidationError("Missing picks for one or more gameweeks")
    overrides = {int(k): integer(v, "purchase price override", 1) for k, v in (purchase_prices or {}).items()}
    initial = _safe_picks(picks_by_event[first], elements)
    holdings = {}
    warnings = ["Public picks are deadline snapshots. Transfers, lineup changes and chip activations made since that deadline may be private; confirm the current squad before using recommendations."]
    for pick in initial:
        pid = pick["element"]
        if pid in overrides:
            price, price_source, inferred = overrides[pid], "user_confirmed_initial_purchase_price", False
        elif first == 1:
            price = elements[pid]["now_cost"] - elements[pid]["cost_change_start"]
            price_source, inferred = "bootstrap.now_cost-minus-cost_change_start_at_GW1", True
        else:
            raise StateValidationError(f"Initial purchase price for player {pid} is unknown: entry started GW{first}; provide --purchase-prices")
        integer(price, f"initial purchase price {pid}", 1)
        holdings[pid] = {"purchase_price": price, "purchase_price_source": price_source,
                         "purchase_event": first, "purchase_time": None, "inferred_purchase_price": inferred}
    if first == 1 and sum(p["purchase_price"] for p in holdings.values()) + rows[0]["bank"] != bootstrap.get("game_settings", {}).get("squad_total_spend", 1000):
        raise StateValidationError("Inferred GW1 acquisition costs plus bank do not reconcile to the initial budget")
    relevant_transfers = [t for t in transfers if first < t["event"] <= through_event]
    used_chips = [{"name": c["name"], "event": c["event"]} for c in history.get("chips", []) if c["event"] <= through_event]
    records, previous_bank, previous_total = [], None, 0
    lineup = None
    for row in rows:
        gw = row["event"]
        payload = picks_by_event[gw]
        picks = _safe_picks(payload, elements)
        active_chip = payload.get("active_chip")
        if active_chip:
            chip_record = {"name": active_chip, "event": gw}
            if chip_record not in used_chips:
                used_chips.append(chip_record)
        declared = [c["name"] for c in used_chips if c["event"] == gw]
        if declared != ([active_chip] if active_chip else []):
            raise StateValidationError(f"GW{gw} chip history and picks disagree")
        for field in ("event", "points", "total_points", "bank", "value", "event_transfers", "event_transfers_cost"):
            if payload["entry_history"].get(field) != row.get(field):
                raise StateValidationError(f"GW{gw} history and picks disagree on {field}")
        moves = sorted([t for t in relevant_transfers if t["event"] == gw], key=lambda t: t["time"])
        reconciliation = "transfer_replay_matches_picks"
        if active_chip == "freehit":
            if moves:
                temporary = replay_transfers(holdings, moves)
                if set(temporary) != {p["element"] for p in picks}:
                    raise StateValidationError(f"GW{gw} Free Hit transfer replay differs from picks")
            else:
                reconciliation = "free_hit_picks_temporary_no_public_transfer_ledger"
            if lineup is None:
                raise StateValidationError("Cannot reconstruct permanent squad from an opening Free Hit")
        else:
            holdings = replay_transfers(holdings, moves)
            if set(holdings) != {p["element"] for p in picks}:
                raise StateValidationError(f"GW{gw} transfer replay differs from public picks")
            if len(moves) != row["event_transfers"]:
                raise StateValidationError(f"GW{gw} transfer count differs from history")
            if previous_bank is not None:
                expected_bank = previous_bank + sum(t["element_out_cost"] - t["element_in_cost"] for t in moves)
                if expected_bank != row["bank"]:
                    raise StateValidationError(f"GW{gw} transfer cash ledger does not match bank")
            previous_bank = row["bank"]
            lineup = _lineup(picks, gw)
        net = row["points"] - row["event_transfers_cost"]
        if previous_total + net != row["total_points"]:
            raise StateValidationError(f"GW{gw} net points do not reconcile with cumulative score")
        previous_total = row["total_points"]
        record = {k: row[k] for k in ("event", "points", "total_points", "bank", "value", "rank", "overall_rank", "points_on_bench") if k in row}
        record.update({"transfer_cost": row["event_transfers_cost"], "net_points": net,
                       "transfer_count": row["event_transfers"], "active_chip": active_chip,
                       "picks": picks, "automatic_subs": [{k: s[k] for k in ("element_in", "element_out", "event") if k in s} for s in payload.get("automatic_subs", [])],
                       "transfers": [{k: t[k] for k in ("event", "time", "element_in", "element_out", "element_in_cost", "element_out_cost")} for t in moves],
                       "reconciled": True, "reconciliation": reconciliation,
                       "source_url": f"{API_BASE}entry/{entry['id']}/event/{gw}/picks/"})
        records.append(record)
    players = []
    for pid, acquisition in sorted(holdings.items()):
        player = elements[pid]
        price = integer(player["now_cost"], f"now_cost {pid}", 1)
        players.append({"id": pid, "name": player["web_name"],
                        "position": POSITIONS[player["element_type"]], "element_type": player["element_type"],
                        "team": player["team"], "team_name": teams[player["team"]]["short_name"],
                        **acquisition, "selling_price": selling_price(acquisition["purchase_price"], price), "now_cost": price})
    max_ft = bootstrap.get("game_settings", {}).get("max_extra_free_transfers", 4) + 1
    observed_bank = previous_bank
    if bank is not None and bank != observed_bank:
        warnings.append(f"User-confirmed bank {bank} differs from last permanent deadline bank {observed_bank}; post-deadline squad changes must be reconciled separately.")
    if any(p["inferred_purchase_price"] for p in players):
        warnings.append("Untransferred GW1 purchase prices were inferred from current price minus season price change; transferred acquisitions use exact public transaction costs.")
    club_overflow = sorted(team for team, count in Counter(p["team"] for p in players).items() if count > 3)
    if club_overflow and allow_club_overflow:
        warnings.append(f"Explicitly accepted existing squad club overflow for club IDs {club_overflow}; the planner must restore a legal club quota on the next transfer.")
    static_url = bootstrap.get("game_config", {}).get("settings", {}).get("static_content_url", "")
    season_match = re.search(r"/(20\d{2})_(\d{2})/", static_url)
    event_info = next((e for e in bootstrap.get("events", []) if e["id"] == through_event), {})
    if event_info and not (event_info.get("finished") and event_info.get("data_checked")):
        warnings.append(f"GW{through_event} is not marked finished and data_checked; points and ranks are provisional.")
    state = {"schema_version": SCHEMA_VERSION, "entry_id": integer(entry["id"], "entry_id", 1),
             "team_name": entry.get("name"),
             "season": f"{season_match[1]}/{season_match[2]}" if season_match else None,
             "imported_at": imported_at or datetime.now(timezone.utc).isoformat(),
             "as_of_event": through_event, "next_event": through_event + 1,
             "bank": observed_bank if bank is None else bank, "free_transfers": free_transfers,
             "players": players, "lineup": lineup,
             "chips": chip_windows(bootstrap, used_chips, through_event + 1),
             "chips_used": sorted(used_chips, key=lambda c: c["event"]),
             "history": records,
             "rules": {"max_free_transfers": max_ft, "transfer_hit": 4,
                       "club_limit": bootstrap.get("game_settings", {}).get("squad_team_limit", 3),
                       "squad_size": bootstrap.get("game_settings", {}).get("squad_squadsize", 15),
                       "money_unit": "GBP_0.1_million"},
             "provenance": {"source": source, "squad_source": "public_deadline_picks_and_transfer_replay",
                            "free_transfers_source": "user_confirmed", "bank_source": "user_confirmed" if bank is not None else "public_last_permanent_deadline",
                            "observed_deadline_bank": observed_bank,
                            "latest_event_finished": event_info.get("finished"),
                            "latest_event_data_checked": event_info.get("data_checked"),
                            "post_deadline_changes_visible": False,
                            "allow_club_overflow": allow_club_overflow,
                            "bootstrap_url": f"{API_BASE}bootstrap-static/"},
             "warnings": warnings}
    validate_state(state, allow_club_overflow=allow_club_overflow)
    return state


def atomic_write_json(path: str | Path, data: dict) -> None:
    """Replace an output only after fully writing and syncing a private file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as out:
            temp_path = Path(out.name)
            json.dump(data, out, indent=2, ensure_ascii=False, allow_nan=False)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
