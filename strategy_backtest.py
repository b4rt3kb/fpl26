"""Realized, stateful replay of precomputed origin-frozen FPL forecasts.

This is an ex-post reconstruction unless the caller supplies genuine deadline
snapshots. Outcomes are never passed to the decision policy. All money in the
state and ledger is integer tenths; forecast ``price`` is in millions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from team_state import free_transfers_next, selling_price


QUOTAS = {"GK": (2, 1, 1), "DEF": (5, 3, 5), "MID": (5, 2, 5), "FWD": (3, 1, 3)}
CHIPS = {"wildcard", "freehit", "bboost", "3xc"}
LIMITATIONS = [
    "Historical prices, player pools and fixture calendars may be reconstructed ex post; this is not a point-in-time account backtest.",
    "Origin-frozen feature provenance must be verified by the forecast producer; the replay cannot infer it from numeric forecasts.",
    "Missing outcome rows score zero (blank or no appearance); completeness of the outcome archive must be established upstream.",
    "The planner uses heuristic chip reserves and constant future prices; realized scoring includes autosubs and vice captain, but its forecast objective does not.",
    "Results are a simulated strategy, not the user's account score; an already-used development season is not an untouched test.",
]


def _ids(values):
    result = [int(x) for x in values]
    if len(result) != len(set(result)):
        raise ValueError("Duplicate player in decision")
    return result


def _formation(xi, metadata):
    counts = Counter(metadata[i]["pos"] for i in xi)
    return len(xi) == 11 and all(low <= counts[pos] <= high for pos, (_, low, high) in QUOTAS.items())


def _validate_squad(squad, metadata, xi=None, club_limit=3):
    squad = _ids(squad)
    if len(squad) != 15 or not set(squad).issubset(metadata):
        raise ValueError("Squad must have 15 known unique players")
    if any(sum(metadata[i]["pos"] == pos for i in squad) != n for pos, (n, _, _) in QUOTAS.items()):
        raise ValueError("Invalid squad position quotas")
    if club_limit is not None and max(Counter(metadata[i]["team"] for i in squad).values()) > club_limit:
        raise ValueError("Club limit exceeded")
    if xi is not None and (len(_ids(xi)) != 11 or not set(xi).issubset(squad) or not _formation(xi, metadata)):
        raise ValueError("Invalid starting XI formation")


def _forecast_metadata(forecasts, gw):
    required = {"element", "gw", "pos", "price", "pred", "team_key", "web_name"}
    if not required.issubset(forecasts):
        raise ValueError("Forecast columns missing: " + str(sorted(required - set(forecasts))))
    if forecasts.duplicated(["element", "gw"]).any():
        raise ValueError("Duplicate forecast player/gameweek")
    if not np.isfinite(forecasts[["price", "pred"]].to_numpy(float)).all() or (forecasts.price <= 0).any():
        raise ValueError("Invalid price or prediction")
    gws = sorted(int(g) for g in forecasts.gw.unique())
    if gws != list(range(gw, gw + len(gws))) or not gws or gws[-1] > 38:
        raise ValueError("Forecast horizon must start at origin and be consecutive")
    if len(forecasts) != forecasts.element.nunique() * len(gws):
        raise ValueError("Forecast must include every candidate in every GW, including blanks")
    if "origin" in forecasts and not forecasts.origin.eq(gw).all():
        raise ValueError("Forecast origin mismatch")
    for _, block in forecasts.groupby("element"):
        if block[["price", "pos", "team_key"]].nunique().max() != 1:
            raise ValueError("Origin prices, positions and clubs must remain frozen across horizon")
    metadata = {}
    for row in forecasts[forecasts.gw.eq(gw)].itertuples():
        price = float(row.price) * 10
        if not np.isclose(price, round(price), atol=1e-7):
            raise ValueError("Price must be an integer number of tenths")
        pos = "GK" if row.pos == "GKP" else row.pos
        if pos not in QUOTAS:
            raise ValueError("Unknown position")
        selectable = getattr(row, "can_select", True)
        if selectable not in (True, False, 0, 1):
            raise ValueError("can_select must be a boolean, not a string or missing value")
        metadata[int(row.element)] = {"pos": pos, "team": int(row.team_key),
                                      "price": int(round(price)), "name": str(row.web_name),
                                      "can_select": bool(selectable), "pred": float(row.pred)}
    return metadata


def aggregate_outcomes(outcomes):
    """Return (GW, element) -> stats. DGW rows require distinct fixture IDs."""
    required = {"element", "gw", "total_points", "minutes"}
    if not required.issubset(outcomes):
        raise ValueError("Outcome columns missing: " + str(sorted(required - set(outcomes))))
    frame = outcomes.copy()
    if frame.duplicated(["element", "gw"]).any():
        if "fixture" not in frame or frame.fixture.isna().any():
            raise ValueError("Multiple player/GW outcomes require fixture IDs")
        if frame.duplicated(["element", "gw", "fixture"]).any():
            raise ValueError("Duplicate player/fixture outcome")
    for col in ("yellow_cards", "red_cards"):
        if col not in frame:
            frame[col] = 0
    cols = ["total_points", "minutes", "yellow_cards", "red_cards"]
    if not np.isfinite(frame[cols].to_numpy(float)).all() or (frame[["minutes", "yellow_cards", "red_cards"]] < 0).any().any():
        raise ValueError("Invalid realized outcome")
    grouped = frame.groupby(["gw", "element"])[cols].sum()
    return {(int(gw), int(pid)): {k: float(v) for k, v in row.items()} for (gw, pid), row in grouped.iterrows()}


def realized_score(decision, metadata, outcomes):
    """Score one GW with formation-preserving bench priority and captain fallback.

    ``outcomes`` maps player IDs to aggregated GW stats. Cards also count as
    participation for autosubs; the captain/vice fallback uses GW minutes.
    No outcome affects the original lineup or bench order.
    """
    squad, xi, bench = (_ids(decision[k]) for k in ("squad", "starting_xi", "bench"))
    _validate_squad(squad, metadata, xi, club_limit=None)
    if len(bench) != 4 or set(bench) != set(squad) - set(xi):
        raise ValueError("Bench must contain the four non-starters in priority order")
    captain, vice = int(decision["captain"]), int(decision["vice_captain"])
    if captain == vice or captain not in xi or vice not in xi:
        raise ValueError("Captain and vice must be distinct starters")
    chip = decision.get("chip")
    if chip is not None and chip not in CHIPS:
        raise ValueError("Unknown chip")
    def stat(pid, field):
        return float(outcomes.get(pid, {}).get(field, 0))
    def played(pid):
        return any(stat(pid, col) > 0 for col in ("minutes", "yellow_cards", "red_cards"))
    substitutions = []
    nominal = list(xi)
    if chip != "bboost":
        starter_gk = next(i for i in xi if metadata[i]["pos"] == "GK")
        bench_gk = next(i for i in bench if metadata[i]["pos"] == "GK")
        if not played(starter_gk) and played(bench_gk):
            nominal[nominal.index(starter_gk)] = bench_gk
            substitutions.append({"out": starter_gk, "in": bench_gk})
        # Explore valid assignments, selecting substitutions lexicographically
        # by bench priority, never by realized points. The search is <=3 subs.
        reserves = [i for i in bench if metadata[i]["pos"] != "GK"]
        absent = {i for i in xi if metadata[i]["pos"] != "GK" and not played(i)}
        possibilities = []
        def search(index, lineup, unused, used, flags):
            if index == len(reserves):
                possibilities.append((tuple(flags), lineup, used))
                return
            incoming = reserves[index]
            if played(incoming):
                for outgoing in sorted(unused, key=lambda i: (metadata[i]["pos"] != metadata[incoming]["pos"], i)):
                    proposed = [incoming if i == outgoing else i for i in lineup]
                    if _formation(proposed, metadata):
                        search(index + 1, proposed, unused - {outgoing}, used + [{"out": outgoing, "in": incoming}], flags + [1])
            search(index + 1, lineup, unused, used, flags + [0])
        search(0, nominal, absent, [], [])
        _, nominal, outfield_subs = max(possibilities, key=lambda item: item[0])
        substitutions += outfield_subs
    scoring = squad if chip == "bboost" else nominal
    applied = captain if stat(captain, "minutes") > 0 else vice if stat(vice, "minutes") > 0 else None
    extra = (2 if chip == "3xc" else 1) * stat(applied, "total_points") if applied is not None else 0
    base = sum(stat(i, "total_points") for i in scoring)
    return {"base_points": base, "captain_bonus": extra, "gross_points": base + extra,
            "captain_applied": applied, "captain_multiplier": (3 if chip == "3xc" else 2) if applied is not None else 0,
            "scoring_players": scoring, "automatic_substitutions": substitutions,
            "bench_points_unused": sum(stat(i, "total_points") for i in squad if i not in scoring)}


def no_transfer_decision(state, forecasts):
    """Best legal XI/captain from the owned squad using forecasts only."""
    gw = int(state["next_event"])
    meta = _forecast_metadata(forecasts, gw)
    owned = sorted(int(p["id"]) for p in state["players"])
    _validate_squad(owned, meta, club_limit=None)
    candidates = []
    for defenders in range(3, 6):
        for mids in range(2, 6):
            forwards = 10 - defenders - mids
            if not 1 <= forwards <= 3:
                continue
            xi = []
            for pos, n in {"GK": 1, "DEF": defenders, "MID": mids, "FWD": forwards}.items():
                xi += sorted((i for i in owned if meta[i]["pos"] == pos), key=lambda i: (-meta[i]["pred"], i))[:n]
            captain, vice = sorted(xi, key=lambda i: (-meta[i]["pred"], i))[:2]
            value = sum(meta[i]["pred"] for i in xi) + meta[captain]["pred"]
            candidates.append((value, xi, captain, vice))
    _, xi, captain, vice = max(candidates, key=lambda item: item[0])
    bench = sorted(set(owned) - set(xi), key=lambda i: (meta[i]["pos"] != "GK", -meta[i]["pred"], i))
    return {"gw": gw, "chip": None, "squad": owned, "permanent_squad": owned,
            "starting_xi": xi, "bench": bench, "captain": captain, "vice_captain": vice,
            "transfer_in": [], "transfer_out": []}


def initial_state_from_forecast(forecasts, *, budget=1000, chips=None, rules=None, bench_weight=0.1):
    """Choose a common starting 15 from the first GW forecast, before outcomes."""
    from fpl_features import pick_squad
    gw = int(forecasts.gw.min())
    meta = _forecast_metadata(forecasts, gw)
    pool = forecasts[forecasts.gw.eq(gw)].copy()
    pool["pos"] = pool.pos.replace({"GKP": "GK"})
    pool = pool[pool.element.map(lambda i: meta[int(i)]["can_select"])]
    selected = pick_squad(pool, budget=budget / 10, bench_weight=bench_weight)
    players = [{"id": int(i), "name": meta[int(i)]["name"], "position": meta[int(i)]["pos"],
                "team": meta[int(i)]["team"], "purchase_price": meta[int(i)]["price"],
                "now_cost": meta[int(i)]["price"], "selling_price": meta[int(i)]["price"]} for i in selected.element]
    return {"next_event": gw, "free_transfers": 1, "bank": int(budget) - sum(p["purchase_price"] for p in players),
            "players": players, "chips": deepcopy(chips or []), "chips_used": [],
            "rules": {"max_free_transfers": 5, "transfer_hit": 4, "club_limit": 3, **(rules or {})},
            "provenance": {"initial_squad": "optimized from first-origin forecasts only", "budget": int(budget)}}


def _refresh_state(state, meta, gw):
    result = deepcopy(state)
    result["next_event"] = gw
    if len(result["players"]) != 15 or len({p["id"] for p in result["players"]}) != 15:
        raise ValueError("Initial holdings must contain 15 unique players")
    for player in result["players"]:
        pid = int(player["id"])
        if pid not in meta:
            raise ValueError(f"Origin forecast omits owned player {pid}")
        m = meta[pid]
        player.update(name=m["name"], position=m["pos"], team=m["team"], now_cost=m["price"],
                      selling_price=selling_price(int(player["purchase_price"]), m["price"]))
    rules = result.get("rules", {})
    override = rules.get("free_transfer_overrides", {}).get(str(gw), rules.get("free_transfer_overrides", {}).get(gw))
    if override is not None:
        result["free_transfers"] = int(override)
    maximum = int(rules.get("max_free_transfers", 5))
    if not 1 <= int(result["free_transfers"]) <= maximum or int(result["bank"]) < 0:
        raise ValueError("Invalid bank/free transfers")
    for chip in result.get("chips", []):
        if chip.get("used_event") or chip.get("status") == "used":
            chip["status"] = "used"
        elif chip.get("status") != "unavailable":
            chip["status"] = "expired" if gw > chip["stop_event"] else "upcoming" if gw < chip["start_event"] else "available"
    return result


def apply_decision(state, forecasts, decision):
    """Independently validate the first action and advance its cash/FT/chip ledger."""
    gw = int(state["next_event"])
    meta = _forecast_metadata(forecasts, gw)
    state = _refresh_state(state, meta, gw)
    if int(decision["gw"]) != gw:
        raise ValueError("Decision gameweek differs from origin")
    previous = {int(p["id"]): p for p in state["players"]}
    roster = _ids(decision["squad"])
    permanent = _ids(decision.get("permanent_squad", roster))
    chip = decision.get("chip")
    if chip is not None and chip not in CHIPS:
        raise ValueError("Unknown chip")
    chip_options = [c for c in state.get("chips", []) if c["name"] == chip and c["start_event"] <= gw <= c["stop_event"]
                    and c.get("status") not in ("used", "expired", "unavailable") and not c.get("used_event")]
    if chip and len(chip_options) != 1:
        raise ValueError("Chip unavailable or ambiguous window")
    if chip == "freehit" and any(c["name"] == "freehit" and int(c.get("event", c.get("used_event", -1))) == gw - 1 for c in state.get("chips_used", [])):
        raise ValueError("Consecutive Free Hits are not permitted")
    ins, outs = sorted(set(permanent) - set(previous)), sorted(set(previous) - set(permanent))
    for key, expected in (("transfer_in", ins), ("transfer_out", outs)):
        if key in decision and sorted(_ids(decision[key])) != expected:
            raise ValueError("Transfer list contradicts permanent roster")
    if chip == "freehit" and set(permanent) != set(previous):
        raise ValueError("Free Hit must revert the permanent squad")
    if chip != "freehit" and set(roster) != set(permanent):
        raise ValueError("Active and permanent squads differ without Free Hit")
    club_limit = int(state.get("rules", {}).get("club_limit", 3))
    # Club moves can grandfather a held squad above three; any acquisition must
    # restore the normal quota. No-transfer comparisons may retain that squad.
    _validate_squad(permanent, meta, club_limit=club_limit if ins else None)
    _validate_squad(roster, meta, decision["starting_xi"], club_limit=club_limit if ins or chip == "freehit" else None)
    acquisitions = set(roster) - set(previous)
    if any(not meta[i]["can_select"] for i in acquisitions):
        raise ValueError("Cannot acquire an unselectable player")
    # Validate bench/captain independently, without any actual outcome data.
    realized_score(decision, meta, {})
    bank_before = int(state["bank"])
    sale = {i: previous[i]["selling_price"] for i in outs}
    purchase = {i: meta[i]["price"] for i in ins}
    bank = bank_before + sum(sale.values()) - sum(purchase.values())
    if bank < 0:
        raise ValueError("Transfers exceed available budget")
    temporary_bank = None
    if chip == "freehit":
        temporary_bank = bank_before + sum(previous[i]["selling_price"] for i in set(previous) - set(roster)) - sum(meta[i]["price"] for i in set(roster) - set(previous))
        if temporary_bank < 0:
            raise ValueError("Free Hit exceeds available budget")
    if "bank" in decision and int(decision["bank"]) != bank:
        raise ValueError("Reported bank contradicts transfer ledger")
    rules = state.get("rules", {})
    free = int(state["free_transfers"])
    hit = 0 if chip in ("freehit", "wildcard") else max(0, len(ins) - free) * int(rules.get("transfer_hit", 4))
    following = free_transfers_next(free, len(ins), chip, int(rules.get("max_free_transfers", 5)))
    if gw == 1:
        following = 1  # initial squad selection does not bank a preseason FT
    override_next = rules.get("free_transfer_overrides", {}).get(str(gw + 1), rules.get("free_transfer_overrides", {}).get(gw + 1))
    if override_next is not None:
        following = int(override_next)
        if not 1 <= following <= int(rules.get("max_free_transfers", 5)):
            raise ValueError("Invalid next-event free transfer override")
    result = deepcopy(state)
    result["players"] = [deepcopy(previous[i]) if i in previous else
                         {"id": i, "name": meta[i]["name"], "position": meta[i]["pos"], "team": meta[i]["team"],
                          "purchase_price": meta[i]["price"], "now_cost": meta[i]["price"], "selling_price": meta[i]["price"],
                          "purchase_event": gw} for i in sorted(permanent)]
    result.update(bank=bank, free_transfers=following, next_event=gw + 1)
    if chip:
        chosen = chip_options[0]
        for c in result["chips"]:
            if c == chosen:
                c.update(status="used", used_event=gw)
        result.setdefault("chips_used", []).append({"name": chip, "event": gw})
    ledger = {"bank_before": bank_before, "bank_after": bank, "temporary_bank": temporary_bank,
              "free_transfers_before": free, "free_transfers_next": following, "hit_cost": hit,
              "transfer_in": ins, "transfer_out": outs, "purchase_costs": purchase, "sale_proceeds": sale,
              "chip": chip, "chip_window_id": chip_options[0].get("id") if chip else None}
    return result, ledger


def replay_strategy(initial_state, forecasts_by_origin, outcomes, *, policy="planner", planner_options=None,
                    decision_fn=None, lock_initial_squad=True):
    """Execute only the current action, score it, then re-plan at the next origin.

    Policies: planner (transfers/chips), no_chips (transfers), no_transfer (same
    holdings, no chips, forecast-based XI/captain). decision_fn(state, forecasts)
    can inject deterministic decisions or a planner-compatible result for tests.
    """
    if policy not in ("planner", "no_chips", "no_transfer"):
        raise ValueError("Unknown replay policy")
    origins = sorted(int(x) for x in forecasts_by_origin)
    first = int(initial_state["next_event"])
    if not origins or origins != list(range(first, origins[-1] + 1)) or origins[-1] > 38:
        raise ValueError("Replay origins must be consecutive from initial state's next_event")
    lookup = aggregate_outcomes(outcomes)
    state = deepcopy(initial_state)
    records = []
    for gw in origins:
        # Whitelist decision-time fields: evaluation CSVs may also carry target
        # columns, which must never reach either plan() or a custom policy.
        allowed = {"element", "gw", "pos", "price", "pred", "team_key", "web_name", "can_select",
                   "origin", "fixture_count", "code", "expected_minutes", "p_no_appearance", "history_cutoff"}
        forecast = forecasts_by_origin[gw][[c for c in forecasts_by_origin[gw] if c in allowed]].copy()
        meta = _forecast_metadata(forecast, gw)
        state = _refresh_state(state, meta, gw)
        solver = None
        diagnostics = {}
        started = perf_counter()
        if decision_fn is not None:
            decision = decision_fn(deepcopy(state), forecast.copy())
        elif policy == "no_transfer" or (lock_initial_squad and gw == 1):
            decision = no_transfer_decision(state, forecast)
        else:
            from plan_transfers import plan
            options = dict(planner_options or {})
            options["allow_chips"] = policy == "planner"
            decision = plan(deepcopy(state), forecast.copy(), **options)
        if "steps" in decision:
            solver = decision.get("solver")
            diagnostics = {key: deepcopy(decision[key]) for key in ("candidate_pruning", "chip_search", "no_chip_baseline", "warnings") if key in decision}
            decision = decision["steps"][0]
        decision_seconds = perf_counter() - started
        new_state, ledger = apply_decision(state, forecast, decision)
        actual = {pid: lookup.get((gw, pid), {}) for pid in decision["squad"]}
        scored = realized_score(decision, meta, actual)
        records.append({"gw": gw, "decision": deepcopy(decision), "ledger": ledger, "realized": scored,
                        "net_points": scored["gross_points"] - ledger["hit_cost"], "solver": solver,
                        "planner_diagnostics": diagnostics, "decision_wall_seconds": decision_seconds})
        state = new_state
    return {"schema_version": 1, "policy": policy, "gameweeks": origins, "full_season": origins == list(range(1, 39)),
            "initial_state": deepcopy(initial_state), "final_state": state, "weeks": records,
            "gross_points": sum(r["realized"]["gross_points"] for r in records),
            "hit_points": sum(r["ledger"]["hit_cost"] for r in records),
            "net_points": sum(r["net_points"] for r in records),
            "permanent_transfers": sum(len(r["ledger"]["transfer_in"]) for r in records),
            "decision_wall_seconds": sum(r["decision_wall_seconds"] for r in records),
            "chips_used": state.get("chips_used", []), "limitations": LIMITATIONS,
            "assumptions": {"initial_squad_locked_in_gw1": lock_initial_squad, "outcomes_passed_to_decision_policy": False,
                            "gw1_chips_disabled_when_initial_squad_locked": lock_initial_squad,
                            "replan_every_gameweek": True, "bench_and_captain_using_forecasts_only": True}}


def compare_strategies(initial_state, forecasts_by_origin, outcomes, *, planner_options=None):
    result = {p: replay_strategy(initial_state, forecasts_by_origin, outcomes, policy=p,
                                planner_options=planner_options) for p in ("planner", "no_chips", "no_transfer")}
    return {"schema_version": 1, "strategies": result,
            "net_advantage_vs_no_transfer": result["planner"]["net_points"] - result["no_transfer"]["net_points"],
            "net_advantage_vs_no_chips": result["planner"]["net_points"] - result["no_chips"]["net_points"],
            "limitations": LIMITATIONS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True)
    parser.add_argument("--forecasts", required=True, help="CSV with origin plus the planner forecast columns")
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--policy", choices=["compare", "planner", "no_chips", "no_transfer"], default="compare")
    args = parser.parse_args()
    forecasts = pd.read_csv(args.forecasts)
    if "origin" not in forecasts:
        parser.error("Forecast CSV requires origin")
    by_origin = {int(gw): frame.copy() for gw, frame in forecasts.groupby("origin")}
    state = json.loads(Path(args.state).read_text())
    outcomes = pd.read_csv(args.outcomes)
    if args.policy == "compare":
        result = compare_strategies(state, by_origin, outcomes, planner_options={"seconds": args.seconds})
    else:
        result = replay_strategy(state, by_origin, outcomes, policy=args.policy, planner_options={"seconds": args.seconds})
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
