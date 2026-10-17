"""Import public FPL history and reconstruct a private, versioned team state.

Example: python import_team.py --entry 123 --through-gw 6 --free-transfers 3
         --bank 0.4 --input-dir path/to/snapshots

Offline snapshots: bootstrap.json, entry.json, history.json, transfers.json,
and picks-gwN.json for every GW from entry.started_event through --through-gw.
No login cookies or account password are needed. Public snapshots cannot reveal
unpublished next-gameweek edits; the output explicitly preserves that boundary.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import sys

from team_state import StateValidationError, atomic_write_json, build_team_state


def money_tenths(value: str) -> int:
    try:
        amount = Decimal(value) * 10
        if not amount.is_finite() or amount < 0 or amount != amount.to_integral_value():
            raise ValueError
        return int(amount)
    except (InvalidOperation, ValueError):
        raise argparse.ArgumentTypeError("Bank must be nonnegative £m with at most one decimal place, e.g. 0.4")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entry", type=int, required=True, help="Public FPL entry ID")
    parser.add_argument("--through-gw", type=int, help="Last public GW to reconstruct; defaults to latest history")
    parser.add_argument("--free-transfers", type=int, required=True, help="User-confirmed available FT for the next GW")
    bank_group = parser.add_mutually_exclusive_group()
    bank_group.add_argument("--bank", type=money_tenths, help="User-confirmed bank in £m, e.g. 0.4 (stored as integer 4)")
    bank_group.add_argument("--bank-tenths", type=int, help="User-confirmed bank in integer £0.1m units, e.g. 4")
    parser.add_argument("--input-dir", type=Path, help="Use saved public API JSON without making network calls")
    parser.add_argument("--output", type=Path, help="Defaults to state/team_<entry>.json; keep this file private")
    parser.add_argument("--purchase-prices", type=Path, help="JSON object mapping initial player IDs to purchase prices in integer tenths; required for late entrants")
    parser.add_argument("--allow-club-overflow", action="store_true", help="Accept an existing squad over club quota due to real-world club transfers")
    args = parser.parse_args(argv)
    if args.entry < 1:
        parser.error("--entry must be positive")
    client = None
    input_records = []
    try:
        if args.input_dir:
            def get(filename, endpoint):
                raw = (args.input_dir / filename).read_bytes()
                input_records.append({"filename": filename, "sha256": hashlib.sha256(raw).hexdigest()})
                return json.loads(raw)
            source = "offline_public_api_snapshots"
        else:
            from fpl_api import APIError, FPLClient
            client = FPLClient(cache_dir="data/private-cache")
            def get(filename, endpoint):
                try:
                    return client.get_json(endpoint, max_age_seconds=0, force=True)
                except APIError as exc:
                    raise StateValidationError(f"Public API request failed for {endpoint}: {exc}") from exc
            source = "public_fpl_api"
        bootstrap = get("bootstrap.json", "bootstrap-static/")
        entry = get("entry.json", f"entry/{args.entry}/")
        if entry["id"] != args.entry:
            raise StateValidationError("Entry ID in snapshot differs from --entry")
        history = get("history.json", f"entry/{args.entry}/history/")
        transfers = get("transfers.json", f"entry/{args.entry}/transfers/")
        if not history.get("current"):
            raise StateValidationError("No public gameweek history is available")
        through = args.through_gw if args.through_gw is not None else max(row["event"] for row in history["current"])
        events = sorted(row["event"] for row in history["current"] if row["event"] <= through)
        picks = {gw: get(f"picks-gw{gw}.json", f"entry/{args.entry}/event/{gw}/picks/") for gw in events}
        overrides = json.loads(args.purchase_prices.read_text()) if args.purchase_prices else None
        state = build_team_state(bootstrap, entry, history, transfers, picks,
                                 free_transfers=args.free_transfers,
                                 bank=args.bank_tenths if args.bank_tenths is not None else args.bank,
                                 through_event=through, purchase_prices=overrides,
                                 allow_club_overflow=args.allow_club_overflow, source=source)
        if client is not None:
            state["provenance"]["fetches"] = client.records
        else:
            state["provenance"]["input_files"] = input_records
        output = args.output or Path(f"state/team_{args.entry}.json")
        atomic_write_json(output, state)
        print(f"Imported {len(state['history'])} gameweeks; {len(state['players'])} players; "
              f"{state['history'][-1]['total_points']} total points; bank £{state['bank'] / 10:.1f}m; "
              f"{state['free_transfers']} FT. Saved {output}")
        for warning in state["warnings"]:
            print(f"Note: {warning}")
        return 0
    except (StateValidationError, KeyError, TypeError, OSError, ValueError) as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
