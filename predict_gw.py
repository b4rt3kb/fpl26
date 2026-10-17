"""Forecast all current players over a horizon, using one frozen data cutoff.

python predict_gw.py --gw 7 --horizon 6 --no-squad

Exports a plain all_players_gwN.csv plus JSON provenance. Blank GWs are explicit
zero rows. Current prices, clubs and injury statuses are held fixed throughout
the horizon: there is no injury return-date or future transfer/price model.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_features import build_season, pick_squad, season_agg
from fpl_model import PointsModel
from fetch_fpl import validate_data_manifest

ELEM = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def historical_seasons(manifest, current_year):
    """Only consume the contiguous history explicitly covered by the manifest."""
    seasons = manifest.get("historical_seasons")
    if not isinstance(seasons, list) or not seasons:
        raise ValueError("Manifest needs at least one historical season")
    expected = [f"{y}-{(y+1)%100:02}" for y in range(current_year-len(seasons), current_year)]
    if seasons != expected:
        raise ValueError("Historical seasons must be ordered, contiguous and end immediately before the current season")
    files = manifest.get("files", {})
    for season in seasons:
        for prefix in ("gw", "teams", "players"):
            if f"{prefix}_{season}.csv" not in files:
                raise ValueError(f"Historical input not covered by manifest: {prefix}_{season}.csv")
    return seasons


def availability(players):
    """Conservative snapshot assumption, held fixed; no return-date inference."""
    result = pd.Series(1.0, index=players.index)
    result.loc[players.status.isin(["i", "s", "u", "n"])] = 0.0
    doubt = players.status.eq("d")
    chance = pd.to_numeric(players.chance_of_playing_next_round, errors="coerce")
    result.loc[doubt] = chance.loc[doubt].fillna(50).clip(0, 100) / 100
    return result


def future_rows(players, teams, fixtures, gameweeks):
    team_names = teams.set_index("id")["name"]
    rows = []
    for fixture in fixtures[fixtures.event.isin(gameweeks)].itertuples():
        if pd.isna(fixture.kickoff_time):
            raise ValueError(f"Fixture {fixture.id} has no scheduled kickoff; refresh fixtures")
        for home in (True, False):
            team = fixture.team_h if home else fixture.team_a
            opponent = fixture.team_a if home else fixture.team_h
            for player in players[players.team.eq(team)].itertuples():
                rows.append({"element": player.id, "code": player.code,
                             "name": f"{player.first_name} {player.second_name}",
                             "position": ELEM[player.element_type], "team": team_names.loc[team],
                             "opponent_team": opponent, "was_home": home,
                             "round": int(fixture.event), "fixture": fixture.id,
                             "value": player.now_cost, "kickoff_time": fixture.kickoff_time,
                             "minutes": np.nan, "total_points": np.nan,
                             "team_h_score": np.nan, "team_a_score": np.nan,
                             "expected_goals": np.nan, "expected_assists": np.nan,
                             "starts": np.nan})
    return pd.DataFrame(rows)


def complete_pool(players, teams, forecast_rows, gameweeks, cutoff):
    """One row per current player and requested GW, including blank teams."""
    columns = ["element", "gw", "pred", "fixture_count"]
    if forecast_rows.empty:
        aggregated = pd.DataFrame(columns=columns)
    else:
        aggregated = forecast_rows.groupby(["element", "gw"], as_index=False).agg(
            pred=("pred", "sum"), fixture_count=("fixture", "nunique"))
    base = players[["id", "code", "web_name", "team", "element_type", "now_cost", "status"]].copy()
    base = base.rename(columns={"id": "element", "team": "team_key"})
    base["key"] = "code:" + base.code.astype(int).astype(str)
    base["club"] = base.team_key.map(teams.set_index("id")["short_name"])
    base["pos"] = base.element_type.map(ELEM)
    base["price"] = base.now_cost / 10
    base["can_select"] = players["can_select"].to_numpy() if "can_select" in players else ~players.status.isin(["u", "n"]).to_numpy()
    base["availability_factor"] = availability(players).to_numpy()
    base = base.drop(columns=["element_type", "now_cost"])
    grid = base.merge(pd.DataFrame({"gw": list(gameweeks)}), how="cross")
    grid = grid.merge(aggregated, on=["element", "gw"], how="left", validate="one_to_one")
    grid["pred"] = grid.pred.fillna(0) * grid.availability_factor
    grid["fixture_count"] = grid.fixture_count.fillna(0).astype(int)
    grid["history_cutoff"] = pd.Timestamp(cutoff).isoformat()
    grid["availability_assumption"] = "current_status_held_fixed_no_return_date_model"
    if not np.isfinite(grid.pred).all() or grid.duplicated(["element", "gw"]).any():
        raise ValueError("Invalid forecast output")
    return grid.sort_values(["gw", "element"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gw", type=int)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--model", choices=["hgb", "two"], default="hgb")
    parser.add_argument("--season", help="Defaults to the validated manifest season")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, help="Full-pool CSV destination; JSON metadata beside it")
    parser.add_argument("--no-squad", action="store_true", help="Skip fresh-squad benchmark (export all forecasts only)")
    parser.add_argument("--max-age-hours", type=float, default=6)
    args = parser.parse_args()
    if not 1 <= args.horizon <= 38:
        parser.error("horizon must be between 1 and 38")
    data = args.data_dir
    manifest = validate_data_manifest(data, max_age_hours=args.max_age_hours, target_gw=args.gw)
    args.season = args.season or manifest["season"]
    if args.season != manifest["season"]:
        raise ValueError("Requested season does not match the validated input manifest")
    bootstrap_path = data / "bootstrap.json"
    if not bootstrap_path.exists():
        raise ValueError("bootstrap.json is required: fetch current official data first")
    bootstrap = json.loads(bootstrap_path.read_text())
    players = pd.DataFrame(bootstrap["elements"])
    players = players[players.element_type.isin(ELEM)].copy().reset_index(drop=True)
    teams = pd.DataFrame(bootstrap["teams"])
    events = bootstrap["events"]
    deadlines = {int(event["id"]): event["deadline_time"] for event in events}
    now = pd.to_datetime(datetime.now(timezone.utc).isoformat(), utc=True)
    upcoming = [gw for gw, deadline in deadlines.items() if pd.to_datetime(deadline, utc=True) > now]
    if not upcoming:
        raise ValueError("No gameweek with a future deadline")
    gw = args.gw if args.gw is not None else min(upcoming)
    if gw not in deadlines or pd.to_datetime(deadlines[gw], utc=True) <= now:
        raise ValueError("Choose a GW with a future deadline; current prices/status cannot replay historical deadlines")
    gameweeks = list(range(gw, min(gw + args.horizon, max(deadlines) + 1)))
    if len(gameweeks) != args.horizon:
        raise ValueError("Requested horizon extends beyond the season")
    cutoff = min(now, pd.to_datetime(deadlines[gw], utc=True))
    year = int(args.season.split("-")[0])
    suffix = f"{year % 100:02}{(year + 1) % 100:02}"
    fixtures = pd.read_csv(data / f"fixtures_{suffix}.csv")
    targets = future_rows(players, teams, fixtures, gameweeks)

    history = []
    aggregates = {}
    for season in historical_seasons(manifest, year):
        old_year = int(season[:4])
        path = data / f"gw_{season}.csv"
        previous = aggregates.get(old_year - 1)
        history.append(build_season(path, data / f"teams_{season}.csv", previous,
                                    season_start=old_year))
        aggregates[old_year] = season_agg(path)
    current_path = data / f"gw_{args.season}.csv"
    if not current_path.exists():
        raise ValueError("Current-season match history missing; refresh official data")
    current = build_season(current_path, data / f"teams_{suffix}.csv",
                           aggregates[year-1], extra_rows=targets,
                           player_map_path=data / f"players_{suffix}.csv",
                           deadlines=deadlines, as_of=cutoff, season_start=year)
    completed = current[(~current._forecast) & current.total_points.notna()
                        & current.minutes.notna()
                        & (current.kickoff_time + pd.Timedelta(hours=2) <= cutoff)]
    if "finished" in completed:
        completed = completed[completed.finished.isin([True, 1, "True"])]
    training = pd.concat(history + [completed], ignore_index=True)
    training = training[training.total_points.notna() & training.minutes.notna()
                        & (training.kickoff_time + pd.Timedelta(hours=2) <= cutoff)]
    forecasts = current[current._forecast].copy()
    model = PointsModel(args.model).fit(training)
    forecasts["pred"] = model.predict(forecasts)
    pool = complete_pool(players, teams, forecasts, gameweeks, cutoff)
    destination = args.output or Path("predictions") / f"all_players_gw{gw}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(destination, index=False, float_format="%.6f")
    metadata = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                "data_manifest": manifest,
                "model": f"v0.4-{args.model}", "season": args.season,
                "gameweeks": gameweeks, "history_cutoff": cutoff.isoformat(),
                "training_rows": len(training), "current_history_rows": len(completed),
                "current_history_latest_gw": int(completed.gw.max()) if len(completed) else None,
                "stable_code_coverage_training": float(training.code.notna().mean()),
                "stable_code_coverage_current": float(players.code.notna().mean()),
                "availability_assumption": "Current injury/status/chance held fixed across horizon; no return-date model.",
                "future_assumptions": "Current prices and clubs held fixed; future forecasts do not update rolling history.",
                "historical_limitations": "Earliest kickoff proxy where event deadlines unavailable; 2h completion margin. Historical prices/status snapshots and rule-equivalent targets are not reconstructed.",
                "historical_strength_features": "Removed due to incompatible API scales and unknown snapshot timing."}
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(pool)} player-GW forecasts to {destination}")
    print(f"History: {len(completed)} current-season rows; cutoff {cutoff.isoformat()}")
    if not args.no_squad:
        candidates = pool[pool.gw.eq(gw) & pool.can_select.eq(True)].copy()
        selected = pick_squad(candidates)
        selected["role"] = np.where(selected.is_cap.eq(1), "XI (C)", np.where(selected.in_xi.eq(1), "XI", "bench"))
        squad_path = destination.parent / f"gw{gw}.csv"
        selected.to_csv(squad_path, index=False, float_format="%.6f")
        print(f"Fresh-squad benchmark saved to {squad_path}; not an account transfer plan.")


if __name__ == "__main__":
    main()
