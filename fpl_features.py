"""Validated player matches, features frozen per GW, and legal-squad MILP.

Historical archives lack deadline snapshots. Without explicit deadlines we use
the earliest kickoff of the GW as a documented proxy, exclude the entire target
GW, and require previous matches to have started at least two hours earlier.
This prevents within-DGW leakage but does not recover old price/status snapshots.
Future fixture rows never enter observed history.
"""
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

POS_CODE = {"GK": 0, "GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
POS_NAME = {0: "GK", 1: "DEF", 2: "MID", 3: "FWD"}
PREV = ["prev_p90", "prev_mins", "prev_xg90", "prev_xa90", "prev_start_share", "prev_pts_total"]
ROLL = {("total_points", 3): "pts_r3", ("total_points", 5): "pts_r5",
        ("total_points", 10): "pts_r10", ("minutes", 3): "mins_r3",
        ("minutes", 5): "mins_r5", ("expected_goals", 5): "xg_r5",
        ("expected_assists", 5): "xa_r5", ("starts", 5): "starts_r5"}
FEATS = PREV[:-1] + list(ROLL.values()) + [
    "own_att_r5", "own_def_r5", "opp_att_r5", "opp_def_r5",
    "price", "pos_code", "was_home", "gw", "scoring_era",
]
# Kept separate so historical HGB/rolling baselines retain their original inputs.
MINUTES_HISTORY = ["mins_r10", "mins_last", "mins_std_r5", "appearance_r5",
                   "sixty_r5", "history_matches", "days_since_last_appearance"]
MINUTES_SCHEDULE = ["forecast_lead_days", "fixtures_in_gw", "fixture_order_in_gw",
                    "team_rest_days", "team_matches_prev7", "team_matches_next7"]
MINUTES_FEATURES = ["prev_mins", "prev_start_share", "mins_r3", "mins_r5", "starts_r5",
                    "price", "pos_code", "was_home", "gw"] + MINUTES_HISTORY + MINUTES_SCHEDULE
EXTENDED_FEATURES = ["xg90_r5", "xa90_r5", "defcon_r5", "defcon_threshold_share_r5", "season_year"]


def _schedule_features(rows, fixture_calendar=None):
    """Read only dates and fixture identities, never outcomes or future lineups.

    Supply the calendar snapshot available at the forecast origin. The archive
    fallback has final fixture dates and therefore is not point-in-time evidence.
    Only Premier League congestion is represented; cups/internationals are absent.
    """
    if fixture_calendar is None:
        calendar = rows[["team_id", "fixture", "gw", "kickoff_time"]].drop_duplicates()
    else:
        required = {"id", "event", "team_h", "team_a", "kickoff_time"}
        if required - set(fixture_calendar):
            raise ValueError("Fixture calendar needs id/event/team_h/team_a/kickoff_time")
        parts = []
        for side in ("team_h", "team_a"):
            part = fixture_calendar[[side, "id", "event", "kickoff_time"]].rename(
                columns={side: "team_id", "id": "fixture", "event": "gw"})
            parts.append(part)
        calendar = pd.concat(parts, ignore_index=True).drop_duplicates()
        calendar["kickoff_time"] = pd.to_datetime(calendar["kickoff_time"], utc=True, errors="raise")
        calendar = calendar[calendar.kickoff_time.notna()].copy()
    if calendar.duplicated(["team_id", "fixture"]).any():
        raise ValueError("Conflicting fixture calendar rows")
    calendar = calendar.sort_values(["team_id", "kickoff_time", "fixture"]).copy()
    by_event = calendar.groupby(["team_id", "gw"])
    calendar["fixtures_in_gw"] = by_event.fixture.transform("nunique")
    calendar["fixture_order_in_gw"] = by_event.cumcount() + 1
    calendar["team_rest_days"] = calendar.groupby("team_id").kickoff_time.diff().dt.total_seconds() / 86400
    day = pd.Timedelta(days=1).value
    for _, club in calendar.groupby("team_id"):
        times = club.kickoff_time.astype("int64").to_numpy()
        calendar.loc[club.index, "team_matches_prev7"] = (
            np.searchsorted(times, times, side="left") - np.searchsorted(times, times-7*day, side="left"))
        calendar.loc[club.index, "team_matches_next7"] = (
            np.searchsorted(times, times+7*day, side="right") - np.searchsorted(times, times, side="right"))
    lookup = calendar.set_index(["team_id", "fixture"])
    keys = pd.MultiIndex.from_frame(rows[["team_id", "fixture"]])
    if not keys.isin(lookup.index).all():
        raise ValueError("Target fixture missing from supplied calendar")
    return {col: lookup[col].reindex(keys).to_numpy() for col in MINUTES_SCHEDULE if col != "forecast_lead_days"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def _player_map(gw_path, player_map_path):
    if player_map_path is not None:
        return Path(player_map_path)
    if gw_path:
        p = Path(gw_path)
        if p.name.startswith("gw_"):
            candidate = p.with_name(p.name.replace("gw_", "players_", 1))
            if candidate.exists():
                return candidate
    return None


def clean_rows(rows, player_map_path=None):
    """Filter managers, deduplicate exact rows, reject conflicting match IDs."""
    d = rows.copy()
    required = {"name", "position", "element", "fixture", "minutes", "total_points"}
    if missing := required - set(d.columns):
        raise ValueError(f"Missing player-match columns: {sorted(missing)}")
    d = d[d["position"].isin(POS_CODE)].copy()
    if d.empty:
        return d
    for col in ("element", "fixture"):
        values = pd.to_numeric(d[col], errors="raise")
        if values.isna().any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Invalid {col} identifiers")
        d[col] = values.astype(int)
    d = d.drop_duplicates()
    duplicate = d.duplicated(["element", "fixture"], keep=False)
    if duplicate.any():
        ids = d.loc[duplicate, ["element", "fixture"]].drop_duplicates().to_dict("records")
        raise ValueError(f"Conflicting duplicate player-fixture rows: {ids[:5]}")
    if "code" not in d:
        d["code"] = np.nan
    if player_map_path is not None:
        mapping = pd.read_csv(player_map_path)[["id", "code"]].drop_duplicates()
        if mapping["id"].duplicated().any():
            raise ValueError("Conflicting historical player id/code mapping")
        mapped = d["element"].map(mapping.set_index("id")["code"])
        existing = pd.to_numeric(d["code"], errors="coerce")
        if (existing.notna() & mapped.notna() & existing.ne(mapped)).any():
            raise ValueError("Player code conflicts with season id/code mapping")
        d["code"] = existing.fillna(mapped)
    d["code"] = pd.to_numeric(d["code"], errors="coerce").astype("Int64")
    d["key"] = [f"code:{int(c)}" if pd.notna(c) else f"name:{norm(n)}"
                for c, n in zip(d["code"], d["name"])]
    d["player_key"] = d["element"].astype(str)
    for c in ("expected_goals", "expected_assists", "starts"):
        if c not in d:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.reset_index(drop=True)


def season_agg(gw_file, player_map_path=None):
    d = clean_rows(pd.read_csv(gw_file), _player_map(gw_file, player_map_path))
    d = d[d["total_points"].notna() & d["minutes"].notna()]
    g = d.groupby("key").agg(
        pts=("total_points", "sum"), mins=("minutes", "sum"),
        xg=("expected_goals", lambda s: s.sum(min_count=1)),
        xa=("expected_assists", lambda s: s.sum(min_count=1)),
        starts=("starts", lambda s: s.sum(min_count=1)),
        games=("total_points", "size")).reset_index()
    p90 = g["mins"].clip(lower=720) / 90.0
    return pd.DataFrame({"key": g["key"], "prev_p90": g["pts"] / p90,
                         "prev_mins": g["mins"], "prev_xg90": g["xg"] / p90,
                         "prev_xa90": g["xa"] / p90,
                         "prev_start_share": g["starts"] / g["games"],
                         "prev_pts_total": g["pts"]})


def build_season(gw_path, teams_path, prev_agg=None, extra_rows=None,
                 player_map_path=None, deadlines=None, as_of=None,
                 season_start=None, fixture_calendar=None):
    """Freeze rolling inputs per event deadline; as_of freezes a forecast horizon.

    deadlines maps GW number to a UTC deadline. _forecast marks synthetic rows.
    fixture_calendar optionally supplies a full origin-time fixture snapshot,
    including matches outside the exported horizon for congestion features.
    """
    mapping = _player_map(gw_path, player_map_path)
    parts = []
    if gw_path:
        historical = clean_rows(pd.read_csv(gw_path), mapping)
        historical["_forecast"] = False
        parts.append(historical)
    if extra_rows is not None and len(extra_rows):
        targets = clean_rows(extra_rows, mapping)
        targets["_forecast"] = True
        parts.append(targets)
    if not parts or not sum(map(len, parts)):
        raise ValueError("No player-match data")
    d = pd.concat(parts, ignore_index=True)
    teams = pd.read_csv(teams_path)
    d["team_id"] = d["team"].map(teams.set_index("name")["id"])
    if d["team_id"].isna().any() or not d["opponent_team"].isin(teams["id"]).all():
        raise ValueError("Unknown team in fixture history")
    d["team_id"] = d["team_id"].astype(int)
    d["gw"] = pd.to_numeric(d["round"], errors="raise").astype(int)
    d["kickoff_time"] = pd.to_datetime(d["kickoff_time"], utc=True, errors="raise")
    if d["kickoff_time"].isna().any():
        raise ValueError("Missing kickoff time")
    if not d["was_home"].isin([True, False, 0, 1, "True", "False"]).all():
        raise ValueError("Invalid home/away flag")
    d["was_home"] = d["was_home"].map(lambda x: x in (True, 1, "True")).astype(int)
    for name, values in _schedule_features(d, fixture_calendar).items():
        d[name] = values
    d["defensive_contribution"] = pd.to_numeric(d.get("defensive_contribution", np.nan), errors="coerce")
    thresholds = d["position"].map({"DEF": 10, "MID": 12, "FWD": 12})
    d["_defcon_threshold"] = np.where(d.defensive_contribution.notna() & thresholds.notna(),
                                      d.defensive_contribution.ge(thresholds).astype(float), np.nan)
    d["gf"] = np.where(d["was_home"], d["team_h_score"], d["team_a_score"])
    d["ga"] = np.where(d["was_home"], d["team_a_score"], d["team_h_score"])
    observed = d[(~d["_forecast"]) & d["total_points"].notna() & d["minutes"].notna()].copy()
    if "finished" in observed:
        observed = observed[observed["finished"].isin([True, 1, "True"])]
    observed = observed.sort_values(["kickoff_time", "fixture", "element"])
    team_observed = observed.drop_duplicates(["team_id", "fixture"])
    deadlines = deadlines or {}
    frozen = pd.to_datetime(as_of, utc=True) if as_of is not None else None
    for c in list(ROLL.values()) + ["own_att_r5", "own_def_r5", "opp_att_r5", "opp_def_r5"] + MINUTES_HISTORY + EXTENDED_FEATURES[:-1]:
        d[c] = np.nan
    for gw, idx in d.groupby("gw").groups.items():
        cutoff = pd.to_datetime(deadlines.get(int(gw), d.loc[idx, "kickoff_time"].min()), utc=True)
        if frozen is not None:
            cutoff = min(cutoff, frozen)
        d.loc[idx, "feature_cutoff"] = cutoff
        hist = observed[(observed["kickoff_time"] + pd.Timedelta(hours=2) <= cutoff)
                        & observed["gw"].ne(gw)]
        recent = hist.groupby("player_key").tail(5)
        recent_groups = recent.groupby("player_key")
        minute_values = {
            "mins_r10": hist.groupby("player_key").tail(10).groupby("player_key").minutes.mean(),
            "mins_last": hist.groupby("player_key").minutes.last(),
            "mins_std_r5": recent_groups.minutes.std(ddof=0),
            "appearance_r5": recent.assign(appearance=recent.minutes.gt(0)).groupby("player_key").appearance.mean(),
            "sixty_r5": recent.assign(sixty=recent.minutes.ge(60)).groupby("player_key").sixty.mean(),
            "history_matches": hist.groupby("player_key").size(),
            "days_since_last_appearance": (cutoff-hist[hist.minutes.gt(0)].groupby("player_key").kickoff_time.max()).dt.total_seconds()/86400,
            "defcon_r5": recent_groups.defensive_contribution.mean(),
            "defcon_threshold_share_r5": recent_groups._defcon_threshold.mean(),
        }
        # Ratio of totals, using only rows with a known source statistic. A
        # zero-minute history or absent xG remains unknown, never fabricated zero.
        for source, name in (("expected_goals", "xg90_r5"), ("expected_assists", "xa90_r5")):
            known = recent[recent[source].notna()]
            totals = known.groupby("player_key")[[source, "minutes"]].sum(min_count=1)
            minute_values[name] = 90*totals[source]/totals.minutes.where(totals.minutes.gt(0))
        for name, values in minute_values.items():
            d.loc[idx, name] = d.loc[idx, "player_key"].map(values).to_numpy()
        for (source, window), feature in ROLL.items():
            values = hist.groupby("player_key").tail(window).groupby("player_key")[source].mean()
            d.loc[idx, feature] = d.loc[idx, "player_key"].map(values).to_numpy()
        team_hist = team_observed[(team_observed["kickoff_time"] + pd.Timedelta(hours=2) <= cutoff)
                                  & team_observed["gw"].ne(gw)]
        team_form = team_hist.groupby("team_id").tail(5).groupby("team_id")[["gf", "ga"]].mean()
        for side, team_col in (("own", "team_id"), ("opp", "opponent_team")):
            for src, kind in (("gf", "att"), ("ga", "def")):
                d.loc[idx, f"{side}_{kind}_r5"] = d.loc[idx, team_col].map(team_form[src]).to_numpy()
    if prev_agg is not None:
        d = d.merge(prev_agg, on="key", how="left", validate="many_to_one")
    else:
        for c in PREV:
            d[c] = np.nan
    d["price"] = pd.to_numeric(d["value"], errors="raise") / 10.0
    d["pos_code"] = d["position"].map(POS_CODE)
    if season_start is None:
        match = re.search(r"(20\d{2})-\d{2}", str(gw_path or ""))
        season_start = int(match.group(1)) if match else int(d["kickoff_time"].min().year)
    d["scoring_era"] = int(season_start >= 2025)
    d["season_year"] = season_start
    d["forecast_lead_days"] = (d.kickoff_time-pd.to_datetime(d.feature_cutoff, utc=True)).dt.total_seconds().clip(lower=0)/86400
    d["played"] = d["minutes"].fillna(0).gt(0)
    d["played60"] = d["minutes"].fillna(0).ge(60)
    return d


def validate_squad(squad, budget=100.0):
    """Reject an illegal or fractional result instead of publishing it."""
    if len(squad) != 15 or squad["in_xi"].sum() != 11 or squad["is_cap"].sum() != 1:
        raise ValueError("Invalid squad/XI/captain count")
    if not squad[["in_xi", "is_cap"]].isin([0, 1]).all().all():
        raise ValueError("Non-binary squad result")
    if (squad["is_cap"] > squad["in_xi"]).any() or squad["price"].sum() > budget + 1e-6:
        raise ValueError("Captain or budget constraint violated")
    if squad.groupby("team_key").size().max() > 3:
        raise ValueError("Club quota violated")
    xi = squad[squad.in_xi.eq(1)]
    for pos, (count, lo, hi) in {"GK": (2, 1, 1), "DEF": (5, 3, 5), "MID": (5, 2, 5), "FWD": (3, 1, 3)}.items():
        if squad.pos.eq(pos).sum() != count or not lo <= xi.pos.eq(pos).sum() <= hi:
            raise ValueError(f"Position constraint violated: {pos}")
    for identity in ("element", "key"):
        if identity in squad and squad[identity].duplicated().any():
            raise ValueError(f"Duplicate selected player: {identity}")


def pick_squad(players, pred_col="pred", budget=100.0, bench_weight=0.1):
    """SciPy/HiGHS MILP: fresh 15-player squad, XI and captain, not transfers."""
    df = players.reset_index(drop=True).copy()
    if len(df) < 15 or not np.isfinite(df[[pred_col, "price"]].to_numpy(float)).all():
        raise ValueError("Insufficient players or non-finite prices/predictions")
    if df["team_key"].isna().any() or not df["pos"].isin(POS_NAME.values()).all():
        raise ValueError("Invalid team/position")
    if (df["price"] <= 0).any() or budget <= 0 or bench_weight < 0:
        raise ValueError("Invalid budget, price, or bench weight")
    for identity in ("element", "key"):
        if identity in df and df[identity].duplicated().any():
            raise ValueError(f"Duplicate candidate identity: {identity}")
    n = len(df)
    rows, lower, upper = [], [], []
    def add(items, lo, hi):
        rows.append(items); lower.append(lo); upper.append(hi)
    add([(i, 1) for i in range(n)], 15, 15)
    add([(n+i, 1) for i in range(n)], 11, 11)
    add([(2*n+i, 1) for i in range(n)], 1, 1)
    add([(i, float(df.price.iloc[i])) for i in range(n)], -np.inf, budget)
    for i in range(n):
        add([(n+i, 1), (i, -1)], -np.inf, 0)
        add([(2*n+i, 1), (n+i, -1)], -np.inf, 0)
    for p, (sq, lo, hi) in {"GK": (2, 1, 1), "DEF": (5, 3, 5), "MID": (5, 2, 5), "FWD": (3, 1, 3)}.items():
        ids = df.index[df.pos.eq(p)].tolist()
        add([(i, 1) for i in ids], sq, sq)
        add([(n+i, 1) for i in ids], lo, hi)
    for _, group in df.groupby("team_key"):
        add([(int(i), 1) for i in group.index], -np.inf, 3)
    matrix = lil_matrix((len(rows), 3*n), dtype=float)
    for r, items in enumerate(rows):
        for c, v in items:
            matrix[r, c] = v
    pred = df[pred_col].to_numpy(float)
    objective = -np.concatenate([bench_weight*pred, (1-bench_weight)*pred, pred])
    result = milp(objective, integrality=np.ones(3*n), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix.tocsr(), lower, upper),
                  options={"time_limit": 30})
    if result.status != 0 or result.x is None:
        raise RuntimeError(f"Squad optimization did not reach an optimum: {result.message}")
    if not np.isfinite(result.x).all() or not np.allclose(result.x, np.rint(result.x), atol=1e-6):
        raise RuntimeError("Solver returned a non-integral solution")
    values = np.rint(result.x).astype(int)
    df["in_squad"], df["in_xi"], df["is_cap"] = values[:n], values[n:2*n], values[2*n:]
    selected = df[df.in_squad.eq(1)].copy()
    validate_squad(selected, budget)
    return selected
