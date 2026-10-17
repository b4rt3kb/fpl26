"""Chronological six-GW forecasts for predictive and legal strategy replays.

Historical prices/universes and the final fixture calendar are reconstructed
from archives, NOT claimed to be real pre-deadline snapshots. No future match
statistics, minutes or points are used to make a decision. See output metadata.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fpl_features import build_season, season_agg
from fpl_model import PointsModel


def season_paths(data, season):
    year = int(season[:4])
    suffix = f"{year%100:02}{(year+1)%100:02}"
    return (data / f"gw_{season}.csv",
            data / (f"teams_{season}.csv" if (data/f"teams_{season}.csv").exists() else f"teams_{suffix}.csv"),
            data / (f"players_{season}.csv" if (data/f"players_{season}.csv").exists() else f"players_{suffix}.csv"))


def horizon_training(data, season, prior, rows):
    """Frozen-origin examples teach leads 2–6; each outcome appears at most once.

    We deliberately retain historical (origin) roster/prices rather than the
    eventual target-row player price. Targets are still finished match labels.
    """
    path, teams, mapping = season_paths(data, season)
    parts = []
    for origin in range(1, int(rows.gw.max()) + 1, 6):
        origin_rows = rows[rows.gw.eq(origin)]
        if origin_rows.empty:
            continue
        cutoff = origin_rows.feature_cutoff.min()
        frozen = build_season(path, teams, prior, player_map_path=mapping, as_of=cutoff,
                              season_start=int(season[:4]))
        frozen = frozen[frozen.gw.between(origin+1, origin+5)].copy()
        known = rows[rows.gw.le(origin)].sort_values(["gw", "kickoff_time"]).groupby("element").tail(1)
        frozen = frozen[frozen.element.isin(known.element)]
        frozen["price"] = frozen.element.map(known.set_index("element").price)
        # A club transfer would introduce future metadata. Omit those labels.
        frozen = frozen[frozen.team_id.eq(frozen.element.map(known.set_index("element").team_id))]
        parts.append(frozen)
    return pd.concat(parts, ignore_index=True) if parts else rows.iloc[:0].copy()


def origin_forecasts(data, season, prior, current, origin, horizon):
    """Construct all target fixtures from the origin roster and final calendar."""
    path, teams_path, mapping = season_paths(data, season)
    first = current[current.gw.eq(origin)]
    if first.empty:
        raise ValueError(f"No event rows for GW{origin}")
    cutoff = first.feature_cutoff.min()
    # Archive row price is a documented approximation to deadline price.
    known = current[current.gw.lt(origin)].sort_values(["gw", "kickoff_time"]).groupby("element").tail(1)
    origin_rows = first.sort_values("kickoff_time").groupby("element").head(1)
    known = pd.concat([known, origin_rows], ignore_index=True).drop_duplicates("element", keep="last")
    gws = list(range(origin, min(int(current.gw.max())+1, 39, origin+horizon)))
    sides = current[current.gw.isin(gws)].drop_duplicates(["team_id", "fixture"])
    extra = []
    for match in sides.itertuples():
        for player in known[known.team_id.eq(match.team_id)].itertuples():
            extra.append({"element": player.element, "code": player.code, "name": player.name,
                          "position": player.position, "team": player.team,
                          "opponent_team": match.opponent_team, "was_home": match.was_home,
                          "round": match.gw, "fixture": match.fixture, "value": player.price*10,
                          "kickoff_time": match.kickoff_time, "minutes": np.nan, "total_points": np.nan,
                          "team_h_score": np.nan, "team_a_score": np.nan,
                          "expected_goals": np.nan, "expected_assists": np.nan, "starts": np.nan})
    expanded = build_season(path, teams_path, prior, player_map_path=mapping,
                            extra_rows=pd.DataFrame(extra), as_of=cutoff, season_start=int(season[:4]))
    future = expanded[expanded._forecast].copy()
    base = known[["element", "key", "name", "position", "team_id", "price"]].rename(
        columns={"name":"web_name", "position":"pos", "team_id":"team_key"})
    base["pos"] = base.pos.replace({"GKP":"GK"})
    base["can_select"] = True  # no invented historical injury/status snapshots
    grid = base.merge(pd.DataFrame({"gw":gws}), how="cross")
    grid["origin"] = origin
    grid["history_cutoff"] = cutoff.isoformat()
    grid["active_prior"] = grid.element.map(known.set_index("element").mins_r5.fillna(known.set_index("element").prev_mins/38)).ge(45)
    return future, grid, cutoff


def point_metrics(rows, column):
    correlations, top20, captain = [], [], []
    for _, group in rows.groupby(["origin", "gw"] if "origin" in rows else "gw"):
        if group[column].nunique()>1 and group.actual.nunique()>1:
            correlations.append(float(spearmanr(group[column], group.actual).statistic))
        top20.append(float(group.nlargest(20,column).actual.mean()))
        cap = group[(group.pos.isin(["MID","FWD"])) & group.price.ge(7)]
        if len(cap):
            captain.append(float(cap.nlargest(1,column).actual.iloc[0]))
    residual = rows[column]-rows.actual
    return {"n":len(rows), "mae":float(residual.abs().mean()),
            "rmse":float(np.sqrt((residual**2).mean())), "bias":float(residual.mean()),
            "spearman_per_gw":float(np.mean(correlations)) if correlations else None,
            "top20_actual_mean":float(np.mean(top20)),
            "captain_raw_points":float(sum(captain))}


def run(data, output, season="2025-26", horizon=6, origins=None, refit_every=4, augment=True):
    output.mkdir(parents=True,exist_ok=True)
    year = int(season[:4]); history=[]; augmented=[]; aggregates={}
    paths=[]
    for y in range(2023,year+1):
        s=f"{y}-{(y+1)%100:02}"
        path, teams, mapping = season_paths(data,s)
        paths.extend([path,teams,mapping])
        rows=build_season(path,teams,aggregates.get(y-1),player_map_path=mapping,season_start=y)
        if y<year:
            history.append(rows)
            if augment:
                augmented.append(horizon_training(data,s,aggregates.get(y-1),rows))
            aggregates[y]=season_agg(path,mapping)
        else:
            current=rows
    past=pd.concat(history,ignore_index=True)
    augmented_past=pd.concat(augmented,ignore_index=True) if augmented else past.iloc[:0]
    prior=aggregates[year-1]
    origins=origins or list(range(1,int(current.gw.max())+1))
    outcomes=current[[c for c in ["element","gw","fixture","total_points","minutes","yellow_cards","red_cards"] if c in current]].copy()
    outcomes.to_csv(output/"outcomes.csv",index=False)
    actual=outcomes.groupby(["element","gw"],as_index=False).agg(actual=("total_points","sum"),actual_minutes=("minutes","sum"))
    all_pools=[]; lineage=[]; minute_checks=[]
    for j,origin in enumerate(origins):
        future,grid,cutoff=origin_forecasts(data,season,prior,current,origin,horizon)
        if j%refit_every==0:
            training=pd.concat([past,current[current.gw.lt(origin)]],ignore_index=True)
            training=training[training.kickoff_time.add(pd.Timedelta(hours=2)).le(cutoff)]
            minutes_training=pd.concat([training,augmented_past],ignore_index=True)
            model=PointsModel("ensemble").fit(training,minutes_training=minutes_training)
            trained_origin=origin
        components=model.predict_components(future)
        for col in components:
            future[col]=components[col].to_numpy()
        future["hgb"]=future.direct_pred
        future["minutes_model"]=future.component_pred
        future["ensemble"]=future.pred
        future["rolling10"]=future.pts_r10.fillna(future.prev_pts_total/38).fillna(1).clip(lower=0)
        scores=["hgb","minutes_model","ensemble","rolling10","expected_minutes"]
        sums=future.groupby(["element","gw"],as_index=False)[scores].sum()
        counts=future.groupby(["element","gw"]).fixture.nunique().rename("fixture_count").reset_index()
        # Independence approximation is labelled; actual DGW no-shows correlate.
        pnone=future.groupby(["element","gw"]).p_no_appearance.prod().rename("p_no_appearance").reset_index()
        pool=grid.merge(sums,on=["element","gw"],how="left").merge(counts,on=["element","gw"],how="left").merge(pnone,on=["element","gw"],how="left")
        pool[scores+['fixture_count']]=pool[scores+['fixture_count']].fillna(0)
        pool['p_no_appearance']=pool.p_no_appearance.fillna(1)
        pool=pool.merge(actual,on=["element","gw"],how="left")
        pool[["actual","actual_minutes"]]=pool[["actual","actual_minutes"]].fillna(0)
        all_pools.append(pool)
        checks=future.merge(outcomes,on=["element","gw","fixture"],suffixes=("_forecast",""))
        checks["origin"]=origin
        checks["baseline_play"]=checks.appearance_r5.fillna(checks.prev_mins.div(38*75).clip(0,1)).fillna(.5)
        checks["baseline_60"]=checks.sixty_r5.fillna(checks.prev_mins.div(38*90).clip(0,1)).fillna(.3)
        checks["baseline_minutes"]=checks.mins_r5.fillna(checks.prev_mins.div(38)).fillna(30).clip(0,90)
        minute_checks.append(checks[["element","gw","fixture","origin","p_no_appearance","p_1_59","p_60_plus","expected_minutes","minutes","baseline_play","baseline_60","baseline_minutes"]])
        lineage.append({"origin":origin,"trained_origin":trained_origin,"cutoff":cutoff.isoformat(),
                        "training_rows":len(training),"minutes_training_rows":len(minutes_training),
                        "players":int(grid.element.nunique()), "minutes_fit":model.minutes_model.diagnostics_})
        pool.to_csv(output/f"origin_{origin:02}.csv",index=False)
        print(f"GW{origin}: {len(grid)} forecasts; fit GW{trained_origin}",flush=True)
    forecasts=pd.concat(all_pools,ignore_index=True)
    forecasts.to_csv(output/"forecasts.csv.gz",index=False,compression="gzip")
    immediate=forecasts[forecasts.gw.eq(forecasts.origin)]
    metrics={}
    for name in ["hgb","minutes_model","ensemble","rolling10"]:
        metrics[name]={"all":point_metrics(immediate,name),
                       "active_prior":point_metrics(immediate[immediate.active_prior],name),
                       "frozen_horizon_active":point_metrics(forecasts[forecasts.active_prior],name)}
    checks=pd.concat(minute_checks,ignore_index=True)
    def minute_metrics(frame):
        labels=np.where(frame.minutes.eq(0),0,np.where(frame.minutes.lt(60),1,2))
        probs=frame[["p_no_appearance","p_1_59","p_60_plus"]].to_numpy()
        return {"n":len(frame),"minutes_mae":float((frame.expected_minutes-frame.minutes).abs().mean()),
                "baseline_minutes_mae":float((frame.baseline_minutes-frame.minutes).abs().mean()),
                "multiclass_brier":float(np.mean(np.sum((probs-np.eye(3)[labels])**2,axis=1))),
                "appearance_brier":float(np.mean(((1-frame.p_no_appearance)-frame.minutes.gt(0))**2)),
                "baseline_appearance_brier":float(np.mean((frame.baseline_play-frame.minutes.gt(0))**2)),
                "p60_brier":float(np.mean((frame.p_60_plus-frame.minutes.ge(60))**2)),
                "baseline_p60_brier":float(np.mean((frame.baseline_60-frame.minutes.ge(60))**2))}
    minutes_metrics={"next_gw":minute_metrics(checks[checks.gw.eq(checks.origin)]),
                     "next_gw_active_prior":minute_metrics(checks[checks.gw.eq(checks.origin)&checks.baseline_minutes.ge(45)]),
                     "by_lead":{str(int(lead)):minute_metrics(block) for lead,block in checks.groupby(checks.gw-checks.origin+1)},
                     "by_lead_active_prior":{str(int(lead)):minute_metrics(block[block.baseline_minutes.ge(45)]) for lead,block in checks.groupby(checks.gw-checks.origin+1)}}
    checks.to_csv(output/"minutes_evaluation.csv.gz",index=False,compression="gzip")
    result={"season":season,"horizon":horizon,"origins":lineage,"metrics":metrics,"minutes":minutes_metrics,
            "protocol":{"refit_every":refit_every,"ensemble_weight":.5,"horizon_augmentation":augment},
            "limitations":["Development replay, not an untouched holdout or proof of future superiority.",
                           "Origin population/prices are archive approximations, final calendar is known ex post.",
                           "No archived injury news; all models evaluated without news adjustment.",
                           "DGW appearance aggregation assumes independence; fixture expectations are additive.",
                           "FPL scoring varies across seasons; no invented full historical rescoring."],
            "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    result["code_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),Path(__file__).with_name("fpl_features.py"),Path(__file__).with_name("fpl_model.py"),Path(__file__).with_name("minutes_model.py")]}
    (output/"evaluation.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,default=Path("."))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--season",default="2025-26")
    parser.add_argument("--horizon",type=int,default=6)
    parser.add_argument("--origins",help="Comma-separated origins; omit for full season")
    parser.add_argument("--refit-every",type=int,default=4)
    parser.add_argument("--no-augmentation",action="store_true")
    args=parser.parse_args()
    if args.horizon<1 or args.refit_every<1:
        parser.error("Horizon and refit interval must be positive")
    run(args.data_dir,args.output,args.season,args.horizon,
        [int(x) for x in args.origins.split(',')] if args.origins else None,args.refit_every,not args.no_augmentation)


if __name__=="__main__":
    main()
