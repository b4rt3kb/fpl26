"""fpl-ml v0.4 — deadline-proxy walk-forward evaluation and fresh-squad benchmark.

Ladder: naive -> Ridge -> gradient boosting -> P(played) x E(pts|played).
Walk-forward: retrain every 4 GWs on everything strictly before the chunk.
Metrics: MAE, mean per-GW Spearman, mean per-GW precision@10.
Backtest: fresh LP squad each GW (oracle transfers), actual points, captain x2.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fpl_features import FEATS, POS_NAME, build_season, pick_squad, season_agg
from fpl_model import PointsModel, ranking_metrics


def main():
    s23 = build_season("gw_2023-24.csv", "teams_2023-24.csv", None)
    s24 = build_season("gw_2024-25.csv", "teams_2024-25.csv",
                       season_agg("gw_2023-24.csv"))
    s25 = build_season("gw_2025-26.csv", "teams_2025-26.csv",
                       season_agg("gw_2024-25.csv"))
    hist = pd.concat([s23, s24], ignore_index=True)

    s25 = s25.sort_index()
    for col in ("p_ridge", "p_hgb", "p_two"):
        s25[col] = np.nan

    for g0 in range(1, 39, 4):
        chunk = (s25["gw"] >= g0) & (s25["gw"] < g0 + 4)
        if not chunk.any():
            continue
        tr = pd.concat([hist, s25[s25["gw"] < g0]], ignore_index=True)
        te = s25[chunk]
        cutoff = te["feature_cutoff"].min()
        tr = tr[(tr["kickoff_time"] + pd.Timedelta(hours=2) <= cutoff)
                & tr["total_points"].notna() & tr["minutes"].notna()]
        print(f"Fit chunk GW{g0}–{min(g0+3, 38)}; {len(tr)} completed training rows", flush=True)

        ridge = make_pipeline(SimpleImputer(strategy="median"),
                              StandardScaler(), Ridge(alpha=1.0))
        ridge.fit(tr[FEATS], tr["total_points"])
        s25.loc[chunk, "p_ridge"] = ridge.predict(te[FEATS]).clip(min=0)

        s25.loc[chunk, "p_hgb"] = PointsModel("hgb").fit(tr).predict(te)
        s25.loc[chunk, "p_two"] = PointsModel("two").fit(tr).predict(te)

    s25["p_naive"] = s25["pts_r5"].fillna(
        s25["prev_pts_total"] / 38).fillna(1.0).clip(lower=0)

    def metrics(col):
        values = ranking_metrics(s25, col)
        return (values["mae_player_gw"], values["spearman_player_gw"],
                values["precision_at_10_player_gw"])

    print("Historical cutoff proxy: earliest GW kickoff, with 2h completion margin.")
    print("Metrics aggregate each player's fixtures into gameweeks before ranking.")
    results = {"version": "v0.4", "holdout_season": "2025-26", "metrics": {},
               "fresh_squad_benchmark": {},
               "limitations": "Earliest-kickoff deadline proxy, current archived prices, unlimited free squad replacement, no autosubs/chips; not a real-account season simulation."}
    print(f"{'model':<28}{'MAE/GW':>7}{'Spearman':>10}{'prec@10':>9}")
    for name, col in [("naive (rolling 5 / prev)", "p_naive"),
                      ("Ridge", "p_ridge"),
                      ("gradient boosting", "p_hgb"),
                      ("P(played) x E(pts|played)", "p_two")]:
        mae, sp, pr = metrics(col)
        results["metrics"][name] = ranking_metrics(s25, col)
        print(f"{name:<28}{mae:>7.3f}{sp:>10.3f}{pr:>9.3f}")

    # --- season backtest: fresh LP squad each GW, actual pts, captain x2 ---
    def backtest(col):
        total = 0.0
        for gw, g in s25.groupby("gw"):
            agg = (g.groupby("key")
                     .agg(pred=(col, "sum"), actual=("total_points", "sum"),
                          price=("price", "first"), pos_code=("pos_code", "first"),
                          team_key=("team_id", "first")).reset_index())
            agg["pos"] = agg["pos_code"].map(POS_NAME)
            sel = pick_squad(agg, pred_col="pred")
            xi = sel[sel["in_xi"] == 1]
            total += xi["actual"].sum() + sel.loc[sel["is_cap"] == 1,
                                                  "actual"].sum()
        return total

    for name, col in [("naive", "p_naive"), ("gradient boosting", "p_hgb"),
                      ("two-stage", "p_two")]:
        total = backtest(col)
        results["fresh_squad_benchmark"][name] = float(total)
        print(f"backtest 25/26 (oracle transfers, no auto-subs) "
              f"{name}: {total:.0f} pts", flush=True)
    output = Path("state/benchmark-v04.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
