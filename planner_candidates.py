"""Exact candidate reduction for the planner's constant-price horizon model.

A nonowned player p is removable when K retained, selectable, nonowned players
of the same club and position cost no more and score at least as much in EVERY
GW, where K is the club quota. Fix an order of those K dominators. Whenever p
appears in a roster, replace it with the first absent dominator. One is absent
because that roster already uses a club slot on p. This replacement map cannot
increase the number of changed players between two rosters: when the selected
replacement differs, the earlier replacement is present in the other roster.
Thus acquisition counts and hits cannot rise. All affected players are initially
nonowned, so fixed prices give no acquisition-price spread; lower roster costs
preserve cash feasibility. Apply the same map to temporary Free Hit rosters.

The argument relies on fixed prices, clubs, positions and selectability across
the supplied horizon. No future-GW averaging or heuristic top-N cut is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def prune_dominated(forecasts: pd.DataFrame, owned_ids, club_limit=3):
    """Return (filtered forecasts, auditable exact-pruning metadata).

    The input frame is never modified. Equal players use the smaller element ID
    as a stable dominance tie-break; sequential removal prevents cyclic pruning.
    All currently owned players survive, including unselectable ones.
    """
    if type(club_limit) is not int or club_limit < 1:
        raise ValueError("club_limit must be a positive integer")
    required = {"element", "gw", "team_key", "pos", "price", "pred"}
    if not required.issubset(forecasts.columns):
        raise ValueError("Dominance pruning requires columns: " + str(sorted(required)))
    if forecasts.empty or forecasts.duplicated(["element", "gw"]).any():
        raise ValueError("Dominance pruning requires nonempty, unique player/GW forecasts")
    if not np.isfinite(forecasts[["price", "pred"]].to_numpy(dtype=float)).all():
        raise ValueError("Dominance pruning requires finite prices and forecasts")
    if (forecasts["price"] <= 0).any():
        raise ValueError("Dominance pruning requires positive prices")
    first = forecasts[forecasts.gw == forecasts.gw.min()].set_index("element")
    ids = set(forecasts.element.unique())
    owned = set(owned_ids)
    if not owned <= ids:
        raise ValueError("Forecast omits owned players")
    if len(first) != len(ids) or len(forecasts) != len(ids) * forecasts.gw.nunique():
        raise ValueError("Each player needs every GW for dominance pruning")
    stable_columns = ["price", "pos", "team_key"]
    if "can_select" in forecasts.columns:
        stable_columns.append("can_select")
        if not forecasts.can_select.map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise ValueError("can_select must contain explicit booleans")
    if (forecasts.groupby("element")[stable_columns].nunique(dropna=False) != 1).any().any():
        raise ValueError("Exact dominance pruning requires constant prices/positions/clubs/selectability")
    xp = forecasts.pivot(index="element", columns="gw", values="pred")
    if xp.isna().any().any():
        raise ValueError("Each player needs every GW for dominance pruning")
    unavailable = {i for i in ids if i not in owned and
                   "can_select" in first.columns and not bool(first.loc[i, "can_select"])}
    retained = ids - unavailable
    removals = []
    for _, group in first.groupby(["team_key", "pos"], sort=False):
        group_ids = sorted(i for i in group.index if i in retained and i not in owned)
        if len(group_ids) <= club_limit:
            continue
        predictions = xp.loc[group_ids].to_numpy(dtype=float)
        prices = first.loc[group_ids, "price"].to_numpy(dtype=float)
        # Matrix[j, i] means j dominates i. Exact comparisons are deliberate:
        # a tolerance could delete a player who is genuinely better in one GW.
        at_least = (predictions[:, None, :] >= predictions[None, :, :]).all(axis=2)
        strict = (predictions[:, None, :] > predictions[None, :, :]).any(axis=2)
        index = np.asarray(group_ids)
        dominance = ((prices[:, None] <= prices[None, :]) & at_least &
                     ((prices[:, None] < prices[None, :]) | strict | (index[:, None] < index[None, :])))
        for offset in range(len(group_ids) - 1, -1, -1):
            player = group_ids[offset]
            witnesses = [other for j, other in enumerate(group_ids)
                         if other in retained and dominance[j, offset]]
            if len(witnesses) >= club_limit:
                retained.remove(player)
                removals.append({"element": int(player),
                                 "dominated_by_at_removal": [int(i) for i in witnesses[:club_limit]]})
    filtered = forecasts[forecasts.element.isin(retained)].copy()
    metadata = {
        "method": "exact_same_club_position_componentwise_dominance",
        "input_players": len(ids), "retained_players": len(retained),
        "removed_unselectable_count": len(unavailable), "removed_dominated_count": len(removals),
        "club_limit": club_limit, "owned_players_retained": len(owned),
        "removed_unselectable": [int(i) for i in sorted(unavailable)],
        "dominated_removals": removals,
        "heuristic": False,
        "assumptions": ["constant prices, clubs, positions and selectability across the horizon",
                        "dominators and dominated candidates were not initially owned",
                        "componentwise dominance in every supplied gameweek"],
    }
    return filtered, metadata
