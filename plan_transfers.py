"""Stateful, finite-horizon FPL planner using native HiGHS with a warm start.

Forecast prices stay constant. Chip reserve values are explicit opportunity-cost
assumptions, NOT a learned claim about the optimal date later in the season.
No action is sent to FPL; this module only writes a reviewable recommendation.
"""
import argparse
from collections import Counter, deque
import copy
from functools import lru_cache
import json
from pathlib import Path
from types import SimpleNamespace

import highspy
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from planner_candidates import prune_dominated


DEFAULT_RESERVES = {"wildcard": 25.0, "freehit": 20.0, "bboost": 18.0, "3xc": 12.0}
QUOTA = {"GK": (2, 1, 1), "DEF": (5, 3, 5), "MID": (5, 2, 5), "FWD": (3, 1, 3)}


class Program:
    def __init__(self):
        self.cost, self.lb, self.ub, self.integer = [], [], [], []
        self.rows, self.cols, self.values, self.lower, self.upper = [], [], [], [], []

    def var(self, reward=0.0, upper=1, integer=True):
        k = len(self.cost)
        self.cost.append(-float(reward))
        self.lb.append(0)
        self.ub.append(upper)
        self.integer.append(int(integer))
        return k

    def row(self, terms, lower=-np.inf, upper=np.inf):
        r = len(self.lower)
        for k, v in terms:
            if v:
                self.rows.append(r)
                self.cols.append(k)
                self.values.append(float(v))
        self.lower.append(lower)
        self.upper.append(upper)

    def eq(self, terms, value=0):
        self.row(terms, value, value)

    def solve(self, seconds, seed=None):
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("Solver seconds must be finite and nonnegative")
        matrix = coo_matrix((self.values, (self.rows, self.cols)),
                            shape=(len(self.lower), len(self.cost))).tocsr()
        solver = highspy.Highs()
        solver.setOptionValue("output_flag", False)
        solver.setOptionValue("threads", 2)
        solver.setOptionValue("time_limit", float(seconds))
        solver.setOptionValue("mip_rel_gap", 0.005)
        n = len(self.cost)
        solver.addVars(n, np.asarray(self.lb), np.asarray(self.ub))
        solver.changeColsCost(n, np.arange(n, dtype=np.int32), np.asarray(self.cost))
        solver.addRows(len(self.lower), np.asarray(self.lower), np.asarray(self.upper),
                       matrix.nnz, matrix.indptr.astype(np.int32),
                       matrix.indices.astype(np.int32), matrix.data)
        integer_ids = np.flatnonzero(self.integer).astype(np.int32)
        solver.changeColsIntegrality(len(integer_ids), integer_ids,
                                    np.full(len(integer_ids), highspy.HighsVarType.kInteger, dtype=np.uint8))
        if seed:
            solver.setSolution(len(seed), np.asarray(list(seed), dtype=np.int32),
                               np.asarray(list(seed.values()), dtype=float))
        solver.run()
        solution, info, status = solver.getSolution(), solver.getInfo(), solver.getModelStatus()
        feasible = (solution.value_valid and info.valid and
                    info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible)
        values = np.asarray(solution.col_value) if feasible else None
        if values is not None:
            residual = matrix @ values
            feasible = (np.isfinite(values).all() and
                        np.all(values >= np.asarray(self.lb) - 1e-5) and
                        np.all(values <= np.asarray(self.ub) + 1e-5) and
                        np.all(residual >= np.asarray(self.lower) - 1e-5) and
                        np.all(residual <= np.asarray(self.upper) + 1e-5) and
                        np.max(np.abs(values[integer_ids] - np.rint(values[integer_ids])), initial=0) <= 1e-5)
        return SimpleNamespace(x=values if feasible else None,
                               status=0 if status == highspy.HighsModelStatus.kOptimal else 1 if feasible else 2,
                               message=solver.modelStatusToString(status),
                               mip_gap=float(info.mip_gap) if np.isfinite(info.mip_gap) else None)


def _position(value):
    return "GK" if value in ("GKP", "GK") else value


def _check_squad(ids, metadata, xi=None, club_limit=3):
    if len(ids) != 15 or len(set(ids)) != 15:
        raise ValueError("Squad must contain 15 unique players")
    for pos, (number, low, high) in QUOTA.items():
        if sum(metadata[i]["pos"] == pos for i in ids) != number:
            raise ValueError("Invalid squad position quota: " + pos)
        if xi is not None and not low <= sum(metadata[i]["pos"] == pos for i in xi) <= high:
            raise ValueError("Invalid XI formation: " + pos)
    teams = [metadata[i]["team"] for i in ids]
    if club_limit is not None and any(teams.count(t) > club_limit for t in set(teams)):
        raise ValueError("Club limit exceeded")
    if xi is not None and (len(xi) != 11 or not set(xi).issubset(ids)):
        raise ValueError("Invalid starting XI")


def _freehit_seed(owned, metadata, first, bank, selling, club_limit):
    """Cheapest legal FH roster by integer min-cost flow, without another MILP.

    Position supplies and club capacities form a bipartite flow network. Keeping
    an owned player costs their sale value, so the cash test preserves spreads.
    This supplies a legal forced-FH incumbent even when the permanent squad has
    grandfathered club overflow. It is only a seed, never the recommendation.
    """
    clubs = sorted({m["team"] for m in metadata.values()})
    positions = {pos: n + 1 for n, pos in enumerate(QUOTA)}
    team_nodes = {club: len(positions) + n + 1 for n, club in enumerate(clubs)}
    sink = len(positions) + len(clubs) + 1
    graph = [[] for _ in range(sink + 1)]

    def add(a, b, capacity, cost):
        forward = [b, capacity, cost, len(graph[b])]
        reverse = [a, 0, -cost, len(graph[a])]
        graph[a].append(forward)
        graph[b].append(reverse)
        return forward

    for pos, (number, _, _) in QUOTA.items():
        add(0, positions[pos], number, 0)
    for team in clubs:
        add(team_nodes[team], sink, club_limit, 0)
    edges = {}
    for pid, info in metadata.items():
        if pid not in owned and "can_select" in first and not bool(first.loc[pid, "can_select"]):
            continue
        cost = selling[pid] if pid in owned else info["price"]
        edges[pid] = add(positions[info["pos"]], team_nodes[info["team"]], 1, cost)
    for _ in range(15):
        distance, previous = [float("inf")] * len(graph), [None] * len(graph)
        distance[0] = 0
        queue, queued = deque([0]), {0}
        while queue:
            node = queue.popleft()
            queued.remove(node)
            for index, edge in enumerate(graph[node]):
                target, capacity, cost, _ = edge
                if capacity and distance[node] + cost < distance[target]:
                    distance[target] = distance[node] + cost
                    previous[target] = node, index
                    if target not in queued:
                        queue.append(target)
                        queued.add(target)
        if previous[sink] is None:
            raise ValueError("No legal Free Hit roster can repair the existing club overflow")
        node = sink
        while node:
            parent, index = previous[node]
            edge = graph[parent][index]
            edge[1] -= 1
            graph[node][edge[3]][1] += 1
            node = parent
    selected = {pid for pid, edge in edges.items() if edge[1] == 0}
    cost = sum(selling[i] if i in owned else metadata[i]["price"] for i in selected)
    if cost > bank + sum(selling.values()):
        raise ValueError("No affordable Free Hit roster can repair the existing club overflow")
    _check_squad(list(selected), metadata, club_limit=club_limit)
    return selected


def _scoring_chip_seed(steps, chips, xp, reserves, discount):
    """Exactly schedule BB/TC on fixed baseline squads before the joint solve.

    These chips leave transfers, bank and FT unchanged. A small matching DP
    prevents a short search from missing a profitable expiring scoring chip.
    WC/FH remain decisions for the full model because they change the squads.
    """
    if not steps:
        return ()
    end = steps[-1]["gw"]

    @lru_cache(None)
    def choose(t, used):
        if t == len(steps):
            return 0.0, ()
        best = choose(t + 1, used)
        step = steps[t]
        if step.get("chip"):
            return best
        for j, chip in enumerate(chips):
            if used & (1 << j) or chip["name"] not in ("bboost", "3xc"):
                continue
            if not int(chip["start_event"]) <= step["gw"] <= int(chip["stop_event"]):
                continue
            bonus = (xp[step["captain"], step["gw"]] if chip["name"] == "3xc"
                     else sum(xp[i, step["gw"]] for i in step["bench"]))
            reserve = reserves[chip["name"]] if int(chip["stop_event"]) > end else 0.0
            future, schedule = choose(t + 1, used | (1 << j))
            gain = discount ** t * bonus - reserve + future
            if gain > best[0] + 1e-9:
                best = gain, ((j, t),) + schedule
        return best

    return choose(0, 0)[1]


def plan(state, forecasts, *, allow_chips=True, chip_reserves=None,
         discount=0.9, ft_value=1.5, seconds=60, forced_chip=None):
    """Optimize transfers, XI, captain and chips, preserving persistent holdings.

    forced_chip optionally contains (name, gameweek), primarily for comparisons.
    All currency is integer tenths internally. Public imported prices have
    provenance in state; callers should verify inferred purchase costs.
    """
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError("Solver seconds must be finite and nonnegative")
    if not np.isfinite(discount) or not 0 < discount <= 1 or not np.isfinite(ft_value) or ft_value < 0:
        raise ValueError("Invalid horizon discount or terminal FT value")
    if chip_reserves is not None:
        if (not set(chip_reserves).issubset(DEFAULT_RESERVES) or
                any(not np.isfinite(float(v)) or float(v) < 0 for v in chip_reserves.values())):
            raise ValueError("Invalid chip reserve values")
    df = forecasts.copy()
    required = {"element", "gw", "pos", "price", "pred", "team_key", "web_name"}
    if not required.issubset(df.columns):
        raise ValueError("Forecast columns missing: " + str(sorted(required - set(df.columns))))
    if df.duplicated(["element", "gw"]).any():
        raise ValueError("Duplicate player/gameweek forecast")
    if not np.isfinite(df[["price", "pred"]].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite price or forecast")
    if (df.price <= 0).any():
        raise ValueError("Prices must be positive")
    df["pos"] = df.pos.map(_position)
    original_owned_ids = {int(p["id"]) for p in state["players"]}
    if not original_owned_ids.issubset(set(df.element.astype(int))):
        raise ValueError("Forecast omits owned players")
    df, pruning = prune_dominated(df, original_owned_ids,
                                  club_limit=int(state.get("rules", {}).get("club_limit", 3)))
    gws = sorted(int(x) for x in df.gw.unique())
    start = int(state["next_event"])
    if not gws or gws != list(range(start, start + len(gws))) or gws[-1] > 38:
        raise ValueError("Forecast must cover consecutive gameweeks from next_event")
    ids = sorted(int(x) for x in df.element.unique())
    if len(df) != len(ids) * len(gws):
        raise ValueError("Each player needs every gameweek, including zero for blanks")
    first = df[df.gw == start].set_index("element")
    owned = {int(p["id"]): p for p in state["players"]}
    if not set(owned).issubset(ids):
        raise ValueError("Forecast omits owned players")
    metadata = {i: {"id": i, "name": str(first.loc[i, "web_name"]),
                    "pos": str(first.loc[i, "pos"]), "team": int(first.loc[i, "team_key"]),
                    "price": int(round(float(first.loc[i, "price"]) * 10))} for i in ids}
    for i, block in df.groupby("element"):
        if block[["price", "pos", "team_key"]].nunique().max() != 1:
            raise ValueError("This planner requires constant prices/positions/clubs across horizon")
    rules = state.get("rules", {})
    max_ft = int(rules.get("max_free_transfers", 5))
    ft_overrides = {int(g): int(v) for g, v in rules.get("free_transfer_overrides", {}).items()}
    if any(not 1 <= g <= 38 or not 1 <= v <= max_ft for g, v in ft_overrides.items()):
        raise ValueError("Invalid dated free-transfer override")
    hit = int(rules.get("transfer_hit", 4))
    club_limit = int(rules.get("club_limit", 3))
    free0, bank0 = int(state["free_transfers"]), int(state["bank"])
    free0 = ft_overrides.get(start, free0)
    if not 1 <= free0 <= max_ft or bank0 < 0:
        raise ValueError("Invalid confirmed free transfers or bank")
    # A real club move may leave an already-owned squad above the normal quota.
    # Holding it is legal; the first permanent transfer must repair every club.
    _check_squad(list(owned), metadata, club_limit=None)
    initial_excess = {team: count - club_limit for team, count in
                      Counter(metadata[i]["team"] for i in owned).items() if count > club_limit}
    sell0 = {}
    for i, p in owned.items():
        buy = int(p["purchase_price"])
        now = metadata[i]["price"]
        sell0[i] = now if now <= buy else buy + (now - buy) // 2
        if now != int(p["now_cost"]) or sell0[i] != int(p["selling_price"]):
            raise ValueError("State prices differ from forecasts; re-import state")
    reserves = dict(DEFAULT_RESERVES)
    if chip_reserves:
        reserves.update({k: float(v) for k, v in chip_reserves.items()})
    chips = [c for c in state.get("chips", []) if c.get("status") not in ("used", "expired", "unavailable")
             and not c.get("used_event") and c["name"] in reserves] if allow_chips else []
    # Establish a strong, feasible no-chip alternative before the larger joint
    # search. This is also the independent fallback if the joint search times out.
    baseline = None
    if chips and forced_chip is None:
        baseline = plan(state, df, allow_chips=False, chip_reserves=reserves,
                        discount=discount, ft_value=ft_value, seconds=min(15, seconds))
    xp = {(int(r.element), int(r.gw)): float(r.pred) for r in df.itertuples()}
    model = Program()
    chip_vars, per_name = {}, {}
    for j, ch in enumerate(chips):
        for t, gw in enumerate(gws):
            if int(ch["start_event"]) <= gw <= int(ch["stop_event"]):
                reserve = reserves[ch["name"]] if int(ch["stop_event"]) > gws[-1] else 0
                k = model.var(reward=-reserve)
                chip_vars[j, t] = k
                per_name.setdefault((ch["name"], t), []).append(k)
        model.row([(v, 1) for (jj, _), v in chip_vars.items() if j == jj], upper=1)
    for t in range(len(gws)):
        model.row([(v, 1) for (_, tt), v in chip_vars.items() if tt == t], upper=1)
        if t:
            model.row([(v, 1) for tt in (t - 1, t) for v in per_name.get(("freehit", tt), [])], upper=1)
    if any(c.get("name") == "freehit" and int(c.get("event", c.get("used_event", -1))) == start - 1
           for c in state.get("chips_used", [])):
        model.eq([(v, 1) for v in per_name.get(("freehit", 0), [])])
    if forced_chip:
        name, gw = forced_chip
        options = per_name.get((name, int(gw) - start), [])
        if not options:
            raise ValueError("Requested chip is unavailable in that gameweek")
        model.eq([(v, 1) for v in options], 1)

    persistent, active, xi, captain = {}, {}, {}, {}
    incoming, outgoing, original, original_out = {}, {}, {}, {}
    bb_seed_vars, tc_seed_vars, keep_seed_vars = {}, {}, {}
    banks, choices = {}, {}
    for t, gw in enumerate(gws):
        weight = discount ** t
        fh = per_name.get(("freehit", t), [])
        wc = per_name.get(("wildcard", t), [])
        bb = per_name.get(("bboost", t), [])
        tc = per_name.get(("3xc", t), [])
        banks[t] = model.var(upper=bank0+sum(metadata[i]["price"] for i in owned), integer=False)
        bb_bonus, tc_bonus = [], []
        for i in ids:
            key = i, t
            persistent[key] = model.var()
            active[key] = model.var(reward=weight * 0.001 * xp[i, gw])
            xi[key] = model.var(reward=weight * 0.999 * xp[i, gw])
            captain[key] = model.var(reward=weight * xp[i, gw])
            # Binary roster transitions and no simultaneous buy/sell already
            # force these values to 0/1; extra integer columns slow the search.
            incoming[key], outgoing[key] = model.var(integer=False), model.var(integer=False)
            # Persistent roster changes only through actual permanent transfers.
            transition = [(persistent[key], 1), (incoming[key], -1), (outgoing[key], 1)]
            if t:
                transition.append((persistent[i, t - 1], -1))
            model.eq(transition, int(i in owned) if t == 0 else 0)
            if t:
                model.row([(outgoing[key], 1), (persistent[i, t - 1], -1)], upper=0)
                model.row([(incoming[key], 1), (persistent[i, t - 1], 1)], upper=1)
            else:
                model.row([(outgoing[key], 1)], upper=int(i in owned))
                model.row([(incoming[key], 1)], upper=int(i not in owned))
            for trans in (incoming[key], outgoing[key]):
                model.row([(trans, 1)] + [(v, 1) for v in fh], upper=1)
            for sign in (1, -1):
                model.row([(active[key], sign), (persistent[key], -sign)] + [(v, -1) for v in fh], upper=0)
            model.row([(xi[key], 1), (active[key], -1)], upper=0)
            model.row([(captain[key], 1), (xi[key], -1)], upper=0)
            if "can_select" in first and not bool(first.loc[i, "can_select"]):
                model.eq([(incoming[key], 1)])
                # A removed player can remain on the persistent roster but may
                # not be acquired for a Free Hit squad.
                previous = [(persistent[i, t - 1], -1)] if t else []
                model.row([(active[key], 1)] + [(v, 1) for v in fh] + previous,
                          upper=1 + (int(i in owned) if not t else 0))
            if i in owned:
                original[key], original_out[key] = model.var(integer=False), model.var(integer=False)
                terms = [(original[key], 1), (original_out[key], 1)]
                if t:
                    terms.append((original[i, t - 1], -1))
                model.eq(terms, 1 if not t else 0)
                model.row([(original[key], 1), (persistent[key], -1)], upper=0)
                model.row([(original_out[key], 1), (outgoing[key], -1)], upper=0)
                if t:
                    model.row([(original_out[key], 1), (original[i, t - 1], -1)], upper=0)
                    model.row([(outgoing[key], 1), (original[i, t - 1], 1), (original_out[key], -1)], upper=1)
            # Bench Boost adds the bench once; Triple Captain adds captain once.
            if bb:
                b = model.var(reward=weight * xp[i, gw], integer=False)
                bb_seed_vars[key] = b
                bb_bonus.append(b)
                model.row([(b, 1), (active[key], -1)], upper=0)
                model.row([(b, 1), (xi[key], 1)], upper=1)
                model.row([(b, 1), (active[key], -1), (xi[key], 1)], upper=0)
                model.row([(b, 1)] + [(v, -1) for v in bb], upper=0)
                model.row([(active[key], 1), (xi[key], -1), (b, -1)] + [(v, 1) for v in bb], upper=1)
            if tc:
                c = model.var(reward=weight * xp[i, gw], integer=False)
                tc_seed_vars[key] = c
                tc_bonus.append(c)
                model.row([(c, 1), (captain[key], -1)], upper=0)
                model.row([(c, 1)] + [(v, -1) for v in tc], upper=0)
                model.row([(captain[key], 1), (c, -1)] + [(v, 1) for v in tc], upper=1)

        # Valid integer identities substantially tighten the chip LP relaxation.
        if bb:
            model.eq([(v, 1) for v in bb_bonus]+[(v, -4) for v in bb])
        if tc:
            model.eq([(v, 1) for v in tc_bonus]+[(v, -1) for v in tc])

        for roster in (persistent, active):
            model.eq([(roster[i, t], 1) for i in ids], 15)
            for pos, (number, _, _) in QUOTA.items():
                model.eq([(roster[i, t], 1) for i in ids if metadata[i]["pos"] == pos], number)
        model.eq([(xi[i, t], 1) for i in ids], 11)
        model.eq([(captain[i, t], 1) for i in ids], 1)
        for pos, (_, low, high) in QUOTA.items():
            model.row([(xi[i, t], 1) for i in ids if metadata[i]["pos"] == pos], low, high)
        money = [(banks[t], 1)]
        if t:
            money.append((banks[t - 1], -1))
        for i in ids:
            price = metadata[i]["price"]
            money += [(incoming[i, t], price), (outgoing[i, t], -price)]
            if i in owned:
                money.append((original_out[i, t], price - sell0[i]))
        model.eq(money, bank0 if not t else 0)
        # Free Hit affordability uses selling proceeds, but keeping an owned
        # player does not incur the spread between their sale and market price.
        if fh:
            # This affordability inequality also holds for ordinary permanent
            # transfers. Keeping it unconditional avoids a loose big-M Free
            # Hit relaxation without excluding any legal roster transition.
            afford = [(active[i, t], metadata[i]["price"]) for i in ids]
            if t:
                afford += [(banks[t - 1], -1)]
                afford += [(persistent[i, t - 1], -metadata[i]["price"]) for i in ids]
            for i in owned:
                spread = metadata[i]["price"] - sell0[i]
                if not spread:
                    continue
                if t:
                    keep = model.var(integer=False)
                    keep_seed_vars[i, t] = keep
                    model.row([(keep, 1), (active[i, t], -1)], upper=0)
                    model.row([(keep, 1), (original[i, t - 1], -1)], upper=0)
                    model.row([(active[i, t], 1), (original[i, t - 1], 1), (keep, -1)], upper=1)
                    afford += [(original[i, t - 1], spread), (keep, -spread)]
                else:
                    afford.append((active[i, t], -spread))
            base = bank0 + sum(sell0.values()) if not t else 0
            model.row(afford, upper=base)

        # Enumerated transitions encode exact transfer banking without unsafe
        # max/min relaxations. WC/FH preserve the existing FT balance.
        choices[t] = []
        transfer_chips = wc + fh
        for free in range(1, max_ft + 1):
            for n in range(16):
                for special in range(2 if transfer_chips else 1):
                    after = ft_overrides.get(gw + 1, free if special else min(max_ft, max(0, free - n) + 1))
                    charged = 0 if special else max(0, n - free)
                    terminal = ft_value * after if t == len(gws) - 1 else 0
                    q = model.var(reward=-weight * hit * charged + terminal)
                    choices[t].append((q, free, n, special, after, charged))
        model.eq([(q, 1) for q, *_ in choices[t]], 1)
        before = [(q, free) for q, free, *_ in choices[t]]
        if t:
            before += [(q, -after) for q, _, _, _, after, _ in choices[t - 1]]
        model.eq(before, free0 if not t else 0)
        model.eq([(q, n) for q, _, n, *_ in choices[t]] + [(incoming[i, t], -1) for i in ids])
        model.eq([(q, special) for q, _, _, special, *_ in choices[t]] + [(v, -1) for v in transfer_chips])

        # Overflow is permitted only on a zero-transfer transition. Inductively
        # zero transfers cannot create it, and any acquisition restores <=3.
        # FH always has zero permanent transfers, but its active roster must be
        # legal while the grandfathered permanent holdings revert unchanged.
        no_transfers = [q for q, _, n, *_ in choices[t] if n == 0]
        for team in {m["team"] for m in metadata.values()}:
            excess = initial_excess.get(team, 0)
            allowance = [(q, -excess) for q in no_transfers]
            for roster in (persistent, active):
                terms = [(roster[i, t], 1) for i in ids if metadata[i]["team"] == team] + allowance
                if roster is active:
                    terms += [(v, excess) for v in fh]
                model.row(terms, upper=club_limit)

    # Supply every integer decision. HiGHS completes continuous auxiliaries by
    # LP, so even a zero-time search starts with a legal incumbent. A forced-chip
    # comparison starts from a no-transfer roster using exactly that chip.
    # A complete feasible seed avoids relying on a timed LP to fill auxiliaries
    # when the requested MIP search budget is zero.
    seed, seed_free = dict.fromkeys(range(len(model.cost)), 0), free0
    seed_previous, seed_original = set(owned), set(owned)
    forced_fh_roster = (_freehit_seed(owned, metadata, first, bank0, sell0, club_limit)
                        if initial_excess and forced_chip and forced_chip[0] == "freehit" else None)
    seed_chips = (set(_scoring_chip_seed(baseline["steps"], chips, xp, reserves, discount))
                  if baseline else set())
    for (j, t), v in chip_vars.items():
        seed[v] = int((j, t) in seed_chips or
                      bool(forced_chip) and (chips[j]["name"], gws[t]) == forced_chip)
    for t, gw in enumerate(gws):
        if baseline:
            step = baseline["steps"][t]
            holdings = set(step["permanent_squad"])
            starters, cap = set(step["starting_xi"]), step["captain"]
            n, seed_cash = len(step["transfer_in"]), step["bank"]
            seed_active = holdings
        else:
            holdings, n, seed_cash = set(owned), 0, bank0
            seed_active = (forced_fh_roster if forced_fh_roster is not None and forced_chip[1] == gw
                           else holdings)
            alternatives = []
            for defenders in range(3, 6):
                for mids in range(2, 6):
                    forwards = 10 - defenders - mids
                    if not 1 <= forwards <= 3:
                        continue
                    lineup = []
                    for pos, count in {"GK": 1, "DEF": defenders, "MID": mids, "FWD": forwards}.items():
                        lineup += sorted((i for i in seed_active if metadata[i]["pos"] == pos),
                                         key=lambda i: (-xp[i, gw], i))[:count]
                    c = max(lineup, key=lambda i: xp[i, gw])
                    alternatives.append((sum(xp[i, gw] for i in lineup) + xp[c, gw], lineup, c))
            _, lineup, cap = max(alternatives, key=lambda item: item[0])
            starters = set(lineup)
        for i in ids:
            seed[persistent[i, t]] = int(i in holdings)
            seed[active[i, t]] = int(i in seed_active)
            seed[xi[i, t]] = int(i in starters)
            seed[captain[i, t]] = int(i == cap)
            seed[incoming[i, t]] = int(i in holdings and i not in seed_previous)
            seed[outgoing[i, t]] = int(i in seed_previous and i not in holdings)
            if i in owned:
                seed[original_out[i, t]] = int(i in seed_original and i not in holdings)
                seed[original[i, t]] = int(i in seed_original and i in holdings)
            if (i, t) in keep_seed_vars:
                seed[keep_seed_vars[i, t]] = int(i in seed_original and i in seed_active)
            if (i, t) in bb_seed_vars:
                seed[bb_seed_vars[i, t]] = int(any(seed[v] for v in per_name.get(("bboost", t), []))
                                                and i in seed_active and i not in starters)
            if (i, t) in tc_seed_vars:
                seed[tc_seed_vars[i, t]] = int(any(seed[v] for v in per_name.get(("3xc", t), [])) and i == cap)
        seed[banks[t]] = seed_cash
        special = int(bool(forced_chip) and forced_chip[1] == gw and forced_chip[0] in ("wildcard", "freehit"))
        for q, before, transfers, c, _, _ in choices[t]:
            seed[q] = int((before, transfers, c) == (seed_free, n, special))
        seed_free = ft_overrides.get(gw + 1, seed_free if special else min(max_ft, max(0, seed_free - n) + 1))
        seed_original &= holdings
        seed_previous = holdings
    result = model.solve(seconds, seed)
    if result.x is None or not np.isfinite(result.x).all() or result.status not in (0, 1):
        if baseline:
            safe = copy.deepcopy(baseline)
            safe["candidate_pruning"] = copy.deepcopy(pruning)
            safe["chip_search"] = {"accepted": False, "message": result.message}
            safe["warnings"].append("Joint chip search returned no feasible plan; using the validated no-chip baseline.")
            return safe
        raise RuntimeError("Planner found no feasible recommendation: " + str(result.message))
    values = result.x
    integers = np.flatnonzero(model.integer)
    if np.max(np.abs(values[integers] - np.rint(values[integers]))) > 1e-5:
        raise RuntimeError("Solver returned a fractional recommendation")
    selected = lambda k: values[k] > 0.5
    steps, previous = [], set(owned)
    original_remaining, cash, free = set(owned), bank0, free0
    used = set()
    previous_chip = "freehit" if any(c.get("name") == "freehit" and int(c.get("event", c.get("used_event", -1))) == start - 1 for c in state.get("chips_used", [])) else None
    for t, gw in enumerate(gws):
        perm = {i for i in ids if selected(persistent[i, t])}
        roster = [i for i in ids if selected(active[i, t])]
        starters = [i for i in ids if selected(xi[i, t])]
        captains = [i for i in ids if selected(captain[i, t])]
        ins, outs = sorted(perm - previous), sorted(previous - perm)
        chosen = [j for (j, tt), k in chip_vars.items() if tt == t and selected(k)]
        chip = chips[chosen[0]]["name"] if chosen else None
        if len(chosen) > 1 or any(j in used for j in chosen):
            raise RuntimeError("Invalid chip use")
        used.update(chosen)
        if chip == "freehit" and previous_chip == "freehit":
            raise RuntimeError("Consecutive Free Hits")
        _check_squad(list(perm), metadata, club_limit=club_limit if ins else None)
        _check_squad(roster, metadata, starters, club_limit if ins or chip == "freehit" else None)
        if not ins and perm != previous:
            raise RuntimeError("Zero-transfer transition changed permanent holdings")
        if len(captains) != 1 or captains[0] not in starters:
            raise RuntimeError("Invalid captain")
        if chip == "freehit":
            if perm != previous:
                raise RuntimeError("Free Hit permanently changed squad")
            liquidation = cash + sum(sell0[i] if i in original_remaining else metadata[i]["price"] for i in previous)
            cost = sum(sell0[i] if i in original_remaining else metadata[i]["price"] for i in roster)
            if cost > liquidation:
                raise RuntimeError("Free Hit exceeds budget")
        elif set(roster) != perm:
            raise RuntimeError("Active squad differs from permanent squad")
        for i in outs:
            cash += sell0[i] if i in original_remaining else metadata[i]["price"]
            original_remaining.discard(i)
        cash -= sum(metadata[i]["price"] for i in ins)
        if cash < 0 or abs(values[banks[t]] - cash) > 1e-4:
            raise RuntimeError("Budget validation failed")
        charged = 0 if chip in ("freehit", "wildcard") else max(0, len(ins) - free)
        next_free = ft_overrides.get(gw + 1, free if chip in ("freehit", "wildcard") else min(max_ft, max(0, free - len(ins)) + 1))
        captain_id = captains[0]
        vice = max((i for i in starters if i != captain_id), key=lambda i: xp[i, gw])
        bench = sorted((i for i in roster if i not in starters), key=lambda i: (metadata[i]["pos"] != "GK", -xp[i, gw]))
        raw_points = sum(xp[i, gw] for i in (roster if chip == "bboost" else starters)) + xp[captain_id, gw] * (2 if chip == "3xc" else 1)
        steps.append({"gw": gw, "chip": chip, "transfer_in": ins, "transfer_out": outs,
                      "temporary_in": sorted(set(roster) - previous) if chip == "freehit" else [],
                      "temporary_out": sorted(previous - set(roster)) if chip == "freehit" else [],
                      "free_transfers_before": free, "free_transfers_next": next_free,
                      "hit_cost": charged * hit, "bank": cash,
                      "permanent_squad": sorted(perm), "squad": roster,
                      "starting_xi": starters, "captain": captain_id, "vice_captain": vice,
                      "bench": bench, "predicted_points": round(raw_points, 3),
                      "net_predicted_points": round(raw_points - charged * hit, 3)})
        previous, free, previous_chip = perm, next_free, chip
    output = {"schema_version": 1, "entry_id": state.get("entry_id"), "gameweeks": gws,
            "candidate_pruning": pruning,
            "solver": {"status": int(result.status), "message": str(result.message),
                       "optimal_within_tolerance": bool(result.status == 0),
                       "mip_gap": result.mip_gap,
                       "backend": "highspy", "scope": "joint_transfers_and_chips" if chips else "transfers_without_chips"},
            "assumptions": {"discount": discount, "terminal_free_transfer_value": ft_value,
                            "chip_reserve_values": reserves, "prices_constant": True,
                            "bench_tiebreak_weight": 0.001,
                            "automatic_substitutions_modelled": False,
                            "vice_captain_contingency_modelled": False,
                            "forecast_uncertainty_modelled": False,
                            "chip_reserves_are_heuristic": True},
            "players": list(metadata.values()), "steps": steps,
            "discounted_net_points": round(sum(discount ** t * s["net_predicted_points"] for t, s in enumerate(steps)), 3),
            "warnings": ["Re-plan after every gameweek and before executing transfers.",
                         "Chip timing depends on heuristic opportunity costs beyond the forecast horizon.",
                         "Expected points are model estimates, not a guaranteed score."]}
    if initial_excess:
        output["assumptions"]["grandfathered_initial_club_excess"] = initial_excess
        output["warnings"].append("Existing club overflow may be held; every permanent transfer and each Free Hit active squad must restore the club quota.")
    def utility(candidate):
        score = sum(discount ** t * (step["net_predicted_points"] +
                    0.001 * sum(xp[i, step["gw"]] for i in step["bench"]))
                    for t, step in enumerate(candidate["steps"]))
        score += ft_value * candidate["steps"][-1]["free_transfers_next"]
        for step in candidate["steps"]:
            if step["chip"] and any(c["name"] == step["chip"] and c["start_event"] <= step["gw"] <= c["stop_event"] and c["stop_event"] > gws[-1] for c in chips):
                score -= reserves[step["chip"]]
        return score
    if baseline:
        output["no_chip_baseline"] = {"discounted_net_points": baseline["discounted_net_points"],
                                      "decision_utility": round(utility(baseline), 4)}
        output["decision_utility"] = round(utility(output), 4)
        # Rounding in exported per-GW points can contribute <0.01 here.
        if utility(output) < utility(baseline) - 0.01:
            safe = copy.deepcopy(baseline)
            safe["candidate_pruning"] = copy.deepcopy(pruning)
            safe["chip_search"] = {"accepted": False, "solver": output["solver"],
                                    "reason": "Returned plan was worse than validated no-chip baseline"}
            safe["warnings"].append("Joint chip plan rejected: worse than the available no-chip baseline.")
            return safe
    return output


def refine_plan_lineups(output, forecasts):
    """Improve only submitted lineups after the transfer MILP has been selected.

    Transfer/chip decisions and their solver bound are unchanged. This is a
    local refinement, not a stochastic globally optimal transfer solution.
    """
    from lineup_uncertainty import refine_lineup
    if "p_no_appearance" not in forecasts:
        return output
    result=copy.deepcopy(output)
    result["milp_original_discounted_net_points"]=output["discounted_net_points"]
    steps=[]
    for original in output["steps"]:
        block=forecasts[forecasts.gw.eq(original["gw"])]
        step=refine_lineup(original,block)
        xp=block.set_index("element").pred.to_dict()
        mean=sum(xp[i] for i in (step["squad"] if step["chip"]=="bboost" else step["starting_xi"]))
        mean+=xp[step["captain"]]*(2 if step["chip"]=="3xc" else 1)
        step["milp_original_predicted_points"]=original["predicted_points"]
        step["predicted_points"]=round(mean,3)
        step["net_predicted_points"]=round(mean-step["hit_cost"],3)
        step["lineup_refinement"]["legacy_predicted_points_are_original_planner_values"]=False
        step["lineup_refinement"]["original_planner_mean_field"]="milp_original_predicted_points"
        steps.append(step)
    result["steps"]=steps
    discount=result["assumptions"]["discount"]
    result["discounted_net_points"]=round(sum(discount**t*s["net_predicted_points"] for t,s in enumerate(steps)),3)
    result["discounted_points_with_contingencies"]=round(sum(discount**t*s["expected_net_points_with_contingencies"] for t,s in enumerate(steps)),3)
    result["assumptions"].update({"automatic_substitutions_modelled":True,
        "vice_captain_contingency_modelled":True,
        "appearance_independence_assumed":True,
        "contingencies_in_transfer_milp_objective":False,
        "lineup_refinement":"fixed squad, original XI plus legal single swaps, exact independent appearance expectation"})
    result["warnings"].append("Lineup contingencies are refined after transfer/chip selection; solver bounds apply to the original deterministic MILP objective.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True)
    parser.add_argument("--forecasts", required=True)
    parser.add_argument("--output", default="state/transfer_plan.json")
    parser.add_argument("--no-chips", action="store_true")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--ft-value", type=float, default=1.5)
    parser.add_argument("--chip-reserves", help="JSON object with opportunity costs in expected points")
    parser.add_argument("--no-lineup-refinement",action="store_true")
    args = parser.parse_args()
    state = json.loads(Path(args.state).read_text())
    forecasts=pd.read_csv(args.forecasts)
    output = plan(state, forecasts, allow_chips=not args.no_chips,
                  chip_reserves=json.loads(args.chip_reserves) if args.chip_reserves else None,
                  discount=args.discount, ft_value=args.ft_value, seconds=args.seconds)
    if not args.no_lineup_refinement:
        output=refine_plan_lineups(output,forecasts)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(destination)
    names = {p["id"]: p["name"] for p in output["players"]}
    for step in output["steps"]:
        print(f"GW{step['gw']}: chip={step['chip']}, FT={step['free_transfers_before']}, "
              f"hit={step['hit_cost']}, bank={step['bank']/10:.1f}m, xP={step['net_predicted_points']:.1f}")
        print("  OUT:", ", ".join(names[i] for i in step['transfer_out']))
        print("  IN: ", ", ".join(names[i] for i in step['transfer_in']))
        print("  Captain:", names[step['captain']])
    print("Saved", destination)


if __name__ == "__main__":
    main()
