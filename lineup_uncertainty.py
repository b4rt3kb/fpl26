"""Exact expected autosubs and captain fallback for a fixed 15-player squad.

All ``pred`` values are UNCONDITIONAL expected GW points.  The approximation
assumes independent Bernoulli appearances across players, and zero points when
a player does not appear.  It does not model correlated rotation/illness or the
rare zero-minute card that blocks an FPL autosub.  For fixture-level input, GW
P(no appearance) is the product of fixture probabilities: an additional, explicit
within-player independence assumption, not a guarantee for a double gameweek.

The computation is exact under these assumptions: Poisson-binomial distributions
of absent starters by position, all eight outfield-bench appearance states, and
formation-valid substitutions in lexicographic bench priority.  No Monte Carlo
or actual outcomes are used.  Refinement optimises bench order/C/VC exactly for
each tested XI; default XI search is only the original plus single legal swaps.
The transfer/chip MILP objective is unchanged and remains a separate objective.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import lru_cache
from itertools import permutations, product
import math

import numpy as np
import pandas as pd


_POSITIONS = ("DEF", "MID", "FWD")
_MINIMUM = (3, 2, 1)
_MAXIMUM = (5, 5, 3)
_QUOTAS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
ASSUMPTIONS = [
    "Independent Bernoulli appearances across players; correlated absence/rotation is not modelled.",
    "Predictions are unconditional GW points; points conditional on any appearance are pred/(1-p_no_appearance).",
    "No-appearance means zero points; rare zero-minute cards affecting autosubs are not modelled.",
    "Exact autosub and captain-fallback expectation under these assumptions; no outcome data or Monte Carlo.",
    "Post-selection lineup refinement only; the transfer/chip optimisation objective is not a stochastic objective.",
]


def _ids(values, label):
    result = []
    for value in values:
        try:
            number = int(value)
            valid = not isinstance(value, bool) and number > 0 and float(value) == number
        except (ValueError, TypeError, OverflowError):
            valid = False
        if not valid:
            raise ValueError(f"{label} must contain positive integer player IDs")
        result.append(number)
    if len(set(result)) != len(result):
        raise ValueError(f"Duplicate player in {label}")
    return result


def _forecast_metadata(frame, gw):
    required = {"element", "pos", "pred", "p_no_appearance"}
    if not required <= set(frame):
        raise ValueError("Lineup uncertainty requires " + ", ".join(sorted(required)))
    frame = frame.copy()
    if "gw" in frame:
        frame = frame[frame.gw.eq(gw)].copy()
    if frame.empty:
        raise ValueError("No forecasts for requested gameweek")
    values = frame[["pred", "p_no_appearance"]].to_numpy(float)
    if not np.isfinite(values).all() or ((values[:, 1] < 0) | (values[:, 1] > 1)).any():
        raise ValueError("Predictions and no-appearance probabilities must be finite, probabilities in [0, 1]")
    duplicate = frame.element.duplicated().any()
    if duplicate and ("fixture" not in frame or frame.fixture.isna().any()
                      or frame.duplicated(["element", "fixture"]).any()):
        raise ValueError("Multiple forecasts for one player require distinct fixture IDs")
    metadata = {}
    for pid, group in frame.groupby("element", sort=False):
        pid = _ids([pid], "forecast")[0]
        positions = group.pos.replace({"GKP": "GK"}).unique()
        if len(positions) != 1 or positions[0] not in _QUOTAS:
            raise ValueError("Invalid or inconsistent forecast positions")
        pred = float(group.pred.sum())
        p_no = float(group.p_no_appearance.prod())
        if p_no == 1 and abs(pred) > 1e-9:
            raise ValueError("A player certain not to appear must have zero expected points")
        metadata[pid] = {"pos": positions[0], "pred": pred, "p_no": p_no}
    return metadata, duplicate


def _legal_xi(xi, metadata):
    counts = Counter(metadata[i]["pos"] for i in xi)
    return (len(xi) == 11 and counts["GK"] == 1
            and all(low <= counts[pos] <= high
                    for pos, low, high in zip(_POSITIONS, _MINIMUM, _MAXIMUM)))


def _validate_decision(step, metadata):
    squad = _ids(step["squad"], "squad")
    xi = _ids(step["starting_xi"], "starting XI")
    bench = _ids(step["bench"], "bench")
    if len(squad) != 15 or not set(squad) <= set(metadata):
        raise ValueError("Squad requires 15 distinct known players")
    if Counter(metadata[i]["pos"] for i in squad) != _QUOTAS:
        raise ValueError("Squad must contain 2 GK, 5 DEF, 5 MID and 3 FWD")
    if not set(xi) <= set(squad) or not _legal_xi(xi, metadata):
        raise ValueError("Starting XI has an invalid formation")
    if len(bench) != 4 or set(bench) != set(squad) - set(xi):
        raise ValueError("Bench must contain exactly the four non-starters")
    c, v = _ids([step["captain"], step["vice_captain"]], "captain and vice")
    if c not in xi or v not in xi:
        raise ValueError("Captain and vice must be starters")
    if step.get("chip") not in (None, "wildcard", "freehit", "bboost", "3xc"):
        raise ValueError("Unknown chip")
    hit = float(step.get("hit_cost", 0))
    if not math.isfinite(hit) or hit < 0:
        raise ValueError("Hit cost must be finite and nonnegative")
    return squad, xi, bench, c, v


def _absent_distribution(probabilities):
    distribution = np.array([1.0])
    for probability in probabilities:
        distribution = np.convolve(distribution, [1 - probability, probability])
    return distribution


@lru_cache(maxsize=100000)
def _selection_flags(nominal, absent, bench_positions, played):
    """Chosen incoming slots; assignment depends on counts, never realised XP."""
    if not bench_positions:
        return ()
    incoming, rest = bench_positions[0], bench_positions[1:]
    candidates = []
    if played[0]:
        for outgoing in range(3):
            if not absent[outgoing]:
                continue
            proposed = list(nominal)
            proposed[incoming] += 1
            proposed[outgoing] -= 1
            if not all(low <= count <= high for count, low, high in zip(proposed, _MINIMUM, _MAXIMUM)):
                continue
            remaining = list(absent)
            remaining[outgoing] -= 1
            candidates.append((1,) + _selection_flags(tuple(proposed), tuple(remaining), rest, played[1:]))
    if candidates:
        return max(candidates)
    return (0,) + _selection_flags(nominal, absent, rest, played[1:])


@lru_cache(maxsize=128)
def _selection_table(nominal, bench_positions):
    absent_states = tuple(product(*(range(n + 1) for n in nominal)))
    bench_states = tuple(product((0, 1), repeat=3))
    table = np.array([[_selection_flags(nominal, absent, bench_positions, played)
                       for played in bench_states] for absent in absent_states], dtype=np.int8)
    return absent_states, bench_states, table


def _outfield_bench_value(xi, order, metadata):
    nominal = tuple(sum(metadata[i]["pos"] == pos for i in xi) for pos in _POSITIONS)
    bench_positions = tuple(_POSITIONS.index(metadata[i]["pos"]) for i in order)
    absent_states, bench_states, selection = _selection_table(nominal, bench_positions)
    distributions = [_absent_distribution([metadata[i]["p_no"] for i in xi if metadata[i]["pos"] == pos])
                     for pos in _POSITIONS]
    absent_weights = np.array([distributions[0][d] * distributions[1][m] * distributions[2][f]
                               for d, m, f in absent_states])
    play = np.array([1 - metadata[i]["p_no"] for i in order])
    states = np.asarray(bench_states)
    bench_weights = np.prod(np.where(states, play, 1 - play), axis=1)
    selection_probabilities = np.einsum("a,b,abk->k", absent_weights, bench_weights, selection)
    conditional_points = np.divide([metadata[i]["pred"] for i in order], play,
                                   out=np.zeros(3), where=play > 0)
    return float(selection_probabilities @ conditional_points)


def _components(squad, xi, bench, captain, vice, chip, metadata):
    starter_points = sum(metadata[i]["pred"] for i in xi)
    gk_sub, outfield_sub, boost_points = 0.0, 0.0, 0.0
    if chip == "bboost":
        boost_points = sum(metadata[i]["pred"] for i in bench)
    else:
        starter_gk = next(i for i in xi if metadata[i]["pos"] == "GK")
        bench_gk = next(i for i in bench if metadata[i]["pos"] == "GK")
        gk_sub = metadata[starter_gk]["p_no"] * metadata[bench_gk]["pred"]
        order = tuple(i for i in bench if metadata[i]["pos"] != "GK")
        outfield_sub = _outfield_bench_value(xi, order, metadata)
    extra_multiplier = 2 if chip == "3xc" else 1
    primary = extra_multiplier * metadata[captain]["pred"]
    fallback = extra_multiplier * metadata[captain]["p_no"] * metadata[vice]["pred"]
    return {"starter_points": float(starter_points), "bench_boost_points": float(boost_points),
            "goalkeeper_autosub_points": float(gk_sub), "outfield_autosub_points": float(outfield_sub),
            "captain_primary_bonus": float(primary), "vice_captain_fallback_bonus": float(fallback),
            "captain_extra_multiplier": extra_multiplier,
            "total": float(starter_points + boost_points + gk_sub + outfield_sub + primary + fallback)}


def expected_lineup_points(step, forecast_for_gw):
    """Score a fixed decision without improving it, using forecasts only."""
    metadata, _ = _forecast_metadata(forecast_for_gw, step["gw"])
    squad, xi, bench, captain, vice = _validate_decision(step, metadata)
    return _components(squad, xi, bench, captain, vice, step.get("chip"), metadata)


def refine_lineup(step, forecast_for_gw, *, improve_xi=True):
    """Return a copy with refined XI, bench order, captain/vice and expected XP.

    Original XI is always included; improve_xi=True additionally tests each
    legal one-player swap with its bench.  This is a local search, not a global
    XI or transfer optimum.  Existing predicted_points/net_predicted_points
    remain the original planner values; new contingency totals are separate.
    """
    metadata, fixture_aggregation = _forecast_metadata(forecast_for_gw, step["gw"])
    squad, original_xi, original_bench, original_c, original_v = _validate_decision(step, metadata)
    chip = step.get("chip")
    original = _components(squad, original_xi, original_bench, original_c, original_v, chip, metadata)
    candidates = [tuple(original_xi)]
    if improve_xi:
        seen = {frozenset(original_xi)}
        for outgoing in original_xi:
            for incoming in original_bench:
                proposed = tuple(incoming if i == outgoing else i for i in original_xi)
                key = frozenset(proposed)
                if key not in seen and _legal_xi(proposed, metadata):
                    candidates.append(proposed)
                    seen.add(key)
    best = (original["total"], original_xi, original_bench, original_c, original_v, original)
    for xi in candidates:
        captain, vice = max(permutations(xi, 2), key=lambda pair: (
            metadata[pair[0]]["pred"] + metadata[pair[0]]["p_no"] * metadata[pair[1]]["pred"],
            metadata[pair[0]]["pred"], -metadata[pair[0]]["p_no"], -pair[0], -pair[1]))
        reserves = [i for i in squad if i not in xi]
        bench_gk = next(i for i in reserves if metadata[i]["pos"] == "GK")
        outfield = [i for i in reserves if metadata[i]["pos"] != "GK"]
        # BB has no autosubs; ordering cannot alter its expectation.
        orders = [tuple(outfield)] if chip == "bboost" else permutations(outfield)
        for order in orders:
            bench = [bench_gk] + list(order)
            parts = _components(squad, xi, bench, captain, vice, chip, metadata)
            if parts["total"] > best[0] + 1e-10:
                best = (parts["total"], list(xi), bench, captain, vice, parts)
    value, xi, bench, captain, vice, parts = best
    output = deepcopy(step)
    output.update(starting_xi=list(xi), bench=list(bench), captain=int(captain), vice_captain=int(vice),
                  expected_points_with_contingencies=value,
                  expected_net_points_with_contingencies=value - float(step.get("hit_cost", 0)),
                  expected_points_components=parts)
    assumptions = list(ASSUMPTIONS)
    if fixture_aggregation:
        assumptions.append("DGW inputs aggregated: GW no-appearance probability is product of fixture no-appearance probabilities; fixture appearances assumed independent.")
    output["lineup_refinement"] = {
        "method": "exact_independent_Bernoulli_autosubs_and_captain_fallback",
        "xi_search_scope": "original_and_single_legal_swaps" if improve_xi else "fixed_original_XI",
        "candidate_xi_count": len(candidates), "transfer_objective_unchanged": True,
        "legacy_predicted_points_are_original_planner_values": True,
        "original_expected_points_with_contingencies": original["total"],
        "expected_improvement": value - original["total"],
        "original_decision": {"starting_xi": original_xi, "bench": original_bench,
                              "captain": original_c, "vice_captain": original_v},
        "assumptions": assumptions,
    }
    return output
