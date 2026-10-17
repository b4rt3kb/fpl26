"""Fixture-level availability from a frozen official FPL snapshot.

The API's ``news`` text is evidence for a reported date, not a guarantee of
fitness or selection.  Return probabilities below are deliberately explicit,
UNCALIBRATED planning scenarios, not medical estimates.  ``a`` means no external
availability penalty; the learned appearance model still decides playing time.

English ``Expected back D Mon [YYYY]`` and ``Suspended until D Mon [YYYY]``
are the only recognised date phrases.  We never infer a ban length from cards
or treat a date elsewhere in arbitrary news as a return date.  A reported date
is interpreted as the first potentially available UK calendar day, including
that day.  This interpretation can require a manual override after club news.

Use apply_fixture_availability BEFORE summing fixtures into gameweeks.  When
appearance probabilities exist, we cap learned P(appearance) at the external
availability scenario instead of multiplying two potentially correlated
probabilities.  Conditional minute-band proportions remain unchanged.  This
fusion is a conservative modelling assumption, not a calibrated joint model.

Overrides are a list of JSON objects, for example::

    [{"element": 371, "fixture": 73, "probability": 0.8,
      "probability_low": 0.6, "probability_high": 0.9,
      "reason": "Reviewed club update", "source_url": "https://..."}]

Optional selectors are fixture, gw, from_date and until_date (inclusive ISO
dates); every supplied selector must match.  Fixture overrides take precedence
over GW, then date range, then player-wide overrides.  Equal-priority overlaps
are rejected.  Overrides change availability, never the learned playing role,
registration/can_select eligibility, or prices.
"""
from __future__ import annotations

from datetime import date
import math
import re
from collections.abc import Mapping

import numpy as np
import pandas as pd


SOURCE_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
POLICIES = {
    "low": {"injury_return": (0.25, 0.50, 0.75), "suspension_return": 0.90},
    "base": {"injury_return": (0.50, 0.75, 0.90), "suspension_return": 0.95},
    "high": {"injury_return": (0.75, 0.90, 0.95), "suspension_return": 1.00},
}
POLICY_NOTES = (
    "Uncalibrated sensitivity scenarios: injury return in days 0-6 / 7-13 / 14+ "
    "uses low .25/.50/.75, base .50/.75/.90, high .75/.90/.95; suspension "
    "return low .90, base .95, high 1.00. Undated future doubt carries the "
    "official chance with low/high +/- .25; unknown injury/ban return remains "
    "censored at zero in every scenario, requiring refreshed news or an override."
)
_MONTHS = {name: i for i, name in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}
_RETURN_RE = re.compile(
    r"\b(?P<kind>Expected\s+back|Suspended\s+until)\s+"
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?P<month>[A-Za-z]+)\.?"
    r"(?:\s*,?\s*(?P<year>\d{4}))?\b", re.IGNORECASE)
_BANDS = ("p_no_appearance", "p_1_59", "p_60_plus")
_SCALED = ("pred", "expected_minutes", "p_start")


def _missing(value):
    return value is None or (not isinstance(value, (dict, list, tuple)) and bool(pd.isna(value)))


def _timestamp(value, label):
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label}: {value!r}") from exc
    if pd.isna(result):
        raise ValueError(f"Missing {label}")
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def _probability(value, label):
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be a finite probability in [0, 1]") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{label} must be a finite probability in [0, 1]")
    return number


def _positive_int(value, label):
    try:
        number = int(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if isinstance(value, bool) or number <= 0 or float(value) != number:
        raise ValueError(f"{label} must be a positive integer")
    return number


def parse_return_news(news, *, as_of, news_added=None):
    """Return parsed date/kind and provenance; unfamiliar/invalid text is unknown.

    Explicit years win.  Otherwise news_added anchors the year; a July-December
    update mentioning January-June rolls into the next year.  Other past dates
    remain past (stale news), rather than being silently moved a year forward.
    Without news_added, the current July-June football season anchors the year.
    Both inferred-year rules are stated in the result and can be overridden.
    """
    as_of = _timestamp(as_of, "as_of")
    news = "" if _missing(news) else str(news)
    result = {"date": None, "kind": None, "date_assumption": "no_recognised_return_date",
              "warning": ""}
    matches = list(_RETURN_RE.finditer(news))
    if not matches:
        return result
    if len(matches) != 1:
        result["warning"] = "multiple_return_dates_require_override"
        return result
    match = matches[0]
    month_text = match["month"].lower()
    month = _MONTHS.get(month_text[:3])
    valid_months = ("january", "february", "march", "april", "may", "june", "july",
                    "august", "september", "october", "november", "december")
    if month is None or (month_text not in _MONTHS and month_text not in valid_months and month_text != "sept"):
        result["warning"] = "invalid_return_month"
        return result
    if match["year"]:
        year = int(match["year"])
        assumption = "year_explicit_in_official_news"
    elif not _missing(news_added):
        anchor = _timestamp(news_added, "news_added")
        if anchor > as_of:
            raise ValueError("news_added is after as_of; cannot use future availability news")
        year = anchor.year + int(anchor.month >= 7 and month <= 6)
        assumption = "year_inferred_from_news_added_with_Jul_Jun_rollover"
    else:
        year = as_of.year - int(as_of.month < 7) + int(month < 7)
        assumption = "year_inferred_from_as_of_Jul_Jun_season_news_timestamp_missing"
    try:
        return_date = date(year, month, int(match["day"]))
    except ValueError:
        result["warning"] = "invalid_return_date"
        return result
    result.update(date=return_date, kind="suspension" if match["kind"].lower().startswith("suspended") else "injury",
                  date_assumption=assumption)
    if return_date < as_of.tz_convert("Europe/London").date():
        result["warning"] = "reported_return_date_has_passed_status_still_requires_refresh"
    return result


def validate_overrides(overrides):
    """Validate all override entries, including those outside the forecast horizon."""
    if overrides is None:
        return []
    if not isinstance(overrides, (list, tuple)):
        raise ValueError("availability overrides must be a list of objects")
    allowed = {"element", "fixture", "gw", "from_date", "until_date", "probability",
               "probability_low", "probability_high", "reason", "source_url"}
    result = []
    for original in overrides:
        if not isinstance(original, Mapping) or set(original) - allowed:
            raise ValueError("Invalid or unknown availability override fields")
        item = dict(original)
        for key in ("element", "probability", "reason"):
            if key not in item:
                raise ValueError(f"Availability override requires {key}")
        for key in ("element", "fixture", "gw"):
            if key in item:
                item[key] = _positive_int(item[key], f"override {key}")
        item["probability"] = _probability(item["probability"], "override probability")
        for key in ("probability_low", "probability_high"):
            item[key] = _probability(item.get(key, item["probability"]), f"override {key}")
        if not item["probability_low"] <= item["probability"] <= item["probability_high"]:
            raise ValueError("Override probabilities must satisfy low <= probability <= high")
        if not isinstance(item["reason"], str) or not item["reason"].strip():
            raise ValueError("Override reason must be nonempty")
        for key in ("from_date", "until_date"):
            if key in item:
                if not isinstance(item[key], str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", item[key]):
                    raise ValueError(f"Override {key} must be an ISO calendar date")
                item[key] = date.fromisoformat(item[key])
        if "from_date" in item and "until_date" in item and item["from_date"] > item["until_date"]:
            raise ValueError("Override from_date is after until_date")
        result.append(item)
    return result


def _matching_override(overrides, element, fixture, gw, kickoff_date):
    matches = []
    for item in overrides:
        if item["element"] != element:
            continue
        if "fixture" in item and item["fixture"] != fixture:
            continue
        if "gw" in item and item["gw"] != gw:
            continue
        if "from_date" in item and kickoff_date < item["from_date"]:
            continue
        if "until_date" in item and kickoff_date > item["until_date"]:
            continue
        rank = 3 if "fixture" in item else 2 if "gw" in item else 1 if ("from_date" in item or "until_date" in item) else 0
        matches.append((rank, item))
    if not matches:
        return None
    rank = max(x[0] for x in matches)
    best = [item for priority, item in matches if priority == rank]
    if len(best) != 1:
        raise ValueError(f"Conflicting availability overrides for element {element}, fixture {fixture}")
    return best[0]


def _availability(player, *, kickoff_time, gw, fixture, as_of, next_event, overrides, policy):
    kickoff = _timestamp(kickoff_time, "fixture kickoff_time")
    as_of = _timestamp(as_of, "as_of")
    if kickoff <= as_of:
        raise ValueError("Availability forecasts require a kickoff after as_of")
    kickoff_date = kickoff.tz_convert("Europe/London").date()
    element = _positive_int(player.get("id", player.get("element")), "player id")
    news = "" if _missing(player.get("news")) else str(player["news"])
    news_added = player.get("news_added")
    if not _missing(news_added) and _timestamp(news_added, "news_added") > as_of:
        raise ValueError("news_added is after as_of; cannot use future availability news")
    parsed = parse_return_news(news, as_of=as_of, news_added=news_added)
    override = _matching_override(overrides, element, fixture, gw, kickoff_date)
    warning = parsed["warning"]
    source_url = SOURCE_URL
    if override is not None:
        probabilities = {"low": override["probability_low"], "base": override["probability"],
                         "high": override["probability_high"]}
        source = "manual_override"
        source_url = override.get("source_url", "")
        assumption = "manual_availability_scenario: " + override["reason"]
    else:
        status = player.get("status")
        chance = player.get("chance_of_playing_next_round")
        chance = None if _missing(chance) else _probability(float(chance) / 100, "official next-round chance")
        if status == "a":
            probabilities = dict.fromkeys(POLICIES, 1.0)
            source, assumption = "official_bootstrap.status", "active_status_no_external_penalty_not_guaranteed_appearance"
        elif status in ("u", "n"):
            probabilities = dict.fromkeys(POLICIES, 0.0)
            source, assumption = "official_bootstrap.status", "unavailable_registration_status_held_fixed"
        elif status in ("i", "s") and parsed["date"] is not None:
            elapsed = (kickoff_date - parsed["date"]).days
            if elapsed < 0:
                probabilities = dict.fromkeys(POLICIES, 0.0)
                assumption = "before_reported_return_date_conservative_zero"
            else:
                band = 0 if elapsed < 7 else 1 if elapsed < 14 else 2
                probabilities = {name: values["suspension_return"] if status == "s" else values["injury_return"][band]
                                 for name, values in POLICIES.items()}
                assumption = "reported_date_is_first_potential_return_day_uncalibrated_" + ("suspension" if status == "s" else "injury") + "_scenario"
                if gw == next_event and chance is not None:
                    warning = ";".join(filter(None, [warning, "fixture_return_scenario_overrides_coarse_official_next_round_chance"]))
            source = "official_bootstrap.news+uncalibrated_return_scenario"
        elif status == "d" or (status in ("i", "s") and chance is not None and chance > 0):
            base = chance if chance is not None else 0.5
            if gw == next_event:
                probabilities = dict.fromkeys(POLICIES, base)
                assumption = "official_next_round_chance_used_as_external_cap" if chance is not None else "doubtful_missing_chance_uncalibrated_half"
            else:
                probabilities = {"low": max(0, base - 0.25), "base": base, "high": min(1, base + 0.25)}
                assumption = "undated_doubt_official_chance_carried_forward_with_uncalibrated_range"
            source = "official_bootstrap.chance_of_playing_next_round" if chance is not None else "uncalibrated_missing_chance_fallback"
        elif status in ("i", "s"):
            probabilities = dict.fromkeys(POLICIES, 0.0)
            source = "official_bootstrap.status+unknown_return_censoring"
            assumption = "unknown_return_conservatively_censored_zero_not_known_absence_for_entire_horizon"
            warning = ";".join(filter(None, [warning, "unknown_return_refresh_news_or_supply_manual_override"]))
        else:
            raise ValueError(f"Unknown official availability status {status!r} for element {element}")
    return {
        "availability_probability": probabilities[policy],
        "availability_probability_low": probabilities["low"],
        "availability_probability_base": probabilities["base"],
        "availability_probability_high": probabilities["high"],
        "availability_factor": probabilities[policy],
        "availability_policy": policy,
        "availability_source": source,
        "availability_source_url": source_url,
        "availability_news": news,
        "availability_news_added": None if _missing(news_added) else str(news_added),
        "availability_return_date": parsed["date"].isoformat() if parsed["date"] is not None else None,
        "availability_date_assumption": parsed["date_assumption"],
        "availability_assumption": assumption,
        "availability_warning": warning,
    }


def availability_for_fixture(player, *, kickoff_time, gw, as_of, next_event,
                             fixture=None, overrides=None, policy="base"):
    """Return the selected probability, low/base/high scenarios and provenance."""
    if policy not in POLICIES:
        raise ValueError("availability policy must be low, base or high")
    return _availability(player, kickoff_time=kickoff_time, gw=_positive_int(gw, "gw"),
                         fixture=None if fixture is None else _positive_int(fixture, "fixture"),
                         as_of=as_of, next_event=_positive_int(next_event, "next_event"),
                         overrides=validate_overrides(overrides), policy=policy)


def apply_fixture_availability(forecast_rows, players, *, as_of, next_event,
                               overrides=None, policy="base"):
    """Copy/adjust fixture forecasts, retaining raw_* columns for reproducibility.

    ``availability_factor`` is the external cap; ``availability_multiplier`` is
    the actual adjustment, min(1, cap / raw Pappearance).  Without learned
    minute bands this falls back to raw multiplication and labels that method.
    Reapplying this function reuses raw_* values rather than compounding caps.
    """
    if policy not in POLICIES:
        raise ValueError("availability policy must be low, base or high")
    as_of = _timestamp(as_of, "as_of")
    next_event = _positive_int(next_event, "next_event")
    overrides = validate_overrides(overrides)
    rows = forecast_rows.copy()
    required = {"element", "fixture", "kickoff_time", "pred"}
    if not required <= set(rows) or not ({"gw", "round"} & set(rows)):
        raise ValueError("Availability requires element, fixture, kickoff_time, pred and gw (or round)")
    if "id" not in players or "status" not in players or players.id.duplicated().any():
        raise ValueError("Availability requires unique player ids and official statuses")
    lookup = players.set_index("id", drop=False).to_dict("index")
    unknown = set(rows.element) - set(lookup)
    unknown_overrides = {item["element"] for item in overrides} - set(lookup)
    if unknown or unknown_overrides:
        raise ValueError(f"Unknown player ids in forecasts/overrides: {sorted(unknown | unknown_overrides)}")
    information = []
    for row in rows.to_dict("records"):
        information.append(_availability(
            lookup[row["element"]], kickoff_time=row["kickoff_time"],
            gw=_positive_int(row.get("gw", row.get("round")), "gw"),
            fixture=_positive_int(row["fixture"], "fixture"), as_of=as_of,
            next_event=next_event, overrides=overrides, policy=policy))
    if not information:
        return rows
    for column in information[0]:
        rows[column] = [item[column] for item in information]
    for column in _BANDS + _SCALED:
        if column in rows and "raw_" + column not in rows:
            rows["raw_" + column] = rows[column].copy()
    present_bands = set(_BANDS) & set(rows)
    if present_bands and present_bands != set(_BANDS):
        raise ValueError("Availability fusion requires all three appearance probability bands")
    if present_bands:
        bands = rows[["raw_" + column for column in _BANDS]].to_numpy(dtype=float)
        if (not np.isfinite(bands).all() or (bands < 0).any() or (bands > 1).any()
                or not np.allclose(bands.sum(axis=1), 1, atol=1e-6, rtol=0)):
            raise ValueError("Appearance probability bands must be finite, in [0, 1], and sum to one")
        play = 1 - bands[:, 0]
        multiplier = np.divide(rows.availability_probability.to_numpy(), play,
                               out=np.zeros_like(play), where=play > 0)
        multiplier = np.minimum(1, multiplier)
        rows["p_1_59"] = bands[:, 1] * multiplier
        rows["p_60_plus"] = bands[:, 2] * multiplier
        rows["p_no_appearance"] = 1 - rows.p_1_59 - rows.p_60_plus
        rows["availability_fusion"] = "cap_learned_appearance_preserve_conditional_bands_uncalibrated"
    else:
        multiplier = rows.availability_probability.to_numpy()
        rows["availability_fusion"] = "legacy_raw_multiplication_no_learned_appearance"
    for column in _SCALED:
        if column in rows:
            values = rows["raw_" + column].to_numpy(dtype=float)
            known = np.isfinite(values)
            if column == "p_start":
                if np.isinf(values).any() or ((values[known] < 0) | (values[known] > 1)).any():
                    raise ValueError("p_start must be a probability or NaN when unsupported")
                if present_bands and (values[known] > play[known] + 1e-6).any():
                    raise ValueError("p_start cannot exceed learned appearance probability")
            elif not known.all():
                raise ValueError(f"{column} must be finite")
            if column == "expected_minutes" and ((values < 0) | (values > 90)).any():
                raise ValueError("expected_minutes must be between zero and 90 per fixture")
            rows[column] = values * multiplier
    rows["availability_multiplier"] = multiplier
    return rows
