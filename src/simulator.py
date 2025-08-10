"""Utilities for parsing match fixtures and running Monte Carlo simulations.

This module reads fixtures exported from SportsClubStats and provides
functions to compute league tables and run a SportsClubStats-style model to
project results.  It powers the public functions exposed in
``__init__``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict

from joblib import Parallel, delayed

try:  # Optional dependency for progress bars
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - tqdm is optional in tests
    tqdm = None

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default simulation parameters
# ---------------------------------------------------------------------------

# Relative advantage multiplier for the home team
# No built-in home advantage is used by default.
DEFAULT_HOME_FIELD_ADVANTAGE = 1.0

# Default number of parallel jobs. Use all available cores.
DEFAULT_JOBS = os.cpu_count() or 1

# Number of simulations to execute per parallel batch when running with
# ``n_jobs`` greater than one. Smaller batches keep memory usage low while
# ensuring deterministic ordering of results.
_BATCH_SIZE = 100



# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------
SCORE_PATTERN = re.compile(
    r"(\d+/\d+/\d+)\s+(.+?)\s+(\d+)-(\d+)\s+(.+?)\s*(?:\(ID:.*)?$"
)
NOSCORE_PATTERN = re.compile(
    r"(\d+/\d+/\d+)\s+(.+?)\s{2,}(.+?)\s*(?:\(ID:.*)?$"
)


def _parse_date(date_str: str) -> pd.Timestamp:
    parts = date_str.split("/")
    year = parts[-1]
    if len(year) == 4:
        return pd.to_datetime(date_str, format="%d/%m/%Y")
    return pd.to_datetime(date_str, format="%m/%d/%y")


def parse_matches(path: str | Path) -> pd.DataFrame:
    """Return a DataFrame of fixtures and results."""
    rows: list[dict] = []
    in_games = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "GamesBegin":
                in_games = True
                continue
            if line.strip() == "GamesEnd":
                break
            if not in_games:
                continue
            line = line.rstrip("\n")
            m = SCORE_PATTERN.match(line)
            if m:
                date_str, home, hs, as_, away = m.groups()
                rows.append(
                    {
                        "date": _parse_date(date_str),
                        "home_team": home.strip(),
                        "away_team": away.strip(),
                        "home_score": int(hs),
                        "away_score": int(as_),
                    }
                )
                continue
            m = NOSCORE_PATTERN.match(line)
            if m:
                date_str, home, away = m.groups()
                rows.append(
                    {
                        "date": _parse_date(date_str),
                        "home_team": home.strip(),
                        "away_team": away.strip(),
                        "home_score": np.nan,
                        "away_score": np.nan,
                    }
                )
    return pd.DataFrame(rows)


def reset_results_from(matches: pd.DataFrame, start_date: str | pd.Timestamp) -> pd.DataFrame:
    """Return a copy of ``matches`` with results on or after ``start_date`` cleared."""

    df = matches.copy()
    start = pd.to_datetime(start_date)
    mask = df["date"] >= start
    df.loc[mask, ["home_score", "away_score"]] = np.nan
    return df


# ---------------------------------------------------------------------------
# Table computation
# ---------------------------------------------------------------------------


def _head_to_head_points(
    matches: pd.DataFrame,
    teams: list[str],
) -> Dict[str, int]:
    points = {t: 0 for t in teams}
    df = matches.dropna(subset=["home_score", "away_score"])
    df = df[df["home_team"].isin(teams) & df["away_team"].isin(teams)]
    for _, row in df.iterrows():
        ht = row["home_team"]
        at = row["away_team"]
        hs = int(row["home_score"])
        as_ = int(row["away_score"])
        if hs > as_:
            points[ht] += 3
        elif hs < as_:
            points[at] += 3
        else:
            points[ht] += 1
            points[at] += 1
    return points


def league_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute league standings from played matches."""
    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    table: Dict[str, Dict[str, float]] = {
        t: {
            "team": t,
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "gf": 0,
            "ga": 0,
        }
        for t in teams
    }

    played = matches.dropna(subset=["home_score", "away_score"])
    for _, row in played.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hs = int(row["home_score"])
        as_ = int(row["away_score"])
        table[home]["played"] += 1
        table[away]["played"] += 1
        table[home]["gf"] += hs
        table[home]["ga"] += as_
        table[away]["gf"] += as_
        table[away]["ga"] += hs
        if hs > as_:
            table[home]["wins"] += 1
            table[home]["points"] = table[home].get("points", 0) + 3
            table[away]["losses"] += 1
            table[away].setdefault("points", 0)
        elif hs < as_:
            table[away]["wins"] += 1
            table[away]["points"] = table[away].get("points", 0) + 3
            table[home]["losses"] += 1
            table[home].setdefault("points", 0)
        else:
            table[home]["draws"] += 1
            table[away]["draws"] += 1
            table[home]["points"] = table[home].get("points", 0) + 1
            table[away]["points"] = table[away].get("points", 0) + 1

    for t in table.values():
        t.setdefault("points", 0)
        t["gd"] = t["gf"] - t["ga"]

    df = pd.DataFrame(table.values())
    df["head_to_head"] = 0
    for _, group in df.groupby(["points", "wins", "gd", "gf"]):
        if len(group) <= 1:
            continue
        teams_tied = group["team"].tolist()
        h2h = _head_to_head_points(played, teams_tied)
        for t, val in h2h.items():
            df.loc[df["team"] == t, "head_to_head"] = val

    df = df.sort_values(
        ["points", "wins", "gd", "gf", "head_to_head", "team"],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _simulate_table(
    played_df: pd.DataFrame,
    remaining: pd.DataFrame,
    rng: np.random.Generator,
    *,
    home_advantage: float = DEFAULT_HOME_FIELD_ADVANTAGE,
    team_params: Dict[str, tuple[float, float]] | None = None,
    home_goals_mean: float = 1.0,
    away_goals_mean: float = 1.0,
 ) -> pd.DataFrame:
    """Simulate remaining fixtures using Poisson goal generation."""

    if home_advantage <= 0:
        raise ValueError("home_advantage must be greater than zero")

    if home_goals_mean <= 0:
        raise ValueError("home_goals_mean must be greater than zero")
    if away_goals_mean <= 0:
        raise ValueError("away_goals_mean must be greater than zero")

    sims: list[dict] = []

    for _, row in remaining.iterrows():
        ha = home_advantage
        if team_params is not None:
            ha *= team_params.get(row["home_team"], (1.0, 1.0))[0] / team_params.get(row["away_team"], (1.0, 1.0))[1]
            away_factor = team_params.get(row["away_team"], (1.0, 1.0))[0] / team_params.get(row["home_team"], (1.0, 1.0))[1]
        else:
            away_factor = 1.0

        hm = home_goals_mean * ha
        am = away_goals_mean * away_factor
        hs = int(rng.poisson(hm))
        as_ = int(rng.poisson(am))

        sims.append(
            {
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_score": hs,
                "away_score": as_,
            }
        )
    all_matches = pd.concat([played_df, pd.DataFrame(sims)], ignore_index=True)
    return league_table(all_matches)


def _iterate_tables(
    played_df: pd.DataFrame,
    remaining: pd.DataFrame,
    rng: np.random.Generator,
    iterations: int,
    *,
    desc: str,
    progress: bool,
    home_advantage: float,
    team_params: Dict[str, tuple[float, float]] | None,
    home_goals_mean: float,
    away_goals_mean: float,
    n_jobs: int,
):
    """Yield successive simulated tables.

    This helper mirrors the parallel and serial execution logic used by the
    public simulation functions to ensure deterministic ordering of results.
    """

    if n_jobs > 1:
        seeds = rng.integers(0, 2**32 - 1, size=iterations)
        pbar = None
        if progress and tqdm is not None:
            pbar = tqdm(seeds, desc=desc, unit="sim")

        def run(seed: int) -> pd.DataFrame:
            return _simulate_table(
                played_df,
                remaining,
                np.random.default_rng(seed),
                home_advantage=home_advantage,
                team_params=team_params,
                home_goals_mean=home_goals_mean,
                away_goals_mean=away_goals_mean,
            )

        for start in range(0, iterations, _BATCH_SIZE):
            batch = seeds[start : start + _BATCH_SIZE]
            results = Parallel(n_jobs=n_jobs)(delayed(run)(s) for s in batch)
            if pbar is not None and hasattr(pbar, "update"):
                pbar.update(len(batch))
            for table in results:
                yield table
        if pbar is not None and hasattr(pbar, "close"):
            pbar.close()
    else:
        iterator = range(iterations)
        if progress and tqdm is not None:
            iterator = tqdm(iterator, desc=desc, unit="sim")
        for _ in iterator:
            yield _simulate_table(
                played_df,
                remaining,
                rng,
                home_advantage=home_advantage,
                team_params=team_params,
                home_goals_mean=home_goals_mean,
                away_goals_mean=away_goals_mean,
            )


# ---------------------------------------------------------------------------
# Public simulation API
# ---------------------------------------------------------------------------


def simulate_chances(
    matches: pd.DataFrame,
    iterations: int = 1000,
    *,
    rng: np.random.Generator | None = None,
    progress: bool = True,
    home_advantage: float = DEFAULT_HOME_FIELD_ADVANTAGE,
    team_params: Dict[str, tuple[float, float]] | None = None,
    home_goals_mean: float = 1.0,
    away_goals_mean: float = 1.0,
    n_jobs: int = DEFAULT_JOBS,
) -> Dict[str, float]:
    """Return title probabilities.

    """

    if rng is None:
        rng = np.random.default_rng()


    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    champs = {t: 0 for t in teams}

    played_df = matches.dropna(subset=["home_score", "away_score"])
    remaining = matches[
        matches["home_score"].isna() | matches["away_score"].isna()
    ]

    for table in _iterate_tables(
        played_df,
        remaining,
        rng,
        iterations,
        desc="Chances",
        progress=progress,
        home_advantage=home_advantage,
        team_params=team_params,
        home_goals_mean=home_goals_mean,
        away_goals_mean=away_goals_mean,
        n_jobs=n_jobs,
    ):
        champs[table.iloc[0]["team"]] += 1

    for t in champs:
        champs[t] /= iterations
    return champs


def simulate_relegation_chances(
    matches: pd.DataFrame,
    iterations: int = 1000,
    *,
    rng: np.random.Generator | None = None,
    progress: bool = True,
    home_advantage: float = DEFAULT_HOME_FIELD_ADVANTAGE,
    team_params: Dict[str, tuple[float, float]] | None = None,
    home_goals_mean: float = 1.0,
    away_goals_mean: float = 1.0,
    n_jobs: int = DEFAULT_JOBS,
) -> Dict[str, float]:
    """Return probabilities of finishing in the bottom four."""

    if rng is None:
        rng = np.random.default_rng()


    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    relegated = {t: 0 for t in teams}

    played_df = matches.dropna(subset=["home_score", "away_score"])
    remaining = matches[
        matches["home_score"].isna() | matches["away_score"].isna()
    ]
    for table in _iterate_tables(
        played_df,
        remaining,
        rng,
        iterations,
        desc="Relegation",
        progress=progress,
        home_advantage=home_advantage,
        team_params=team_params,
        home_goals_mean=home_goals_mean,
        away_goals_mean=away_goals_mean,
        n_jobs=n_jobs,
    ):
        for team in table.tail(4)["team"]:
            relegated[team] += 1

    for t in relegated:
        relegated[t] /= iterations
    return relegated


def simulate_final_table(
    matches: pd.DataFrame,
    iterations: int = 1000,
    *,
    rng: np.random.Generator | None = None,
    progress: bool = True,
    home_advantage: float = DEFAULT_HOME_FIELD_ADVANTAGE,
    team_params: Dict[str, tuple[float, float]] | None = None,
    home_goals_mean: float = 1.0,
    away_goals_mean: float = 1.0,
    n_jobs: int = DEFAULT_JOBS,
) -> pd.DataFrame:
    """Project average finishing position and points."""

    if rng is None:
        rng = np.random.default_rng()


    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    pos_totals = {t: 0.0 for t in teams}
    points_totals = {t: 0.0 for t in teams}

    played_df = matches.dropna(subset=["home_score", "away_score"])
    remaining = matches[
        matches["home_score"].isna() | matches["away_score"].isna()
    ]

    for table in _iterate_tables(
        played_df,
        remaining,
        rng,
        iterations,
        desc="Final table",
        progress=progress,
        home_advantage=home_advantage,
        team_params=team_params,
        home_goals_mean=home_goals_mean,
        away_goals_mean=away_goals_mean,
        n_jobs=n_jobs,
    ):
        for idx, row in table.iterrows():
            pos_totals[row["team"]] += idx + 1
            points_totals[row["team"]] += row["points"]

    results = []
    for team in teams:
        results.append(
            {
                "team": team,
                "position": pos_totals[team] / iterations,
                "points": points_totals[team] / iterations,
            }
        )

    df = pd.DataFrame(results)
    df = df.sort_values("position").reset_index(drop=True)
    return df


def summary_table(
    matches: pd.DataFrame,
    iterations: int = 1000,
    *,
    rng: np.random.Generator | None = None,
    progress: bool = True,
    home_advantage: float = DEFAULT_HOME_FIELD_ADVANTAGE,
    team_params: Dict[str, tuple[float, float]] | None = None,
    home_goals_mean: float = 1.0,
    away_goals_mean: float = 1.0,
    n_jobs: int = DEFAULT_JOBS,
) -> pd.DataFrame:
    """Return a combined projection table ranked by expected points.

    The ``position`` column corresponds to the rank after sorting by the
    expected point totals.
    """

    if rng is None:
        rng = np.random.default_rng()


    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    title_counts = {t: 0 for t in teams}
    relegated = {t: 0 for t in teams}
    points_totals = {t: 0.0 for t in teams}
    wins_totals = {t: 0.0 for t in teams}
    gd_totals = {t: 0.0 for t in teams}

    played_df = matches.dropna(subset=["home_score", "away_score"])
    remaining = matches[
        matches["home_score"].isna() | matches["away_score"].isna()
    ]

    for table in _iterate_tables(
        played_df,
        remaining,
        rng,
        iterations,
        desc="Summary",
        progress=progress,
        home_advantage=home_advantage,
        team_params=team_params,
        home_goals_mean=home_goals_mean,
        away_goals_mean=away_goals_mean,
        n_jobs=n_jobs,
    ):
        title_counts[table.iloc[0]["team"]] += 1
        for team in table.tail(4)["team"]:
            relegated[team] += 1
        for _, row in table.iterrows():
            points_totals[row["team"]] += row["points"]
            wins_totals[row["team"]] += row["wins"]
            gd_totals[row["team"]] += row["gd"]

    rows = []
    for team in teams:
        rows.append(
            {
                "team": team,
                "points": points_totals[team] / iterations,
                "wins": wins_totals[team] / iterations,
                "gd": gd_totals[team] / iterations,
                "title": title_counts[team] / iterations,
                "relegation": relegated[team] / iterations,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["points", "wins", "gd"], ascending=[False, False, False]).reset_index(drop=True)
    df["position"] = range(1, len(df) + 1)
    df["points"] = df["points"].round().astype(int)
    df["wins"] = df["wins"].round().astype(int)
    df["gd"] = df["gd"].round().astype(int)
    return df[["position", "team", "points", "wins", "gd", "title", "relegation"]]
