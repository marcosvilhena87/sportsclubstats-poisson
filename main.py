"""Command-line interface for running Brasileir\u00e3o simulations."""

# pylint: disable=wrong-import-position

import os
import sys
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import argparse

import numpy as np
import pandas as pd

from simulator import (
    parse_matches,
    summary_table,
    DEFAULT_JOBS,
    DEFAULT_HOME_FIELD_ADVANTAGE,
)
from calibration import estimate_parameters



def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate Brasileirão 2025 title odds")
    parser.add_argument(
        "--file", default="data/Brasileirao2025A.txt", help="fixture file path"
    )
    parser.add_argument(
        "--simulations", type=int, default=5000, help="number of simulation runs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for repeatable simulations",
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        default=True,
        help="disable the progress bar during simulations",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help="number of parallel workers",
    )
    parser.add_argument(
        "--home-advantage",
        type=float,
        default=DEFAULT_HOME_FIELD_ADVANTAGE,
        help="relative advantage multiplier for the home team",
    )
    parser.add_argument(
        "--home-goals-mean",
        type=float,
        default=1.0,
        help="expected goals for the home side (Poisson scoring)",
    )
    parser.add_argument(
        "--away-goals-mean",
        type=float,
        default=1.0,
        help="expected goals for the away side (Poisson scoring)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=4,
        help="number of top spots to report",
    )
    parser.add_argument(
        "--auto-calibrate",
        action="store_true",
        help="estimate parameters from past seasons",
    )
    parser.add_argument(
        "--from-date",
        dest="from_date",
        default=None,
        help="ignore results on or after this YYYY-MM-DD date",
    )
    parser.add_argument(
        "--html-output",
        default=os.path.join(os.path.dirname(__file__), "brasileirao.html"),
        help="path to save summary table as HTML",
    )
    args = parser.parse_args()

    if args.auto_calibrate:
        pattern = os.path.join("data", "Brasileirao????A.txt")
        season_files = sorted(glob.glob(pattern))
        season_files = [
            f for f in season_files if os.path.abspath(f) != os.path.abspath(args.file)
        ]
        _, args.home_advantage = estimate_parameters(season_files)

    matches = parse_matches(args.file)
    if args.from_date:
        from_date = pd.to_datetime(args.from_date)
        matches.loc[matches["date"] >= from_date, ["home_score", "away_score"]] = np.nan
    rng = np.random.default_rng(args.seed) if args.seed is not None else None
    home_adv = args.home_advantage

    summary = summary_table(
        matches,
        iterations=args.simulations,
        rng=rng,
        progress=args.progress,
        home_advantage=home_adv,
        home_goals_mean=args.home_goals_mean,
        away_goals_mean=args.away_goals_mean,
        n_jobs=args.jobs,
        top_n=args.top_n,
    )
    if args.html_output:
        summary.to_html(args.html_output, index=False)

    POINTS_W = max(len("xPts"), len(str(summary["points"].max())))
    WINS_W = max(len("xWins"), len(str(summary["wins"].max())))
    GD_W = max(len("xGD"), len(str(summary["gd"].max())))
    TITLE_W = max(len("Title"), max(len(f"{p:.2%}") for p in summary["title"]))
    top_col = f"top{args.top_n}"
    TOP_W = max(len(f"Top{args.top_n}"), max(len(f"{p:.2%}") for p in summary[top_col]))
    REL_W = max(len("Relegation"), max(len(f"{p:.2%}") for p in summary["relegation"]))
    print(
        f"{'Pos':>3}  {'Team':15s} "
        f"{'xPts':^{POINTS_W}} {'xWins':^{WINS_W}} {'xGD':^{GD_W}} "
        f"{'Title':^{TITLE_W}} {f'Top{args.top_n}':^{TOP_W}} {'Relegation':^{REL_W}}"
    )
    for _, row in summary.iterrows():
        title = f"{row['title']:.2%}"
        top_val = f"{row[top_col]:.2%}"
        releg = f"{row['relegation']:.2%}"
        print(
            f"{row['position']:>2d}   {row['team']:15s} "
            f"{row['points']:^{POINTS_W}d} {row['wins']:^{WINS_W}d} {row['gd']:^{GD_W}d} "
            f"{title:^{TITLE_W}} {top_val:^{TOP_W}} {releg:^{REL_W}}"
        )

if __name__ == "__main__":
    main()
