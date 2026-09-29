"""Merge summary.csv files of several models into one table.

    python -m negprobe.compare results --out results/all_models.csv
"""
import argparse
from pathlib import Path

import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="folder with one sub-folder per model")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    files = sorted(Path(args.results).glob("*/summary.csv"))
    if not files:
        raise SystemExit(f"no summary.csv found under {args.results}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cols = ["model", "subset", "n_pairs", "spearman_rho", "top1_overlap",
            "delta_gold_logp", "gold_rank_dropped", "p@1_affirmative", "p@1_negated",
            "spearman_rho_micro", "spearman_rho_micro_ci95",
            "delta_gold_logp_micro", "delta_micro_ci95"]
    table = df[cols].sort_values(["subset", "model"])
    out = args.out or Path(args.results) / "all_models.csv"
    table.to_csv(out, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(table.round(3).to_string(index=False))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
