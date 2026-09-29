"""Turn results into the paper's figures and tables.

    python -m negprobe.figures results --out figures
    python -m negprobe.figures results --truth results_truth --out figures

Produces (PDF for LaTeX + PNG to look at):
  fig1_model_comparison  rho / overlap / Delta per model and subset
  fig2_scaling           metrics against parameter count (Pythia and other families)
  fig3_per_relation      per-relation overlap vs Delta, the spread across relations
  fig4_truth             true/false accuracy by polarity (only with --truth)
  table1_main.tex        the main results table, ready to \\input
  table2_relations.csv   per-relation numbers for the appendix
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .metrics import exact_match_p1

# categorical palette, checked for colour-blind separation
INK, MUTED, GRID = "#1A1A1A", "#5A6474", "#D8DCE2"
COLORS = ["#1F6FB2", "#C8761B", "#1E8F7A", "#B23A34"]
MARKERS = ["o", "s", "^", "D"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.facecolor": "white",
})

# parameter counts in millions, for the scaling figure
SIZES = {
    "bert-base-cased": 110, "bert-large-cased": 340,
    "roberta-base": 125, "roberta-large": 355,
    "pythia-70m": 70, "pythia-160m": 160, "pythia-410m": 410, "pythia-1b": 1000,
    "pythia-1.4b": 1400, "pythia-2.8b": 2800, "pythia-6.9b": 6900, "pythia-12b": 12000,
    "Qwen2.5-0.5B": 500, "Qwen2.5-1.5B": 1500, "Qwen2.5-3B": 3000, "Qwen2.5-7B": 7000,
    "Qwen2.5-14B": 14000, "Qwen2.5-32B": 32000, "Qwen2.5-72B": 72000,
    "Llama-3.2-1B": 1200, "Llama-3.2-3B": 3200, "Llama-3.1-8B": 8000, "Llama-3.1-70B": 70000,
}


def short(model: str) -> str:
    return model.split("/")[-1]


def params_m(model: str) -> float:
    s = short(model)
    base = s.replace("-Instruct", "").replace("-instruct", "")
    if base in SIZES:
        return SIZES[base]
    m = re.search(r"(\d+(?:\.\d+)?)\s*([bBmM])", base)  # e.g. "-7B", "410m"
    if m:
        return float(m.group(1)) * (1000 if m.group(2).lower() == "b" else 1)
    return float("nan")


def family(model: str) -> str:
    s = short(model).lower()
    for f in ("bert", "roberta", "pythia", "qwen", "llama", "mistral"):
        if f in s:
            return "bert" if s.startswith("bert") else f
    return "other"


def is_instruct(model: str) -> bool:
    return "instruct" in model.lower() or "-it" in model.lower()


def load_summaries(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("*/summary.csv"))
    if not files:
        raise SystemExit(f"no summary.csv under {root}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["short"] = df.model.map(short)
    df["params_m"] = df.model.map(params_m)
    df["family"] = df.model.map(family)
    df["instruct"] = df.model.map(is_instruct)
    return df


def save(fig, out: Path, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def fig_model_comparison(df: pd.DataFrame, out: Path):
    metrics = [("spearman_rho", "Rank correlation\naffirmative vs negated"),
               ("top1_overlap", "Same top-1 prediction"),
               ("delta_gold_logp", "Change in log P(correct answer)")]
    subsets = sorted(df.subset.unique())
    models = list(dict.fromkeys(df.sort_values("params_m")["short"]))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    y = np.arange(len(models))
    h = 0.8 / max(len(subsets), 1)
    for ax, (col, title) in zip(axes, metrics):
        for k, sub in enumerate(subsets):
            d = df[df.subset == sub].set_index("short").reindex(models)[col]
            ax.barh(y + k * h - 0.4 + h / 2, d.values, height=h * 0.86,
                    color=COLORS[k], label=sub, zorder=3)
        ax.set_yticks(y, models if ax is axes[0] else [""] * len(models))
        ax.set_title(title, loc="left")
        ax.xaxis.grid(True, zorder=0)
        ax.invert_yaxis()
        if col == "delta_gold_logp":
            ax.axvline(0, color=MUTED, lw=0.8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", bbox_to_anchor=(0.99, 1.06),
               ncol=len(subsets), fontsize=8)
    fig.suptitle("Negation sensitivity by model", x=0.02, y=1.06, ha="left", fontsize=11)
    save(fig, out, "fig1_model_comparison")


def fig_scaling(df: pd.DataFrame, out: Path):
    d = df.dropna(subset=["params_m"])
    if d.empty:
        return
    metrics = [("top1_overlap", "Same top-1 prediction"),
               ("delta_gold_logp", "Change in log P(correct answer)"),
               ("p@1_affirmative", "P@1, affirmative")]
    subsets = sorted(d.subset.unique())
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharex=True)
    fams = sorted(d.family.unique())
    for ax, (col, title) in zip(axes, metrics):
        for i, fam in enumerate(fams):
            for k, sub in enumerate(subsets):
                s = d[(d.family == fam) & (d.subset == sub)].sort_values("params_m")
                if s.empty:
                    continue
                style = "--" if sub != subsets[0] else "-"
                ax.plot(s.params_m, s[col], style, color=COLORS[i % len(COLORS)],
                        marker=MARKERS[k % len(MARKERS)], ms=5, lw=1.6, zorder=3,
                        label=f"{fam} · {sub}")
        ax.set_xscale("log")
        ax.set_xlabel("parameters (millions, log scale)")
        ax.set_title(title, loc="left")
        ax.grid(True, zorder=0)
        if col == "delta_gold_logp":
            ax.axhline(0, color=MUTED, lw=0.8)
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle("Does scale change how models handle negation?", x=0.02, y=1.06,
                 ha="left", fontsize=11)
    save(fig, out, "fig2_scaling")


def fig_per_relation(root: Path, out: Path, subset: str = "TREx"):
    frames = []
    for f in sorted(list(root.glob("*/pairs.jsonl")) + list(root.glob("*/pairs.jsonl.gz"))):
        d = exact_match_p1(pd.read_json(f, lines=True))
        d = d[d.subset == subset]
        if d.empty:
            continue
        g = d.groupby("relation")[["overlap", "delta", "p1_pos"]].mean()
        g["model"] = short(d.model.iloc[0])
        frames.append(g.reset_index())
    if not frames:
        return
    rel = pd.concat(frames)
    models = sorted(rel.model.unique())
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, m in enumerate(models):
        s = rel[rel.model == m]
        ax.scatter(s.delta, s.overlap, s=26, color=COLORS[i % len(COLORS)],
                   edgecolor="white", linewidth=0.8, label=m, zorder=3)
    worst = rel[rel.model == models[0]].nlargest(3, "overlap")
    for _, r in worst.iterrows():
        ax.annotate(r.relation, (r.delta, r.overlap), fontsize=7, color=MUTED,
                    xytext=(4, 4), textcoords="offset points")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("change in log P(correct answer) under negation")
    ax.set_ylabel("share with the same top-1 prediction")
    ax.set_title(f"Negation sensitivity varies by relation ({subset})", loc="left")
    ax.grid(True, zorder=0)
    ax.legend(fontsize=8)
    save(fig, out, "fig3_per_relation")


def fig_truth(truth_root: Path, out: Path):
    """One row per model and subset. Values are exactly those in summary.csv (no averaging
    across subsets), so the figure matches the tables."""
    files = sorted(truth_root.glob("*/summary.csv"))
    if not files:
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["row"] = df.model.map(short) + " · " + df.subset
    df = df.sort_values(["model", "subset"], ascending=[True, False])
    rows = list(df["row"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.9 + 0.9 * len(rows)))
    y = np.arange(len(rows))
    for k, (col, lab) in enumerate([("acc_affirmative", "affirmative"),
                                    ("acc_negated", "negated")]):
        axes[0].barh(y + k * 0.4 - 0.2, df[col].values, height=0.34, color=COLORS[k],
                     label=lab, zorder=3)
    axes[0].axvline(0.5, color=MUTED, lw=0.8, ls=":")
    axes[0].set_yticks(y, rows)
    axes[0].set_xlim(0, 1.0)  # room for the legend to the right of the bars
    axes[0].set_xlabel("accuracy (0.5 = chance)")
    axes[0].set_title("True/false judgements", loc="left")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].invert_yaxis()
    axes[0].xaxis.grid(True, zorder=0)

    for k, col in enumerate(["pair_consistency", "says_true_rate"]):
        axes[1].barh(y + k * 0.4 - 0.2, df[col].values, height=0.34,
                     color=COLORS[k + 2], label=col.replace("_", " "), zorder=3)
    axes[1].axvline(0.5, color=MUTED, lw=0.8, ls=":")
    axes[1].set_yticks(y, [""] * len(rows))
    axes[1].set_xlim(0, 0.9)
    axes[1].set_xlabel("share of statement pairs / of items")
    axes[1].set_title('Consistency and rate of "True" answers', loc="left")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].invert_yaxis()
    axes[1].xaxis.grid(True, zorder=0)
    save(fig, out, "fig4_truth")


def tables(df: pd.DataFrame, root: Path, out: Path):
    cols = ["short", "subset", "n_pairs", "spearman_rho", "top1_overlap",
            "delta_gold_logp", "p@1_affirmative", "p@1_negated"]
    t = df[cols].sort_values(["subset", "short"]).round(3)
    t.columns = ["Model", "Data", "N", r"$\rho$", "Overlap", r"$\Delta$", "P@1", "P@1 (neg)"]
    (out / "table1_main.tex").write_text(t.to_latex(
        index=False, escape=False, column_format="llrrrrrr",
        caption="Negation probing results. $\\rho$ and overlap compare the affirmative and "
                "negated distributions; $\\Delta$ is the change in the log probability of the "
                "correct answer under negation.",
        label="tab:main"))
    frames = []
    for f in sorted(list(root.glob("*/pairs.jsonl")) + list(root.glob("*/pairs.jsonl.gz"))):
        d = exact_match_p1(pd.read_json(f, lines=True))
        g = (d.groupby(["subset", "relation"])[["rho", "overlap", "delta", "p1_pos", "p1_neg"]]
             .mean().round(3).reset_index())
        g.insert(0, "model", short(d.model.iloc[0]))
        frames.append(g)
    if frames:
        pd.concat(frames).to_csv(out / "table2_relations.csv", index=False)
    print("  table1_main.tex / table2_relations.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="folder with one sub-folder per model (Experiments A+B)")
    ap.add_argument("--truth", default=None, help="results folder of Experiment C")
    ap.add_argument("--subset", default="TREx", help="subset for the per-relation figure")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args(argv)

    root, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_summaries(root)
    print(f"{df.model.nunique()} models, {len(df)} rows. writing to {out}/")
    fig_model_comparison(df, out)
    fig_scaling(df, out)
    fig_per_relation(root, out, args.subset)
    if args.truth:
        fig_truth(Path(args.truth), out)
    tables(df, root, out)
    (out / "figures_meta.json").write_text(json.dumps(
        {"models": sorted(df.model.unique()), "subsets": sorted(df.subset.unique())}, indent=2))


if __name__ == "__main__":
    main()
