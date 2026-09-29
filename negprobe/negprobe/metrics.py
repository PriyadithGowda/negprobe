"""Per-pair metrics and aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def _ranks(x: torch.Tensor) -> torch.Tensor:
    """Row-wise ranks (0 = smallest). Ties are practically absent in float logits."""
    order = x.argsort(dim=-1)
    r = torch.empty_like(order, dtype=torch.float32)
    src = torch.arange(x.shape[-1], device=x.device, dtype=torch.float32).expand_as(order)
    r.scatter_(-1, order, src)
    return r


def spearman_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Spearman rho between matching rows of a and b (same as scipy.stats.spearmanr
    when there are no ties). Computed on the GPU, which matters for 150k vocabularies."""
    ra, rb = _ranks(a), _ranks(b)
    ra = ra - ra.mean(-1, keepdim=True)
    rb = rb - rb.mean(-1, keepdim=True)
    return (ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1))


def rank_of(dist: torch.Tensor, pos: int | None) -> float:
    """1-based rank of vocabulary position ``pos`` in ``dist`` (higher logp = better)."""
    if pos is None:
        return float("nan")
    return float((dist > dist[pos]).sum().item() + 1)


def exact_match_p1(df: pd.DataFrame) -> pd.DataFrame:
    """P@1 as in LAMA: the top-ranked candidate *is* the gold object (whole word).

    For masked models this equals ``gold rank == 1``. For causal models the rank is
    taken on the gold object's first token, which would also count a hit when only a
    prefix matches ("Green" for "Greenland") and, in bfloat16, when scores tie; the
    exact-match definition avoids both and is defined for every pair.
    """
    df = df.copy()
    gold = df["obj"].astype(str).str.strip()
    df["p1_pos"] = (df["top1_pos"].astype(str).str.strip() == gold).astype(float)
    df["p1_neg"] = (df["top1_neg"].astype(str).str.strip() == gold).astype(float)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-pair results.

    Macro averages as in Petroni et al. (2019): mean over the pairs of each relation,
    then mean over relations within a subset. Micro averages over all pairs are
    reported too.
    """
    cols = {
        "rho": "spearman_rho",
        "overlap": "top1_overlap",
        "delta": "delta_gold_logp",
        "rank_drop": "gold_rank_dropped",
        "p1_pos": "p@1_affirmative",
        "p1_neg": "p@1_negated",
    }
    rows = []
    for subset, g in df.groupby("subset"):
        per_rel = g.groupby("relation")[list(cols)].mean()
        row = {"subset": subset, "n_pairs": len(g), "n_relations": len(per_rel)}
        for c, name in cols.items():
            row[f"{name}"] = per_rel[c].mean()
            row[f"{name}_micro"] = g[c].mean()
        # 95% bootstrap CIs over pairs (they belong to the *_micro means)
        row["spearman_rho_micro_ci95"] = _bootstrap(g, "rho")
        row["delta_micro_ci95"] = _bootstrap(g, "delta")
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap(g: pd.DataFrame, col: str, n: int = 1000, seed: int = 0) -> str:
    vals = g[col].dropna().to_numpy()
    if len(vals) < 2:
        return ""
    rng = np.random.default_rng(seed)
    means = [rng.choice(vals, len(vals)).mean() for _ in range(n)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return f"[{lo:.3f}, {hi:.3f}]"
