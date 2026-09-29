"""Run Experiments A and B for one model.

Example:
    python -m negprobe.run --model bert-base-cased --data data/negated_data \
        --vocab data/common_vocab_cased.txt --out results
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .data import load_pairs, load_vocab
from .metrics import exact_match_p1, rank_of, spearman_rows, summarize
from .scorers import load_scorer


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face model id or local path")
    ap.add_argument("--kind", default="auto", choices=["auto", "mlm", "causal"])
    ap.add_argument("--data", required=True, help="negated LAMA folder (Facebook release or LAMA_primed_negated/data)")
    ap.add_argument("--subsets", nargs="*", default=None, help="e.g. TREx GoogleRE ConceptNet SQuAD")
    ap.add_argument("--trex-templates", default=None, help="optional relations.jsonl with template/template_negated")
    ap.add_argument("--vocab", default=None, help="LAMA common_vocab_cased.txt (recommended for comparability)")
    ap.add_argument("--only-mask-final", action="store_true",
                    help="keep only pairs usable by causal LMs (always on for causal models)")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N pairs (for testing)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    pairs = load_pairs(args.data, args.subsets, args.trex_templates)
    vocab = load_vocab(args.vocab)
    scorer = load_scorer(args.model, args.kind, device=args.device, dtype=args.dtype,
                         vocab=vocab, load_in_4bit=args.load_in_4bit,
                         trust_remote_code=args.trust_remote_code)
    if scorer.is_causal or args.only_mask_final:
        before = len(pairs)
        pairs = [p for p in pairs if p.mask_final]
        print(f"kept {len(pairs)}/{before} pairs where [MASK] is sentence-final")
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"{args.model}: {len(pairs)} pairs, restricted vocab size {len(scorer.index_list)}")

    out_dir = Path(args.out) / slug(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    t0 = time.time()
    for i in tqdm(range(0, len(pairs), args.batch_size)):
        batch = pairs[i: i + args.batch_size]
        objs = [p.obj for p in batch]
        d_pos, g_pos = scorer.score([p.pos for p in batch], objs)
        d_neg, g_neg = scorer.score([p.neg for p in batch], objs)
        rho = spearman_rows(d_pos, d_neg).cpu()
        top_pos = d_pos.argmax(-1).cpu()
        top_neg = d_neg.argmax(-1).cpu()
        for j, p in enumerate(batch):
            gids = scorer.gold_ids(p.obj)
            # rank is defined on the first gold token (the only one for MLMs)
            gpos = scorer.restricted_position(gids[0]) if gids else None
            if not scorer.is_causal and len(gids) != 1:
                gpos = None
            r_pos, r_neg = rank_of(d_pos[j], gpos), rank_of(d_neg[j], gpos)
            gp, gn = g_pos[j].item(), g_neg[j].item()
            records.append({
                **p.to_dict(),
                "model": args.model,
                "rho": rho[j].item(),
                "overlap": int(top_pos[j] == top_neg[j]),
                "top1_pos": scorer.decode(scorer.index_list[top_pos[j]].item()),
                "top1_neg": scorer.decode(scorer.index_list[top_neg[j]].item()),
                "gold_logp_pos": gp,
                "gold_logp_neg": gn,
                "delta": gn - gp if not (math.isnan(gp) or math.isnan(gn)) else float("nan"),
                "gold_rank_pos": r_pos,
                "gold_rank_neg": r_neg,
                "rank_drop": float(r_neg > r_pos) if not math.isnan(r_pos) else float("nan"),
                "p1_pos": float(r_pos == 1) if not math.isnan(r_pos) else float("nan"),
                "p1_neg": float(r_neg == 1) if not math.isnan(r_neg) else float("nan"),
            })

    df = exact_match_p1(pd.DataFrame(records))  # P@1 = top-1 candidate is the gold object
    df.to_json(out_dir / "pairs.jsonl", orient="records", lines=True, force_ascii=False)
    summary = summarize(df)
    summary.insert(0, "model", args.model)
    summary.to_csv(out_dir / "summary.csv", index=False)
    meta = {"model": args.model, "kind": "causal" if scorer.is_causal else "mlm",
            "n_pairs": len(df), "vocab_size": len(scorer.index_list),
            "common_vocab": bool(vocab), "seconds": round(time.time() - t0, 1),
            "args": vars(args), "torch": torch.__version__}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary[["subset", "n_pairs", "n_relations", "spearman_rho", "top1_overlap",
                       "delta_gold_logp", "gold_rank_dropped", "p@1_affirmative"]].round(3))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
