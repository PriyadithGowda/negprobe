"""Experiment C: do models judge negated statements correctly?

For every fact a 2x2 design is built:

    polarity (affirmative / negated)  x  object (gold / distractor)

    Albert Einstein was born in Ulm.          -> True
    Albert Einstein was not born in Ulm.      -> False
    Albert Einstein was born in Lyon.         -> False
    Albert Einstein was not born in Lyon.     -> True

The model answers True/False; the answer is read from the probabilities of the
True and False tokens, so nothing has to be parsed and the result is deterministic.
Instruction-tuned models are prompted through their chat template, base models
with a few fixed examples.

    python -m negprobe.truth --model Qwen/Qwen2.5-7B-Instruct --data data/negated_data \
        --subsets TREx --per-relation 100

Causal LMs only (masked LMs do not produce a continuation to score).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import MASK, load_pairs
from .metrics import _bootstrap
from .run import slug

FEWSHOT = [
    ("Paris is the capital of France.", "True"),
    ("Paris is not the capital of France.", "False"),
    ("The Nile is located in Asia.", "False"),
    ("The Nile is not located in Asia.", "True"),
]
INSTRUCTION = ("Decide whether the statement is true or false. "
               "Answer with one word: True or False.")

TRUE_FORMS = ["True", " True", "true", " true", "TRUE"]
FALSE_FORMS = ["False", " False", "false", " false", "FALSE"]


def statement(sentence: str, obj: str) -> str:
    s = sentence.replace(MASK, obj)
    s = s.replace(" .", ".").replace(" ,", ",")  # LAMA templates space out punctuation
    return s[0].upper() + s[1:] if s else s


def build_items(pairs, per_relation: int | None, seed: int = 0):
    """One item per (fact, polarity, object) cell -> 4 items per fact."""
    rng = random.Random(seed)
    by_rel = defaultdict(list)
    for p in pairs:
        by_rel[(p.subset, p.relation)].append(p)

    items = []
    for (subset, relation), ps in by_rel.items():
        objects = sorted({p.obj for p in ps})
        if len(objects) < 2:
            continue  # no distractor available
        if per_relation and len(ps) > per_relation:
            ps = rng.sample(ps, per_relation)
        for p in ps:
            distractor = p.obj
            while distractor == p.obj:
                distractor = rng.choice(objects)
            for polarity, sent in (("affirmative", p.pos), ("negated", p.neg)):
                for kind, obj in (("gold", p.obj), ("distractor", distractor)):
                    # gold is true iff affirmative; distractor is true iff negated
                    truth = (polarity == "affirmative") == (kind == "gold")
                    items.append({
                        "subset": subset, "relation": relation, "uid": p.uid,
                        "sub": p.sub, "obj": obj, "gold_obj": p.obj,
                        "polarity": polarity, "object_kind": kind,
                        "statement": statement(sent, obj), "answer": truth,
                    })
    return items


class TruthScorer:
    def __init__(self, model_name, device="auto", dtype="auto", chat="auto",
                 load_in_4bit=False, trust_remote_code=False):
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        kwargs = dict(trust_remote_code=trust_remote_code)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            kwargs["device_map"] = "auto"
        else:
            if dtype == "auto" and not torch.cuda.is_available():
                dtype = "fp32"  # fp16 checkpoints overflow to NaN on CPU/MPS
            kwargs["dtype"] = {"auto": "auto", "fp32": torch.float32,
                               "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
            if device == "auto" and torch.cuda.is_available():
                kwargs["device_map"] = "auto"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()
        except TypeError:  # transformers < 4.56
            if "dtype" in kwargs:
                kwargs["torch_dtype"] = kwargs.pop("dtype")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()
        if not load_in_4bit and device != "auto":
            self.model.to(device)
        self.device = next(self.model.parameters()).device
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.chat = bool(getattr(self.tok, "chat_template", None)) if chat == "auto" else chat == "yes"
        self.true_ids = self._first_ids(TRUE_FORMS)
        self.false_ids = self._first_ids(FALSE_FORMS)

    def _first_ids(self, forms):
        ids = {self.tok.encode(f, add_special_tokens=False)[0] for f in forms
               if self.tok.encode(f, add_special_tokens=False)}
        return sorted(ids)

    def prompt(self, statement: str) -> str:
        if self.chat:
            msg = [{"role": "user", "content": f"{INSTRUCTION}\n\nStatement: {statement}"}]
            return self.tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        shots = "".join(f"Statement: {s}\nAnswer: {a}\n\n" for s, a in FEWSHOT)
        return f"{INSTRUCTION}\n\n{shots}Statement: {statement}\nAnswer:"

    @torch.no_grad()
    def score(self, statements: list[str]):
        """Returns (p_true, p_false) normalised over the True/False tokens."""
        texts = [self.prompt(s) for s in statements]
        enc = [self.tok.encode(t, add_special_tokens=not self.chat) for t in texts]
        L = max(map(len, enc))
        pad = self.tok.pad_token_id
        ids = torch.full((len(enc), L), pad, dtype=torch.long)
        attn = torch.zeros((len(enc), L), dtype=torch.long)
        for i, e in enumerate(enc):  # left padding
            ids[i, L - len(e):] = torch.tensor(e)
            attn[i, L - len(e):] = 1
        ids, attn = ids.to(self.device), attn.to(self.device)
        position_ids = (attn.cumsum(-1) - 1).clamp_min(0)
        try:
            logits = self.model(input_ids=ids, attention_mask=attn,
                                position_ids=position_ids).logits[:, -1].float()
        except TypeError:
            logits = self.model(input_ids=ids, attention_mask=attn).logits[:, -1].float()
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"NaN/inf logits — re-run with --dtype fp32.")
        probs = torch.softmax(logits, dim=-1)
        pt = probs[:, self.true_ids].sum(-1)
        pf = probs[:, self.false_ids].sum(-1)
        total = (pt + pf).clamp_min(1e-12)
        return (pt / total).cpu(), (pf / total).cpu()


def summarize_truth(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per cell, negation gap, pair consistency, knowledge-filtered accuracy."""
    rows = []
    for subset, g in df.groupby("subset"):
        acc = g.groupby(["polarity", "object_kind"]).correct.mean()
        aff = g[g.polarity == "affirmative"].correct.mean()
        neg = g[g.polarity == "negated"].correct.mean()

        # consistency: the model flips its verdict between "X ..." and "X not ..."
        piv = g.pivot_table(index=["relation", "uid", "obj"], columns="polarity",
                            values="said_true", aggfunc="first")
        consistency = ((piv.get("affirmative") != piv.get("negated")).mean()
                       if {"affirmative", "negated"} <= set(piv.columns) else float("nan"))

        # only facts the model knows: affirmative+gold answered correctly
        known = g[(g.polarity == "affirmative") & (g.object_kind == "gold") & g.correct]
        keys = set(zip(known.relation, known.uid))
        is_known = pd.Series([k in keys for k in zip(g.relation, g.uid)], index=g.index)
        kn = g[is_known & (g.polarity == "negated")]

        rows.append({
            "subset": subset, "n_items": len(g), "n_facts": g.uid.nunique(),
            "acc_affirmative": aff, "acc_negated": neg, "negation_gap": aff - neg,
            "acc_affirmative_gold": acc.get(("affirmative", "gold"), float("nan")),
            "acc_affirmative_distractor": acc.get(("affirmative", "distractor"), float("nan")),
            "acc_negated_gold": acc.get(("negated", "gold"), float("nan")),
            "acc_negated_distractor": acc.get(("negated", "distractor"), float("nan")),
            "pair_consistency": consistency,
            "acc_negated_known_facts": kn.correct.mean() if len(kn) else float("nan"),
            "says_true_rate": g.said_true.mean(),  # rate of "True" answers (response bias)
            "acc_ci95": _bootstrap(g, "correct"),
        })
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--subsets", nargs="*", default=None)
    ap.add_argument("--trex-templates", default=None)
    ap.add_argument("--per-relation", type=int, default=100,
                    help="facts per relation (each gives 4 items); 0 = all")
    ap.add_argument("--chat", default="auto", choices=["auto", "yes", "no"],
                    help="auto: chat template when the tokenizer has one")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_truth")
    args = ap.parse_args(argv)

    pairs = load_pairs(args.data, args.subsets, args.trex_templates)
    items = build_items(pairs, args.per_relation or None, args.seed)
    scorer = TruthScorer(args.model, device=args.device, dtype=args.dtype, chat=args.chat,
                         load_in_4bit=args.load_in_4bit, trust_remote_code=args.trust_remote_code)
    print(f"{args.model}: {len(items)} items "
          f"({'chat template' if scorer.chat else 'few-shot prompt'})")
    print("example prompt:\n" + scorer.prompt(items[0]["statement"]))

    t0 = time.time()
    for i in tqdm(range(0, len(items), args.batch_size)):
        batch = items[i: i + args.batch_size]
        pt, pf = scorer.score([b["statement"] for b in batch])
        for j, b in enumerate(batch):
            b["p_true"] = round(pt[j].item(), 5)
            b["said_true"] = bool(pt[j] > pf[j])
            b["correct"] = int(b["said_true"] == b["answer"])

    df = pd.DataFrame(items)
    out_dir = Path(args.out) / slug(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_json(out_dir / "items.jsonl", orient="records", lines=True, force_ascii=False)
    summary = summarize_truth(df)
    summary.insert(0, "model", args.model)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "meta.json").write_text(json.dumps(
        {"model": args.model, "chat": scorer.chat, "n_items": len(df),
         "seconds": round(time.time() - t0, 1), "args": vars(args)}, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary[["subset", "n_items", "acc_affirmative", "acc_negated", "negation_gap",
                       "pair_consistency", "acc_negated_known_facts", "says_true_rate"]].round(3))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
