"""Experiment D: free generation under negation.

The model is asked to complete a negated sentence. If it fills in the very fact
being denied ("Einstein was not born in ___" -> "Ulm"), it is treating the negated
sentence like the affirmative one. The affirmative version is run as a control:
it measures how often the model knows the fact at all.

    python -m negprobe.generate --model Qwen/Qwen2.5-7B-Instruct --data data/negated_data \
        --subsets TREx --per-relation 50

Reported per subset:
  gold_rate_affirmative  the model completes the affirmative sentence correctly (knowledge)
  gold_rate_negated      it gives the true object even though the sentence denies it (the error)
  error_rate_known       the same, restricted to facts it gets right when affirmative
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import MASK, load_pairs
from .run import slug
from .truth import build_items

PROMPT = ("Complete the sentence with a single plausible answer. "
          "Reply with the answer only.\n\nSentence: {sentence}\nAnswer:")


def normalise(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def contains_gold(output: str, gold: str) -> bool:
    o, g = normalise(output), normalise(gold)
    return bool(g) and re.search(rf"\b{re.escape(g)}\b", o) is not None


class Generator:
    def __init__(self, model_name, device="auto", dtype="auto", max_new_tokens=12,
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
        except TypeError:
            if "dtype" in kwargs:
                kwargs["torch_dtype"] = kwargs.pop("dtype")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()
        if not load_in_4bit and device != "auto":
            self.model.to(device)
        self.device = next(self.model.parameters()).device
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.chat = bool(getattr(self.tok, "chat_template", None))
        self.max_new_tokens = max_new_tokens

    def prompt(self, sentence: str) -> str:
        text = PROMPT.format(sentence=sentence)
        if self.chat:
            return self.tok.apply_chat_template([{"role": "user", "content": text}],
                                                tokenize=False, add_generation_prompt=True)
        return text

    @torch.no_grad()
    def complete(self, sentences: list[str]) -> list[str]:
        enc = self.tok([self.prompt(s) for s in sentences], return_tensors="pt",
                       padding=True, add_special_tokens=not self.chat).to(self.device)
        out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                  do_sample=False, pad_token_id=self.tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        return [t.strip().split("\n")[0] for t in self.tok.batch_decode(gen, skip_special_tokens=True)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--subsets", nargs="*", default=None)
    ap.add_argument("--trex-templates", default=None)
    ap.add_argument("--per-relation", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_generate")
    args = ap.parse_args(argv)

    pairs = load_pairs(args.data, args.subsets, args.trex_templates)
    # reuse the sampling of Experiment C, but only the gold-object cells
    items = [i for i in build_items(pairs, args.per_relation or None, args.seed)
             if i["object_kind"] == "gold"]
    for it in items:
        it["sentence"] = it["statement"].replace(it["obj"], "___", 1)
        it.pop("statement")

    gen = Generator(args.model, device=args.device, dtype=args.dtype,
                    max_new_tokens=args.max_new_tokens, load_in_4bit=args.load_in_4bit,
                    trust_remote_code=args.trust_remote_code)
    print(f"{args.model}: {len(items)} prompts")
    print("example prompt:\n" + gen.prompt(items[0]["sentence"]))

    t0 = time.time()
    for i in tqdm(range(0, len(items), args.batch_size)):
        batch = items[i: i + args.batch_size]
        for b, o in zip(batch, gen.complete([x["sentence"] for x in batch])):
            b["output"] = o
            b["gold_in_output"] = int(contains_gold(o, b["gold_obj"]))

    df = pd.DataFrame(items)
    out_dir = Path(args.out) / slug(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_json(out_dir / "generations.jsonl", orient="records", lines=True, force_ascii=False)

    rows = []
    for subset, g in df.groupby("subset"):
        aff = g[g.polarity == "affirmative"]
        neg = g[g.polarity == "negated"]
        known = set(zip(aff[aff.gold_in_output == 1].relation, aff[aff.gold_in_output == 1].uid))
        neg_known = neg[[k in known for k in zip(neg.relation, neg.uid)]]
        rows.append({
            "model": args.model, "subset": subset, "n_facts": g.uid.nunique(),
            "gold_rate_affirmative": aff.gold_in_output.mean(),
            "gold_rate_negated": neg.gold_in_output.mean(),
            "error_rate_known": neg_known.gold_in_output.mean() if len(neg_known) else float("nan"),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "meta.json").write_text(json.dumps(
        {"model": args.model, "chat": gen.chat, "n_prompts": len(df),
         "seconds": round(time.time() - t0, 1), "args": vars(args)}, indent=2))
    print(summary.round(3).to_string(index=False))
    print("\nexamples where the model answers the denied fact:")
    bad = df[(df.polarity == "negated") & (df.gold_in_output == 1)].head(5)
    for _, r in bad.iterrows():
        print(f"  {r.sentence}  ->  {r.output}   (gold {r.gold_obj})")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
