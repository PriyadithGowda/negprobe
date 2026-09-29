# How to verify this work

Three levels, from cheapest to most thorough. Level 1 takes a minute and needs no GPU.

## Level 1 — check every number in the paper against the committed outputs

```bash
pip install pandas
python verify_numbers.py            # add --scan to list decimals in the paper that have no check
```

The script pairs each result number **as printed in `paper/negation_paper.tex`** with the value
recomputed from the per-item outputs, and passes a check only when the recomputed value rounds to the
printed one. It covers every table (1–5), every result number in the text, the released-template
figures, the truth-judgement analysis, and the agreement between the two independent BERT runs.
Output ends with a line such as

```
224 passed, 0 failed, 224 checks
```

`--scan` lists the decimals in the paper that no check covers; the only ones left are the 0.5 chance
level, model names (1.4b, 1.5B) and LaTeX figure widths.

Inputs (all committed, gzipped):

| Path | Contents |
|---|---|
| `results/<model>/pairs.jsonl.gz` | one row per probe pair, repaired templates (BERT, Pythia) |
| `results_released/<model>/pairs.jsonl.gz` | BERT with the released templates (Table 1 "as released", `DATA_ISSUE.md`) |
| `results_truth/Qwen_Qwen2.5-1.5B{,-Instruct}/items.jsonl.gz` | one row per true/false statement, base and instruction-tuned model |
| `logs/colab_2026-09-28/` | console logs of the four Colab runs behind the Pythia and Qwen results |
| `results*/<model>/summary.csv`, `meta.json` | aggregates, bootstrap intervals, arguments, vocabulary size, runtime |

## Level 2 — spot-check individual claims by hand

```python
import pandas as pd
d = pd.read_json("results/bert-base-cased/pairs.jsonl.gz", lines=True)

# the replication claim: same top-1 filler for the affirmative and negated query
t = d[d.subset == "TREx"]
print(t.groupby("relation").overlap.mean().mean())      # 0.541

# look at the actual sentences
print(t[t.overlap == 1][["pos", "top1_pos", "neg", "top1_neg", "obj"]].head())

# the template error, as released
r = pd.read_json("results_released/bert-base-cased/pairs.jsonl.gz", lines=True)
print(r[r.relation == "P103"][["pos", "neg"]].head(3))

# the truth-judgement cells
q = pd.read_json("results_truth/Qwen_Qwen2.5-1.5B-Instruct/items.jsonl.gz", lines=True)
print(q[q.subset == "TREx"].groupby(["polarity", "object_kind"]).correct.mean())
```

Columns are documented in the main `README.md`.

## Level 3 — re-run everything from scratch

```bash
bash setup_mac.sh                 # or: python -m venv .venv && pip install -r requirements.txt
bash reproduce_all.sh             # SKIP_GPU=1 runs only what a CPU handles in reasonable time
```

`reproduce_all.sh` downloads the probes, re-runs every experiment in the paper in the configuration
used there (repaired templates, float32), rebuilds the figures, runs `verify_numbers.py`, and writes:

| File | Contents |
|---|---|
| `ENVIRONMENT.txt` | OS, Python, full `pip freeze`, torch build, GPU |
| `logs/*.log` | stdout of each run, including progress and warnings |
| `CHECKSUMS.txt` | sha256 of the input templates and every result file |

Measured runtimes (from `meta.json`): BERT-base 8 min and BERT-large 47 min on an Apple M1 CPU;
on a Colab T4 GPU, Pythia-410m 4.8 min, Pythia-1.4b 15.4 min, and the truth-judgement runs 8.3 min
(instruct) and 13.1 min (base).

The cloze runs involve no sampling: the committed released- and repaired-template BERT runs were made
independently and agree exactly on all 37,595 pairs outside P103 and P190. Across different hardware or
torch versions, the last printed digit can still differ; a re-run on a GPU in half precision will not be
bit-identical to the CPU runs.

## What cannot be verified from this repository

* The probes are not redistributed; `get_data.sh` downloads them from the original source.
  `DATA_ISSUE.md` records the SHA-256 of the `relations.jsonl` the results were produced with.
* Models are downloaded from the Hugging Face Hub at run time and are not pinned by revision, so a model
  that is updated upstream could change results.
* Two development incidents (the NaN run of Pythia-410m, mentioned in the paper, and a base-model run
  through a chat template, mentioned only in `REPRODUCE.md`) left no committed outputs; the paper
  reports only the qualitative observation from the first.
