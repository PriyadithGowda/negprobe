# negprobe: negation in factual probes, from BERT to modern language models

Code, outputs and paper for the term paper

> **Birds Can Still Not Fly? Negation in Factual Probes from BERT to Modern Language Models**
> Priyadith Hosahalli Nagesh, Universität Trier.
> Advanced Topics in Computational Text and Media Sciences #1, SoSe 2026.
> Paper: [`negprobe/paper/negation_paper.pdf`](paper/negation_paper.pdf)

The paper replicates the negated LAMA experiments of
[Kassner & Schütze (2020)](https://aclanthology.org/2020.acl-main.698/) and extends them in three ways:

1. **Δ**, a probability-based measure of how far a model pushes the denied fact down;
2. a **matched comparison** of masked (BERT) and autoregressive (Pythia) models on identical items;
3. a **truth-judgement probe** that asks a model whether a negated statement is true.

It also documents and corrects an **error in the released relation templates** (two of the 41 T-REx
relations receive the wrong negated sentence).

---

## Main findings

- **The replication holds.** BERT-base returns the same top answer for a query and its negation in
  51.9 % of T-REx pairs with the templates as released, and in 54.1 % with the corrected templates
  (BERT-large: 46.9 % and 49.1 %).
- **Overlap and Δ measure different things.** Pythia changes its top answer more often than BERT,
  but lowers the denied fact far less (Δ ≈ −0.08 against −0.49 for BERT-base).
- **Scale does not bring negation sensitivity.** From Pythia-410m to Pythia-1.4b, factual accuracy rises
  while Δ stays near zero.
- **Truth judgements do better than the cloze probe suggests.** Qwen2.5-1.5B-Instruct gets 62.2 % of
  negated statements right (chance is 50 %), but its separation of true from false statements drops from
  55 to 24 points under negation. Its base model barely registers the negation (a 5-point gap).

### Cross-model comparison

All 28,764 mask-final T-REx pairs (35 relations), identical for every model, corrected templates.
P@1 = the top-ranked candidate is the gold object.

| Model | ρ | Overlap | Δ | P@1 | P@1 negated |
|---|---|---|---|---|---|
| BERT-base | 0.880 | 0.517 | −0.491 | 0.363 | 0.277 |
| BERT-large | 0.842 | 0.487 | −0.817 | 0.376 | 0.292 |
| Pythia-410m | 0.900 | 0.452 | −0.083 | 0.226 | 0.221 |
| Pythia-1.4b | 0.899 | 0.455 | −0.016 | 0.239 | 0.231 |

### Truth judgement (T-REx, 8,200 statements)

| Model | Prompt | Acc. affirmative | Acc. negated | Pair consistency | Says "True" |
|---|---|---|---|---|---|
| Qwen2.5-1.5B (base) | four examples | 0.749 | 0.525 | 0.378 | 0.192 |
| Qwen2.5-1.5B-Instruct | chat template | 0.773 | 0.622 | 0.601 | 0.314 |

The design is balanced (half the statements are true), so a bias towards one answer cannot raise
accuracy above 0.5. The two models are prompted differently, so their difference does not isolate the
effect of instruction tuning. All tables are in [`docs/RESULTS.md`](docs/RESULTS.md).

---

## Repository layout

| Path | What it is |
|---|---|
| `paper/` | LaTeX source, figures and the compiled PDF |
| `negprobe/` | the Python package (see *Code* below) |
| `results/` | cloze runs (BERT, Pythia): one row per probe pair in `pairs.jsonl.gz`, plus `summary.csv` and `meta.json` |
| `results_released/` | BERT runs with the templates as released (the "as released" rows of Table 1) |
| `results_truth/` | truth-judgement runs (Qwen base and instruct): one row per statement in `items.jsonl.gz` |
| `logs/` | console logs of the Colab runs (Pythia and Qwen) |
| `figures/` | figures and tables as generated from the outputs |
| `docs/RESULTS.md` | every table, recomputed from the outputs |
| `docs/DATA_ISSUE.md` | the template error: evidence, effect and fix |
| `docs/REPRODUCE.md` | the exact command behind each table, with runtimes and hardware notes |
| `docs/VERIFY.md` | how to check the results, from one minute to a full re-run |
| `verify_numbers.py` | checks every result number printed in the paper against the outputs |
| `reproduce_all.sh` | re-runs every experiment and records environment and checksums |
| `negprobe_colab.ipynb` | self-contained notebook for a Colab GPU |

---

## Verify the results (1 minute, no GPU)

```bash
pip install pandas
python verify_numbers.py        # expected last line: 224 passed, 0 failed, 224 checks
```

The script recomputes each number printed in the paper from the committed per-item outputs and checks
that it matches. See [`docs/VERIFY.md`](docs/VERIFY.md) for spot checks by hand and a full re-run.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash get_data.sh                  # downloads the negated LAMA probes and the LAMA vocabulary

# Experiments A + B: rank metrics and Δ
python -m negprobe.run --model bert-base-cased --data data/negated_data \
    --vocab data/common_vocab_cased.txt --subsets GoogleRE TREx --out results

# Experiment C: truth judgement (causal models; add --chat no for base models)
python -m negprobe.truth --model Qwen/Qwen2.5-1.5B-Instruct --data data/negated_data \
    --subsets TREx GoogleRE --per-relation 50 --out results_truth

# figures and tables
python -m negprobe.figures results --truth results_truth --out figures
```

`bash reproduce_all.sh` runs every experiment in the paper (`SKIP_GPU=1` on a CPU-only machine).
Runtimes: BERT-base 8 min and BERT-large 47 min on an Apple M1 CPU; Pythia-410m 5 min, Pythia-1.4b
15 min and each Qwen run 8–13 min on a Colab T4 GPU. All reported runs use float32.

---

## What the experiments measure

**A: replication.** Spearman rank correlation (ρ) and top-1 overlap between the predictions for a
query and its negation, as in Kassner & Schütze.

**B: the negation shift Δ.** `Δ = log P(gold | negated) − log P(gold | affirmative)`. A negated query
has no single correct answer, so a changed top prediction proves little; Δ asks directly whether the
model lowers the fact that was denied.

**C: truth judgement.** Each fact becomes four statements (affirmative or negated, with the true object
or a distractor). The verdict is read from the probabilities of the tokens `True` and `False`, so no
output has to be parsed.

**D: free generation** (`negprobe/generate.py`). Implemented, but not reported in the paper.

## Code

```
negprobe/
  data.py       loads the probes; TEMPLATE_FIXES and check_templates() for the template error
  scorers.py    masked and causal LM scorers, with precision and finiteness checks
  metrics.py    ρ, ranks, exact-match P@1, bootstrap intervals, aggregation
  run.py        Experiments A + B   -> results/<model>/
  truth.py      Experiment C        -> results_truth/<model>/
  generate.py   Experiment D        -> results_generate/<model>/
  figures.py    figures and LaTeX tables
  compare.py    merges summaries across models
```

## Good to know

1. **Half precision without a CUDA GPU produced NaN** (float16 on CPU, bfloat16 on an Apple M1),
   which looks like perfect accuracy. The default `--dtype auto` uses float32 when there is no CUDA
   device, and the code stops on any non-finite score.
2. **Qwen2.5 base models ship a chat template** but are not instruction-tuned; use `--chat no` so they
   get the four-example prompt.
3. **The corrected templates are the default.** Set `NEGPROBE_RELEASED_TEMPLATES=1` to use the
   templates as released.

## Data and licence

The probes come from Petroni et al. (2019) and Kassner & Schütze (2020) and are downloaded by
`get_data.sh`; this repository does not redistribute them. Models are loaded from the Hugging Face Hub.
Code: MIT licence (see `LICENSE`).
