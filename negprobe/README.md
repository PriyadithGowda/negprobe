# negprobe — negation in factual probes, from BERT to modern LMs

Code and results for the term paper *Birds Can Still Not Fly? Negation in Factual Probes from BERT to
Modern Language Models* (Universität Trier, Advanced Topics in Computational Text and Media Sciences #1,
SoSe 2026).

The paper replicates [Kassner & Schütze (2020)](https://aclanthology.org/2020.acl-main.698/) and extends
it with a probability-based measure of negation sensitivity, a matched comparison of masked and
autoregressive models, and a truth-judgement probe. It also documents and corrects an error in the
released relation templates.

## Contents

| Path | What it is |
|---|---|
| `negprobe/` | the package (see *Code layout* below) |
| `paper/` | LaTeX source, figures and the compiled PDF |
| `results/`, `results_truth/` | per-model outputs (`pairs.jsonl.gz` / `items.jsonl.gz`), `summary.csv`, `meta.json` |
| `results_released/` | BERT runs with the released (erroneous) templates, for Table 1 and `DATA_ISSUE.md` |
| `docs/RESULTS.md` | every number in the paper, with the file it comes from |
| `docs/DATA_ISSUE.md` | the template error: evidence, effect, fix |
| `docs/REPRODUCE.md` | step-by-step reproduction, including hardware pitfalls |
| `docs/VERIFY.md` | **how to verify this work** — three levels, from one minute to a full re-run |
| `verify_numbers.py` | checks every result number printed in the paper against the committed outputs |
| `reproduce_all.sh` | re-runs every experiment, records environment and checksums |
| `negprobe_colab.ipynb` | self-contained GPU notebook (no upload needed) |

Per-item outputs are committed gzipped (`results/*/pairs.jsonl.gz`, `results_truth/*/items.jsonl.gz`),
so every claim can be traced to the individual probes behind it without re-running anything.

## Verifying the results

```bash
python verify_numbers.py     # checks every result number in the paper, prints pass/fail
bash reproduce_all.sh        # re-runs every experiment from scratch
```

See [`docs/VERIFY.md`](docs/VERIFY.md).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash get_data.sh                      # negated LAMA probes + LAMA common vocabulary

# Experiments A+B — rank metrics and the negation shift
python -m negprobe.run   --model bert-base-cased --data data/negated_data \
    --vocab data/common_vocab_cased.txt --subsets GoogleRE TREx --out results

# Experiment C — true/false judgement (causal LMs only)
python -m negprobe.truth --model Qwen/Qwen2.5-1.5B-Instruct --data data/negated_data \
    --subsets TREx GoogleRE --per-relation 50

# Experiment D — free generation under negation
python -m negprobe.generate --model Qwen/Qwen2.5-1.5B-Instruct --data data/negated_data \
    --subsets TREx --per-relation 50

# figures and LaTeX tables
python -m negprobe.figures results --truth results_truth --out figures
```

## What the experiments measure

**A — replication metrics.** Spearman rank correlation and top-1 overlap between the affirmative and
negated distributions, as in Kassner & Schütze.

**B — the negation shift.** `Δ = log P(gold | negated) − log P(gold | affirmative)`. Under negation the
query has no unique correct answer, so a changed top prediction proves little; Δ is anchored to the
denied fact and stays interpretable. It separates models that the rank metrics score alike.

**C — truth judgement.** Each fact becomes four statements (affirmative/negated × gold/distractor). The
verdict is read from the probabilities of the `True` and `False` tokens, so nothing is parsed. Reports
accuracy by polarity, the negation gap, pair consistency, accuracy restricted to facts the model knows,
and the rate of "True" answers.

**D — free generation.** The model completes a negated sentence; supplying the denied fact is the error,
with the affirmative completion as a knowledge control.

## Headline results

Comparable subset — all 28,764 mask-final T-REx pairs, 35 relations, identical items for every model,
repaired templates (P@1 = the top-ranked candidate is the gold object):

| Model | ρ | Overlap | Δ | P@1 | P@1 negated |
|---|---|---|---|---|---|
| BERT-base | 0.880 | 0.517 | −0.491 | 0.363 | 0.277 |
| BERT-large | 0.842 | 0.487 | −0.817 | 0.376 | 0.292 |
| Pythia-410m | 0.900 | 0.452 | −0.083 | 0.226 | 0.221 |
| Pythia-1.4b | 0.899 | 0.455 | −0.016 | 0.239 | 0.231 |

Truth judgement on T-REx (8,200 statements, repaired templates):

| Model | Prompt | Acc. affirmative | Acc. negated | Pair consistency | Says "True" |
|---|---|---|---|---|---|
| Qwen2.5-1.5B (base) | four examples | 0.749 | 0.525 | 0.378 | 0.192 |
| Qwen2.5-1.5B-Instruct | chat template | 0.773 | 0.622 | 0.601 | 0.314 |

The design is balanced, so a response bias alone cannot lift accuracy above 0.5. The instruction-tuned
model does register negation, but its gap in "True" answers between true and false statements falls from
55 points (affirmative) to 24 points (negated), and a strong bias towards "False" makes the cells uneven
(0.911 on false negated statements, 0.332 on true ones). The base model's gap falls from 50 points to 5.
The two models are prompted differently, so this comparison does not isolate the effect of post-training.

Full tables, including the released-vs-repaired template comparison, are in
[`docs/RESULTS.md`](docs/RESULTS.md).

## Code layout

```
negprobe/
  data.py       loads the probes; TEMPLATE_FIXES + check_templates()
  scorers.py    MaskedLMScorer / CausalLMScorer, precision and finiteness guards
  metrics.py    Spearman on GPU, ranks, bootstrap CIs, aggregation
  run.py        Experiments A+B      -> results/<model>/
  truth.py      Experiment C         -> results_truth/<model>/
  generate.py   Experiment D         -> results_generate/<model>/
  figures.py    figures + LaTeX table
  compare.py    merges summaries across models
```

## Three things worth knowing

1. **Half precision without CUDA produced NaN in our runs** (float16 on CPU, bfloat16 on an Apple M1),
   which silently produces perfect-looking rankings. With the default `--dtype auto` the code uses fp32
   when there is no CUDA device, and it always aborts on non-finite scores. On a Mac, keep the default.
2. **Base models ship chat templates too.** Qwen2.5 base has a chat template in its tokenizer
   configuration but no instruction tuning; pass `--chat no` so it gets the four-example prompt.
3. **All reported runs use the repaired templates.** Set `NEGPROBE_RELEASED_TEMPLATES=1` to run with
   the templates as released (used for the "as released" BERT rows of Table 1).

## Citation of the probes

The probes are from Petroni et al. (2019) and Kassner & Schütze (2020); this repository redistributes no
data, only the code that downloads and corrects it.
