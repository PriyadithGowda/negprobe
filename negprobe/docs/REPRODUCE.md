# Reproducing the results

Every number in the paper comes from these commands. Runtimes are measured, not estimated
(see `RESULTS.md` §6).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash get_data.sh     # downloads negated_data.tar.gz and common_vocab_cased.txt into ./data
```

`get_data.sh` also clones the LAMA repositories for reference; only `data/negated_data` and
`data/common_vocab_cased.txt` are needed.

## The runs behind each table

```bash
# Tables 1, 2, 5 and Figures 1-3 — Experiments A+B, repaired templates (the default)
python -m negprobe.run --model bert-base-cased  --data data/negated_data \
    --vocab data/common_vocab_cased.txt --subsets GoogleRE TREx --out results      # ~8 min, CPU
python -m negprobe.run --model bert-large-cased --data data/negated_data \
    --vocab data/common_vocab_cased.txt --subsets GoogleRE TREx --out results      # ~47 min, CPU

# the "as released" rows of Table 1 and DATA_ISSUE.md: same runs with the fix switched off
python -c "import negprobe.data as D; D.TEMPLATE_FIXES = {}; from negprobe.run import main; \
  main(['--model','bert-base-cased','--data','data/negated_data','--vocab','data/common_vocab_cased.txt', \
        '--subsets','GoogleRE','TREx','--out','results_released'])"      # repeat for bert-large-cased

# autoregressive models (scored on mask-final pairs only), repaired templates, float32
python -m negprobe.run --model EleutherAI/pythia-410m --data data/negated_data \
    --vocab data/common_vocab_cased.txt --subsets GoogleRE TREx --dtype fp32 --batch-size 64 --out results  # 4.8 min, Colab T4
python -m negprobe.run --model EleutherAI/pythia-1.4b --data data/negated_data \
    --vocab data/common_vocab_cased.txt --subsets GoogleRE TREx --dtype fp32 --batch-size 32 --out results  # 15.4 min, Colab T4

# Tables 3 and 4, Figure 4 — Experiment C, repaired templates, float32
python -m negprobe.truth --model Qwen/Qwen2.5-1.5B-Instruct --data data/negated_data \
    --subsets TREx GoogleRE --per-relation 50 --dtype fp32 --out results_truth    # 8.3 min, Colab T4
python -m negprobe.truth --model Qwen/Qwen2.5-1.5B --data data/negated_data \
    --subsets TREx GoogleRE --per-relation 50 --chat no --dtype fp32 --out results_truth   # base, four examples; 13.1 min, Colab T4

# Experiment D (implemented, not run for the paper)
python -m negprobe.generate --model Qwen/Qwen2.5-1.5B-Instruct --data data/negated_data \
    --subsets TREx --per-relation 50 --dtype bf16

# figures and LaTeX tables
python -m negprobe.figures results --truth results_truth --out figures
```

## The comparable-subset table

Models differ in which items they can score, so the cross-model table restricts to a common set:
the mask-final pairs (what an autoregressive model can score), with repaired templates for every model.
28,764 T-REx pairs across 35 relations (P190's repaired template does not end in the object).

```python
import pandas as pd
from negprobe.metrics import exact_match_p1
d = exact_match_p1(pd.read_json("results/bert-base-cased/pairs.jsonl.gz", lines=True))
t = d[(d.subset == "TREx") & d.mask_final]
print(t.groupby("relation")[["rho", "overlap", "delta", "p1_pos", "p1_neg"]].mean().mean().round(3))
```

## Hardware notes, learned the hard way

**Half precision without CUDA produced NaN in our runs.** Pythia-410m in float16 on CPU returned
non-finite logits; NaN comparisons are false, so every rank reads as 1 and the run looks like perfect
accuracy. That first Pythia-410m run was discarded (its outputs were not kept). With `--dtype auto`
(the default) the code now uses fp32 whenever no CUDA device is present, and it aborts on non-finite
scores whatever the dtype.

**`--dtype bf16` on an Apple M1 also produced NaN** (caught by the guard above). On a Mac, use the
default fp32 on CPU; that is how the BERT runs in the paper were made (Pythia and Qwen: fp32 on a Colab T4). We did not test
half precision on MPS.

**Base models and chat templates.** Qwen2.5 base ships a chat template in its tokenizer configuration
but is not instruction tuned, so `--chat auto` would prompt it through that template. Pass `--chat no`
for base models so they get the four-example prompt, as in the paper's base-model run. (During development a base-model run through the
chat template answered almost only "False"; its outputs were not kept, so the paper reports nothing
from it.)

**Free Colab disconnects.** Two runs were lost mid-way and the daily GPU quota ran out. Download results
after every single model: `negprobe_colab.ipynb` ends with a zip-and-download cell for that reason.

## Determinism

The cloze runs involve no sampling. The committed released-template and repaired-template BERT runs were
made independently and agree exactly on all 37,595 pairs outside P103 and P190 (checked by
`verify_numbers.py`). The truth-judgement items are drawn with a fixed seed (`--seed`, default 0).
