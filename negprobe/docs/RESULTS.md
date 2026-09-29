# Results

Every number here is recomputed from the committed per-item outputs; `verify_numbers.py` checks each
number printed in the paper against the same files. Last checked: 28 Sep 2026.

Averaging follows Petroni et al. (2019): mean within a relation, then across relations (macro).
P@1 is exact match: the top-ranked candidate is the gold object (`negprobe.metrics.exact_match_p1`).
For Pythia this differs from the first-token rank stored in the `p1_*` columns of the committed
`pairs.jsonl.gz`; every table and figure uses the exact-match definition.

All runs reported here and in the paper use the repaired templates (`docs/DATA_ISSUE.md`) and float32.
BERT ran on an Apple M1 CPU; Pythia and Qwen were re-run on a Colab T4 GPU on 28 Sep 2026
(logs in `logs/colab_2026-09-28/`).

## 1. Main table (per model, per subset)

| model       | subset   |   n_pairs |   n_relations |   rho |   overlap |   delta |   gold_rank_dropped |   p1_pos |   p1_neg |
|:------------|:---------|----------:|--------------:|------:|----------:|--------:|--------------------:|---------:|---------:|
| BERT-base   | GoogleRE |      5528 |             3 | 0.834 |     0.113 |  -0.814 |               0.76  |    0.099 |    0.028 |
| BERT-large  | GoogleRE |      5528 |             3 | 0.776 |     0.112 |  -0.84  |               0.749 |    0.105 |    0.028 |
| Pythia-410m | GoogleRE |      5528 |             3 | 0.881 |     0.259 |  -0.748 |               0.698 |    0.045 |    0.013 |
| Pythia-1.4b | GoogleRE |      5528 |             3 | 0.856 |     0.148 |  -1.18  |               0.755 |    0.055 |    0.016 |
| BERT-base   | TREx     |     34039 |            41 | 0.887 |     0.541 |  -0.568 |               0.467 |    0.311 |    0.237 |
| BERT-large  | TREx     |     34039 |            41 | 0.852 |     0.491 |  -0.84  |               0.439 |    0.323 |    0.251 |
| Pythia-410m | TREx     |     28764 |            35 | 0.9   |     0.452 |  -0.083 |               0.4   |    0.226 |    0.221 |
| Pythia-1.4b | TREx     |     28764 |            35 | 0.899 |     0.455 |  -0.016 |               0.375 |    0.239 |    0.231 |

Source: `results/<model>/pairs.jsonl.gz` (aggregates also in `summary.csv`). Pythia rows cover only
pairs whose mask is sentence-final (28,764 of 34,039 on T-REx), which is what an autoregressive model can
score. `gold_rank_dropped` = share of pairs where the gold object ranks lower under negation.

## 2. Comparable subset — identical items for every model

All 28,764 mask-final T-REx pairs, 35 relations (P190's repaired template is not mask-final), repaired
templates for every model (paper, Table 2).

| model       |   n_pairs |   n_relations |   rho |   overlap |   delta |   p1_pos |   p1_neg |
|:------------|----------:|--------------:|------:|----------:|--------:|---------:|---------:|
| BERT-base   |     28764 |            35 | 0.88  |     0.517 |  -0.491 |    0.363 |    0.277 |
| BERT-large  |     28764 |            35 | 0.842 |     0.487 |  -0.817 |    0.376 |    0.292 |
| Pythia-410m |     28764 |            35 | 0.9   |     0.452 |  -0.083 |    0.226 |    0.221 |
| Pythia-1.4b |     28764 |            35 | 0.899 |     0.455 |  -0.016 |    0.239 |    0.231 |

Without P36 (*capital*), whose Δ is far larger for BERT-large than for any other model:

| model       |   delta without P36 |
|:------------|--------------------:|
| BERT-base   |              -0.407 |
| BERT-large  |              -0.548 |
| Pythia-410m |               0.001 |
| Pythia-1.4b |               0.026 |

## 3. Effect of the template repair (BERT, T-REx, all 41 relations)

Source: `results/<model>/` (repaired) and `results_released/<model>/` (as released). The two runs
are bit-identical on every pair outside P103 and P190.

| model      | templates   |   rho |   overlap |   delta |   p1_pos |   p1_neg |
|:-----------|:------------|------:|----------:|--------:|---------:|---------:|
| BERT-base  | repaired    | 0.887 |     0.541 |  -0.568 |    0.311 |    0.237 |
| BERT-base  | as released | 0.866 |     0.519 |  -0.735 |    0.311 |    0.237 |
| BERT-large | repaired    | 0.852 |     0.491 |  -0.84  |    0.323 |    0.251 |
| BERT-large | as released | 0.832 |     0.469 |  -1.018 |    0.323 |    0.251 |

## 4. Experiment C — truth judgement

| model                                 | subset   |   n_items |   acc_affirmative |   acc_negated |   negation_gap |   pair_consistency |   acc_negated_known_facts |   says_true_rate |
|:--------------------------------------|:---------|----------:|------------------:|--------------:|---------------:|-------------------:|--------------------------:|-----------------:|
| Qwen2.5-1.5B (base, four examples)    | GoogleRE |       600 |             0.64  |         0.507 |          0.133 |              0.327 |                     0.507 |            0.163 |
| Qwen2.5-1.5B (base, four examples)    | TREx     |      8200 |             0.749 |         0.525 |          0.224 |              0.378 |                     0.538 |            0.192 |
| Qwen2.5-1.5B-Instruct (chat template) | GoogleRE |       600 |             0.653 |         0.567 |          0.087 |              0.433 |                     0.645 |            0.257 |
| Qwen2.5-1.5B-Instruct (chat template) | TREx     |      8200 |             0.773 |         0.622 |          0.151 |              0.601 |                     0.66  |            0.314 |

Accuracy by cell (the correct answer is "True" for affirmative/gold and negated/distractor):

| model                                 | subset   |   acc_affirmative_gold |   acc_affirmative_distractor |   acc_negated_gold |   acc_negated_distractor |
|:--------------------------------------|:---------|-----------------------:|-----------------------------:|-------------------:|-------------------------:|
| Qwen2.5-1.5B (base, four examples)    | GoogleRE |                  0.46  |                        0.82  |              1     |                    0.013 |
| Qwen2.5-1.5B (base, four examples)    | TREx     |                  0.569 |                        0.929 |              0.96  |                    0.089 |
| Qwen2.5-1.5B-Instruct (chat template) | GoogleRE |                  0.46  |                        0.847 |              0.86  |                    0.273 |
| Qwen2.5-1.5B-Instruct (chat template) | TREx     |                  0.69  |                        0.856 |              0.911 |                    0.332 |

Rate of "True" answers by cell and bias-corrected sensitivity d′ (T-REx):

| model                                 |   True: affirm. gold |   True: affirm. distractor |   True: neg. distractor |   True: neg. gold |   d' affirm. |   d' neg. |
|:--------------------------------------|---------------------:|---------------------------:|------------------------:|------------------:|-------------:|----------:|
| Qwen2.5-1.5B (base, four examples)    |                0.569 |                      0.071 |                   0.089 |             0.04  |         1.64 |      0.41 |
| Qwen2.5-1.5B-Instruct (chat template) |                0.69  |                      0.144 |                   0.332 |             0.089 |         1.56 |      0.91 |

The design is balanced, so 0.5 is chance and an unbiased model would answer "True" half the time. The
two models are prompted differently, so their difference cannot be attributed to post-training alone.

## 5. Per-relation extremes (BERT-base, T-REx, repaired templates)

Least negation-sensitive (highest overlap):

| relation   |   rho |   overlap |   delta |   p1_pos |   p1_neg |
|:-----------|------:|----------:|--------:|---------:|---------:|
| P463       | 0.928 |     0.947 |  -0.328 |    0.671 |    0.693 |
| P1376      | 0.961 |     0.915 |  -0.122 |    0.739 |    0.701 |
| P190       | 0.992 |     0.892 |  -0.043 |    0.024 |    0.02  |
| P1001      | 0.94  |     0.887 |  -0.172 |    0.705 |    0.692 |
| P176       | 0.9   |     0.867 |  -0.167 |    0.856 |    0.832 |

Most negation-sensitive (lowest overlap):

| relation   |   rho |   overlap |   delta |   p1_pos |   p1_neg |
|:-----------|------:|----------:|--------:|---------:|---------:|
| P103       | 0.567 |     0     |  -4.734 |    0.722 |    0     |
| P364       | 0.716 |     0.013 |  -4.35  |    0.445 |    0.008 |
| P1303      | 0.739 |     0.027 |   0.307 |    0.076 |    0.009 |
| P19        | 0.893 |     0.132 |  -0.777 |    0.211 |    0.04  |
| P108       | 0.928 |     0.141 |  -1.774 |    0.068 |    0.052 |

Correlation between per-relation overlap and Delta: 0.578 (n=41 relations).

Google-RE per relation:

| relation       |   rho |   overlap |   delta |   p1_pos |   p1_neg |
|:---------------|------:|----------:|--------:|---------:|---------:|
| date_of_birth  | 0.712 |     0.011 |  -0.249 |    0.016 |    0.009 |
| place_of_birth | 0.893 |     0.112 |  -0.581 |    0.149 |    0.058 |
| place_of_death | 0.898 |     0.217 |  -1.613 |    0.131 |    0.016 |

## 6. Run metadata

| Model | kind | input pairs / items | restricted vocab | runtime (s) | hardware, precision |
|---|---|---|---|---|---|
| bert-base-cased | mlm | 39567 | 21018 | 505.7 | Apple M1 CPU, float32 |
| bert-large-cased | mlm | 39567 | 21018 | 2824.1 | Apple M1 CPU, float32 |
| EleutherAI/pythia-410m | causal | 34292 | 15800 | 286.6 | Colab T4 GPU, float32 |
| EleutherAI/pythia-1.4b | causal | 34292 | 15800 | 922.2 | Colab T4 GPU, float32 |
| Qwen2.5-1.5B (truth) | causal | 8800 items | – | 787.1 | Colab T4 GPU, float32 |
| Qwen2.5-1.5B-Instruct (truth) | causal | 8800 items | – | 498.0 | Colab T4 GPU, float32 |

All cloze runs use the LAMA common vocabulary, restricted per model to entries that are a single token
under its own tokenizer. Input pairs count all pairs; causal models score only the mask-final ones
(28,764 T-REx + 5,528 Google-RE). The cloze runs involve no sampling: the released and repaired BERT runs,
made independently, agree exactly on all pairs outside P103 and P190.
