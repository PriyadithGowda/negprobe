# An error in the released negated-LAMA templates

## Summary

`relations.jsonl` in the negated LAMA release (`negated_data.tar.gz`) assigns the **wrong negated
template to two of the 41 T-REx relations**. For those relations the "negated" sentence is a different
statement, not a negation of the affirmative one, so any negation effect measured on them is spurious.

## Evidence

Printing the file in order shows a one-row shift at lines 37–39 (numbered from 1):

```
36 P127   owned by                     T: [X] is owned by [Y] .               N: [X] is not owned by [Y] .
37 P103   native language              T: The native language of [X] is [Y] . N: [X] is not owned by [Y] .        <- P127's
38 P190   twinned administrative body  T: [X] and [Y] are twin cities .       N: The native language of [X] is not [Y] .  <- P103's
```

P103 receives P127's negated template; P190 receives P103's. P190 has no correct negated template in the
file at all. No other relation is affected: a token-overlap check between each affirmative template and
its negated counterpart flags exactly these two.

## Effect on results

BERT-base, T-REx, all 41 relations, macro averages:

| templates | rho | overlap | Delta |
|---|---|---|---|
| as released | 0.866 | 0.519 | −0.735 |
| repaired | 0.887 | 0.541 | −0.568 |

Roughly a fifth of the measured negation effect in the released configuration comes from these two
relations. As released they show the largest effects in the whole probe (overlap 0.000 with Delta of
−6.54 and −5.10), which is what a model does when asked to complete an unrelated sentence.

Once repaired the two relations separate cleanly, and in opposite directions:

| relation | overlap | Delta | reading |
|---|---|---|---|
| P190 (twin cities) | 0.892 | −0.04 | among the least negation-sensitive relations; the released figure was entirely an artefact |
| P103 (native language) | 0.000 | −4.73 | genuinely negation-sensitive: 72.2 % correct affirmative, 0.0 % negated |

So the released data was wrong in two ways at once: it invented an effect for P190 and it exaggerated
a real one for P103 (Delta −6.54 as released against −4.73 repaired). Excluding the relations would have discarded a true finding; only a re-run with
correct templates distinguishes the cases.

## Fix

`negprobe/data.py`:

```python
TEMPLATE_FIXES = {
    "P103": "The native language of [X] is not [Y] .",
    "P190": "[X] and [Y] are not twin cities .",
}
```

`check_templates()` compares every negated template with its affirmative counterpart and warns on a
mismatch. It flags exactly P103 and P190 before the fix and nothing after. The fix is applied by default;
set `negprobe.data.TEMPLATE_FIXES = {}` to reproduce the released behaviour (as `reproduce_all.sh` does).

## Scope of the claim

We verified the error in the copy distributed as `negated_data.tar.gz`
(`https://dl.fbaipublicfiles.com/LAMA/negated_data.tar.gz`, retrieved September 2026); the
`relations.jsonl` we used has SHA-256
`154be499a67d5a681bdeaff3bce578a64064c6ce73e471523c6423071e3e5298`. The per-pair outputs of the
released-template BERT runs are committed in `results_released/`, and `verify_numbers.py` recomputes
every figure on this page from them. We have **not**
surveyed which published results depend on it, and make no claim about that.
