#!/usr/bin/env python
"""Check every result number printed in the paper against the committed outputs.

    python verify_numbers.py            # run all checks, exit 1 on any failure
    python verify_numbers.py --scan     # also list decimals in the paper not covered by a check

Each check pairs a number *as printed in paper/negation_paper.tex* with the value recomputed from
the per-item outputs (results*/**/pairs.jsonl.gz, results_truth/**/items.jsonl.gz) or, for bootstrap
intervals and run metadata, from summary.csv / meta.json. A check passes when the recomputed value
rounds to the printed one (tolerance = half a unit in the last printed digit).
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
EXCLUDED = {"P103", "P190"}  # relations hit by the released-template error (docs/DATA_ISSUE.md)
M = ["rho", "overlap", "delta", "p1_pos", "p1_neg"]


def load(path: str) -> pd.DataFrame:
    p = ROOT / path
    if not p.exists() and (ROOT / (path + ".gz")).exists():
        p = ROOT / (path + ".gz")
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt") as f:
        return pd.DataFrame([json.loads(line) for line in f])


def _allow_import_without_torch():
    """verify_numbers.py needs only pandas. The negprobe aggregation functions it reuses
    (exact_match_p1, summarize_truth) do not touch torch or transformers, but their modules import
    them for model scoring; if those packages are absent, stand-ins are registered so the import works."""
    import importlib.util
    import types

    class _Any:
        def __getattr__(self, name):
            return _Any()

        def __call__(self, *a, **k):
            return a[0] if len(a) == 1 and callable(a[0]) and not k else _Any()

    for name in ["torch", "transformers", "tqdm"]:
        if importlib.util.find_spec(name) is None:
            stub = types.ModuleType(name)
            stub.__getattr__ = lambda attr: _Any()
            if name == "tqdm":
                stub.tqdm = lambda it, *a, **k: it
            sys.modules[name] = stub


def macro(d, cols=M):
    return d.groupby("relation")[list(cols)].mean().mean()


def micro(d, cols=M):
    return d[list(cols)].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    args = ap.parse_args()

    P = {  # per-pair outputs
        "bb": load("results/bert-base-cased/pairs.jsonl"),
        "bl": load("results/bert-large-cased/pairs.jsonl"),
        "bb_rel": load("results_released/bert-base-cased/pairs.jsonl"),
        "bl_rel": load("results_released/bert-large-cased/pairs.jsonl"),
        "p410": load("results/EleutherAI_pythia-410m/pairs.jsonl"),
        "p14": load("results/EleutherAI_pythia-1.4b/pairs.jsonl"),
    }
    sys.path.insert(0, str(ROOT))
    _allow_import_without_torch()
    from negprobe.metrics import exact_match_p1  # P@1 = top-ranked candidate is the gold object
    P = {k: exact_match_p1(v) for k, v in P.items()}
    Q = load("results_truth/Qwen_Qwen2.5-1.5B-Instruct/items.jsonl")   # instruction-tuned, chat template
    QB = load("results_truth/Qwen_Qwen2.5-1.5B/items.jsonl")            # base model, four-example prompt
    meta = {m: json.loads((ROOT / f"results/{m}/meta.json").read_text())
            for m in ["bert-base-cased", "EleutherAI_pythia-410m"]}
    summ = {m: pd.read_csv(ROOT / f"results/{m}/summary.csv").set_index("subset")
            for m in ["bert-base-cased", "bert-large-cased"]}

    def sub(k, s):
        return P[k][P[k].subset == s]

    def common(k):
        t = sub(k, "TREx")
        return t[t.mask_final]  # repaired templates: 28,764 pairs, 35 relations (P190 is not mask-final)

    def rel(k, r, col, s="TREx"):
        t = sub(k, s)
        return t[t.relation == r][col].mean()

    checks = []  # (label, number as printed in the paper, recomputed value)

    def c(label, printed, value):
        checks.append((label, str(printed), float(value)))

    # ---------------- Table 1
    table1 = {
        ("BERT-base", "bb", "GoogleRE"): ["0.834", "0.113", "-0.814", "0.099", "0.028"],
        ("BERT-large", "bl", "GoogleRE"): ["0.776", "0.112", "-0.840", "0.105", "0.028"],
        ("Pythia-410m", "p410", "GoogleRE"): ["0.881", "0.259", "-0.748", "0.045", "0.013"],
        ("BERT-base released", "bb_rel", "TREx"): ["0.866", "0.519", "-0.735", "0.311", "0.237"],
        ("BERT-large released", "bl_rel", "TREx"): ["0.832", "0.469", "-1.018", "0.323", "0.251"],
        ("BERT-base repaired", "bb", "TREx"): ["0.887", "0.541", "-0.568", "0.311", "0.237"],
        ("BERT-large repaired", "bl", "TREx"): ["0.852", "0.491", "-0.840", "0.323", "0.251"],
    }
    for (name, k, s), printed in table1.items():
        d = sub(k, s)
        g = macro(d)
        for col, pr in zip(M, printed):
            c(f"Table 1 {name} {s} {col}", pr, g[col])
        c(f"Table 1 {name} {s} N", {"GoogleRE": "5528", "TREx": "34039"}[s], len(d))

    # ---------------- Table 2 (common item set)
    for name, k, printed in [("BERT-base", "bb", ["0.880", "0.517", "-0.491", "0.363", "0.277"]),
                             ("BERT-large", "bl", ["0.842", "0.487", "-0.817", "0.376", "0.292"]),
                             ("Pythia-410m", "p410", ["0.900", "0.452", "-0.083", "0.226", "0.221"]),
                             ("Pythia-1.4b", "p14", ["0.899", "0.455", "-0.016", "0.239", "0.231"])]:
        t = common(k)
        g = macro(t)
        for col, pr in zip(M, printed):
            c(f"Table 2 {name} {col}", pr, g[col])
        c(f"Table 2 {name} n pairs", "28764", len(t))
        c(f"Table 2 {name} n relations", "35", t.relation.nunique())
    ids = [set(zip(common(k).relation, common(k).uid)) for k in ["bb", "bl", "p410", "p14"]]
    c("Table 2 identical item sets for all four models (1 = true)", "1", all(x == ids[0] for x in ids))

    # ---------------- Table 3, Table 4 (truth judgement)
    sys.path.insert(0, str(ROOT))
    from negprobe.truth import summarize_truth  # the function that wrote summary.csv
    S = summarize_truth(Q).set_index("subset")
    SB = summarize_truth(QB).set_index("subset")
    for lab, SS, s, printed in [("Base", SB, "GoogleRE", ["600", "0.640", "0.507", "0.133", "0.327", "0.163"]),
                                ("Base", SB, "TREx", ["8200", "0.749", "0.525", "0.224", "0.378", "0.192"]),
                                ("Instruct", S, "GoogleRE", ["600", "0.653", "0.567", "0.087", "0.433", "0.257"]),
                                ("Instruct", S, "TREx", ["8200", "0.773", "0.622", "0.151", "0.601", "0.314"])]:
        for col, pr in zip(["n_items", "acc_affirmative", "acc_negated", "negation_gap",
                            "pair_consistency", "says_true_rate"], printed):
            c(f"Table 3 {lab} {s} {col}", pr, SS.loc[s, col])
    cellcols = ["acc_affirmative_gold", "acc_affirmative_distractor", "acc_negated_gold", "acc_negated_distractor"]
    for lab, SS, s, printed in [("Instruct", S, "TREx", ["0.690", "0.856", "0.911", "0.332"]),
                                ("Instruct", S, "GoogleRE", ["0.460", "0.847", "0.860", "0.273"]),
                                ("Base", SB, "TREx", ["0.569", "0.929", "0.960", "0.089"])]:
        for cell, pr in zip(cellcols, printed):
            c(f"Table 4 {lab} {s} {cell}", pr, SS.loc[s, cell])
    # caption: for every model and subset, the two "True"-correct cells are the two worst
    for lab, SS in [("Instruct", S), ("Base", SB)]:
        for s in ["TREx", "GoogleRE"]:
            v = SS.loc[s, cellcols].astype(float).sort_values()
            ok = set(v.index[:2]) == {"acc_affirmative_gold", "acc_negated_distractor"}
            c(f"Table 4 caption: True-correct cells are the two worst, {lab} {s} (1 = true)", "1", ok)
    qt = Q[Q.subset == "TREx"]
    sizes = qt.groupby(["polarity", "object_kind"]).size()
    c("Table 4 items per cell (T-REx), max", "2050", sizes.max())
    c("Table 4 items per cell (T-REx), min", "2050", sizes.min())
    qbt = QB[QB.subset == "TREx"]
    c("Table 4 identical statements for base and instruct (1 = true)", "1",
      len(Q) == len(QB) and sorted(Q.statement) == sorted(QB.statement))

    # ---------------- Table 5 (macro vs micro, T-REx, repaired)
    for name, k, printed in [("BERT-base", "bb", ["0.887", "0.883", "0.541", "0.525", "-0.568", "-0.590"]),
                             ("BERT-large", "bl", ["0.852", "0.849", "0.491", "0.479", "-0.840", "-0.906"])]:
        t = sub(k, "TREx")
        ma, mi = macro(t), micro(t)
        vals = [ma.rho, mi.rho, ma.overlap, mi.overlap, ma.delta, mi.delta]
        for lab, pr, v in zip(["rho macro", "rho micro", "overlap macro", "overlap micro",
                               "delta macro", "delta micro"], printed, vals):
            c(f"Table 5 {name} {lab}", pr, v)

    # ---------------- Sections 1 and 4 (data)
    bbt = sub("bb", "TREx")
    c("Sec 4 Google-RE relations", "3", sub("bb", "GoogleRE").relation.nunique())
    c("Sec 4 T-REx relations", "41", bbt.relation.nunique())
    c("Sec 1 all BERT pairs (Google-RE + T-REx)", "39567", len(P["bb"]))
    c("Sec 4 mask-final T-REx pairs", "28764", bbt.mask_final.sum())
    c("Sec 4 mask-final share (%)", "84.5", 100 * bbt.mask_final.mean())
    c("Sec 4 Pythia T-REx pairs scored", "28764", len(sub("p410", "TREx")))
    c("Sec 4 Pythia Google-RE pairs scored (all)", "5528", len(sub("p410", "GoogleRE")))
    for k in ["p410", "p14"]:
        c(f"Fig 3 caption: {k} T-REx relations with mask-final pairs", "35", sub(k, "TREx").relation.nunique())
    c("Sec 4 BERT restricted vocabulary", "21018", meta["bert-base-cased"]["vocab_size"])
    c("Sec 4 Pythia restricted vocabulary", "15800", meta["EleutherAI_pythia-410m"]["vocab_size"])

    # ---------------- Section 5 (largest macro/micro differences, 'up to')
    diffs = {col: [] for col in M}
    for k in ["bb", "bl", "p410", "p14"]:
        for s in ["GoogleRE", "TREx"]:
            d = sub(k, s)
            ma, mi = macro(d), micro(d)
            for col in M:
                diffs[col].append(abs(ma[col] - mi[col]))
    where = {col: [] for col in M}
    for k in ["bb", "bl", "p410", "p14"]:
        for s_ in ["GoogleRE", "TREx"]:
            d = sub(k, s_)
            for col in M:
                where[col].append((abs(macro(d)[col] - micro(d)[col]), s_))
    for col in ["overlap", "delta"]:
        c(f"Sec 5 largest |macro - micro| {col} is on Google-RE (1 = true)", "1", max(where[col])[1] == "GoogleRE")
    c("Sec 5 max |macro - micro| overlap ('up to 0.05')", "0.05", max(diffs["overlap"]))
    c("Sec 5 max |macro - micro| delta ('up to 0.27')", "0.27", max(diffs["delta"]))

    # ---------------- Sections 6.1 and 6.2
    bbr, blr, blt = sub("bb_rel", "TREx"), sub("bl_rel", "TREx"), sub("bl", "TREx")
    c("Sec 6.1 BERT-base overlap, released (%)", "51.9", 100 * macro(bbr).overlap)
    c("Sec 6.1 BERT-base rho, released", "0.866", macro(bbr).rho)
    c("Sec 6.1 BERT-large overlap, released (%)", "46.9", 100 * macro(blr).overlap)
    c("Sec 6.1 BERT-large rho, released", "0.832", macro(blr).rho)
    c("Sec 6.1 BERT-base overlap, repaired (%)", "54.1", 100 * macro(bbt).overlap)
    c("Sec 6.1 BERT-large overlap, repaired (%)", "49.1", 100 * macro(blt).overlap)
    c("Sec 6.2 overlap drop base -> large (points)", "5", 100 * (macro(bbt).overlap - macro(blt).overlap))
    for m, lo, hi in [("bert-base-cased", "-0.606", "-0.573"), ("bert-large-cased", "-0.934", "-0.879")]:
        ci = json.loads(summ[m].loc["TREx", "delta_micro_ci95"])
        c(f"Sec 6.2 {m} delta 95% CI, low", lo, ci[0])
        c(f"Sec 6.2 {m} delta 95% CI, high", hi, ci[1])
    c("Sec 6.2 BERT-base micro delta", "-0.590", micro(bbt).delta)
    c("Sec 6.2 BERT-large micro delta", "-0.906", micro(blt).delta)
    c("Sec 6.2 BERT-base P36 delta", "-3.35", rel("bb", "P36", "delta"))
    c("Sec 6.2 BERT-large P36 delta", "-9.98", rel("bl", "P36", "delta"))
    c("Sec 6.2 BERT-base macro delta without P36", "-0.498", macro(bbt[bbt.relation != "P36"]).delta)
    c("Sec 6.2 BERT-large macro delta without P36", "-0.611", macro(blt[blt.relation != "P36"]).delta)
    c("Sec 6.2 BERT-base median pair delta", "-0.199", bbt.delta.median())
    c("Sec 6.2 BERT-large median pair delta", "-0.185", blt.delta.median())
    r = bbt.groupby("relation")[["overlap", "delta"]].mean()
    c("Sec 6.2 corr(overlap, delta) over 41 relations", "0.578", r.overlap.corr(r.delta))
    for rr, ov, dl in [("P463", "0.947", None), ("P1376", "0.915", None),
                       ("P103", "0.000", "-4.73"), ("P364", "0.013", "-4.35")]:
        c(f"Sec 6.2 {rr} overlap", ov, rel("bb", rr, "overlap"))
        if dl:
            c(f"Sec 6.2 {rr} delta", dl, rel("bb", rr, "delta"))
    c("Sec 6.2 P103 P@1 affirmative (%)", "72.2", 100 * rel("bb", "P103", "p1_pos"))
    c("Sec 6.2 P103 P@1 negated (%)", "0.0", 100 * rel("bb", "P103", "p1_neg"))
    for rr, ov in [("date_of_birth", "0.011"), ("place_of_birth", "0.112"), ("place_of_death", "0.217")]:
        c(f"Sec 6.2 Google-RE {rr} overlap", ov, rel("bb", rr, "overlap", "GoogleRE"))

    # ---------------- Section 6.3 (template error)
    for rr, ov, dl in [("P103", "0.000", "-6.54"), ("P190", "0.000", "-5.10")]:
        c(f"Sec 6.3 released {rr} overlap", ov, rel("bb_rel", rr, "overlap"))
        c(f"Sec 6.3 released {rr} delta", dl, rel("bb_rel", rr, "delta"))
    top2 = set(bbr.groupby("relation").delta.mean().sort_values().index[:2])
    c("Sec 6.3 P103 and P190 are the two largest released effects (1 = true)", "1", top2 == EXCLUDED)
    c("Sec 6.3 overlap gain from the repair, base ('about two points')", "2",
      100 * (macro(bbt).overlap - macro(bbr).overlap))
    c("Sec 6.3 overlap gain from the repair, large ('about two points')", "2",
      100 * (macro(blt).overlap - macro(blr).overlap))
    for name, rep_, rel_ in [("base", bbt, bbr), ("large", blt, blr)]:
        share = 1 - macro(rep_).delta / macro(rel_).delta
        c(f"Sec 6.3 share of released delta due to P103/P190, {name} ('roughly a fifth', 0.15-0.25)",
          "0.2", 0.2 if 0.15 <= share <= 0.25 else share)
    c("Sec 6.3 repaired P190 overlap", "0.892", rel("bb", "P190", "overlap"))
    c("Sec 6.3 repaired P190 delta", "-0.04", rel("bb", "P190", "delta"))
    ranks = bbt.groupby("relation").overlap.mean().rank(ascending=False)
    c("Sec 6.3 P190 rank by overlap among 41 relations ('one of the least sensitive')", "3", ranks["P190"])

    # ---------------- Sections 6.5 and 6.6
    for k, pr in [("bb", "-0.407"), ("bl", "-0.548")]:
        t = common(k)
        c(f"Sec 6.6 {k} common-set delta without P36", pr, macro(t[t.relation != "P36"]).delta)
    for k, pr in [("p410", "0.001"), ("p14", "0.026")]:
        t = common(k)
        c(f"Sec 6.6 {k} common-set delta without P36", pr, macro(t[t.relation != "P36"]).delta)
    c("Sec 6.6 Pythia Google-RE delta, 410m", "-0.748", macro(sub("p410", "GoogleRE")).delta)
    c("Sec 6.6 Pythia Google-RE delta, 1.4b", "-1.180", macro(sub("p14", "GoogleRE")).delta)

    # ---------------- Section 6.7 (instruct = Q, base = QB; T-REx)
    from statistics import NormalDist
    z = NormalDist().inv_cdf
    c("Sec 6.7 T-REx negated accuracy (%)", "62.2", 100 * S.loc["TREx", "acc_negated"])
    c("Sec 6.7 T-REx negation gap (points)", "15.1", 100 * S.loc["TREx", "negation_gap"])
    c("Sec 6.7 pair consistency", "0.601", S.loc["TREx", "pair_consistency"])
    c("Sec 6.7 same verdict for statement and negation ('about two fifths', 0.35-0.45)", "0.4",
      0.4 if 0.35 <= 1 - S.loc["TREx", "pair_consistency"] <= 0.45 else 1 - S.loc["TREx", "pair_consistency"])
    c("Sec 6.7 negated accuracy on known facts", "0.660", S.loc["TREx", "acc_negated_known_facts"])
    c("Sec 6.7 overall accuracy", "0.697", qt.correct.mean())
    c("Sec 6.7 says True overall, T-REx (%)", "31.4", 100 * qt.said_true.mean())
    tr = qt.groupby(["polarity", "object_kind"]).said_true.mean()
    c("Sec 6.7 says True, affirmative gold (%)", "69.0", 100 * tr[("affirmative", "gold")])
    c("Sec 6.7 says True, affirmative distractor (%)", "14.4", 100 * tr[("affirmative", "distractor")])
    c("Sec 6.7 affirmative margin (points)", "55",
      100 * (tr[("affirmative", "gold")] - tr[("affirmative", "distractor")]))
    c("Sec 6.7 says True, negated distractor (%)", "33.2", 100 * tr[("negated", "distractor")])
    c("Sec 6.7 says True, negated gold (%)", "8.9", 100 * tr[("negated", "gold")])
    c("Sec 6.7 negated margin (points)", "24",
      100 * (tr[("negated", "distractor")] - tr[("negated", "gold")]))
    neg_acc_identity = 0.5 + 0.5 * (tr[("negated", "distractor")] - tr[("negated", "gold")])
    c("Sec 6.7 negated accuracy = 0.5 + half the negated margin", "0.622", neg_acc_identity)
    tp = qt.groupby("polarity").said_true.mean()
    c("Sec 6.7 says True, affirmative statements (%)", "42", 100 * tp["affirmative"])
    c("Sec 6.7 says True, negated statements (%)", "21", 100 * tp["negated"])
    da = z(tr[("affirmative", "gold")]) - z(tr[("affirmative", "distractor")])
    dn = z(tr[("negated", "distractor")]) - z(tr[("negated", "gold")])
    c("Sec 6.7 d' affirmative", "1.56", da)
    c("Sec 6.7 d' negated", "0.91", dn)
    c("Sec 6.7 d' ratio ('about three fifths', 0.55-0.65)", "0.6", 0.6 if 0.55 <= dn / da <= 0.65 else dn / da)
    tb = qbt.groupby(["polarity", "object_kind"]).said_true.mean()
    c("Sec 6.7 base says True, affirmative gold (%)", "56.9", 100 * tb[("affirmative", "gold")])
    c("Sec 6.7 base says True, affirmative distractor (%)", "7.1", 100 * tb[("affirmative", "distractor")])
    c("Sec 6.7 base affirmative margin (points)", "50",
      100 * (tb[("affirmative", "gold")] - tb[("affirmative", "distractor")]))
    c("Sec 6.7 base says True, negated distractor (%)", "8.9", 100 * tb[("negated", "distractor")])
    c("Sec 6.7 base says True, negated gold (%)", "4.0", 100 * tb[("negated", "gold")])
    c("Sec 6.7 base negated margin (points)", "5",
      100 * (tb[("negated", "distractor")] - tb[("negated", "gold")]))
    c("Sec 6.7 base d' affirmative", "1.64", z(tb[("affirmative", "gold")]) - z(tb[("affirmative", "distractor")]))
    c("Sec 6.7 base d' negated", "0.41", z(tb[("negated", "distractor")]) - z(tb[("negated", "gold")]))
    c("Sec 6.7 base negated accuracy", "0.525", SB.loc["TREx", "acc_negated"])
    c("Sec 6.7 base pair consistency", "0.378", SB.loc["TREx", "pair_consistency"])
    c("Abstract base negated accuracy (%)", "52.5", 100 * SB.loc["TREx", "acc_negated"])

    # ---------------- Section 7 (two independent BERT runs agree on unaffected pairs)
    for k_rep, k_rel in [("bb", "bb_rel"), ("bl", "bl_rel")]:
        x = P[k_rep].merge(P[k_rel], on=["subset", "relation", "uid", "obj"], suffixes=("", "_r"))
        x = x[~x.relation.isin(EXCLUDED)]
        c(f"Sec 7 {k_rep}: max |difference| between the two runs on unaffected pairs", "0",
          max((x[col] - x[col + "_r"]).abs().max() for col in M))
        c(f"Sec 7 {k_rep}: pairs compared", str(len(x)), len(x))

    # ---------------- report
    fails = 0
    for label, printed, value in checks:
        dec = len(printed.split(".")[1]) if "." in printed else 0
        tol = 0.5 * 10 ** (-dec) + 1e-9
        ok = abs(value - float(printed)) <= tol
        fails += not ok
        print(f"{'OK  ' if ok else 'FAIL'} | {label:<86} paper {printed:>7}  data {value:10.4f}")
    print(f"\n{len(checks) - fails} passed, {fails} failed, {len(checks)} checks")

    if args.scan:
        tex = (ROOT / "paper/negation_paper.tex").read_text()
        body = tex[tex.index("\\begin{abstract}"):tex.index("\\begin{thebibliography}")]
        body = re.sub(r"(?<!\\)%.*", "", body)
        covered = {p.lstrip("-") for _, p, _ in checks}
        nums = set(re.findall(r"(?<![\w.])-?\d+\.\d+", body))
        missing = sorted({n.lstrip("-") for n in nums} - covered)
        print("\nDecimals in the paper without a check:", missing or "none")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
