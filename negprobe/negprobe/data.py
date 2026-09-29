"""Load negated LAMA probes into one flat list of affirmative/negated pairs.

Two input layouts are supported:

1. Facebook's full negated LAMA release (``negated_data.tar.gz``):
   Google_RE / ConceptNet / Squad samples carry a ``negated`` field;
   T-REx samples are negated with ``template_negated`` from ``relations.jsonl``.
2. Kassner's LAMA_primed_negated repo (``data/<Subset>/<variant>/*.jsonl``):
   every sample carries ``masked_negations``. The negations are identical
   across the three misprime variants, so only one variant is read.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path

MASK = "[MASK]"

# Everything after the mask must be punctuation for a causal LM to predict it
# as the next token (e.g. "... born in [MASK] ." or "(born [MASK]).").
_TAIL_OK = re.compile(r"^[\s\.\)\],;:!?\"']*$")


@dataclass
class Pair:
    subset: str
    relation: str
    uid: str
    sub: str
    obj: str
    pos: str  # affirmative sentence containing exactly one [MASK]
    neg: str  # negated sentence containing exactly one [MASK]

    @property
    def mask_final(self) -> bool:
        """True if [MASK] is the last content token in both sentences."""
        return all(_TAIL_OK.match(s.split(MASK, 1)[1]) for s in (self.pos, self.neg))

    def to_dict(self):
        d = asdict(self)
        d["mask_final"] = self.mask_final
        return d


def _join(sentences) -> str:
    if isinstance(sentences, str):
        return sentences.strip()
    return " ".join(s.strip() for s in sentences)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _fill(template: str, sub: str) -> str:
    return template.replace("[X]", sub).replace("[Y]", MASK)


# The released relations.jsonl has an off-by-one error: P103 carries P127's
# negated template and P190 carries P103's, so for these two relations the
# "negated" sentence is a different statement rather than a negation.
# Detected by comparing each template with its negated counterpart; see
# check_templates(). The corrected templates are used unless the environment variable
# NEGPROBE_RELEASED_TEMPLATES=1 is set, which reproduces the released behaviour.
TEMPLATE_FIXES = {
    "P103": "The native language of [X] is not [Y] .",
    "P190": "[X] and [Y] are not twin cities .",
}
if os.environ.get("NEGPROBE_RELEASED_TEMPLATES") == "1":
    TEMPLATE_FIXES = {}

# Google-RE has no negated field and no relations.jsonl entry; these are the
# templates hard-coded in LAMA's scripts/run_experiments.py.
GOOGLE_RE_TEMPLATES = {
    "place_of_birth": ("[X] was born in [Y] .", "[X] was not born in [Y] ."),
    "date_of_birth": ("[X] (born [Y]).", "[X] (not born [Y])."),
    "place_of_death": ("[X] died in [Y] .", "[X] did not die in [Y] ."),
}


def _canonical_subset(name: str) -> str:
    n = name.lower().replace("-", "").replace("_", "")
    if n.startswith("trex"):
        return "TREx"
    if n.startswith("googlere"):
        return "GoogleRE"
    if n.startswith("conceptnet"):
        return "ConceptNet"
    if n.startswith("squad"):
        return "SQuAD"
    return name


def _read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _make_pair(subset, relation, sample, pos, neg, idx) -> Pair | None:
    if not pos or not neg:
        return None
    pos, neg = _clean(pos), _clean(neg)
    # keep only the first mask, as LAMA does ("score only first mask")
    if pos.count(MASK) < 1 or neg.count(MASK) < 1:
        return None
    uid = str(sample.get("uuid", sample.get("id", idx)))
    return Pair(
        subset=subset,
        relation=relation,
        uid=uid,
        sub=str(sample.get("sub_label", "")),
        obj=str(sample["obj_label"]).strip(),
        pos=pos,
        neg=neg,
    )


def load_kassner_repo(root: Path, variant: str = "random") -> list[Pair]:
    """Read ``LAMA_primed_negated/data``."""
    pairs = []
    for subset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        vdir = subset_dir / variant
        if not vdir.exists():
            continue
        subset = _canonical_subset(subset_dir.name)
        for f in sorted(vdir.glob("*.jsonl")):
            for i, s in enumerate(_read_jsonl(f)):
                relation = s.get("pred", f.stem)
                p = _make_pair(subset, relation, s,
                               _join(s["masked_sentences"]),
                               _join(s.get("masked_negations", [])), i)
                if p:
                    pairs.append(p)
    return pairs


def check_templates(templates: dict, verbose: bool = True) -> list[str]:
    """Flag relations whose negated template is not the affirmative one plus a negation.

    Returns the list of suspicious relation ids. Uses token overlap, so it catches
    a swapped template but not a subtly reworded one.
    """
    def strip_neg(t):
        t = re.sub(r"\b(not|never|n't)\b", " ", t)
        t = re.sub(r"\b(does|did|do)\b", " ", t)
        return set(re.sub(r"\s+", " ", t).lower().strip(" .").split())

    bad = []
    for rel, r in templates.items():
        a, b = strip_neg(r["template"]), strip_neg(r["template_negated"])
        if len(a | b) and len(a & b) / len(a | b) < 0.5:
            bad.append(rel)
    if bad and verbose:
        print(f"warning: negated template looks wrong for {sorted(bad)} "
              f"(known issue in the released relations.jsonl)")
    return bad


def load_facebook_release(root: Path, fix_templates: bool = True) -> list[Pair]:
    """Read the extracted ``negated_data.tar.gz`` (folder with one dir per subset).

    The layout is detected loosely, so small differences in folder names do
    not matter. T-REx needs a ``relations.jsonl`` somewhere under ``root``.
    """
    templates = {}
    for rel_file in root.rglob("relations.jsonl"):
        for r in _read_jsonl(rel_file):
            if r.get("template") and r.get("template_negated"):
                templates[r["relation"]] = dict(r)
    check_templates(templates)
    if fix_templates:
        for rel, fixed in TEMPLATE_FIXES.items():
            if rel in templates:
                templates[rel]["template_negated"] = fixed

    pairs = []
    for f in sorted(root.rglob("*.jsonl")):
        if f.name == "relations.jsonl":
            continue
        top = f.relative_to(root).parts[0]
        subset = _canonical_subset(top)
        relation = f.stem.replace("_test", "")
        for i, s in enumerate(_read_jsonl(f)):
            pos = _join(s.get("masked_sentences", []))
            rel_id = s.get("predicate_id", relation)
            if "negated" in s and s["negated"]:
                # ConceptNet and SQuAD carry the negated sentence directly
                neg = _join(s["negated"])
            elif "masked_negations" in s:
                neg = _join(s["masked_negations"])
            elif rel_id in templates and s.get("sub_label"):
                # T-REx: build both sides from the relation templates so that
                # they differ only in the negation (as LAMA does)
                pos = _fill(templates[rel_id]["template"], s["sub_label"])
                neg = _fill(templates[rel_id]["template_negated"], s["sub_label"])
            elif relation in GOOGLE_RE_TEMPLATES and s.get("sub_label"):
                pos_t, neg_t = GOOGLE_RE_TEMPLATES[relation]
                pos, neg = _fill(pos_t, s["sub_label"]), _fill(neg_t, s["sub_label"])
            else:
                continue
            relation_name = s.get("pred", relation)
            if str(relation_name).startswith("/"):  # Google-RE Freebase ids
                relation_name = relation
            p = _make_pair(subset, relation_name, s, pos, neg, i)
            if p:
                pairs.append(p)
    return pairs


def load_pairs(path: str | Path, subsets: list[str] | None = None,
               trex_templates: str | Path | None = None,
               fix_templates: bool = True) -> list[Pair]:
    """Auto-detect the layout and return all pairs.

    ``trex_templates``: optional LAMA ``relations.jsonl`` (affirmative + negated
    templates). When given, T-REx pairs are rebuilt from the templates so that
    affirmative and negated sentences differ only in the negation.
    """
    root = Path(path)
    is_kassner = any((d / "random").is_dir() for d in root.iterdir() if d.is_dir())
    pairs = (load_kassner_repo(root) if is_kassner
             else load_facebook_release(root, fix_templates=fix_templates))

    if trex_templates:
        tmpl = {r["relation"]: dict(r) for r in _read_jsonl(Path(trex_templates))}
        if fix_templates:
            for rel, fixed in TEMPLATE_FIXES.items():
                if rel in tmpl:
                    tmpl[rel]["template_negated"] = fixed
        for p in pairs:
            r = tmpl.get(p.relation)
            if p.subset == "TREx" and r and r.get("template_negated") and p.sub:
                p.pos = _fill(r["template"], p.sub)
                p.neg = _fill(r["template_negated"], p.sub)

    if subsets:
        wanted = {_canonical_subset(s) for s in subsets}
        pairs = [p for p in pairs if p.subset in wanted]

    # de-duplicate (same relation + uid)
    seen, out = set(), []
    for p in pairs:
        key = (p.subset, p.relation, p.uid, p.pos, p.neg)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def load_vocab(path: str | Path | None) -> list[str] | None:
    """LAMA common vocabulary (``common_vocab_cased.txt``), one token per line."""
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return [w.strip() for w in f if w.strip()]
