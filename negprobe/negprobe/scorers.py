"""Model wrappers that return, for a batch of cloze sentences:

* the log-probability distribution over the (restricted) vocabulary at the
  [MASK] position  -> used for Spearman rho and rank-1 overlap (Experiment A)
* the log-probability of the gold object at that position
  -> used for the negation shift Delta (Experiment B)

Masked LMs (BERT, RoBERTa, ...) predict the [MASK] token directly.
Causal LMs (Pythia, Llama, Qwen, ...) predict the token after the text that
precedes [MASK]; only sentences where [MASK] is the last content token are
valid for them (see ``Pair.mask_final``).
"""
from __future__ import annotations

import math

import torch
from transformers import (AutoModelForCausalLM, AutoModelForMaskedLM,
                          AutoTokenizer)

from .data import MASK


def _check_finite(logits, model_name: str):
    """A checkpoint stored in fp16 overflows to NaN on CPU/MPS; fail loudly instead
    of writing a file full of meaningless ranks."""
    if not torch.isfinite(logits).all():
        raise RuntimeError(
            f"{model_name}: the model produced NaN/inf logits. This usually means a "
            "half-precision checkpoint is running on CPU or MPS. Re-run with --dtype fp32.")


def _dtype(name: str):
    return {"auto": "auto", "fp32": torch.float32, "fp16": torch.float16,
            "bf16": torch.bfloat16}[name]


class _Base:
    is_causal = False

    def __init__(self, model_name, device="auto", dtype="auto", vocab=None,
                 load_in_4bit=False, trust_remote_code=False):
        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        kwargs = dict(trust_remote_code=trust_remote_code)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            kwargs["device_map"] = "auto"
        else:
            # "auto" would keep a checkpoint stored in fp16, which overflows to NaN
            # on CPU and MPS. Only use half precision when there is a CUDA GPU.
            if dtype == "auto" and not torch.cuda.is_available():
                dtype = "fp32"
            kwargs["dtype"] = _dtype(dtype)
            if device == "auto":
                kwargs["device_map"] = "auto" if torch.cuda.is_available() else None
        try:
            self.model = self._model_cls().from_pretrained(model_name, **kwargs).eval()
        except TypeError:  # transformers < 4.56 spells it torch_dtype
            if "dtype" in kwargs:
                kwargs["torch_dtype"] = kwargs.pop("dtype")
            self.model = self._model_cls().from_pretrained(model_name, **kwargs).eval()
        if not load_in_4bit and device not in ("auto",):
            self.model.to(device)
        self.device = next(self.model.parameters()).device
        self.lowercase = bool(getattr(self.tok, "do_lower_case", False))
        self.index_list = self._build_index_list(vocab)

    # ---- vocabulary -------------------------------------------------------
    def _word_ids(self, word: str) -> list[int]:
        raise NotImplementedError

    def _build_index_list(self, vocab):
        """Token ids the distribution is restricted to.

        With a LAMA common vocab: ids of the words that are a single token for
        this model (words that are not are dropped, as in LAMA).
        Without: the whole vocabulary minus special tokens.
        """
        if vocab:
            ids, seen = [], set()
            for w in vocab:
                t = self._word_ids(w.lower() if self.lowercase else w)
                if len(t) == 1 and t[0] not in seen:
                    seen.add(t[0])
                    ids.append(t[0])
            idx = torch.tensor(ids)
        else:
            special = set(self.tok.all_special_ids)
            n = self.model.get_output_embeddings().weight.shape[0]
            idx = torch.tensor([i for i in range(n) if i not in special])
        self._pos_in_index = {int(t): i for i, t in enumerate(idx)}
        return idx.to(self.device)

    def decode(self, token_id: int) -> str:
        return self.tok.decode([token_id]).strip()

    def gold_ids(self, obj: str) -> list[int]:
        return self._word_ids(obj.lower() if self.lowercase else obj)

    def restricted_position(self, token_id: int):
        """Position of ``token_id`` in the restricted distribution, or None."""
        return self._pos_in_index.get(int(token_id))

    def _prep(self, s: str) -> str:
        return s.lower().replace(MASK.lower(), MASK) if self.lowercase else s


class MaskedLMScorer(_Base):
    def _model_cls(self):
        return AutoModelForMaskedLM

    def _word_ids(self, word):
        # RoBERTa-style BPE marks a preceding space inside the token
        if self.tok.mask_token == "<mask>":
            return self.tok.encode(" " + word, add_special_tokens=False)
        return self.tok.encode(word, add_special_tokens=False)

    @torch.no_grad()
    def score(self, sentences: list[str], objs: list[str]):
        """Returns (restricted log-prob matrix [B, V'], gold log-probs [B])."""
        texts = []
        for s in sentences:
            s = self._prep(s)
            head, tail = s.split(MASK, 1)
            tail = tail.replace(MASK, self.tok.mask_token)  # later masks stay masked
            if self.tok.mask_token == "<mask>":
                head = head.rstrip()  # "<mask>" absorbs the preceding space
                texts.append(head + self.tok.mask_token + tail)
            else:
                texts.append(head + self.tok.mask_token + tail)
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=512).to(self.device)
        logits = self.model(**enc).logits
        mask_pos = (enc["input_ids"] == self.tok.mask_token_id).int().argmax(dim=1)
        rows = logits[torch.arange(len(texts), device=self.device), mask_pos]
        _check_finite(rows, self.name)
        logp = torch.log_softmax(rows.float(), dim=-1)

        gold = torch.full((len(texts),), float("nan"))
        for i, o in enumerate(objs):
            ids = self.gold_ids(o)
            if len(ids) == 1:  # multi-token objects are skipped, as in LAMA
                gold[i] = logp[i, ids[0]].item()
        return logp[:, self.index_list], gold


class CausalLMScorer(_Base):
    is_causal = True

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def _model_cls(self):
        return AutoModelForCausalLM

    def _word_ids(self, word):
        return self.tok.encode(" " + word, add_special_tokens=False)

    def _prefix(self, s):
        return self._prep(s).split(MASK, 1)[0].rstrip()

    @torch.no_grad()
    def score(self, sentences: list[str], objs: list[str]):
        """Next-token distribution after the prefix, and the summed
        log-probability of the full (possibly multi-token) gold object."""
        prefixes = [self._prefix(s) for s in sentences]
        bos = [self.tok.bos_token_id] if self.tok.bos_token_id is not None else []
        seqs, starts, gold_lens = [], [], []
        for p, o in zip(prefixes, objs):
            p_ids = bos + self.tok.encode(p, add_special_tokens=False)
            o_ids = self.gold_ids(o)
            seqs.append(p_ids + o_ids)
            starts.append(len(p_ids))
            gold_lens.append(len(o_ids))

        # left-pad by hand so position bookkeeping stays simple
        L = max(map(len, seqs))
        pad = self.tok.pad_token_id
        input_ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        attn = torch.zeros((len(seqs), L), dtype=torch.long)
        offsets = []
        for i, s in enumerate(seqs):
            off = L - len(s)
            input_ids[i, off:] = torch.tensor(s)
            attn[i, off:] = 1
            offsets.append(off)
        input_ids, attn = input_ids.to(self.device), attn.to(self.device)
        # with left padding, positions must be counted from the first real token
        position_ids = (attn.cumsum(-1) - 1).clamp_min(0)
        try:
            logits = self.model(input_ids=input_ids, attention_mask=attn,
                                position_ids=position_ids).logits.float()
        except TypeError:  # a few architectures do not take position_ids
            logits = self.model(input_ids=input_ids, attention_mask=attn).logits.float()
        _check_finite(logits, self.name)
        logp_all = torch.log_softmax(logits, dim=-1)

        next_rows, gold = [], torch.zeros(len(seqs))
        for i in range(len(seqs)):
            first = offsets[i] + starts[i]           # position of first gold token
            next_rows.append(logp_all[i, first - 1])  # prediction for that position
            lp = 0.0
            for k in range(gold_lens[i]):
                tok_id = input_ids[i, first + k]
                lp += logp_all[i, first + k - 1, tok_id].item()
            gold[i] = lp if gold_lens[i] else math.nan
        next_rows = torch.stack(next_rows)
        return next_rows[:, self.index_list], gold


def load_scorer(model_name: str, kind: str = "auto", **kw):
    """``kind``: 'mlm', 'causal' or 'auto' (decided from the model config)."""
    if kind == "auto":
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=kw.get("trust_remote_code", False))
        archs = " ".join(cfg.architectures or []).lower()
        kind = "mlm" if ("masked" in archs or cfg.model_type in
                         {"bert", "roberta", "distilbert", "albert", "electra",
                          "xlm-roberta", "deberta", "deberta-v2", "modernbert"}) else "causal"
    return (MaskedLMScorer if kind == "mlm" else CausalLMScorer)(model_name, **kw)
