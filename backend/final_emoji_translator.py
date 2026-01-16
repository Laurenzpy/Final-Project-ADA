# final_emoji_translator.py
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# Normalization helpers
# ----------------------------
_WS_RE = re.compile(r"\s+")


def normalize_emoji_seq(s: str) -> str:
    """
    Normalize emoji input so TM exact-match works reliably.

    Handles both:
      - "🎵 🐶 🏠" (space separated)
      - "🎵🐶🏠"    (no spaces, best-effort split by codepoints)

    Note: Emoji grapheme clusters can be multi-codepoint, but in our dataset
    most sequences are single-codepoint emojis or already spaced.
    """
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""

    # collapse whitespace
    s = _WS_RE.sub(" ", s).strip()

    # if there are spaces -> treat as tokens
    if " " in s:
        toks = [t for t in s.split(" ") if t]
        return " ".join(toks)

    # otherwise best-effort split into codepoints
    toks = list(s)
    return " ".join([t for t in toks if t.strip()])


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = _WS_RE.sub(" ", s)
    return s


# ----------------------------
# Config + Translator
# ----------------------------
@dataclass
class HybridConfig:
    # Translation Memory from TRAIN ONLY (stage1-4)
    tm_train_paths: List[str]

    # T5 fallback model dir (local)
    t5_model_dir: str = "artifacts/t5_e2t"

    # Retrieval thresholds (cosine sim in TF-IDF space)
    retrieval_high_conf: float = 0.60
    retrieval_low_conf: float = 0.35

    # NEW:
    # Use T5 already when similarity is below this value (if available).
    # If None, defaults to retrieval_low_conf (i.e., old behavior: T5 only under "low").
    #
    # Typical values:
    # - 0.35  -> T5 only for very low similarity (rare)
    # - 0.50  -> T5 often
    # - 0.55  -> T5 very often (recommended if you want it used more)
    # - 0.60  -> T5 for everything not "high" retrieval
    t5_fallback_below_conf: Optional[float] = 0.55

    # Safety: allow disabling T5 completely
    enable_t5_fallback: bool = True

    # Device selection: "auto" | "cpu" | "mps" | "cuda"
    device: str = "auto"

    # generation settings
    max_new_tokens: int = 32
    num_beams: int = 4


class FinalEmojiTranslator:
    """
    Single Source of Truth pipeline:

    1) Build TM from Stage1-4 (emoji->text) ONLY (no leakage)
    2) Inference (Stage5):
       - exact match -> direct
       - retrieval high -> retrieval result
       - else:
           - if similarity < t5_fallback_below_conf: try T5
           - otherwise: retrieval low
       - if similarity < retrieval_low_conf and T5 not used/available: retrieval very_low (grounded)
    """

    def __init__(self, cfg: HybridConfig):
        self.cfg = cfg

        # Load TM pairs
        self.tm_exact: Dict[str, str] = {}
        self._tm_inputs: List[str] = []
        self._tm_outputs: List[str] = []

        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tm_matrix = None  # sparse matrix

        # Lazy-loaded T5
        self._t5_tokenizer = None
        self._t5_model = None
        self._t5_device = None
        self._t5_available = False

        self._load_tm()
        self._build_retrieval_index()
        self._init_t5_if_possible()

    # ----------------------------
    # TM build
    # ----------------------------
    def _load_tm(self) -> None:
        rows = []
        for p in self.cfg.tm_train_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"TM train file not found: {p}")
            df = pd.read_csv(p)
            if "input" not in df.columns or "output" not in df.columns:
                raise ValueError(f"CSV must contain columns input,output: {p}")
            rows.append(df[["input", "output"]])

        df = pd.concat(rows, ignore_index=True)
        df["input"] = df["input"].astype(str).map(normalize_emoji_seq)
        df["output"] = df["output"].astype(str).map(normalize_text)

        grouped = df.groupby("input")["output"].apply(list).reset_index()

        def choose_canonical(outputs: List[str]) -> str:
            outs = [normalize_text(o) for o in outputs if normalize_text(o)]
            if not outs:
                return ""
            counts: Dict[str, int] = {}
            for o in outs:
                counts[o] = counts.get(o, 0) + 1
            mx = max(counts.values())
            cand = [o for o, c in counts.items() if c == mx]
            cand.sort(key=lambda s: (len(s.split()), len(s)))
            return cand[0]

        grouped["output"] = grouped["output"].apply(choose_canonical)
        grouped = grouped[grouped["input"].str.len() > 0]
        grouped = grouped[grouped["output"].str.len() > 0].reset_index(drop=True)

        self.tm_exact = dict(zip(grouped["input"].tolist(), grouped["output"].tolist()))
        self._tm_inputs = grouped["input"].tolist()
        self._tm_outputs = grouped["output"].tolist()

    def _build_retrieval_index(self) -> None:
        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"[^ ]+",
            lowercase=False,
            min_df=1,
        )
        self._tm_matrix = self._vectorizer.fit_transform(self._tm_inputs)

    # ----------------------------
    # T5
    # ----------------------------
    def _resolve_device(self) -> str:
        if self.cfg.device != "auto":
            return self.cfg.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _init_t5_if_possible(self) -> None:
        if not self.cfg.enable_t5_fallback:
            self._t5_available = False
            return
        if not os.path.isdir(self.cfg.t5_model_dir):
            self._t5_available = False
            return

        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        device = self._resolve_device()

        self._t5_tokenizer = AutoTokenizer.from_pretrained(self.cfg.t5_model_dir)
        self._t5_model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.t5_model_dir)

        import torch
        self._t5_device = device
        self._t5_model.to(device)
        self._t5_model.eval()
        self._t5_available = True

    def _t5_generate(self, emoji_in: str) -> str:
        if not self._t5_available:
            return ""
        import torch

        inp = normalize_emoji_seq(emoji_in)
        inputs = self._t5_tokenizer([inp], return_tensors="pt", truncation=True)
        inputs = {k: v.to(self._t5_device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self._t5_model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                num_beams=self.cfg.num_beams,
                early_stopping=True,
            )

        txt = self._t5_tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        return normalize_text(txt)

    # ----------------------------
    # API
    # ----------------------------
    def translate(self, emoji_in: str) -> Dict[str, Any]:
        """
        Returns:
          {
            "input": ...,
            "prediction": ...,
            "mode": "tm_exact" | "tm_retrieval_high" | "tm_retrieval_low" | "tm_retrieval_very_low" | "t5_fallback",
            "retrieval_score": float,
            "retrieval_match_input": str,
            "retrieval_match_output": str,
            "latency_ms": float
          }
        """
        t0 = time.perf_counter()
        norm_in = normalize_emoji_seq(emoji_in)

        # 1) exact match
        if norm_in in self.tm_exact:
            pred = self.tm_exact[norm_in]
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "input": emoji_in,
                "prediction": pred,
                "mode": "tm_exact",
                "retrieval_score": 1.0,
                "retrieval_match_input": norm_in,
                "retrieval_match_output": pred,
                "latency_ms": latency_ms,
            }

        # 2) retrieval
        assert self._vectorizer is not None and self._tm_matrix is not None
        q = self._vectorizer.transform([norm_in])
        sims = cosine_similarity(q, self._tm_matrix).reshape(-1)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_out = self._tm_outputs[best_idx]
        best_inp = self._tm_inputs[best_idx]

        # Determine T5 trigger threshold
        t5_trigger = self.cfg.t5_fallback_below_conf
        if t5_trigger is None:
            t5_trigger = self.cfg.retrieval_low_conf

        # 3) decision
        if best_score >= self.cfg.retrieval_high_conf:
            mode = "tm_retrieval_high"
            pred = best_out
        else:
            # If we want T5 more often: try T5 already below t5_trigger
            if self._t5_available and best_score < float(t5_trigger):
                t5_pred = self._t5_generate(norm_in)
                if t5_pred:
                    mode = "t5_fallback"
                    pred = t5_pred
                else:
                    # grounded fallback
                    if best_score >= self.cfg.retrieval_low_conf:
                        mode = "tm_retrieval_low"
                    else:
                        mode = "tm_retrieval_very_low"
                    pred = best_out
            else:
                # retrieval path
                if best_score >= self.cfg.retrieval_low_conf:
                    mode = "tm_retrieval_low"
                    pred = best_out
                else:
                    mode = "tm_retrieval_very_low"
                    pred = best_out

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "input": emoji_in,
            "prediction": pred,
            "mode": mode,
            "retrieval_score": best_score,
            "retrieval_match_input": best_inp,
            "retrieval_match_output": best_out,
            "latency_ms": latency_ms,
        }