from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PREFIX = "emoji2text: "


def norm(s: str) -> str:
    return " ".join(str(s).strip().split())


@dataclass
class HybridConfig:
    # Training-only memory for retrieval fallback
    tm_train_paths: List[str]

    # local HuggingFace folder (your trained T5)
    t5_model_dir: Optional[str] = None

    enable_t5: bool = True
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"

    max_new_tokens: int = 64
    num_beams: int = 4


class FinalEmojiTranslator:
    """
    FINAL (simple):
      - Primary: T5 generator (trained with PREFIX)
      - Fallback: retrieval (TF-IDF over emoji sequences) using training-only memory
    """

    def __init__(self, cfg: HybridConfig):
        self.cfg = cfg
        self._t5_available = False
        self._load_tm()
        self._init_t5()

    def _load_tm(self) -> None:
        dfs = []
        for p in self.cfg.tm_train_paths:
            if not os.path.exists(p):
                continue
            df = pd.read_csv(p)
            if "input" in df.columns and "output" in df.columns:
                tmp = df[["input", "output"]].astype(str).copy()
                tmp.columns = ["emoji", "text"]
                dfs.append(tmp)

        if not dfs:
            raise RuntimeError(
                "No usable TM data loaded. Expected CSV(s) with columns input,output."
            )

        self.tm = pd.concat(dfs, ignore_index=True)
        self.tm["emoji"] = self.tm["emoji"].fillna("").astype(str).map(norm)
        self.tm["text"] = self.tm["text"].fillna("").astype(str).map(norm)

        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
        self.X = self.vectorizer.fit_transform(self.tm["emoji"])

    def _auto_device(self) -> str:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _init_t5(self) -> None:
        if not self.cfg.enable_t5:
            return
        if not self.cfg.t5_model_dir or not os.path.isdir(self.cfg.t5_model_dir):
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            self.t5_tokenizer = AutoTokenizer.from_pretrained(self.cfg.t5_model_dir)
            self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.t5_model_dir)

            dev = self._auto_device() if self.cfg.device == "auto" else self.cfg.device
            self.device = dev

            self.t5_model.to(self.device)
            self.t5_model.eval()
            self._t5_available = True
        except Exception:
            self._t5_available = False

    def _retrieval(self, emoji_in: str) -> Dict[str, Any]:
        q = self.vectorizer.transform([norm(emoji_in)])
        sims = cosine_similarity(q, self.X)[0]
        idx = int(np.argmax(sims))
        return {
            "best_score": float(sims[idx]),
            "best_inp": self.tm.iloc[idx]["emoji"],
            "best_out": self.tm.iloc[idx]["text"],
        }

    def _t5_generate(self, emoji_in: str) -> Optional[str]:
        if not self._t5_available:
            return None

        import torch

        src = norm(PREFIX + emoji_in)  # ✅ IMPORTANT: prefix like in training
        enc = self.t5_tokenizer(src, return_tensors="pt", truncation=True, max_length=64).to(self.device)
        with torch.no_grad():
            out = self.t5_model.generate(
                **enc,
                max_new_tokens=self.cfg.max_new_tokens,
                num_beams=self.cfg.num_beams,
                do_sample=False,
            )
        return norm(self.t5_tokenizer.decode(out[0], skip_special_tokens=True))

    def translate(self, emoji_in: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        emoji_in = norm(str(emoji_in))

        ret = self._retrieval(emoji_in)
        pred = ""
        mode = ""

        t5_pred = self._t5_generate(emoji_in)
        if t5_pred:
            pred = t5_pred
            mode = "t5"
        else:
            pred = ret["best_out"]
            mode = "retrieval_fallback"

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "input": emoji_in,
            "prediction": pred,
            "mode": mode,
            "retrieval_score": ret["best_score"],
            "retrieval_match_input": ret["best_inp"],
            "retrieval_match_output": ret["best_out"],
            "latency_ms": latency_ms,
        }