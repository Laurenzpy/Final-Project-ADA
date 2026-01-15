import os
import glob
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import requests


@dataclass
class RetrievalHit:
    score: float
    emoji_input: str
    meaning: str


class EmojiTranslator:
    """
    Unified pipeline (single system), leakage-safe:

      Stage 0: Normalize emoji sequence.
      Stage 1: Exact match from translation memory (built ONLY from stage1-4).
      Stage 2: Similarity retrieval (TF-IDF over emoji tokens). If score >= threshold -> return.
      Stage 3: Retrieval (low-confidence): if retrieval exists but < threshold -> still return best meaning
               (avoids degenerate T5 "prompt echo" behavior).
      Stage 4: T5 fallback: only if retrieval yields no usable meaning.
      Stage 5: Optional LLM fallback (Ollama) if enabled and output looks degenerate.

    Rationale:
      Your fine-tuned T5 is not instruction-following; it tends to copy context lists when prompted.
      So we keep retrieval as the grounded stage, and use T5 only as last-resort fallback.
    """

    def __init__(
        self,
        model_dir: str = "artifacts/t5_e2t",
        memory_glob: str = "data/emoji_dataset_stage[1-4]_e2t.csv",
        max_input_len: int = 64,
        max_new_tokens: int = 32,
        retrieval_threshold: float = 0.70,
        retrieval_topk: int = 5,
        device: Optional[str] = None,
        enable_ollama_fallback: bool = False,
        ollama_url: str = "http://localhost:11434/api/generate",
        ollama_model: str = "llama3",
    ):
        self.model_dir = model_dir
        self.memory_glob = memory_glob
        self.max_input_len = max_input_len
        self.max_new_tokens = max_new_tokens

        self.retrieval_threshold = retrieval_threshold
        self.retrieval_topk = retrieval_topk

        self.enable_ollama_fallback = enable_ollama_fallback
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

        self._load_translation_memory()
        self._build_retrieval_index()
        self._load_t5(device=device)

    # -------------------------
    # Stage 0: normalization
    # -------------------------
    @staticmethod
    def normalize_emoji_seq(s: str) -> str:
        s = str(s).strip()
        s = " ".join(s.split())
        return s

    # -------------------------
    # Stage 1: exact match memory
    # -------------------------
    def _load_translation_memory(self) -> None:
        files = sorted(glob.glob(self.memory_glob))
        if not files:
            raise FileNotFoundError(f"No memory files found for glob: {self.memory_glob}")

        dfs = []
        for f in files:
            df = pd.read_csv(f)
            if "input" not in df.columns or "output" not in df.columns:
                raise ValueError(f"{f} must contain columns: input, output")
            df = df[["input", "output"]].copy()
            df["input"] = df["input"].astype(str).map(self.normalize_emoji_seq)
            df["output"] = df["output"].astype(str).str.strip()
            dfs.append(df)

        full = pd.concat(dfs, ignore_index=True)

        seq2meaning: Dict[str, str] = {}
        for inp, grp in full.groupby("input"):
            outs = [o for o in grp["output"].tolist() if isinstance(o, str) and o.strip()]
            if not outs:
                continue
            best = min(outs, key=lambda x: len(x))
            seq2meaning[inp] = best

        self.seq2meaning = seq2meaning
        self._memory_sequences = list(seq2meaning.keys())

    # -------------------------
    # Stage 2: similarity retrieval
    # -------------------------
    def _build_retrieval_index(self) -> None:
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\S+",
            lowercase=False,
            ngram_range=(1, 2),
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self._memory_sequences)

    def retrieve(self, emoji_seq: str) -> List[RetrievalHit]:
        q = self.normalize_emoji_seq(emoji_seq)
        if not self._memory_sequences:
            return []

        q_vec = self.vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        topk = min(self.retrieval_topk, len(sims))
        idxs = sims.argsort()[-topk:][::-1]

        hits: List[RetrievalHit] = []
        for idx in idxs:
            seq = self._memory_sequences[idx]
            hits.append(
                RetrievalHit(
                    score=float(sims[idx]),
                    emoji_input=seq,
                    meaning=self.seq2meaning.get(seq, ""),
                )
            )
        return hits

    # -------------------------
    # Stage 4: T5 fallback
    # -------------------------
    def _load_t5(self, device: Optional[str] = None) -> None:
        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(f"MODEL_DIR not found: {self.model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir)

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    def _t5_generate(self, emoji_input: str, num_beams: int = 5) -> str:
        inputs = self.tokenizer(
            emoji_input,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_len,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        return self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

    # -------------------------
    # Optional Stage 5: Ollama fallback
    # -------------------------
    def _ollama_generate(self, emoji_seq: str) -> str:
        prompt = (
            "Translate the following emoji sequence into ONE short, literal English sentence. "
            "Do not invent extra details.\n\n"
            f"Emoji: {emoji_seq}\n"
            "Answer:"
        )
        payload = {"model": self.ollama_model, "prompt": prompt, "stream": False}
        r = requests.post(self.ollama_url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        return str(data.get("response", "")).strip()

    # -------------------------
    # Unified translate()
    # -------------------------
    def translate(self, emoji_seq: str) -> Tuple[str, str, Dict[str, Any]]:
        q = self.normalize_emoji_seq(emoji_seq)
        debug: Dict[str, Any] = {"input": q}

        # Stage 1: exact match
        if q in self.seq2meaning:
            out = self.seq2meaning[q]
            debug["exact_match"] = True
            return out, "exact_match", debug

        debug["exact_match"] = False

        # Stage 2: retrieval hits
        hits = self.retrieve(q)
        debug["retrieval_hits"] = [h.__dict__ for h in hits]

        # If we have a top meaning at all, we can return it (grounded).
        # High-confidence retrieval
        if hits and hits[0].meaning:
            if hits[0].score >= self.retrieval_threshold:
                return hits[0].meaning, "retrieval", debug
            # Stage 3: low-confidence retrieval still better than ungrounded T5
            return hits[0].meaning, "retrieval_lowconf", debug

        # Stage 4: T5 fallback (only if retrieval gave nothing usable)
        out = self._t5_generate(q, num_beams=5)
        debug["t5_used"] = True
        debug["t5_output"] = out

        # Optional Stage 5: if degenerate and enabled
        if self.enable_ollama_fallback:
            degenerate = (len(out.split()) <= 3) or (out.lower() in {"the man sat on the couch", "the sailor sat in the sun"})
            debug["degenerate"] = degenerate
            if degenerate:
                try:
                    llm_out = self._ollama_generate(q)
                    debug["ollama_output"] = llm_out
                    return llm_out, "ollama_fallback", debug
                except Exception as e:
                    debug["ollama_error"] = str(e)

        return out, "t5_fallback", debug