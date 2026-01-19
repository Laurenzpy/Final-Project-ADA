from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class HybridConfig:
    # CSVs mit Emoji2Text-Pairs (wird weiterhin geladen, aber nur als Notnagel-Fallback)
    tm_train_paths: List[str]

    # lokaler HuggingFace Ordner (euer trainiertes T5)
    t5_model_dir: Optional[str] = None

    # --- RETRIEVAL PARAMS (für Notnagel-Fallback + Debug; werden nicht mehr für die Entscheidung genutzt) ---
    retrieval_high_conf: float = 0.60
    retrieval_low_conf: float = 0.35
    t5_fallback_below_conf: float = 0.55  # bleibt drin, wird in T5-only nicht mehr gebraucht

    # T5 soll jetzt der Hauptpfad sein
    enable_t5_fallback: bool = True
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    max_new_tokens: int = 32
    num_beams: int = 4


class FinalEmojiTranslator:
    """
    T5-ONLY (Primary): Das trainierte T5-Modell ist das eigentliche System.
    Retrieval/TM wird nur noch als Notnagel genutzt:
      - falls T5 nicht verfügbar ist (nicht geladen / Ordner fehlt)
      - oder falls die Generierung aus irgendeinem Grund fehlschlägt

    WICHTIG: Deine e2t CSVs haben Spalten: input,output
    (aus deiner ZIP: data/emoji_dataset_stage*_e2t.csv)
    """

    def __init__(self, cfg: HybridConfig):
        self.cfg = cfg
        self._t5_available = False

        # Wir laden TM weiterhin, aber es ist nur Fallback/Debug.
        self._load_tm()
        self._init_t5()

    def _load_tm(self) -> None:
        dfs = []
        debug_infos = []

        # ✅ passt direkt zu deinen CSVs (input,output)
        emoji_cols = ["emoji", "emoji_sequence", "input", "source", "src", "x"]
        text_cols = ["text", "target", "output", "translation", "tgt", "y"]

        for p in self.cfg.tm_train_paths:
            try:
                df = pd.read_csv(p)
                debug_infos.append((p, list(df.columns)))

                e_col = next((c for c in emoji_cols if c in df.columns), None)
                t_col = next((c for c in text_cols if c in df.columns), None)

                if e_col and t_col:
                    tmp = df[[e_col, t_col]].copy()
                    tmp.columns = ["emoji", "text"]
                    dfs.append(tmp)
            except Exception as e:
                debug_infos.append((p, f"READ_FAIL: {type(e).__name__}: {e}"))

        if not dfs:
            msg_lines = [
                "No usable TM data loaded. None of the CSV files contained a valid (emoji/text) column pair.",
                "Expected one emoji column from: " + ", ".join(emoji_cols),
                "Expected one text  column from: " + ", ".join(text_cols),
                "Files inspected (path -> columns):",
            ]
            for info in debug_infos[:50]:
                msg_lines.append(f" - {info[0]} -> {info[1]}")
            raise RuntimeError("\n".join(msg_lines))

        self.tm = pd.concat(dfs, ignore_index=True)
        self.tm["emoji"] = self.tm["emoji"].astype(str).fillna("")
        self.tm["text"] = self.tm["text"].astype(str).fillna("")
        self.tm["emoji_norm"] = self.tm["emoji"]

        # Char ngram TF-IDF funktioniert gut für Emoji-Sequenzen
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
        self.X = self.vectorizer.fit_transform(self.tm["emoji_norm"])

    def _init_t5(self) -> None:
        # In T5-only wollen wir T5 wirklich laden; wenn’s nicht geht, fällt er später auf Retrieval zurück.
        if not self.cfg.enable_t5_fallback:
            return
        if not self.cfg.t5_model_dir:
            return
        if not os.path.isdir(self.cfg.t5_model_dir):
            # Ordner fehlt → T5 aus
            return

        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch

            self.t5_tokenizer = AutoTokenizer.from_pretrained(self.cfg.t5_model_dir)
            self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.t5_model_dir)

            if self.cfg.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.cfg.device

            self.t5_model.to(self.device)
            self._t5_available = True
        except Exception:
            self._t5_available = False

    def _t5_generate(self, emoji_seq: str) -> Optional[str]:
        if not self._t5_available:
            return None

        import torch

        inputs = self.t5_tokenizer(emoji_seq, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.t5_model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                num_beams=self.cfg.num_beams,
            )
        return self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _retrieval_fallback(self, norm_in: str) -> Dict[str, Any]:
        """
        Retrieval ist nur noch Fallback + Debug:
        liefert best_out + score + match.
        """
        q = self.vectorizer.transform([norm_in])
        sims = cosine_similarity(q, self.X)[0]
        idx = int(np.argmax(sims))
        best_score = float(sims[idx])
        best_inp = self.tm.iloc[idx]["emoji"]
        best_out = self.tm.iloc[idx]["text"]
        return {
            "best_score": best_score,
            "best_inp": best_inp,
            "best_out": best_out,
        }

    def translate(self, emoji_in: str) -> Dict[str, Any]:
        """
        T5-ONLY:
          1) Versuche immer T5
          2) Wenn T5 nicht verfügbar oder Generation None/leer → Retrieval-Notnagel
        """
        t0 = time.perf_counter()
        norm_in = str(emoji_in)

        # Retrieval-Infos immer berechnen? -> kostets etwas, aber hilft Debug/Eval.
        # Wenn du es noch schneller willst, kann man es nur im Fallback rechnen.
        ret = self._retrieval_fallback(norm_in)
        best_score = float(ret["best_score"])
        best_inp = ret["best_inp"]
        best_out = ret["best_out"]

        pred = ""
        mode = ""

        # ✅ Primary: T5
        if self._t5_available:
            t5_pred = self._t5_generate(norm_in)
            if t5_pred and t5_pred.strip():
                pred = t5_pred.strip()
                mode = "t5_only"
            else:
                pred = best_out
                mode = "t5_failed_fallback_retrieval"
        else:
            pred = best_out
            mode = "t5_unavailable_fallback_retrieval"

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "input": emoji_in,
            "prediction": pred,
            "mode": mode,
            # wir geben Retrieval-Debug weiterhin mit aus (für Analyse/Eval)
            "retrieval_score": best_score,
            "retrieval_match_input": best_inp,
            "retrieval_match_output": best_out,
            "latency_ms": latency_ms,
        }