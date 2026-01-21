from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PREFIX = "emoji2text: "


def norm(s: str) -> str:
    return " ".join(str(s).strip().split())


def safe_read_e2t(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "input" not in df.columns or "output" not in df.columns:
        raise ValueError(f"Expected columns input,output in {path}")
    df = df[["input", "output"]].astype(str)
    df["input"] = df["input"].fillna("").map(norm)
    df["output"] = df["output"].fillna("").map(norm)
    df = df[(df["input"] != "") & (df["output"] != "")]
    return df.reset_index(drop=True)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def best_of_multi_chrf(pred: str, refs: List[str]) -> float:
    try:
        import sacrebleu
        scores = [float(sacrebleu.sentence_chrf(pred, [r]).score) for r in refs]
        return float(max(scores)) if scores else float("nan")
    except Exception:
        return float("nan")


def best_of_multi_semantic(pred: str, refs: List[str]) -> float:
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return float("nan")

    if not hasattr(best_of_multi_semantic, "_model"):
        best_of_multi_semantic._model = SentenceTransformer("all-MiniLM-L6-v2")  # type: ignore
    model = best_of_multi_semantic._model  # type: ignore

    emb_p = model.encode([pred], normalize_embeddings=True)
    emb_r = model.encode(refs, normalize_embeddings=True)
    sims = cosine_similarity(emb_p, emb_r)[0]
    return float(np.max(sims)) if sims.size else float("nan")


class RetrievalBaseline:
    """
    Simple TF-IDF over emoji inputs -> returns output of nearest neighbor in training memory.
    IMPORTANT: memory must be training-only to avoid leakage.
    """
    def __init__(self, mem_df: pd.DataFrame):
        self.mem = mem_df.copy()
        self.mem["input"] = self.mem["input"].map(norm)
        self.mem["output"] = self.mem["output"].map(norm)

        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
        self.X = self.vectorizer.fit_transform(self.mem["input"])

    def predict(self, emoji_in: str) -> str:
        q = self.vectorizer.transform([norm(emoji_in)])
        sims = cosine_similarity(q, self.X)[0]
        idx = int(np.argmax(sims))
        return str(self.mem.iloc[idx]["output"])


class T5Generator:
    def __init__(self, model_dir: str):
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, emoji_in: str) -> str:
        src = norm(PREFIX + emoji_in)
        enc = self.tokenizer(src, return_tensors="pt", truncation=True, max_length=64).to(self.device)
        out = self.model.generate(**enc, max_new_tokens=64, num_beams=4, do_sample=False)
        return norm(self.tokenizer.decode(out[0], skip_special_tokens=True))


def eval_model(
    name: str,
    predictor,
    unique_inputs: List[str],
    refs_by_input: Dict[str, List[str]],
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    rows = []
    em_scores = []
    chrf_scores = []
    sem_scores = []

    for inp in tqdm(unique_inputs, desc=name):
        refs = refs_by_input[inp]
        pred = predictor.predict(inp)

        em = float(pred in set(refs))
        chrf = best_of_multi_chrf(pred, refs)
        sem = best_of_multi_semantic(pred, refs)

        em_scores.append(em)
        chrf_scores.append(chrf)
        sem_scores.append(sem)

        rows.append(
            {
                "input": inp,
                f"pred_{name}": pred,
                "refs": " ||| ".join(refs),
                f"multi_ref_exact_match_{name}": em,
                f"best_sentence_chrf_{name}": chrf,
                f"best_semantic_sim_{name}": sem,
            }
        )

    summary = {
        "multi_ref_exact_match": float(np.mean(em_scores)) if em_scores else float("nan"),
        "avg_best_sentence_chrf": float(np.nanmean(chrf_scores)) if chrf_scores else float("nan"),
        "avg_best_semantic_sim": float(np.nanmean(sem_scores)) if sem_scores else float("nan"),
    }
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_path", required=True)
    ap.add_argument("--t5_dir", default="artifacts/t5_e2t")
    ap.add_argument("--memory_path", default="data/final_train_e2t.csv", help="Training-only memory for retrieval baseline (avoid leakage).")
    ap.add_argument("--out_dir", default="eval_results_final")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_test = safe_read_e2t(args.test_path)
    df_test["input"] = df_test["input"].map(norm)
    df_test["output"] = df_test["output"].map(norm)

    refs_by_input: Dict[str, List[str]] = (
        df_test.groupby("input")["output"].apply(lambda x: sorted(list(set(x.tolist())))).to_dict()
    )
    unique_inputs = sorted(refs_by_input.keys())

    # Training-only memory for retrieval baseline
    if not os.path.exists(args.memory_path):
        raise FileNotFoundError(f"Missing memory_path: {args.memory_path} (run prepare_data.py first)")
    mem_df = safe_read_e2t(args.memory_path)

    retrieval = RetrievalBaseline(mem_df)
    t5 = T5Generator(args.t5_dir)

    summ_retr, rows_retr = eval_model("retrieval", retrieval, unique_inputs, refs_by_input)
    summ_t5, rows_t5 = eval_model("t5", t5, unique_inputs, refs_by_input)

    # merge rows on input
    df_retr = pd.DataFrame(rows_retr)
    df_t5 = pd.DataFrame(rows_t5)
    merged = df_retr.merge(df_t5.drop(columns=["refs"]), on="input", how="left")
    merged.to_csv(os.path.join(args.out_dir, "predictions.csv"), index=False)

    comparison = {
        "test_path": args.test_path,
        "memory_path": args.memory_path,
        "num_unique_inputs": int(len(unique_inputs)),
        "retrieval_only": summ_retr,
        "t5_generator": summ_t5,
        "has_sacrebleu": bool(not np.isnan(summ_t5["avg_best_sentence_chrf"])),
        "has_sentence_transformers": bool(not np.isnan(summ_t5["avg_best_semantic_sim"])),
    }

    with open(os.path.join(args.out_dir, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(json.dumps(comparison, indent=2))
    print(f"✅ Saved to {args.out_dir}")


if __name__ == "__main__":
    main()