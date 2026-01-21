from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

RANK_PREFIX = "rank: "


def normalize_ws(s: str) -> str:
    return " ".join(str(s).strip().split())


def safe_read_e2t(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "input" not in df.columns or "output" not in df.columns:
        raise ValueError(f"Expected columns input,output in {path}")
    df = df[["input", "output"]].astype(str)
    df["input"] = df["input"].fillna("").map(normalize_ws)
    df["output"] = df["output"].fillna("").map(normalize_ws)
    df = df[(df["input"] != "") & (df["output"] != "")]
    return df.reset_index(drop=True)


def choose_canonical(outputs: List[str]) -> str:
    """
    Canonical target for one emoji input:
    1) most frequent target
    2) tie-break: shortest (words), then shortest (chars), then lexicographic
    """
    outs = [normalize_ws(o) for o in outputs if normalize_ws(o)]
    if not outs:
        return ""
    counts: Dict[str, int] = {}
    for o in outs:
        counts[o] = counts.get(o, 0) + 1
    mx = max(counts.values())
    cand = [o for o, c in counts.items() if c == mx]
    cand.sort(key=lambda s: (len(s.split()), len(s), s))
    return cand[0]


def canonicalize_df(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("input")["output"].apply(list).reset_index()
    g["output"] = g["output"].apply(choose_canonical)
    g = g[(g["input"] != "") & (g["output"] != "")]
    return g.reset_index(drop=True)


@dataclass
class Retriever:
    df_mem: pd.DataFrame
    ngram_range: Tuple[int, int] = (1, 3)

    def __post_init__(self):
        self.df_mem = self.df_mem.copy()
        self.df_mem["input"] = self.df_mem["input"].astype(str).map(normalize_ws)
        self.df_mem["output"] = self.df_mem["output"].astype(str).map(normalize_ws)

        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=self.ngram_range)
        self.X = self.vectorizer.fit_transform(self.df_mem["input"])

    def topk(self, query_emoji: str, k: int) -> List[int]:
        q = self.vectorizer.transform([normalize_ws(query_emoji)])
        sims = cosine_similarity(q, self.X)[0]
        idxs = np.argsort(-sims)[:k]
        return [int(i) for i in idxs]


def make_ranker_example(emoji_in: str, candidates: List[str], label_idx: int) -> Tuple[str, str]:
    lines = [f"{RANK_PREFIX}{emoji_in}"]
    for i, c in enumerate(candidates, start=1):
        lines.append(f"c{i}: {c}")
    src = "\n".join(lines)
    tgt = f"c{label_idx + 1}"
    return src, tgt


def build_ranker_dataset(df_canon: pd.DataFrame, retriever: Retriever, k: int) -> pd.DataFrame:
    mem_outputs = retriever.df_mem["output"].tolist()

    rows: List[Dict[str, Any]] = []
    for _, r in df_canon.iterrows():
        emoji_in = r["input"]
        gold = normalize_ws(r["output"])

        idxs = retriever.topk(emoji_in, k=k)
        cand_texts = [normalize_ws(mem_outputs[i]) for i in idxs]

        # dedup preserving order
        seen = set()
        dedup = []
        for c in cand_texts:
            if c and c not in seen:
                seen.add(c)
                dedup.append(c)
        cand_texts = dedup

        # pad to k
        while len(cand_texts) < k:
            cand_texts.append(gold)
        cand_texts = cand_texts[:k]

        # ensure gold included
        if gold in cand_texts:
            label_idx = cand_texts.index(gold)
        else:
            cand_texts[-1] = gold
            label_idx = k - 1

        src, tgt = make_ranker_example(emoji_in, cand_texts, label_idx)
        rows.append({"src": src, "tgt": tgt, "input": emoji_in, "gold": gold})

    return pd.DataFrame(rows)


def build_training_args(*, output_dir: str, learning_rate: float, epochs: int, train_bs: int, eval_bs: int, seed: int):
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        predict_with_generate=True,
        logging_steps=50,
        logging_first_step=True,
        save_total_limit=2,
        report_to=[],
        fp16=False,
        seed=seed,
    )

    if "evaluation_strategy" in sig:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in sig:
        kwargs["eval_strategy"] = "epoch"

    if "save_strategy" in sig:
        kwargs["save_strategy"] = "epoch"

    if "load_best_model_at_end" in sig:
        kwargs["load_best_model_at_end"] = True
    if "metric_for_best_model" in sig:
        kwargs["metric_for_best_model"] = "accuracy"
    if "greater_is_better" in sig:
        kwargs["greater_is_better"] = True

    if "generation_max_length" in sig:
        kwargs["generation_max_length"] = 6  # "c16" fits

    return Seq2SeqTrainingArguments(**kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()

    # NEW: train/dev paths
    ap.add_argument("--train_path", default="data/iid_train_e2t.csv")
    ap.add_argument("--dev_path", default="data/iid_dev_e2t.csv")

    # Memory default: stages 1-4 (retrieval pool)
    ap.add_argument(
        "--memory_paths",
        nargs="+",
        default=[
            "data/emoji_dataset_stage1_e2t.csv",
            "data/emoji_dataset_stage2_e2t.csv",
            "data/emoji_dataset_stage3_e2t.csv",
            "data/emoji_dataset_stage4_e2t.csv",
        ],
    )

    ap.add_argument("--model_name", default="t5-small")
    ap.add_argument("--out_dir", default="artifacts/t5_ranker")
    ap.add_argument("--checkpoint_dir", default="checkpoints/t5_ranker")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train_bs", type=int, default=8)
    ap.add_argument("--eval_bs", type=int, default=8)

    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--max_src_len", type=int, default=256)
    ap.add_argument("--max_tgt_len", type=int, default=8)
    args = ap.parse_args()

    # Load memory pool (original pairs -> more candidate diversity)
    for p in args.memory_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing memory file: {p}")
    df_mem_raw = pd.concat([safe_read_e2t(p) for p in args.memory_paths], ignore_index=True)

    # Load train/dev (IID files)
    if not Path(args.train_path).exists():
        raise FileNotFoundError(f"Missing train_path: {args.train_path}")
    if not Path(args.dev_path).exists():
        raise FileNotFoundError(f"Missing dev_path: {args.dev_path}")

    df_train_raw = safe_read_e2t(args.train_path)
    df_dev_raw = safe_read_e2t(args.dev_path)

    df_train_canon = canonicalize_df(df_train_raw)
    df_dev_canon = canonicalize_df(df_dev_raw)

    retriever = Retriever(df_mem=df_mem_raw)

    print(">>> Building ranker datasets (canonical targets)...")
    train_rank = build_ranker_dataset(df_train_canon, retriever=retriever, k=args.k)
    dev_rank = build_ranker_dataset(df_dev_canon, retriever=retriever, k=args.k)
    print(f">>> train_rank: {len(train_rank)}  dev_rank: {len(dev_rank)}  K={args.k}")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_rank.head(50).to_csv(os.path.join(args.out_dir, "ranker_train_examples_head.csv"), index=False)
    dev_rank.head(50).to_csv(os.path.join(args.out_dir, "ranker_dev_examples_head.csv"), index=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    train_ds = Dataset.from_pandas(train_rank[["src", "tgt"]])
    dev_ds = Dataset.from_pandas(dev_rank[["src", "tgt"]])

    def preprocess(batch):
        x = tokenizer(batch["src"], truncation=True, max_length=args.max_src_len)
        y = tokenizer(text_target=batch["tgt"], truncation=True, max_length=args.max_tgt_len)
        x["labels"] = y["input_ids"]
        return x

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    dev_tok = dev_ds.map(preprocess, batched=True, remove_columns=dev_ds.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.asarray(preds)
        if preds.ndim == 3:
            preds = preds.argmax(axis=-1)

        pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        pred_texts = [normalize_ws(p).lower() for p in pred_texts]

        labels = np.asarray(labels)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        gold_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        gold_texts = [normalize_ws(g).lower() for g in gold_texts]

        acc = float(np.mean([p == g for p, g in zip(pred_texts, gold_texts)]))
        return {"accuracy": acc}

    training_args = build_training_args(
        output_dir=args.checkpoint_dir,
        learning_rate=args.lr,
        epochs=args.epochs,
        train_bs=args.train_bs,
        eval_bs=args.eval_bs,
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print(">>> training start")
    trainer.train()

    print(">>> saving ranker model")
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    run_info = {
        "model_name": args.model_name,
        "task": "RAG ranker (choose best candidate c1..cK) with canonical targets",
        "k": args.k,
        "seed": args.seed,
        "train_path": args.train_path,
        "dev_path": args.dev_path,
        "memory_paths": args.memory_paths,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "max_src_len": args.max_src_len,
        "note": "IID train/dev used; memory pool provides candidate diversity. Canonical targets avoid punishing valid alternatives.",
    }
    with open(os.path.join(args.out_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    print(f"✅ Saved to {args.out_dir}")


if __name__ == "__main__":
    main()