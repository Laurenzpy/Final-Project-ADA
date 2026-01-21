from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

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


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_training_args(*, output_dir: str, lr: float, epochs: int, train_bs: int, eval_bs: int, seed: int):
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        learning_rate=lr,
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

    if "generation_max_length" in sig:
        kwargs["generation_max_length"] = 64

    return Seq2SeqTrainingArguments(**kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="data/final_train_e2t.csv")
    ap.add_argument("--dev_path", default="data/final_dev_e2t.csv")
    ap.add_argument("--model_name", default="t5-small")
    ap.add_argument("--out_dir", default="artifacts/t5_e2t")
    ap.add_argument("--checkpoint_dir", default="checkpoints/t5_e2t")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train_bs", type=int, default=8)
    ap.add_argument("--eval_bs", type=int, default=8)

    ap.add_argument("--max_src_len", type=int, default=64)
    ap.add_argument("--max_tgt_len", type=int, default=64)
    args = ap.parse_args()

    for p in [args.train_path, args.dev_path]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing: {p}")

    df_train = safe_read_e2t(args.train_path)
    df_dev = safe_read_e2t(args.dev_path)

    df_train["src"] = (PREFIX + df_train["input"]).map(norm)
    df_train["tgt"] = df_train["output"].map(norm)
    df_dev["src"] = (PREFIX + df_dev["input"]).map(norm)
    df_dev["tgt"] = df_dev["output"].map(norm)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    train_ds = Dataset.from_pandas(df_train[["src", "tgt"]])
    dev_ds = Dataset.from_pandas(df_dev[["src", "tgt"]])

    def preprocess(batch):
        x = tokenizer(batch["src"], truncation=True, max_length=args.max_src_len)
        y = tokenizer(text_target=batch["tgt"], truncation=True, max_length=args.max_tgt_len)
        x["labels"] = y["input_ids"]
        return x

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    dev_tok = dev_ds.map(preprocess, batched=True, remove_columns=dev_ds.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def compute_metrics(eval_pred) -> Dict[str, float]:
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.asarray(preds)
        if preds.ndim == 3:
            preds = preds.argmax(axis=-1)

        pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        pred_texts = [norm(p).lower() for p in pred_texts]

        labels = np.asarray(labels)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        gold_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        gold_texts = [norm(g).lower() for g in gold_texts]

        em = float(np.mean([p == g for p, g in zip(pred_texts, gold_texts)]))
        return {"exact_match": em}

    training_args = build_training_args(
        output_dir=args.checkpoint_dir,
        lr=args.lr,
        epochs=args.epochs,
        train_bs=args.train_bs,
        eval_bs=args.eval_bs,
        seed=args.seed,
    )

    print(f"Using device: {get_device()}")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    run_info = {
        "model_name": args.model_name,
        "task": "T5 generator fine-tuning: emoji -> text",
        "train_path": args.train_path,
        "dev_path": args.dev_path,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "max_src_len": args.max_src_len,
        "max_tgt_len": args.max_tgt_len,
        "prefix": PREFIX,
        "note": "Main evaluation is IID within Stage5 (train/dev/test split by input). Stage6 optional OOD test.",
    }
    with open(os.path.join(args.out_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    print(f"✅ Saved generator to {args.out_dir}")


if __name__ == "__main__":
    main()