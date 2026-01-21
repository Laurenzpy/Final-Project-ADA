# train_seq2seq.py
"""
T5 fine-tuning (Emoji -> Text) for the final submission.

Key points (paper-correct):
- Train: Stages 1–4 (emoji->text)
- Validation/Dev: Stage 6
- Test/Eval: Stage 5 (ONLY in eval script)
- Uses a consistent T5-style task prefix:  "emoji2text: <EMOJIS>"

This file is the single source of truth that builds:
  artifacts/t5_e2t

Run:
  python train_seq2seq.py
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Use sacrebleu for stable BLEU during training
import sacrebleu


TASK_PREFIX = "emoji2text: "


def choose_canonical_output(outputs: list[str]) -> str:
    """Most frequent output; tie-breaker: shorter."""
    outputs = [" ".join(str(o).split()) for o in outputs if str(o).strip()]
    if not outputs:
        return ""
    counts: dict[str, int] = {}
    for o in outputs:
        counts[o] = counts.get(o, 0) + 1
    mx = max(counts.values())
    cand = [o for o, c in counts.items() if c == mx]
    cand.sort(key=lambda s: (len(s.split()), len(s)))
    return cand[0]


def load_and_dedup(csv_paths: list[str]) -> pd.DataFrame:
    """
    Load e2t CSVs with columns input, output:
      input  = emoji sequence
      output = text
    Deduplicate by input (emoji sequence) by choosing the most frequent target.
    """
    df = pd.concat([pd.read_csv(p)[["input", "output"]] for p in csv_paths], ignore_index=True)
    df["input"] = df["input"].astype(str).fillna("")
    df["output"] = df["output"].astype(str).fillna("")

    grouped = df.groupby("input")["output"].apply(list).reset_index()
    grouped["output"] = grouped["output"].apply(choose_canonical_output)
    return grouped.reset_index(drop=True)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def build_training_args(
    *,
    output_dir: str,
    learning_rate: float,
    epochs: int,
    train_bs: int,
    eval_bs: int,
    seed: int,
) -> Seq2SeqTrainingArguments:
    """HF version-safe TrainingArguments builder."""
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters

    args_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        predict_with_generate=True,
        logging_steps=25,
        logging_first_step=True,
        save_total_limit=2,
        report_to=[],
        fp16=False,
        seed=seed,
    )

    # Compat for different HF versions
    if "evaluation_strategy" in sig:
        args_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in sig:
        args_kwargs["eval_strategy"] = "epoch"

    if "save_strategy" in sig:
        args_kwargs["save_strategy"] = "epoch"

    if "generation_max_length" in sig:
        args_kwargs["generation_max_length"] = 32

    if "load_best_model_at_end" in sig:
        args_kwargs["load_best_model_at_end"] = True
    if "metric_for_best_model" in sig:
        args_kwargs["metric_for_best_model"] = "bleu"
    if "greater_is_better" in sig:
        args_kwargs["greater_is_better"] = True

    return Seq2SeqTrainingArguments(**args_kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--model_name", default="t5-small")
    ap.add_argument("--out_dir", default="artifacts/t5_e2t")
    ap.add_argument("--checkpoint_dir", default="checkpoints/t5_e2t")
    ap.add_argument("--seed", type=int, default=42)

    # UPDATED hyperparams (more conservative, longer training)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--train_bs", type=int, default=8)
    ap.add_argument("--eval_bs", type=int, default=8)

    ap.add_argument("--max_src_len", type=int, default=64)
    ap.add_argument("--max_tgt_len", type=int, default=64)
    args = ap.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_files = [str(data_dir / f"emoji_dataset_stage{i}_e2t.csv") for i in range(1, 5)]
    val_files = [str(data_dir / "emoji_dataset_stage6_e2t.csv")]

    for p in train_files + val_files:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing dataset file: {p}")

    # No leakage:
    # Train on stage 1-4, validate on stage 6, keep stage 5 for final evaluation only.
    train_df = load_and_dedup(train_files)
    val_df = load_and_dedup(val_files)

    print(">>> T5 fine-tuning (emoji->text)")
    print(">>> TASK_PREFIX :", TASK_PREFIX)
    print(">>> train files :", train_files)
    print(">>> val files   :", val_files)
    print(">>> train size  :", len(train_df))
    print(">>> val size    :", len(val_df))
    print(">>> model       :", args.model_name)
    print(">>> out_dir     :", args.out_dir)
    print(">>> lr/epochs   :", args.lr, args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    def preprocess(batch):
        # Consistent task prefix for T5
        inputs = [TASK_PREFIX + x for x in batch["input"]]
        targets = batch["output"]

        model_inputs = tokenizer(
            inputs,
            text_target=targets,
            max_length=args.max_src_len,
            truncation=True,
            padding=False,
        )

        # robust fallback if labels not populated
        if "labels" not in model_inputs:
            labels = tokenizer(
                targets,
                max_length=args.max_tgt_len,
                truncation=True,
                padding=False,
            )["input_ids"]
            model_inputs["labels"] = labels
        return model_inputs

    print(">>> tokenizing...", flush=True)
    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.asarray(preds)
        if preds.ndim == 3:
            preds = preds.argmax(axis=-1)

        preds = preds.astype(np.int32)
        labels = np.asarray(labels)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id).astype(np.int32)

        pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

        bleu = sacrebleu.corpus_bleu(pred_texts, [label_texts]).score
        return {"bleu": float(bleu)}

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
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(">>> training start", flush=True)
    trainer.train()

    print(">>> saving model", flush=True)
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    run_info = {
        "task_prefix": TASK_PREFIX,
        "model_name": args.model_name,
        "seed": args.seed,
        "train_files": train_files,
        "val_files": val_files,
        "heldout_test": "data/emoji_dataset_stage5_e2t.csv",
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "train_bs": args.train_bs,
        "eval_bs": args.eval_bs,
        "max_src_len": args.max_src_len,
        "max_tgt_len": args.max_tgt_len,
        "note": "No leakage: Stage 5 not used in training/validation. T5 uses task prefix 'emoji2text: '.",
    }
    with open(os.path.join(args.out_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    print("✅ Saved model to:", args.out_dir, flush=True)


if __name__ == "__main__":
    main()