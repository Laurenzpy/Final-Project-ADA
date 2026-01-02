#!/usr/bin/env python3
"""
Train a seq2seq model to translate emoji sequences -> English text.

Data files (expected in the same folder):
  - emoji_dataset_stage1.csv ... emoji_dataset_stage6.csv (columns: input, output)
      input  = English sentence
      output = emoji sequence (space-separated)
  - merged_emoji_sample.csv (emoji definitions; optional augmentation)

By default we:
  - flip columns to train emoji -> English
  - drop single-emoji outputs
  - keep only 2..6 emoji tokens per source
"""

import os
import glob
import argparse
import inspect
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from packaging import version

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import Dataset, DatasetDict

# Optional metrics (safe if installed)
try:
    import evaluate
    _HAS_EVALUATE = True
except Exception:
    _HAS_EVALUATE = False


def load_stage_csvs(data_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(data_dir, "emoji_dataset_stage*.csv")))
    if not paths:
        raise FileNotFoundError(f"No emoji_dataset_stage*.csv found in: {data_dir}")

    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        if not {"input", "output"}.issubset(df.columns):
            raise ValueError(f"{p} must have columns ['input','output'], found: {df.columns.tolist()}")
        dfs.append(df[["input", "output"]])

    merged = pd.concat(dfs, ignore_index=True).dropna()
    return merged


def load_emoji_definitions(defs_path: str) -> Dict[str, str]:
    """
    merged_emoji_sample.csv contains columns like: emoji, name, definition, keywords, ...
    We'll build a mapping emoji -> short definition string.
    """
    if not os.path.exists(defs_path):
        return {}

    df = pd.read_csv(defs_path)
    if "emoji" not in df.columns:
        return {}

    # Prefer 'definition', else 'name'
    def_col = "definition" if "definition" in df.columns else ("name" if "name" in df.columns else None)
    if def_col is None:
        return {}

    mapping = {}
    for _, r in df.iterrows():
        e = str(r["emoji"])
        d = str(r[def_col]) if pd.notna(r[def_col]) else ""
        d = d.strip()
        if e and d:
            mapping[e] = d
    return mapping


def count_emoji_tokens(emoji_seq: str) -> int:
    """
    Your dataset uses space-separated emojis like: "🎵 💃"
    So token count = number of space-split parts.
    """
    return len(str(emoji_seq).strip().split())


def maybe_augment_source(
    emoji_seq: str,
    emoji2def: Dict[str, str],
    p_augment: float,
    rng: np.random.Generator,
) -> str:
    """
    Optional: sometimes append emoji definitions to the source to help learning.

    Example:
      "🎵 💃 :: musical note; dancing"

    IMPORTANT: We only do this stochastically, so the model still learns to work
    when inference provides emojis only.
    """
    if not emoji2def or p_augment <= 0:
        return emoji_seq

    if rng.random() > p_augment:
        return emoji_seq

    toks = str(emoji_seq).strip().split()
    defs = [emoji2def.get(t, "") for t in toks]
    defs = [d for d in defs if d]
    if not defs:
        return emoji_seq

    return f"{emoji_seq} :: " + " ; ".join(defs)


def make_dataset(
    df: pd.DataFrame,
    emoji2def: Dict[str, str],
    min_emojis: int,
    max_emojis: int,
    p_augment: float,
    seed: int,
) -> DatasetDict:
    """
    Build HuggingFace DatasetDict with train/validation/test.
    We flip: source = df.output (emoji sequence), target = df.input (English).
    """
    df = df.copy()
    df["source"] = df["output"].astype(str).str.strip()
    df["target"] = df["input"].astype(str).str.strip()

    # Filter by emoji sequence length
    df["n_emoji"] = df["source"].apply(count_emoji_tokens)
    df = df[(df["n_emoji"] >= min_emojis) & (df["n_emoji"] <= max_emojis)]
    df = df.drop(columns=["n_emoji"])

    # Shuffle
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    if p_augment > 0 and emoji2def:
        df["source"] = df["source"].apply(lambda s: maybe_augment_source(s, emoji2def, p_augment, rng))

    ds = Dataset.from_pandas(df[["source", "target"]])

    # 90/5/5 split
    ds_train_test = ds.train_test_split(test_size=0.10, seed=seed)
    ds_val_test = ds_train_test["test"].train_test_split(test_size=0.50, seed=seed)

    return DatasetDict(
        train=ds_train_test["train"],
        validation=ds_val_test["train"],
        test=ds_val_test["test"],
    )


def build_training_args_kwargs() -> Dict[str, Any]:
    """
    Transformers renamed `evaluation_strategy` -> `eval_strategy` in newer versions,
    which causes the exact TypeError you saw. :contentReference[oaicite:1]{index=1}

    We inspect the signature at runtime and pick the correct keyword.
    """
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    params = sig.parameters

    kwargs = {}

    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"

    return kwargs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="/mnt/data", help="Folder containing the CSVs")
    ap.add_argument("--defs_csv", type=str, default="/mnt/data/merged_emoji_sample.csv")
    ap.add_argument("--model_name", type=str, default="facebook/bart-base")
    ap.add_argument("--out_dir", type=str, default="./emoji2text_bart")
    ap.add_argument("--min_emojis", type=int, default=2)
    ap.add_argument("--max_emojis", type=int, default=6)
    ap.add_argument("--p_augment_defs", type=float, default=0.3, help="Prob. to append definitions to source (0 disables)")
    ap.add_argument("--max_source_len", type=int, default=64)
    ap.add_argument("--max_target_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--num_epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--fp16", action="store_true", help="Enable fp16 if you have CUDA")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Transformers version: {transformers.__version__}")
    print("Loading data...")
    df = load_stage_csvs(args.data_dir)

    emoji2def = load_emoji_definitions(args.defs_csv)
    if emoji2def:
        print(f"Loaded {len(emoji2def)} emoji definitions from {args.defs_csv}")
    else:
        print("No emoji definitions loaded (augmentation disabled unless you provide defs).")

    dsd = make_dataset(
        df=df,
        emoji2def=emoji2def,
        min_emojis=args.min_emojis,
        max_emojis=args.max_emojis,
        p_augment=args.p_augment_defs,
        seed=args.seed,
    )

    print(dsd)

    print("Loading model/tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["source"],
            max_length=args.max_source_len,
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target"],
                max_length=args.max_target_len,
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dsd.map(preprocess, batched=True, remove_columns=dsd["train"].column_names)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    compute_metrics = None
    if _HAS_EVALUATE:
        rouge = evaluate.load("rouge")
        bleu = evaluate.load("sacrebleu")

        def _metrics(eval_pred):
            preds, labels = eval_pred
            # Replace -100 with pad_token_id so we can decode
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
            label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # sacrebleu expects list of refs per prediction
            bleu_res = bleu.compute(predictions=pred_texts, references=[[t] for t in label_texts])
            rouge_res = rouge.compute(predictions=pred_texts, references=label_texts)

            return {
                "bleu": bleu_res["score"],
                "rouge1": rouge_res["rouge1"],
                "rougeL": rouge_res["rougeL"],
            }

        compute_metrics = _metrics
        print("Metrics enabled: ROUGE + SacreBLEU")
    else:
        print("Metrics disabled (install `evaluate` + `sacrebleu` to enable).")

    extra_kwargs = build_training_args_kwargs()

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        predict_with_generate=True,
        logging_steps=50,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=2,
        warmup_steps=args.warmup_steps,
        report_to="none",
        seed=args.seed,
        fp16=args.fp16,
        **extra_kwargs,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    print(test_metrics)

    print("Saving model...")
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    print("\nDone. Example inference:")
    example = "🎵 💃"
    inputs = tokenizer(example, return_tensors="pt")
    out_ids = model.generate(**inputs, max_new_tokens=40, num_beams=4)
    print(example, "->", tokenizer.decode(out_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
