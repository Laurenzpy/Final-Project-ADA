import os
import inspect
import numpy as np
import pandas as pd

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

DATA_DIR = "data"
TRAIN_FILES = [os.path.join(DATA_DIR, f"emoji_dataset_stage{i}_e2t.csv") for i in range(1, 5)]
VAL_FILES   = [os.path.join(DATA_DIR, "emoji_dataset_stage5_e2t.csv")]

MODEL_NAME = "t5-small"
OUT_DIR = "artifacts/t5_e2t"


def choose_canonical_output(outputs):
    outputs = [" ".join(str(o).split()) for o in outputs if str(o).strip()]
    if not outputs:
        return ""
    counts = {}
    for o in outputs:
        counts[o] = counts.get(o, 0) + 1
    mx = max(counts.values())
    cand = [o for o, c in counts.items() if c == mx]
    cand.sort(key=lambda s: (len(s.split()), len(s)))
    return cand[0]


def load_and_dedup(csv_paths):
    df = pd.concat([pd.read_csv(p)[["input", "output"]] for p in csv_paths], ignore_index=True)
    df["input"] = df["input"].astype(str)
    df["output"] = df["output"].astype(str)

    grouped = df.groupby("input")["output"].apply(list).reset_index()
    grouped["output"] = grouped["output"].apply(choose_canonical_output)
    return grouped.reset_index(drop=True)


def main():
    print(">>> SCRIPT ENTERED", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    train_df = load_and_dedup(TRAIN_FILES)
    val_df   = load_and_dedup(VAL_FILES)

    print(">>> train size:", len(train_df), flush=True)
    print(">>> val size  :", len(val_df), flush=True)
    print(">>> model     :", MODEL_NAME, flush=True)

    print(">>> loading tokenizer/model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    train_ds = Dataset.from_pandas(train_df)
    val_ds   = Dataset.from_pandas(val_df)

    max_src_len = 64
    max_tgt_len = 64

    def preprocess(batch):
        inputs = batch["input"]
        targets = batch["output"]

        # ✅ Modern: targets über text_target (statt as_target_tokenizer)
        model_inputs = tokenizer(
            inputs,
            text_target=targets,
            max_length=max_src_len,
            truncation=True,
            padding=False,
        )

        # labels werden automatisch als input_ids für text_target gesetzt
        # aber: max_length für target separat clampen (robust)
        # -> wenn tokenizer das schon passend macht, ist es ok; sonst kürzen wir:
        if "labels" in model_inputs:
            # manchmal sind labels direkt da
            pass
        else:
            # fallback: manche tokenizer legen targets in "labels" anders ab
            labels = tokenizer(
                targets,
                max_length=max_tgt_len,
                truncation=True,
                padding=False,
            )["input_ids"]
            model_inputs["labels"] = labels

        return model_inputs

    print(">>> tokenizing...", flush=True)
    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    val_tok   = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # ---- METRICS (ohne evaluate -> stabiler)
    # pip install sacrebleu
    import sacrebleu
    print(">>> metrics: sacrebleu enabled", flush=True)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.asarray(preds)

        # preds kann logits sein: [B, T, V]
        if preds.ndim == 3:
            preds = preds.argmax(axis=-1)

        preds = preds.astype(np.int32)
        preds = np.clip(preds, 0, tokenizer.vocab_size - 1)

        labels = np.asarray(labels)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id).astype(np.int32)
        labels = np.clip(labels, 0, tokenizer.vocab_size - 1)

        pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

        bleu = sacrebleu.corpus_bleu(pred_texts, [label_texts]).score
        return {"bleu": float(bleu)}

    # ---- TRAINING ARGS (versions-sicher)
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters

    args_kwargs = dict(
        output_dir="checkpoints/t5_e2t",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        num_train_epochs=5,
        predict_with_generate=True,
        logging_steps=25,
        logging_first_step=True,
        save_total_limit=2,
        report_to=[],
        fp16=False,
    )

    # eval strategy heißt je nach Version anders
    if "evaluation_strategy" in sig:
        args_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in sig:
        args_kwargs["eval_strategy"] = "epoch"

    # save strategy
    if "save_strategy" in sig:
        args_kwargs["save_strategy"] = "epoch"

    # generation length heißt manchmal anders
    if "generation_max_length" in sig:
        args_kwargs["generation_max_length"] = 32

    # best model laden (nicht jede Version kann das)
    if "load_best_model_at_end" in sig:
        args_kwargs["load_best_model_at_end"] = True
    if "metric_for_best_model" in sig:
        args_kwargs["metric_for_best_model"] = "bleu"
    if "greater_is_better" in sig:
        args_kwargs["greater_is_better"] = True

    training_args = Seq2SeqTrainingArguments(**args_kwargs)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(">>> Training start", flush=True)
    trainer.train()

    print(">>> Saving model", flush=True)
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print("✅ Saved:", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()