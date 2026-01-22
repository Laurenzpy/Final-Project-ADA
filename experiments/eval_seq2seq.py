import os
import random
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Wo dein fine-tuned Modell liegt
MODEL_DIR = "artifacts/t5_e2t"

# Eval-Datei (Stage5 = Test/Val in deinem Setup)
EVAL_FILE = os.path.join("data", "emoji_dataset_stage5_e2t.csv")

# Wie viele Beispiele ausgeben
N_SHOW = 30

def generate_text(model, tokenizer, emoji_input: str):
    # T5 bekommt einfach das Emoji-String als Input
    inputs = tokenizer(
        emoji_input,
        return_tensors="pt",
        truncation=True,
        max_length=64,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Gute Defaults gegen "the the the" / Wiederholungen:
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=32,
        num_beams=5,
        do_sample=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )

    pred = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return pred

def main():
    assert os.path.isdir(MODEL_DIR), f"MODEL_DIR not found: {MODEL_DIR}. Hast du train_seq2seq.py erfolgreich laufen lassen?"
    assert os.path.isfile(EVAL_FILE), f"EVAL_FILE not found: {EVAL_FILE}"

    print(">>> loading model from:", MODEL_DIR, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

    # Device: MPS wenn verfügbar, sonst CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()
    print(">>> device:", device, flush=True)

    df = pd.read_csv(EVAL_FILE)[["input", "output"]]
    df["input"] = df["input"].astype(str)
    df["output"] = df["output"].astype(str)

    # Zufällig sampeln (aber stabil reproduzierbar)
    random.seed(42)
    idxs = list(range(len(df)))
    random.shuffle(idxs)

    shown = 0
    for i in idxs:
        inp = df.loc[i, "input"]
        gt = df.loc[i, "output"]

        pred = generate_text(model, tokenizer, inp)

        print("IN :", inp)
        print("GT :", gt)
        print("PR :", pred)
        print("-" * 60)

        shown += 1
        if shown >= N_SHOW:
            break

if __name__ == "__main__":
    main()