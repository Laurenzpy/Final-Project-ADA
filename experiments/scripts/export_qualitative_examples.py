import os
import random
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = "artifacts/t5_e2t"
EVAL_FILE = os.path.join("data", "emoji_dataset_stage5_e2t.csv")

def generate_text(model, tokenizer, emoji_input: str, num_beams: int):
    inputs = tokenizer(
        emoji_input,
        return_tensors="pt",
        truncation=True,
        max_length=64,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=32,
        num_beams=num_beams,
        do_sample=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="How many examples to export")
    ap.add_argument("--beam", type=int, default=5, help="Beam size for beam decoding")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="report_assets/samples/qualitative_examples.csv")
    args = ap.parse_args()

    assert os.path.isdir(MODEL_DIR), f"MODEL_DIR not found: {MODEL_DIR}"
    assert os.path.isfile(EVAL_FILE), f"EVAL_FILE not found: {EVAL_FILE}"

    print(">>> loading model from:", MODEL_DIR, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    print(">>> device:", device, flush=True)

    df = pd.read_csv(EVAL_FILE)[["input", "output"]].copy()
    df["input"] = df["input"].astype(str)
    df["output"] = df["output"].astype(str)

    random.seed(args.seed)
    idxs = list(range(len(df)))
    random.shuffle(idxs)

    rows = []
    for j, i in enumerate(idxs[:args.n]):
        inp = df.loc[i, "input"]
        gt = df.loc[i, "output"]

        pred_greedy = generate_text(model, tokenizer, inp, num_beams=1)
        pred_beam = generate_text(model, tokenizer, inp, num_beams=args.beam)

        rows.append({
            "idx": j,
            "emoji_input": inp,
            "reference": gt,
            "pred_greedy": pred_greedy,
            "pred_beam": pred_beam
        })

        print(f"[{j+1}/{args.n}] done", flush=True)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("✅ saved:", out_path)

if __name__ == "__main__":
    main()