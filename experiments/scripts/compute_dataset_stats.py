import os
import json
import argparse
import pandas as pd

EVAL_FILE = os.path.join("data", "emoji_dataset_stage5_e2t.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report_assets/tables/dataset_stats.json")
    ap.add_argument("--max_emoji_len", type=int, default=64)
    ap.add_argument("--max_text_len", type=int, default=32)
    args = ap.parse_args()

    assert os.path.isfile(EVAL_FILE), f"EVAL_FILE not found: {EVAL_FILE}"

    df = pd.read_csv(EVAL_FILE)[["input", "output"]].copy()
    df["input"] = df["input"].astype(str)
    df["output"] = df["output"].astype(str)

    emoji_lens = df["input"].apply(lambda s: len(s.split()))
    text_lens = df["output"].apply(lambda s: len(s.split()))

    stats = {
        "file": EVAL_FILE,
        "n_samples": int(len(df)),
        "avg_emoji_len_tokens": float(emoji_lens.mean()),
        "median_emoji_len_tokens": float(emoji_lens.median()),
        "max_emoji_len_tokens": int(emoji_lens.max()),
        "avg_text_len_words": float(text_lens.mean()),
        "median_text_len_words": float(text_lens.median()),
        "max_text_len_words": int(text_lens.max()),
        "max_input_len_used_in_tokenizer": args.max_emoji_len,
        "max_new_tokens_generation": args.max_text_len,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("✅ saved:", args.out)

if __name__ == "__main__":
    main()