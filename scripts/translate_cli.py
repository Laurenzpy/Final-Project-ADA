import os
import argparse
import pandas as pd

from translator.pipeline import EmojiTranslator

EVAL_FILE = os.path.join("data", "emoji_dataset_stage5_e2t.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=None, help="Row index from eval CSV")
    ap.add_argument("--n_random", type=int, default=0, help="Show N random samples")
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--ollama", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(EVAL_FILE):
        raise SystemExit(f"Missing dataset file: {EVAL_FILE}")

    df = pd.read_csv(EVAL_FILE)[["input", "output"]].copy()
    df["input"] = df["input"].astype(str)
    df["output"] = df["output"].astype(str)

    translator = EmojiTranslator(
        retrieval_threshold=args.threshold,
        retrieval_topk=args.topk,
        enable_ollama_fallback=args.ollama,
    )

    if args.n_random > 0:
        samples = df.sample(n=min(args.n_random, len(df)), random_state=42)
        for i, row in samples.iterrows():
            pred, stage, _ = translator.translate(row["input"])
            print("=" * 70)
            print("ROW :", i)
            print("IN  :", row["input"])
            print("GT  :", row["output"])
            print("STAGE:", stage)
            print("OUT :", pred)
        return

    if args.idx is None:
        raise SystemExit("Please provide --idx <row_index> or --n_random <N>")

    if args.idx < 0 or args.idx >= len(df):
        raise SystemExit(f"--idx must be between 0 and {len(df)-1}")

    row = df.loc[args.idx]
    pred, stage, _ = translator.translate(row["input"])

    print("ROW :", args.idx)
    print("IN  :", row["input"])
    print("GT  :", row["output"])
    print("STAGE:", stage)
    print("OUT :", pred)

if __name__ == "__main__":
    main()