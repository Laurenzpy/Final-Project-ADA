import os
import random
import argparse
import pandas as pd

from translator.pipeline import EmojiTranslator

EVAL_FILE = os.path.join("data", "emoji_dataset_stage5_e2t.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="report_assets/samples/pipeline_predictions_300.csv")
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--ollama", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(EVAL_FILE):
        raise SystemExit(f"Missing eval file: {EVAL_FILE}")

    df = pd.read_csv(EVAL_FILE)[["input", "output"]].copy()
    df["input"] = df["input"].astype(str)
    df["output"] = df["output"].astype(str)

    random.seed(args.seed)
    idxs = list(range(len(df)))
    random.shuffle(idxs)
    idxs = idxs[:args.n]

    tr = EmojiTranslator(
        retrieval_threshold=args.threshold,
        retrieval_topk=args.topk,
        enable_ollama_fallback=args.ollama,
    )

    rows = []
    for j, i in enumerate(idxs, start=1):
        inp = df.loc[i, "input"]
        gt = df.loc[i, "output"]

        pred, stage, debug = tr.translate(inp)

        rows.append({
            "idx": i,
            "emoji_input": inp,
            "reference": gt,
            "pred_pipeline": pred,
            "stage": stage,
        })

        if j % 25 == 0:
            print(f"[{j}/{args.n}] done", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("✅ saved:", args.out)

if __name__ == "__main__":
    main()