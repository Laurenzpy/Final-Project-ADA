import os
import argparse
import pandas as pd
from collections import Counter

def summarize(preds):
    preds = [str(x).strip() for x in preds]
    lens = [len(p.split()) for p in preds]
    c = Counter(preds)
    top_pred, top_count = c.most_common(1)[0]
    return {
        "avg_pred_len_words": sum(lens) / len(lens) if lens else 0.0,
        "unique_rate": len(c) / len(preds) if preds else 0.0,
        "top1_freq": top_count / len(preds) if preds else 0.0,
        "top_prediction_preview": top_pred[:100],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="report_assets/samples/qualitative_examples.csv")
    ap.add_argument("--out_csv", default="report_assets/tables/main_results.csv")
    args = ap.parse_args()

    assert os.path.isfile(args.in_csv), f"Missing: {args.in_csv}. Run export_qualitative_examples first."

    df = pd.read_csv(args.in_csv)

    s_g = summarize(df["pred_greedy"].tolist())
    s_b = summarize(df["pred_beam"].tolist())

    rows = [
        {"model": "t5_e2t", "decoding": "greedy", **s_g},
        {"model": "t5_e2t", "decoding": "beam", **s_b},
    ]

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print("✅ saved:", args.out_csv)

if __name__ == "__main__":
    main()