import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qual_csv", default="report_assets/samples/qualitative_examples.csv")
    ap.add_argument("--stats_json", default="report_assets/tables/dataset_stats.json")
    ap.add_argument("--fig_dir", default="report_assets/figures")
    args = ap.parse_args()

    assert os.path.isfile(args.qual_csv), f"Missing: {args.qual_csv}"
    assert os.path.isfile(args.stats_json), f"Missing: {args.stats_json}"

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.qual_csv)

    # (1) Pred length distribution
    greedy_lens = [len(str(x).split()) for x in df["pred_greedy"]]
    beam_lens = [len(str(x).split()) for x in df["pred_beam"]]

    plt.figure()
    plt.hist(greedy_lens, bins=20, alpha=0.7, label="greedy")
    plt.hist(beam_lens, bins=20, alpha=0.7, label="beam")
    plt.xlabel("Predicted length (words)")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(fig_dir / "pred_length_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()

    # (2) Collapse rate (Top-1 sentence frequency)
    def top1_rate(col):
        preds = [str(x).strip() for x in df[col].tolist()]
        c = Counter(preds)
        top_pred, top_count = c.most_common(1)[0]
        return top_pred, top_count / len(preds)

    top_g, rate_g = top1_rate("pred_greedy")
    top_b, rate_b = top1_rate("pred_beam")

    plt.figure()
    plt.bar(["greedy", "beam"], [rate_g, rate_b])
    plt.ylabel("Top-1 sentence frequency")
    plt.savefig(fig_dir / "collapse_rate.png", dpi=200, bbox_inches="tight")
    plt.close()

    # (3) Dataset length distributions (from stats)
    with open(args.stats_json, "r", encoding="utf-8") as f:
        stats = json.load(f)

    # A tiny “summary figure” as text box
    plt.figure()
    plt.axis("off")
    txt = (
        f"Dataset: {stats.get('file','')}\n"
        f"N samples: {stats.get('n_samples')}\n"
        f"Avg emoji len (tokens): {stats.get('avg_emoji_len_tokens'):.2f}\n"
        f"Avg text len (words): {stats.get('avg_text_len_words'):.2f}\n"
        f"Max emoji len (tokens): {stats.get('max_emoji_len_tokens')}\n"
        f"Max text len (words): {stats.get('max_text_len_words')}\n"
    )
    plt.text(0.01, 0.99, txt, va="top")
    plt.savefig(fig_dir / "dataset_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("✅ saved figures to:", fig_dir)
    print("Top greedy:", top_g)
    print("Top beam:", top_b)

if __name__ == "__main__":
    main()