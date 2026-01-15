import os
import json
import argparse
import pandas as pd
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

def compute_bleu(references, predictions):
    # sacrebleu expects references as list-of-lists: [refs] for single-reference corpora
    bleu = BLEU(effective_order=True)
    score = bleu.corpus_score(predictions, [references])
    return float(score.score)  # 0..100

def compute_rouge(references, predictions):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for ref, pred in zip(references, predictions):
        s = scorer.score(ref, pred)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    return {
        "rouge1_f": float(sum(r1) / len(r1)) if r1 else 0.0,
        "rouge2_f": float(sum(r2) / len(r2)) if r2 else 0.0,
        "rougeL_f": float(sum(rl) / len(rl)) if rl else 0.0,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="report_assets/samples/qualitative_examples.csv")
    ap.add_argument("--main_results_csv", default="report_assets/tables/main_results.csv")
    ap.add_argument("--out_json", default="report_assets/tables/text_metrics.json")
    args = ap.parse_args()

    assert os.path.isfile(args.in_csv), f"Missing: {args.in_csv}"

    df = pd.read_csv(args.in_csv)
    refs = df["reference"].astype(str).tolist()

    results = {}

    for col, name in [("pred_greedy", "greedy"), ("pred_beam", "beam")]:
        preds = df[col].astype(str).tolist()

        bleu = compute_bleu(refs, preds)
        rouge = compute_rouge(refs, preds)

        results[name] = {
            "bleu": bleu,          # 0..100
            **rouge,              # 0..1
            "n_samples": int(len(preds)),
        }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("✅ saved:", args.out_json)

    # Optionally: merge into main_results.csv if it exists
    if os.path.isfile(args.main_results_csv):
        mr = pd.read_csv(args.main_results_csv)

        def attach(decoding_key, decoding_label):
            row_idx = mr.index[mr["decoding"] == decoding_label]
            if len(row_idx) == 0:
                return
            i = row_idx[0]
            mr.loc[i, "bleu"] = results[decoding_key]["bleu"]
            mr.loc[i, "rouge1_f"] = results[decoding_key]["rouge1_f"]
            mr.loc[i, "rouge2_f"] = results[decoding_key]["rouge2_f"]
            mr.loc[i, "rougeL_f"] = results[decoding_key]["rougeL_f"]
            mr.loc[i, "n_metric_samples"] = results[decoding_key]["n_samples"]

        attach("greedy", "greedy")
        attach("beam", "beam")

        mr.to_csv(args.main_results_csv, index=False)
        print("✅ updated:", args.main_results_csv)
    else:
        print("ℹ️ main_results.csv not found, skipped update.")

if __name__ == "__main__":
    main()