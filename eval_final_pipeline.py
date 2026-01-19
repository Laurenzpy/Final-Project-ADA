# eval_final_pipeline.py
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# ✅ WICHTIG: evaluiere exakt den Backend-Translator
from backend.final_emoji_translator import HybridConfig, FinalEmojiTranslator

# Metrics
try:
    import sacrebleu
    HAS_SACREBLEU = True
except Exception:
    sacrebleu = None
    HAS_SACREBLEU = False

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except Exception:
    rouge_scorer = None
    HAS_ROUGE = False


# ----------------------------
# Paths
# ----------------------------
STAGE5_PATH = "data/emoji_dataset_stage5_e2t.csv"

TM_TRAIN_PATHS = [
    "data/emoji_dataset_stage1_e2t.csv",
    "data/emoji_dataset_stage2_e2t.csv",
    "data/emoji_dataset_stage3_e2t.csv",
    "data/emoji_dataset_stage4_e2t.csv",
]

T5_DIR = "artifacts/t5_e2t"
OUT_DIR = "eval_results"
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------
# Helpers
# ----------------------------
def safe_read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "input" not in df.columns or "output" not in df.columns:
        raise ValueError(f"Expected columns input,output in {path}")
    return df.astype(str)


def corpus_bleu(refs: List[str], hyps: List[str]) -> float:
    if not HAS_SACREBLEU:
        return float("nan")
    return float(sacrebleu.corpus_bleu(hyps, [refs]).score)  # 0..100


def corpus_chrf(refs: List[str], hyps: List[str]) -> float:
    if not HAS_SACREBLEU:
        return float("nan")
    return float(sacrebleu.corpus_chrf(hyps, [refs]).score)  # 0..100


def corpus_rougeL(refs: List[str], hyps: List[str]) -> float:
    if not HAS_ROUGE:
        return float("nan")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for r, h in zip(refs, hyps):
        s = scorer.score(r, h)["rougeL"].fmeasure
        scores.append(s)
    return float(np.mean(scores))  # 0..1


def sentence_bleu(ref: str, hyp: str) -> float:
    if not HAS_SACREBLEU:
        return float("nan")
    try:
        return float(sacrebleu.sentence_bleu(hyp, [ref]).score)
    except Exception:
        return 0.0


def quantiles_ms(x: List[float]) -> Dict[str, float]:
    arr = np.array(x, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.quantile(arr, 0.50)),
        "p90_ms": float(np.quantile(arr, 0.90)),
        "p99_ms": float(np.quantile(arr, 0.99)),
    }


def bucket(score: float) -> str:
    if score >= 0.80:
        return ">=0.80"
    if score >= 0.60:
        return "0.60-0.79"
    if score >= 0.35:
        return "0.35-0.59"
    if score >= 0.20:
        return "0.20-0.34"
    return "<0.20"


# ----------------------------
# Evaluation
# ----------------------------
def run_eval(
    name: str,
    cfg: HybridConfig,
    stage5: pd.DataFrame,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    print(f"\n=== RUN: {name} ===")
    tr = FinalEmojiTranslator(cfg)

    if max_samples is not None and max_samples > 0:
        stage5 = stage5.sample(n=min(max_samples, len(stage5)), random_state=42).reset_index(drop=True)

    rows, latencies = [], []
    mode_counts, score_buckets, bleu_by_mode = {}, {}, {}

    refs = stage5["output"].tolist()
    hyps = []

    for _, r in tqdm(stage5.iterrows(), total=len(stage5)):
        out = tr.translate(r["input"])

        pred = out.get("prediction", "")
        mode = out.get("mode", "unknown")
        score = float(out.get("retrieval_score", 0.0))
        lat = float(out.get("latency_ms", float("nan")))

        hyps.append(pred)
        latencies.append(lat)

        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        score_buckets[bucket(score)] = score_buckets.get(bucket(score), 0) + 1

        sb = sentence_bleu(r["output"], pred)
        bleu_by_mode.setdefault(mode, []).append(sb)

        rows.append({
            "input": r["input"],
            "gt": r["output"],
            "pred": pred,
            "mode": mode,
            "retrieval_score": score,
            "latency_ms": lat,
        })

    df = pd.DataFrame(rows)

    # qualitative (nur wenn sacrebleu da ist)
    if HAS_SACREBLEU:
        df["sentence_bleu"] = [sentence_bleu(g, p) for g, p in zip(df["gt"], df["pred"])]
        df.sort_values("sentence_bleu", ascending=False).head(25).to_csv(
            f"{OUT_DIR}/qualitative_best__{name}.csv", index=False
        )
        df.sort_values("sentence_bleu").head(25).to_csv(
            f"{OUT_DIR}/qualitative_worst__{name}.csv", index=False
        )

    summary = {
        "run": name,
        "config": asdict(cfg),
        "num_samples": int(len(stage5)),
        "corpus_bleu": corpus_bleu(refs, hyps),
        "corpus_chrf": corpus_chrf(refs, hyps),
        "corpus_rougeL_f": corpus_rougeL(refs, hyps),
        "exact_match_accuracy": float((df["gt"] == df["pred"]).mean()),
        "mode_distribution": mode_counts,
        "avg_sentence_bleu_by_mode": {k: float(np.mean(v)) for k, v in bleu_by_mode.items()},
        "retrieval_score_buckets": score_buckets,
        "latency": quantiles_ms(latencies),
        "avg_pred_len_words": float(np.mean([len(x.split()) for x in hyps])),
        "avg_gt_len_words": float(np.mean([len(x.split()) for x in refs])),
        "has_sacrebleu": bool(HAS_SACREBLEU),
        "has_rouge": bool(HAS_ROUGE),
    }

    df.to_csv(f"{OUT_DIR}/final_pipeline_outputs__{name}.csv", index=False)
    with open(f"{OUT_DIR}/final_pipeline_summary__{name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


def main():
    stage5 = safe_read_csv(STAGE5_PATH)
    print(f"\nLoaded {len(stage5)} Stage-5 samples from: {STAGE5_PATH}")

    summaries: Dict[str, Any] = {}

    # 1) Retrieval only (Baseline)
    summaries["retrieval_only"] = run_eval(
        "retrieval_only",
        HybridConfig(
            tm_train_paths=TM_TRAIN_PATHS,
            t5_model_dir=T5_DIR,
            enable_t5_fallback=False,
        ),
        stage5,
    )

    # 2) Hybrid: T5 only when retrieval score is really low
    summaries["hybrid_t5_under_0_55"] = run_eval(
        "hybrid_t5_under_0_55",
        HybridConfig(
            tm_train_paths=TM_TRAIN_PATHS,
            t5_model_dir=T5_DIR,
            t5_fallback_below_conf=0.55,
            enable_t5_fallback=True,
        ),
        stage5,
    )

    # 3) Hybrid more aggressive
    summaries["hybrid_t5_under_0_35"] = run_eval(
        "hybrid_t5_under_0_35",
        HybridConfig(
            tm_train_paths=TM_TRAIN_PATHS,
            t5_model_dir=T5_DIR,
            t5_fallback_below_conf=0.35,
            enable_t5_fallback=True,
        ),
        stage5,
    )

    with open(f"{OUT_DIR}/final_pipeline_comparison.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()