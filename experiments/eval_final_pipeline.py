from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from backend.final_emoji_translator import HybridConfig, FinalEmojiTranslator

try:
    import sacrebleu
    HAS_SACREBLEU = True
except Exception:
    sacrebleu = None
    HAS_SACREBLEU = False


def safe_read_e2t(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "input" not in df.columns or "output" not in df.columns:
        raise ValueError(f"Expected columns input,output in {path}")
    df = df[["input", "output"]].astype(str)
    df["input"] = df["input"].fillna("").astype(str)
    df["output"] = df["output"].fillna("").astype(str)
    return df


def norm(s: str) -> str:
    return " ".join(str(s).strip().split())


def quantiles_ms(latencies: List[float]) -> Dict[str, float]:
    arr = np.array([x for x in latencies if np.isfinite(x)], dtype=np.float64)
    if arr.size == 0:
        return {"mean_ms": float("nan"), "p50_ms": float("nan"), "p90_ms": float("nan")}
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.quantile(arr, 0.50)),
        "p90_ms": float(np.quantile(arr, 0.90)),
    }


def best_sentence_chrf(pred: str, refs: List[str]) -> float:
    if not HAS_SACREBLEU:
        return float("nan")
    scores = []
    for r in refs:
        scores.append(float(sacrebleu.sentence_chrf(pred, [r]).score))
    return float(max(scores)) if scores else float("nan")


def run_one(
    name: str,
    cfg: HybridConfig,
    test_df: pd.DataFrame,
    refs_by_input: Dict[str, List[str]],
    out_dir: str,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    print(f"\n=== EVAL RUN: {name} ===")

    df = test_df.copy()
    if max_samples is not None and max_samples > 0:
        df = df.sample(n=min(max_samples, len(df)), random_state=seed).reset_index(drop=True)

    tr = FinalEmojiTranslator(cfg)

    rows = []
    lat = []
    em_hits = []
    best_chrf = []

    for _, r in tqdm(df.iterrows(), total=len(df)):
        inp = norm(r["input"])
        gt_single = norm(r["output"])
        refs = [norm(x) for x in refs_by_input.get(inp, [gt_single])]

        out = tr.translate(inp)
        pred = norm(out.get("prediction", ""))

        lat.append(float(out.get("latency_ms", float("nan"))))

        hit = float(pred in set(refs))
        em_hits.append(hit)

        best_chrf.append(best_sentence_chrf(pred, refs))

        rows.append(
            {
                "input": inp,
                "gt_single": gt_single,
                "pred": pred,
                "mode": out.get("mode", ""),
                "multi_refs": " ||| ".join(refs),
                "multi_ref_exact_match": hit,
                "best_sentence_chrf": best_chrf[-1],
                "retrieval_score": out.get("retrieval_score", 0.0),
                "ranker_choice": out.get("ranker_choice", None),
                "latency_ms": out.get("latency_ms", None),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(out_dir, f"predictions__{name}.csv"), index=False)

    summary = {
        "run": name,
        "num_samples": int(len(out_df)),
        "config": asdict(cfg),
        "multi_ref_exact_match": float(np.mean(em_hits)) if em_hits else float("nan"),
        "avg_best_sentence_chrf": float(np.nanmean(best_chrf)) if best_chrf else float("nan"),
        "latency": quantiles_ms(lat),
        "has_sacrebleu": bool(HAS_SACREBLEU),
    }

    with open(os.path.join(out_dir, f"summary__{name}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_path", required=True, help="CSV with columns input,output")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--ranker_dir", default="artifacts/t5_ranker")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--tm_paths",
        nargs="+",
        default=[
            "data/emoji_dataset_stage1_e2t.csv",
            "data/emoji_dataset_stage2_e2t.csv",
            "data/emoji_dataset_stage3_e2t.csv",
            "data/emoji_dataset_stage4_e2t.csv",
        ],
    )
    ap.add_argument("--top_k", type=int, default=16)
    args = ap.parse_args()

    # auto out_dir if not provided
    out_dir = args.out_dir or f"eval_results_{PathSafe(args.test_path)}_k{args.top_k}"
    args.out_dir = out_dir

    os.makedirs(args.out_dir, exist_ok=True)

    # proof file
    with open(os.path.join(args.out_dir, "TOP_K.txt"), "w", encoding="utf-8") as f:
        f.write(str(args.top_k) + "\n")
    with open(os.path.join(args.out_dir, "TEST_PATH.txt"), "w", encoding="utf-8") as f:
        f.write(str(args.test_path) + "\n")

    test_df = safe_read_e2t(args.test_path)
    test_df["input"] = test_df["input"].map(norm)
    test_df["output"] = test_df["output"].map(norm)

    # multi-reference set from the test set itself (fair for one-to-many)
    refs_by_input: Dict[str, List[str]] = (
        test_df.groupby("input")["output"].apply(lambda x: sorted(list(set(x.tolist())))).to_dict()
    )

    print(f"Loaded test: {args.test_path} (n={len(test_df)})")
    max_samples = args.max_samples if args.max_samples and args.max_samples > 0 else None

    summaries: Dict[str, Any] = {}

    summaries["retrieval_only"] = run_one(
        name="retrieval_only",
        cfg=HybridConfig(
            tm_train_paths=args.tm_paths,
            ranker_model_dir=args.ranker_dir,
            use_ranker=False,
            require_ranker=False,
            top_k=args.top_k,
            compute_retrieval_debug=False,
        ),
        test_df=test_df,
        refs_by_input=refs_by_input,
        out_dir=args.out_dir,
        max_samples=max_samples,
        seed=args.seed,
    )

    summaries["rag_ranker"] = run_one(
        name="rag_ranker",
        cfg=HybridConfig(
            tm_train_paths=args.tm_paths,
            ranker_model_dir=args.ranker_dir,
            use_ranker=True,
            require_ranker=True,
            top_k=args.top_k,
            compute_retrieval_debug=False,
        ),
        test_df=test_df,
        refs_by_input=refs_by_input,
        out_dir=args.out_dir,
        max_samples=max_samples,
        seed=args.seed,
    )

    with open(os.path.join(args.out_dir, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    with open(os.path.join(args.out_dir, f"comparison_k{args.top_k}.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print("\n=== DONE ===")
    print(f">>> Saved: {os.path.join(args.out_dir, 'comparison.json')}")


def PathSafe(p: str) -> str:
    # simple readable slug for filenames
    base = os.path.basename(p)
    base = base.replace(".csv", "").replace(" ", "_")
    return base


if __name__ == "__main__":
    main()