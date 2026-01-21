from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd


def norm(s: str) -> str:
    return " ".join(str(s).strip().split())


def safe_read_e2t(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "input" not in df.columns or "output" not in df.columns:
        raise ValueError(f"Expected columns input,output in {path}")
    df = df[["input", "output"]].astype(str)
    df["input"] = df["input"].fillna("").map(norm)
    df["output"] = df["output"].fillna("").map(norm)
    df = df[(df["input"] != "") & (df["output"] != "")]
    return df.reset_index(drop=True)


def group_split_by_input(
    df: pd.DataFrame, train_frac: float, dev_frac: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by unique input to avoid leakage.
    All rows for a given input stay within one split.
    test_frac = 1 - train_frac - dev_frac
    """
    if train_frac <= 0 or dev_frac <= 0 or (train_frac + dev_frac) >= 1.0:
        raise ValueError("Need 0<train_frac, 0<dev_frac, train_frac+dev_frac<1")

    rng = np.random.RandomState(seed)
    inputs = df["input"].unique().tolist()
    rng.shuffle(inputs)

    n = len(inputs)
    n_train = int(round(n * train_frac))
    n_dev = int(round(n * dev_frac))
    n_train = min(n_train, n)
    n_dev = min(n_dev, n - n_train)

    train_inputs = set(inputs[:n_train])
    dev_inputs = set(inputs[n_train : n_train + n_dev])
    test_inputs = set(inputs[n_train + n_dev :])

    train_df = df[df["input"].isin(train_inputs)].reset_index(drop=True)
    dev_df = df[df["input"].isin(dev_inputs)].reset_index(drop=True)
    test_df = df[df["input"].isin(test_inputs)].reset_index(drop=True)
    return train_df, dev_df, test_df


def stats(name: str, df: pd.DataFrame) -> None:
    u = int(df["input"].nunique()) if len(df) else 0
    r = int(len(df))
    avg = (r / u) if u > 0 else float("nan")
    print(f" - {name}: rows={r} unique_inputs={u} avg_outputs_per_input={avg:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--seed", type=int, default=42)

    # ✅ NEW DEFAULTS: Stage5 split is now 60/20/20 (by input)
    ap.add_argument("--stage5_train_frac", type=float, default=0.60)
    ap.add_argument("--stage5_dev_frac", type=float, default=0.20)

    ap.add_argument("--out_train", default="data/final_train_e2t.csv")
    ap.add_argument("--out_dev", default="data/final_dev_e2t.csv")
    ap.add_argument("--out_test_stage5", default="data/final_test_stage5_e2t.csv")
    ap.add_argument("--out_stage5_train", default="data/final_stage5_train_e2t.csv")

    args = ap.parse_args()
    data_dir = Path(args.data_dir)

    # Load Stage 1-4 for base training
    stage14_paths: List[Path] = [
        data_dir / "emoji_dataset_stage1_e2t.csv",
        data_dir / "emoji_dataset_stage2_e2t.csv",
        data_dir / "emoji_dataset_stage3_e2t.csv",
        data_dir / "emoji_dataset_stage4_e2t.csv",
    ]
    for p in stage14_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    df_stage14 = pd.concat([safe_read_e2t(str(p)) for p in stage14_paths], ignore_index=True)

    # Load Stage 5 and split within Stage 5 (IID for Stage 5)
    stage5_path = data_dir / "emoji_dataset_stage5_e2t.csv"
    if not stage5_path.exists():
        raise FileNotFoundError(f"Missing: {stage5_path}")
    df_stage5 = safe_read_e2t(str(stage5_path))

    s5_train, s5_dev, s5_test = group_split_by_input(
        df_stage5,
        train_frac=args.stage5_train_frac,
        dev_frac=args.stage5_dev_frac,
        seed=args.seed,
    )

    # Final train = Stage1-4 + Stage5_train
    final_train = pd.concat([df_stage14, s5_train], ignore_index=True).reset_index(drop=True)
    final_dev = s5_dev.copy()
    final_test = s5_test.copy()

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    final_train.to_csv(args.out_train, index=False)
    final_dev.to_csv(args.out_dev, index=False)
    final_test.to_csv(args.out_test_stage5, index=False)
    s5_train.to_csv(args.out_stage5_train, index=False)  # helpful for debugging / reporting

    print("✅ Final datasets written:")
    print("Train includes Stage1-4 + Stage5_train (IID Stage5 main task)")
    stats("final_train", final_train)
    stats("final_dev(Stage5_dev)", final_dev)
    stats("final_test_stage5(Stage5_test)", final_test)
    stats("stage5_train_only", s5_train)

    print("\nFiles:")
    print(f" - {args.out_train}")
    print(f" - {args.out_dev}")
    print(f" - {args.out_test_stage5}")
    print(f" - {args.out_stage5_train}")


if __name__ == "__main__":
    main()