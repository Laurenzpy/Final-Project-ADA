from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

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


def split_by_input(
    df: pd.DataFrame, train_frac: float, dev_frac: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by unique 'input' (group split) to avoid leakage.
    All rows for a given input go to the same split.
    """
    rng = np.random.RandomState(seed)
    unique_inputs = df["input"].unique().tolist()
    rng.shuffle(unique_inputs)

    n = len(unique_inputs)
    n_train = int(round(n * train_frac))
    n_dev = int(round(n * dev_frac))
    n_train = min(n_train, n)
    n_dev = min(n_dev, n - n_train)

    train_inputs = set(unique_inputs[:n_train])
    dev_inputs = set(unique_inputs[n_train : n_train + n_dev])
    test_inputs = set(unique_inputs[n_train + n_dev :])

    train_df = df[df["input"].isin(train_inputs)].reset_index(drop=True)
    dev_df = df[df["input"].isin(dev_inputs)].reset_index(drop=True)
    test_df = df[df["input"].isin(test_inputs)].reset_index(drop=True)

    return train_df, dev_df, test_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--stages", nargs="+", default=["1", "2", "3", "4"], help="Stages to include for IID split (default: 1 2 3 4)")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--dev_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_prefix", default="iid", help="Output prefix: creates data/<prefix>_train_e2t.csv etc.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)

    paths: List[Path] = []
    for s in args.stages:
        p = data_dir / f"emoji_dataset_stage{s}_e2t.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
        paths.append(p)

    df_all = pd.concat([safe_read_e2t(str(p)) for p in paths], ignore_index=True)

    train_df, dev_df, test_df = split_by_input(df_all, args.train_frac, args.dev_frac, args.seed)

    out_train = data_dir / f"{args.out_prefix}_train_e2t.csv"
    out_dev = data_dir / f"{args.out_prefix}_dev_e2t.csv"
    out_test = data_dir / f"{args.out_prefix}_test_e2t.csv"

    train_df.to_csv(out_train, index=False)
    dev_df.to_csv(out_dev, index=False)
    test_df.to_csv(out_test, index=False)

    # quick stats
    def stats(x: pd.DataFrame) -> Dict[str, int]:
        return {"rows": int(len(x)), "unique_inputs": int(x["input"].nunique())}

    print("✅ IID splits written:")
    print(f" - {out_train}  {stats(train_df)}")
    print(f" - {out_dev}    {stats(dev_df)}")
    print(f" - {out_test}   {stats(test_df)}")


if __name__ == "__main__":
    main()