import os
import pandas as pd

DATA_DIR = "data"

for i in range(1, 7):
    in_path = os.path.join(DATA_DIR, f"emoji_dataset_stage{i}.csv")
    out_path = os.path.join(DATA_DIR, f"emoji_dataset_stage{i}_e2t.csv")

    df = pd.read_csv(in_path)
    df2 = pd.DataFrame({
        "input": df["output"],   # emojis -> input
        "output": df["input"],   # text -> output
    })
    df2.to_csv(out_path, index=False)
    print("wrote", out_path, "rows:", len(df2))
