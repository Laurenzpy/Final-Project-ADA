import os
import json
import subprocess
import torch
import time

def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return None

def main():
    info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_hash(),
        "python_version": os.popen("python --version").read().strip(),
        "torch_version": torch.__version__,
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
        "model_dir": "artifacts/t5_e2t",
        "eval_file": "data/emoji_dataset_stage5_e2t.csv"
    }

    os.makedirs("report_assets/logs", exist_ok=True)
    with open("report_assets/logs/run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("✅ saved: report_assets/logs/run_info.json")

if __name__ == "__main__":
    main()