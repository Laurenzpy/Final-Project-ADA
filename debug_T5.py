import os
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


MODEL_DIR = "artifacts/t5_e2t"


def list_dir(path: str, max_items: int = 50):
    if not os.path.exists(path):
        print(f"[ERROR] Pfad existiert nicht: {path}")
        return
    items = sorted(os.listdir(path))
    print(f"\n[INFO] Inhalt von {path} ({len(items)} Dateien/Ordner):")
    for i, it in enumerate(items[:max_items]):
        print(" -", it)
    if len(items) > max_items:
        print(f" ... (+{len(items) - max_items} weitere)")


def show_config(model_dir: str):
    cfg_path = os.path.join(model_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        print("\n[INFO] config.json (Auszug):")
        keys = ["model_type", "vocab_size", "d_model", "num_layers", "num_heads"]
        for k in keys:
            if k in cfg:
                print(f" - {k}: {cfg[k]}")
    else:
        print("\n[WARN] Keine config.json gefunden.")


def main():
    print("[INFO] MODEL_DIR =", MODEL_DIR)
    list_dir(MODEL_DIR)
    show_config(MODEL_DIR)

    # Tokenizer & Modell laden
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    model.eval()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("\n[INFO] device:", device)

    # Test inputs
    inputs = [
        "🍕 ❤️",
        "🚗 🔧",
        "🎵 😊",
        "😀",
        "🚀",
        "❤️",
        "🍕"
    ]

    print("\n[INFO] Tokenizer sanity check:")
    for x in inputs[:3]:
        enc = tokenizer(x, return_tensors="pt")
        ids = enc["input_ids"][0].tolist()
        toks = tokenizer.convert_ids_to_tokens(ids)
        print(f" - {x} -> ids[:20]={ids[:20]} | toks[:20]={toks[:20]}")

    def run_generate(gen_kwargs, title):
        print(f"\n===== {title} =====")
        with torch.no_grad():
            for x in inputs:
                enc = tokenizer(x, return_tensors="pt").to(device)

                out = model.generate(
                    **enc,
                    **gen_kwargs
                )

                decoded = tokenizer.decode(out[0], skip_special_tokens=True)

                # Zusätzlich: Logits check am ersten Decoding-Step (zeigt ob Modell "immer gleich" ist)
                # Wir nehmen einmal greedy: next-token distribution für den ersten Schritt
                out_greedy = model.generate(**enc, max_new_tokens=1, do_sample=False)
                decoded1 = tokenizer.decode(out_greedy[0], skip_special_tokens=True)

                print(f"{x} -> {decoded}")
                print(f"    first_step_greedy -> {decoded1}")

    # 1) Greedy (deterministisch)
    run_generate(
        gen_kwargs=dict(
            max_new_tokens=30,
            do_sample=False,
            num_beams=1
        ),
        title="Greedy (do_sample=False)"
    )

    # 2) Beam Search (falls greedy kollabiert)
    run_generate(
        gen_kwargs=dict(
            max_new_tokens=30,
            do_sample=False,
            num_beams=5,
            early_stopping=True
        ),
        title="Beam Search (num_beams=5)"
    )

    # 3) Sampling (um zu sehen ob überhaupt Varianz möglich ist)
    run_generate(
        gen_kwargs=dict(
            max_new_tokens=30,
            do_sample=True,
            top_p=0.9,
            temperature=1.0
        ),
        title="Sampling (top_p=0.9)"
    )

    print("\n[HINWEIS] Wenn ALLES (Greedy/Beam/Sampling) immer denselben Satz liefert,")
    print("dann ist das Modell sehr wahrscheinlich kollabiert oder du lädst nicht den richtigen Checkpoint.")


if __name__ == "__main__":
    main()