import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = "bart_emoji2text"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    while True:
        s = input("Emoji sequence (1–6, space-separated) > ").strip()
        if not s:
            continue

        inputs = tok(s, return_tensors="pt").to(DEVICE)
        out = model.generate(
            **inputs,
            num_beams=5,
            max_new_tokens=32,
            no_repeat_ngram_size=3,  # helps stop repetition loops
        )
        print("->", tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
