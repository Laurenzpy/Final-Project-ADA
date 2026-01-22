from transformers import T5Tokenizer, T5ForConditionalGeneration
from tokenizers import emojis_to_text_aliases  # <-- nutzt jetzt eure zentrale Funktion

model_dir = "artifacts/t5_e2t"

tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)
model.eval()

inputs = ["🍕 ❤️", "🚗 🔧", "🎵 😊"]

for x in inputs:
    x_text = emojis_to_text_aliases(x)
    ids = tokenizer(x_text, return_tensors="pt")
    out = model.generate(**ids, max_new_tokens=20)
    print(f"{x}  (as '{x_text}') -> {tokenizer.decode(out[0], skip_special_tokens=True)}")