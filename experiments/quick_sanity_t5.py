import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

TASK_PREFIX = "emoji2text: "
MODEL_NAME = "t5-small"

# Choose device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)

# 1) Load tiny batch
df = pd.read_csv("data/emoji_dataset_stage1_e2t.csv")[["input", "output"]].dropna().astype(str).head(16).copy()
df["input"] = df["input"].apply(lambda x: TASK_PREFIX + x)

print("Sample rows (what the model sees):")
for i in range(min(3, len(df))):
    print(f"\nSRC: {df.iloc[i]['input']}")
    print(f"TGT: {df.iloc[i]['output']}")

ds = Dataset.from_pandas(df)

# 2) Tokenize
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

def preprocess(batch):
    x = tok(batch["input"], truncation=True, max_length=64)
    y = tok(text_target=batch["output"], truncation=True, max_length=64)
    x["labels"] = y["input_ids"]
    return x

ds_tok = ds.map(preprocess, batched=True, remove_columns=ds.column_names)

# 3) Train briefly on same tiny set
args = Seq2SeqTrainingArguments(
    output_dir="quick_sanity_out",
    per_device_train_batch_size=8,
    num_train_epochs=30,
    learning_rate=5e-4,
    logging_steps=5,
    save_total_limit=1,
    report_to=[],
    fp16=False,
    predict_with_generate=False,  # IMPORTANT: avoid generate inside trainer on MPS
    dataloader_pin_memory=False,  # IMPORTANT: avoids MPS pin_memory warning
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=ds_tok,
    data_collator=DataCollatorForSeq2Seq(tok, model=model),
)

trainer.train()

# 4) Generate a few predictions (device-safe)
print("\n=== After overfit (should be close to targets) ===")
model.eval()

for i in range(3):
    src = df.iloc[i]["input"]
    tgt = df.iloc[i]["output"]

    enc = tok(src, return_tensors="pt", truncation=True, max_length=64)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}  # move tensors to same device as model

    with torch.no_grad():
        out = model.generate(**enc, num_beams=4, max_new_tokens=32)

    pred = tok.decode(out[0].detach().cpu(), skip_special_tokens=True)
    print(f"\nSRC: {src}")
    print(f"TGT: {tgt}")
    print(f"PRD: {pred}")