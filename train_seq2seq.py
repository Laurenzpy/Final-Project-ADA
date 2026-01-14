import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from seq2seq.tokenizers import tokenize_emojis, tokenize_text
from seq2seq.vocab import build_vocab
from seq2seq.dataset import Emoji2TextDataset, collate_batch
from seq2seq.model import Encoder, Decoder, Seq2Seq

DATA_DIR = "data"

TRAIN_FILES = [os.path.join(DATA_DIR, f"emoji_dataset_stage{i}_e2t.csv") for i in range(1, 5)]
VAL_FILES   = [os.path.join(DATA_DIR, "emoji_dataset_stage5_e2t.csv")]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokens_emoji2text(csv_paths):
    """
    Build vocab for EMOJI -> TEXT
    src = emojis
    tgt = text
    """
    src_tokens, tgt_tokens = [], []
    for p in csv_paths:
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            src_tokens.append(tokenize_emojis(str(r["input"])))
            tgt_tokens.append(tokenize_text(str(r["output"])))
    return src_tokens, tgt_tokens


def main():
    print(">>> Training started", flush=True)
    print(">>> DEVICE:", DEVICE, flush=True)

    # 1) Build vocabs from TRAIN only (emoji -> text)
    src_tokens, tgt_tokens = load_tokens_emoji2text(TRAIN_FILES)

    src_vocab = build_vocab(src_tokens, max_size=5000)
    tgt_vocab = build_vocab(tgt_tokens, max_size=20000)

    print(f">>> vocab built: src={len(src_vocab.itos)} tgt={len(tgt_vocab.itos)}", flush=True)

    os.makedirs("artifacts", exist_ok=True)
    src_vocab.save("artifacts/src_vocab.json")
    tgt_vocab.save("artifacts/tgt_vocab.json")

    # 2) Datasets / loaders  (IMPORTANT: direction!)
    train_ds = Emoji2TextDataset(
        TRAIN_FILES, src_vocab, tgt_vocab, direction="emoji2text"
    )
    val_ds = Emoji2TextDataset(
        VAL_FILES, src_vocab, tgt_vocab, direction="emoji2text"
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, src_vocab.pad_id, tgt_vocab.pad_id)
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, src_vocab.pad_id, tgt_vocab.pad_id)
    )

    print(
        f">>> dataloaders built: train_batches={len(train_dl)} val_batches={len(val_dl)}",
        flush=True
    )

    # 3) Model
    encoder = Encoder(len(src_vocab.itos), emb_dim=128, hid_dim=256)
    decoder = Decoder(len(tgt_vocab.itos), emb_dim=128, hid_dim=256)
    model = Seq2Seq(encoder, decoder, pad_id=tgt_vocab.pad_id).to(DEVICE)

    # 4) Optimizer + Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id)

    best_val_loss = float("inf")

    # 5) Training loop
    EPOCHS = 25
    for epoch in range(1, EPOCHS + 1):
        print(f"\n>>> Epoch {epoch}/{EPOCHS}", flush=True)

        model.train()
        train_loss_sum = 0.0

        for src_pad, src_lens, tgt_pad, _ in tqdm(train_dl, desc=f"Epoch {epoch} [train]"):
            src_pad = src_pad.to(DEVICE)
            src_lens = src_lens.to(DEVICE)
            tgt_pad = tgt_pad.to(DEVICE)

            optimizer.zero_grad()

            tf = max(0.1, 1.0 - (epoch - 1) / (EPOCHS * 0.8))  # von ~1.0 runter bis 0.1
            logits = model(src_pad, src_lens, tgt_pad, teacher_forcing_ratio=tf)

            print(f">>> teacher_forcing={tf:.2f}", flush=True)
            
            gold = tgt_pad[:, 1:]

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                gold.reshape(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()

        train_loss = train_loss_sum / max(1, len(train_dl))

        # 6) Validation
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for src_pad, src_lens, tgt_pad, _ in tqdm(val_dl, desc=f"Epoch {epoch} [val]"):
                src_pad = src_pad.to(DEVICE)
                src_lens = src_lens.to(DEVICE)
                tgt_pad = tgt_pad.to(DEVICE)

                logits = model(
                    src_pad,
                    src_lens,
                    tgt_pad,
                    teacher_forcing_ratio=0.0
                )

                gold = tgt_pad[:, 1:]
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    gold.reshape(-1)
                )
                val_loss_sum += loss.item()

        val_loss = val_loss_sum / max(1, len(val_dl))

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}",
            flush=True
        )

        # 7) Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "src_vocab_itos": src_vocab.itos,
                    "tgt_vocab_itos": tgt_vocab.itos,
                    "emb_dim": 128,
                    "hid_dim": 256,
                },
                "artifacts/seq2seq_best.pt"
            )
            print("✅ Saved artifacts/seq2seq_best.pt", flush=True)

    print("\n>>> Training finished", flush=True)
    


if __name__ == "__main__":
    main()
