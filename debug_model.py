import os
import torch
from torch.utils.data import DataLoader

from seq2seq.vocab import build_vocab
from seq2seq.dataset import Emoji2TextDataset, collate_batch
from seq2seq.tokenizers import tokenize_emojis, tokenize_text
from seq2seq.model import Encoder, Decoder, Seq2Seq

DATA_DIR = "data"
TRAIN_FILES = [os.path.join(DATA_DIR, f"emoji_dataset_stage{i}.csv") for i in range(1, 5)]

# build vocab
import pandas as pd
src, tgt = [], []
for p in TRAIN_FILES:
    df = pd.read_csv(p)
    for _, r in df.iterrows():
        src.append(tokenize_emojis(r["input"]))
        tgt.append(tokenize_text(r["output"]))

src_vocab = build_vocab(src, max_size=5000)
tgt_vocab = build_vocab(tgt, max_size=8000)

ds = Emoji2TextDataset(TRAIN_FILES, src_vocab, tgt_vocab)
dl = DataLoader(ds, batch_size=4,
                collate_fn=lambda b: collate_batch(b, src_vocab.pad_id, tgt_vocab.pad_id))

src_pad, src_lens, tgt_pad, tgt_lens = next(iter(dl))

encoder = Encoder(len(src_vocab.itos), emb_dim=128, hid_dim=256)
decoder = Decoder(len(tgt_vocab.itos), emb_dim=128, hid_dim=256)
model = Seq2Seq(encoder, decoder, pad_id=tgt_vocab.pad_id)

logits = model(src_pad, src_lens, tgt_pad)

print("logits shape:", logits.shape)
