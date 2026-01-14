import os
import pandas as pd
from torch.utils.data import DataLoader

from seq2seq.tokenizers import tokenize_emojis, tokenize_text
from seq2seq.vocab import build_vocab
from seq2seq.dataset import Emoji2TextDataset, collate_batch

DATA_DIR = "data"
TRAIN_FILES = [os.path.join(DATA_DIR, f"emoji_dataset_stage{i}.csv") for i in range(1, 5)]

def load_tokens(csv_paths):
    src, tgt = [], []
    for p in csv_paths:
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            src.append(tokenize_emojis(r["input"]))
            tgt.append(tokenize_text(r["output"]))
    return src, tgt

src_tokens, tgt_tokens = load_tokens(TRAIN_FILES)
src_vocab = build_vocab(src_tokens, max_size=5000)
tgt_vocab = build_vocab(tgt_tokens, max_size=8000)

ds = Emoji2TextDataset(TRAIN_FILES, src_vocab, tgt_vocab)
dl = DataLoader(ds, batch_size=4, shuffle=True,
                collate_fn=lambda b: collate_batch(b, src_vocab.pad_id, tgt_vocab.pad_id))

src_pad, src_lens, tgt_pad, tgt_lens = next(iter(dl))

print("src_pad shape:", src_pad.shape)
print("src_lens:", src_lens.tolist())
print("tgt_pad shape:", tgt_pad.shape)
print("tgt_lens:", tgt_lens.tolist())

# decode first sample to check
print("\nDecoded sample 0:")
src0 = [t for t in src_vocab.decode(src_pad[0].tolist(), stop_at_eos=False) if t != "<PAD>"]
tgt0 = [t for t in tgt_vocab.decode(tgt_pad[0].tolist(), stop_at_eos=True)]
print("SRC:", "".join(src0))
print("TGT:", " ".join(tgt0))
