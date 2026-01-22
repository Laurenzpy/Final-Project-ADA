import os
import json
import math
import random
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import regex  # important: better emoji/grapheme splitting than Python's re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Paths (yours)
# ---------------------------
STAGE_PATHS = [
    "emoji_dataset_stage1.csv",
    "emoji_dataset_stage2.csv",
    "emoji_dataset_stage3.csv",
    "emoji_dataset_stage4.csv",
    "emoji_dataset_stage5.csv",
    "emoji_dataset_stage6.csv",
]
MERGED_EMOJI_PATH = "merged_emoji_sample.csv"

# ---------------------------
# Tokenization helpers
# ---------------------------
def split_emoji_sequence(s: str) -> List[str]:
    """
    Your stage outputs are space-separated emoji tokens (e.g., "🎵 💃").
    This also handles no-space inputs like "🎵💃" using grapheme clusters.
    """
    s = str(s).strip()
    if not s:
        return []
    if any(ch.isspace() for ch in s):
        return [tok for tok in s.split() if tok]
    # No spaces: split into grapheme clusters (handles multi-codepoint emoji like ☪️)
    clusters = regex.findall(r"\X", s)
    return [c for c in clusters if c.strip()]


_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[.,!?;:]")

def tokenize_english(text: str) -> List[str]:
    text = str(text).strip().lower()
    return _WORD_RE.findall(text)


# ---------------------------
# Vocab
# ---------------------------
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    @property
    def pad(self) -> int:
        return self.stoi["<pad>"]

    @property
    def bos(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos(self) -> int:
        return self.stoi["<eos>"]

    @property
    def unk(self) -> int:
        return self.stoi["<unk>"]

    def encode(self, tokens: List[str], add_bos_eos: bool = True) -> List[int]:
        ids = [self.stoi.get(t, self.unk) for t in tokens]
        if add_bos_eos:
            return [self.bos] + ids + [self.eos]
        return ids

    def decode(self, ids: List[int], stop_at_eos: bool = True) -> List[str]:
        out = []
        for i in ids:
            if stop_at_eos and i == self.eos:
                break
            if i in (self.pad, self.bos):
                continue
            out.append(self.itos[i] if i < len(self.itos) else "<unk>")
        return out


def build_vocab_from_counter(counter, min_freq: int = 1) -> Vocab:
    itos = SPECIALS.copy()
    for tok, freq in counter.most_common():
        if freq >= min_freq and tok not in itos:
            itos.append(tok)
    stoi = {t: i for i, t in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


# ---------------------------
# Data loading (reverse stages + add emoji definitions)
# ---------------------------
def load_training_pairs() -> List[Tuple[List[str], List[str]]]:
    """
    Returns list of (src_emoji_tokens, tgt_word_tokens)
    """
    pairs: List[Tuple[List[str], List[str]]] = []

    # 1) Reverse stage datasets: input=English, output=emoji sequence
    for p in STAGE_PATHS:
        df = pd.read_csv(p)
        # expected columns: input, output
        for _, row in df.iterrows():
            eng = row["input"]
            emo = row["output"]
            src = split_emoji_sequence(emo)
            tgt = tokenize_english(eng)
            if 1 <= len(src) <= 6 and len(tgt) > 0:
                pairs.append((src, tgt))

    # 2) Add single emoji → definition pairs from merged_emoji_sample.csv
    merged = pd.read_csv(MERGED_EMOJI_PATH)
    # expected columns include: emoji, definition, name
    for _, row in merged.iterrows():
        e = row.get("emoji", None)
        d = row.get("definition", None)
        if pd.isna(e) or pd.isna(d):
            continue
        src = split_emoji_sequence(str(e))
        tgt = tokenize_english(str(d))
        if len(src) == 1 and len(tgt) > 0:
            pairs.append((src, tgt))

    return pairs


# ---------------------------
# Dataset / Collate
# ---------------------------
class Emoji2TextDataset(Dataset):
    def __init__(self, pairs, src_vocab: Vocab, tgt_vocab: Vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_tokens, tgt_tokens = self.pairs[idx]
        src_ids = self.src_vocab.encode(src_tokens, add_bos_eos=True)
        tgt_ids = self.tgt_vocab.encode(tgt_tokens, add_bos_eos=True)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_batch(batch, src_pad: int, tgt_pad: int):
    srcs, tgts = zip(*batch)
    src_lens = [len(x) for x in srcs]
    tgt_lens = [len(x) for x in tgts]

    src_max = max(src_lens)
    tgt_max = max(tgt_lens)

    src_batch = torch.full((len(batch), src_max), src_pad, dtype=torch.long)
    tgt_batch = torch.full((len(batch), tgt_max), tgt_pad, dtype=torch.long)

    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_batch[i, : len(s)] = s
        tgt_batch[i, : len(t)] = t

    return src_batch, tgt_batch


# ---------------------------
# Seq2Seq Transformer
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pad_idx_src: int = 0,
        pad_idx_tgt: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx_src = pad_idx_src
        self.pad_idx_tgt = pad_idx_tgt

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx_src)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx_tgt)

        self.pos = PositionalEncoding(d_model, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, T, D)
        )

        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

    def make_padding_mask(self, x: torch.Tensor, pad_idx: int) -> torch.Tensor:
        # True where padding
        return (x == pad_idx)

    def forward(self, src: torch.Tensor, tgt_inp: torch.Tensor):
        """
        src: (B, S)
        tgt_inp: (B, T)   (teacher-forced decoder input)
        """
        src_key_padding_mask = self.make_padding_mask(src, self.pad_idx_src)  # (B, S)
        tgt_key_padding_mask = self.make_padding_mask(tgt_inp, self.pad_idx_tgt)  # (B, T)

        # causal mask for decoder self-attention
        T = tgt_inp.size(1)
        tgt_mask = torch.triu(torch.ones((T, T), device=tgt_inp.device), diagonal=1).bool()

        src_emb = self.pos(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos(self.tgt_embed(tgt_inp) * math.sqrt(self.d_model))

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = self.output_proj(out)  # (B, T, V)
        return logits

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, bos_idx: int, eos_idx: int, max_len: int = 40):
        """
        src: (B, S)
        returns: (B, T_out) token ids
        """
        self.eval()
        B = src.size(0)
        device = src.device

        ys = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)

        for _ in range(max_len):
            logits = self.forward(src, ys)  # (B, t, V)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # (B, 1)
            ys = torch.cat([ys, next_token], dim=1)
            if torch.all(next_token.squeeze(1) == eos_idx):
                break

        return ys


# ---------------------------
# Metrics (simple + robust)
# ---------------------------
def token_f1(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    from collections import Counter
    pc = Counter(pred_tokens)
    rc = Counter(ref_tokens)
    overlap = sum((pc & rc).values())
    prec = overlap / max(1, len(pred_tokens))
    rec = overlap / max(1, len(ref_tokens))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# ---------------------------
# Train / Eval
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
    out_dir="emoji2text_model",
    epochs=8,
    batch_size=64,
    lr=3e-4,
    d_model=256,
    nhead=8,
    enc_layers=3,
    dec_layers=3,
    ff=512,
    dropout=0.1,
    min_word_freq=1,
    seed=42,
):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    pairs = load_training_pairs()
    random.shuffle(pairs)

    # split
    n = len(pairs)
    n_val = max(1, int(0.1 * n))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    # build vocabs
    from collections import Counter
    src_counter = Counter()
    tgt_counter = Counter()
    for src, tgt in train_pairs:
        src_counter.update(src)
        tgt_counter.update(tgt)

    src_vocab = build_vocab_from_counter(src_counter, min_freq=1)
    tgt_vocab = build_vocab_from_counter(tgt_counter, min_freq=min_word_freq)

    # save vocabs
    with open(os.path.join(out_dir, "src_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(src_vocab.stoi, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "tgt_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(tgt_vocab.stoi, f, ensure_ascii=False, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = Emoji2TextDataset(train_pairs, src_vocab, tgt_vocab)
    val_ds = Emoji2TextDataset(val_pairs, src_vocab, tgt_vocab)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, src_vocab.pad, tgt_vocab.pad),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, src_vocab.pad, tgt_vocab.pad),
    )

    model = Seq2SeqTransformer(
        src_vocab_size=len(src_vocab.itos),
        tgt_vocab_size=len(tgt_vocab.itos),
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        dim_feedforward=ff,
        dropout=dropout,
        pad_idx_src=src_vocab.pad,
        pad_idx_tgt=tgt_vocab.pad,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val = -1.0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for src, tgt in train_loader:
            src = src.to(device)
            tgt = tgt.to(device)

            # teacher forcing: decoder input is tgt[:-1], labels are tgt[1:]
            tgt_inp = tgt[:, :-1]
            tgt_y = tgt[:, 1:]

            logits = model(src, tgt_inp)  # (B, T-1, V)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_y.reshape(-1))

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))

        # eval
        model.eval()
        f1s = []
        exact = 0
        count = 0

        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(device)
                tgt = tgt.to(device)

                pred_ids = model.greedy_decode(src, bos_idx=tgt_vocab.bos, eos_idx=tgt_vocab.eos, max_len=40)

                for i in range(src.size(0)):
                    pred_tokens = tgt_vocab.decode(pred_ids[i].tolist(), stop_at_eos=True)
                    ref_tokens = tgt_vocab.decode(tgt[i].tolist(), stop_at_eos=True)

                    f1s.append(token_f1(pred_tokens, ref_tokens))
                    if pred_tokens == ref_tokens:
                        exact += 1
                    count += 1

        val_f1 = float(np.mean(f1s)) if f1s else 0.0
        val_exact = exact / max(1, count)

        print(f"Epoch {ep:02d} | train_loss={avg_loss:.4f} | val_tokenF1={val_f1:.4f} | val_exact={val_exact:.4f}")

        # save best
        if val_f1 > best_val:
            best_val = val_f1
            torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    print(f"Done. Best val_tokenF1={best_val:.4f}. Model saved to {out_dir}/model.pt")


# ---------------------------
# Load + translate
# ---------------------------
def load_model(model_dir="emoji2text_model", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(model_dir, "src_vocab.json"), "r", encoding="utf-8") as f:
        src_stoi = json.load(f)
    with open(os.path.join(model_dir, "tgt_vocab.json"), "r", encoding="utf-8") as f:
        tgt_stoi = json.load(f)

    # rebuild itos
    src_itos = [None] * (max(src_stoi.values()) + 1)
    for k, v in src_stoi.items():
        src_itos[v] = k
    tgt_itos = [None] * (max(tgt_stoi.values()) + 1)
    for k, v in tgt_stoi.items():
        tgt_itos[v] = k

    src_vocab = Vocab(stoi=src_stoi, itos=src_itos)
    tgt_vocab = Vocab(stoi=tgt_stoi, itos=tgt_itos)

    model = Seq2SeqTransformer(
        src_vocab_size=len(src_vocab.itos),
        tgt_vocab_size=len(tgt_vocab.itos),
        d_model=256, nhead=8, num_encoder_layers=3, num_decoder_layers=3,
        dim_feedforward=512, dropout=0.1,
        pad_idx_src=src_vocab.pad, pad_idx_tgt=tgt_vocab.pad,
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location=device))
    model.eval()
    return model, src_vocab, tgt_vocab, device


@torch.no_grad()
def translate(model, src_vocab, tgt_vocab, device, emoji_sequence: str) -> str:
    src_tokens = split_emoji_sequence(emoji_sequence)
    if not (1 <= len(src_tokens) <= 6):
        return "Input must be 1–6 emojis."

    src_ids = torch.tensor([src_vocab.encode(src_tokens, add_bos_eos=True)], dtype=torch.long, device=device)
    pred_ids = model.greedy_decode(src_ids, bos_idx=tgt_vocab.bos, eos_idx=tgt_vocab.eos, max_len=40)
    pred_tokens = tgt_vocab.decode(pred_ids[0].tolist(), stop_at_eos=True)
    return " ".join(pred_tokens)


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    # Train
    train(
        out_dir="emoji2text_model",
        epochs=8,
        batch_size=64,
        lr=3e-4,
        d_model=256,
        nhead=8,
        enc_layers=3,
        dec_layers=3,
        ff=512,
        dropout=0.1,
        min_word_freq=1,
        seed=42,
    )

    # Demo inference
    model, src_vocab, tgt_vocab, device = load_model("emoji2text_model")
    for ex in ["🎵", "🎵 💃", "🚗 ⛰️ 🌲 🏞️", "🥚 🧀 🍞 🥛 😋", "✈️🧳🏝️🏊🥥😴"]:
        print(ex, "->", translate(model, src_vocab, tgt_vocab, device, ex))
