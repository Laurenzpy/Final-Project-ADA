import pandas as pd

# Load the merged emoji dataframe from a parquet file
merged_emoji_df = pd.read_parquet("merged_emoji_df.parquet")
print(merged_emoji_df.shape)

import random
import re
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import regex  # pip package "regex" (VS Code: pip install regex)

# ----------------- Config -----------------
MAX_EMOJIS = 3
SYNTHETIC_SAMPLES = 20000   # used only if no real sequence column exists
BATCH_SIZE = 128
EPOCHS = 8
LR = 2e-3
EMB_DIM = 128
HID_DIM = 256
TEACHER_FORCING = 0.7
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED)
torch.manual_seed(SEED)

# ----------------- Helpers -----------------
def simple_word_tokenize(text: str):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9'\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []

class Vocab:
    def __init__(self, tokens_list, min_freq=1, specials=("<pad>", "<bos>", "<eos>", "<unk>")):
        self.itos = list(specials)
        counter = Counter(tokens_list)
        for tok, freq in counter.most_common():
            if freq >= min_freq and tok not in self.itos:
                self.itos.append(tok)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        unk = self.stoi["<unk>"]
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids):
        out = []
        for i in ids:
            tok = self.itos[i]
            if tok == "<eos>":
                break
            if tok not in ("<bos>", "<pad>"):
                out.append(tok)
        return " ".join(out)

def pad_sequences(seqs, pad_id):
    max_len = max(len(s) for s in seqs)
    padded = [s + [pad_id] * (max_len - len(s)) for s in seqs]
    return torch.tensor(padded, dtype=torch.long)

def split_emoji_sequence(s):
    """
    Robustly splits a string that contains emojis (possibly no separators)
    into grapheme clusters and keeps those that contain Emoji codepoints.
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    if isinstance(s, list):
        return [str(x) for x in s]

    s = str(s).strip()
    if not s:
        return []

    clusters = regex.findall(r"\X", s)  # grapheme clusters
    emojis = [c for c in clusters if regex.search(r"\p{Emoji}", c)]
    return emojis

def pick_first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ----------------- Expect merged_emoji_df to exist -----------------
# merged_emoji_df is your preprocessed dataframe in memory
df = merged_emoji_df.copy()

# ----------------- Decide where sequences + targets come from -----------------
sequence_col = pick_first_existing_col(df, [
    "emoji_sequence", "emoji_seq", "sequence", "emojis", "emoji_string", "input"
])

target_col = pick_first_existing_col(df, [
    "target_text", "text", "translation", "meaning", "label", "definition", "name"
])
if target_col is None:
    raise ValueError("Could not find a target text column (e.g., name/definition/text).")

pairs = []

if sequence_col is not None:
    # Mode A: real sequences exist in one column
    for _, row in df[[sequence_col, target_col]].dropna().iterrows():
        ems = split_emoji_sequence(row[sequence_col])
        if 1 <= len(ems) <= MAX_EMOJIS:
            tgt = str(row[target_col]).strip()
            if tgt:
                pairs.append((ems, tgt))

    if len(pairs) == 0:
        raise ValueError(
            f"Found sequence column '{sequence_col}' but couldn't parse any valid 1–{MAX_EMOJIS} emoji sequences."
        )
else:
    # Mode B: no sequence column -> synthetic sequences sampled from single emojis
    if "emoji" not in df.columns:
        raise ValueError("No sequence column found AND no single 'emoji' column found to synthesize sequences from.")

    emoji_list = df["emoji"].dropna().astype(str).tolist()
    emoji_to_text = dict(zip(df["emoji"].astype(str), df[target_col].astype(str)))

    def make_synthetic_pair():
        k = random.randint(1, MAX_EMOJIS)
        ems = random.sample(emoji_list, k)
        tgt = " | ".join(emoji_to_text[e] for e in ems)  # simple compositional target
        return ems, tgt

    pairs = [make_synthetic_pair() for _ in range(SYNTHETIC_SAMPLES)]

# ----------------- Train/val split -----------------
random.shuffle(pairs)
split = int(0.9 * len(pairs))
train_pairs = pairs[:split]
val_pairs = pairs[split:]

# ----------------- Build emoji vocab (input) -----------------
# Collect all emojis seen in pairs (important if real sequences contain emojis not in df["emoji"])
all_emojis = sorted({e for ems, _ in pairs for e in ems})
emoji_vocab_itos = ["<pad>"] + all_emojis
emoji_stoi = {e: i for i, e in enumerate(emoji_vocab_itos)}
EMO_PAD = emoji_stoi["<pad>"]

def encode_emojis(ems):
    return [emoji_stoi[e] for e in ems if e in emoji_stoi]  # safe

# ----------------- Build word vocab (output) -----------------
all_word_tokens = []
for _, tgt in train_pairs:
    all_word_tokens.extend(simple_word_tokenize(tgt))

word_vocab = Vocab(all_word_tokens, min_freq=1)
PAD = word_vocab.stoi["<pad>"]
BOS = word_vocab.stoi["<bos>"]
EOS = word_vocab.stoi["<eos>"]

def encode_target(text):
    toks = simple_word_tokenize(text)
    return [BOS] + word_vocab.encode(toks) + [EOS]

# ----------------- Dataset -----------------
class Emoji2TextDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        ems, tgt = self.pairs[idx]
        x = encode_emojis(ems)
        y = encode_target(tgt)
        return x, y

def collate_fn(batch):
    xs, ys = zip(*batch)
    x_pad = pad_sequences(list(xs), EMO_PAD)
    y_pad = pad_sequences(list(ys), PAD)
    return x_pad, y_pad

train_loader = DataLoader(Emoji2TextDataset(train_pairs), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(Emoji2TextDataset(val_pairs), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ----------------- Model -----------------
class Encoder(nn.Module):
    def __init__(self, input_vocab_size, emb_dim, hid_dim, pad_id):
        super().__init__()
        self.emb = nn.Embedding(input_vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
    def forward(self, x):
        e = self.emb(x)
        _, h = self.rnn(e)
        return h

class Decoder(nn.Module):
    def __init__(self, output_vocab_size, emb_dim, hid_dim, pad_id):
        super().__init__()
        self.emb = nn.Embedding(output_vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, output_vocab_size)
    def forward(self, y_in, h):
        e = self.emb(y_in)
        out, h = self.rnn(e, h)
        logits = self.fc(out)
        return logits, h

class Seq2Seq(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec
    def forward(self, x, y):
        h = self.enc(x)
        y_in = y[:, :-1]
        logits, _ = self.dec(y_in, h)
        return logits
    @torch.no_grad()
    def generate(self, emoji_seq, max_len=25):
        ems = split_emoji_sequence(emoji_seq) if isinstance(emoji_seq, str) else list(emoji_seq)
        x_ids = encode_emojis(ems)
        if len(x_ids) == 0:
            return ""
        x = torch.tensor([x_ids], dtype=torch.long, device=DEVICE)
        h = self.enc(x)

        y_ids = [BOS]
        for _ in range(max_len):
            y_in = torch.tensor([[y_ids[-1]]], dtype=torch.long, device=DEVICE)
            logits, h = self.dec(y_in, h)
            next_id = int(torch.argmax(logits[0, -1]).item())
            y_ids.append(next_id)
            if next_id == EOS:
                break
        return word_vocab.decode(y_ids)

enc = Encoder(len(emoji_vocab_itos), EMB_DIM, HID_DIM, EMO_PAD)
dec = Decoder(len(word_vocab), EMB_DIM, HID_DIM, PAD)
model = Seq2Seq(enc, dec).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=PAD)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def run_epoch(loader, train=True):
    model.train(train)
    total_loss, total_tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if train:
            optimizer.zero_grad()

        logits = model(x, y)                 # [B, Ty-1, V]
        target = y[:, 1:]                    # [B, Ty-1]
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B*T, V), target.reshape(B*T))

        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        non_pad = (target != PAD).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad

    return total_loss / max(1, total_tokens)

for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(train_loader, train=True)
    val_loss = run_epoch(val_loader, train=False)
    print(f"Epoch {epoch:02d} | train loss/token={train_loss:.4f} | val loss/token={val_loss:.4f}")

# Quick test (pass either a string like "😀🔥" or a list like ["😀","🔥"])
print(model.generate("😀🔥"))

def predict(emoji_seq: str) -> str:
    model.eval()
    return model.generate(emoji_seq, max_len=25)

tests = ["😂", "🔥", "😂🔥", "🍕❤️", "🎉🥳🔥"]
for t in tests:
    print(t, "->", predict(t))
