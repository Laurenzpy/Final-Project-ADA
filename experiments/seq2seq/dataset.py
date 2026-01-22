import pandas as pd
import torch
from torch.utils.data import Dataset

from seq2seq.tokenizers import tokenize_emojis, tokenize_text
from seq2seq.vocab import Vocab


def _choose_canonical_output(outputs: list[str]) -> str:
    """
    Pick ONE canonical target sentence for a given input.
    Strategy:
      1) most frequent (mode)
      2) tie-break: shortest (often less noisy)
    """
    outs = [" ".join(str(o).split()) for o in outputs if str(o).strip()]
    if not outs:
        return ""

    counts: dict[str, int] = {}
    for o in outs:
        counts[o] = counts.get(o, 0) + 1

    max_c = max(counts.values())
    candidates = [o for o, c in counts.items() if c == max_c]
    candidates.sort(key=lambda s: (len(s.split()), len(s)))
    return candidates[0]


class Emoji2TextDataset(Dataset):
    """
    Flexible dataset for either:
      - emoji2text: input=emojis, output=text  (intended project)
      - text2emoji: input=text, output=emojis  (optional)

    Important option:
      - deduplicate=True: makes mapping input -> ONE output (fixes one-to-many instability)
    """
    def __init__(
        self,
        csv_paths: list[str],
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        direction: str = "emoji2text",
        deduplicate: bool = True,
    ):
        frames = []
        for p in csv_paths:
            df = pd.read_csv(p)
            if "input" not in df.columns or "output" not in df.columns:
                raise ValueError(f"{p} must have columns: input, output")
            frames.append(df[["input", "output"]])

        df = pd.concat(frames, ignore_index=True)
        df["input"] = df["input"].astype(str)
        df["output"] = df["output"].astype(str)

        if direction not in {"emoji2text", "text2emoji"}:
            raise ValueError("direction must be 'emoji2text' or 'text2emoji'")
        self.direction = direction

        # ✅ Key fix: reduce one-to-many
        if deduplicate:
            grouped = df.groupby("input")["output"].apply(list).reset_index()
            grouped["output"] = grouped["output"].apply(_choose_canonical_output)
            df = grouped

        self.df = df.reset_index(drop=True)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        if self.direction == "emoji2text":
            src_tokens = tokenize_emojis(str(row["input"]))
            tgt_tokens = tokenize_text(str(row["output"]))
        else:
            src_tokens = tokenize_text(str(row["input"]))
            tgt_tokens = tokenize_emojis(str(row["output"]))

        src_ids = self.src_vocab.encode(src_tokens)
        tgt_ids = [self.tgt_vocab.sos_id] + self.tgt_vocab.encode(tgt_tokens) + [self.tgt_vocab.eos_id]

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_batch(batch, pad_id_src: int, pad_id_tgt: int):
    src_seqs, tgt_seqs = zip(*batch)

    src_lens = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
    tgt_lens = torch.tensor([len(t) for t in tgt_seqs], dtype=torch.long)

    max_src = int(src_lens.max()) if len(src_lens) else 0
    max_tgt = int(tgt_lens.max()) if len(tgt_lens) else 0

    src_pad = torch.full((len(batch), max_src), pad_id_src, dtype=torch.long)
    tgt_pad = torch.full((len(batch), max_tgt), pad_id_tgt, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs)):
        src_pad[i, : len(s)] = s
        tgt_pad[i, : len(t)] = t

    return src_pad, src_lens, tgt_pad, tgt_lens
