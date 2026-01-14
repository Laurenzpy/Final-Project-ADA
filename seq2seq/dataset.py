import pandas as pd
import torch
from torch.utils.data import Dataset

from seq2seq.tokenizers import tokenize_emojis, tokenize_text
from seq2seq.vocab import Vocab


class Emoji2TextDataset(Dataset):
    """
    Flexible dataset for either:
      - emoji2text: input=emojis, output=text  (your intended project)
      - text2emoji: input=text, output=emojis  (what your stage6 eval looked like)
    """
    def __init__(
        self,
        csv_paths: list[str],
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        direction: str = "emoji2text",
    ):
        frames = []
        for p in csv_paths:
            df = pd.read_csv(p)
            if "input" not in df.columns or "output" not in df.columns:
                raise ValueError(f"{p} must have columns: input, output")
            frames.append(df[["input", "output"]])
        self.df = pd.concat(frames, ignore_index=True)

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        if direction not in {"emoji2text", "text2emoji"}:
            raise ValueError("direction must be 'emoji2text' or 'text2emoji'")
        self.direction = direction

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        if self.direction == "emoji2text":
            # ✅ Intended task: emojis -> meaning text
            src_tokens = tokenize_emojis(str(row["input"]))
            tgt_tokens = tokenize_text(str(row["output"]))
        else:
            # Reverse task: text -> emojis
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
