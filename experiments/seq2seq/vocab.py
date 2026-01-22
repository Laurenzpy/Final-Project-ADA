from collections import Counter
from dataclasses import dataclass
import json

PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
UNK = "<UNK>"

SPECIALS = [PAD, SOS, EOS, UNK]

@dataclass
class Vocab:
    stoi: dict
    itos: list

    @property
    def pad_id(self): return self.stoi[PAD]
    @property
    def sos_id(self): return self.stoi[SOS]
    @property
    def eos_id(self): return self.stoi[EOS]
    @property
    def unk_id(self): return self.stoi[UNK]

    def encode(self, tokens):
        return [self.stoi.get(t, self.unk_id) for t in tokens]

    def decode(self, ids, stop_at_eos=True):
        out = []
        for i in ids:
            if stop_at_eos and i == self.eos_id:
                break
            out.append(self.itos[i] if 0 <= i < len(self.itos) else UNK)
        return out

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            itos = json.load(f)
        stoi = {t: i for i, t in enumerate(itos)}
        return Vocab(stoi, itos)

def build_vocab(token_lists, min_freq=1, max_size=None):
    counter = Counter()
    for toks in token_lists:
        counter.update(toks)

    itos = SPECIALS.copy()
    items = [(t, c) for t, c in counter.items() if c >= min_freq and t not in SPECIALS]
    items.sort(key=lambda x: (-x[1], x[0]))

    if max_size is not None:
        items = items[: max(0, max_size - len(itos))]

    itos.extend([t for t, _ in items])
    stoi = {t: i for i, t in enumerate(itos)}
    return Vocab(stoi, itos)
