import re
import math
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "checkpoints/emoji_seq2text_2to6.pt"

def normalize_emoji_token(e: str) -> str:
    return str(e).replace("\ufe0f", "").strip()

def tokenize_emojis_user(s: str, max_emojis: int) -> list[str]:
    s = str(s).strip()
    if not s:
        return []
    if " " in s:
        return [normalize_emoji_token(t) for t in s.split() if t.strip()][:max_emojis]
    try:
        import regex as reg
        clusters = reg.findall(r"\X", s)
        emojis = [normalize_emoji_token(c) for c in clusters if reg.search(r"\p{Emoji}", c)]
        return emojis[:max_emojis]
    except Exception:
        return [normalize_emoji_token(s)][:max_emojis]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])

class Seq2TextTransformer(nn.Module):
    def __init__(self, emo_vocab_size, txt_vocab_size, emo_pad, txt_pad, cfg):
        super().__init__()
        d = cfg["D_MODEL"]
        self.emo_pad = emo_pad
        self.txt_pad = txt_pad
        self.emo_emb = nn.Embedding(emo_vocab_size, d, padding_idx=emo_pad)
        self.txt_emb = nn.Embedding(txt_vocab_size, d, padding_idx=txt_pad)
        self.pos = PositionalEncoding(d, dropout=cfg["DROPOUT"], max_len=64)
        self.tr = nn.Transformer(
            d_model=d,
            nhead=cfg["NHEAD"],
            num_encoder_layers=cfg["NUM_LAYERS"],
            num_decoder_layers=cfg["NUM_LAYERS"],
            dim_feedforward=cfg["FF_DIM"],
            dropout=cfg["DROPOUT"],
            batch_first=True,
        )
        self.fc = nn.Linear(d, txt_vocab_size)

    def forward(self, src, tgt_in):
        src_pad = (src == self.emo_pad)
        tgt_pad = (tgt_in == self.txt_pad)
        T = tgt_in.size(1)
        tgt_mask = torch.triu(torch.ones(T, T, device=tgt_in.device), diagonal=1).bool()
        src_e = self.pos(self.emo_emb(src))
        tgt_e = self.pos(self.txt_emb(tgt_in))
        out = self.tr(
            src=src_e, tgt=tgt_e, tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        return self.fc(out)

@torch.no_grad()
def beam_generate(model, src, BOS, EOS, beam_size=5, max_len=18, len_penalty=0.7):
    beams = [([BOS], 0.0)]
    finished = []
    for _ in range(max_len):
        new_beams = []
        for tokens, score in beams:
            if tokens[-1] == EOS:
                finished.append((tokens, score))
                continue
            ys = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
            logits = model(src, ys)[0, -1]
            logp = torch.log_softmax(logits, dim=-1)
            topk = torch.topk(logp, k=beam_size)
            for nxt, lp in zip(topk.indices.tolist(), topk.values.tolist()):
                new_beams.append((tokens + [nxt], score + lp))
        new_beams.sort(key=lambda x: x[1] / (len(x[0]) ** len_penalty), reverse=True)
        beams = new_beams[:beam_size]
    finished.extend(beams)
    finished.sort(key=lambda x: x[1] / (len(x[0]) ** len_penalty), reverse=True)
    return finished[0][0]

def main():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    emoji_itos = ckpt["emoji_vocab_itos"]
    text_itos = ckpt["text_vocab_itos"]
    cfg = ckpt["config"]

    emoji_stoi = {t:i for i,t in enumerate(emoji_itos)}
    text_stoi = {t:i for i,t in enumerate(text_itos)}

    EMO_PAD = emoji_stoi["<pad>"]
    TXT_PAD = text_stoi["<pad>"]
    BOS = text_stoi["<bos>"]
    EOS = text_stoi["<eos>"]

    model = Seq2TextTransformer(len(emoji_itos), len(text_itos), EMO_PAD, TXT_PAD, cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    def decode(ids):
        out = []
        for i in ids:
            tok = text_itos[i]
            if tok == "<eos>":
                break
            if tok not in ("<pad>", "<bos>"):
                out.append(tok)
        return " ".join(out)

    while True:
        s = input("Emoji seq (2–6, space-separated recommended) > ").strip()
        ems = tokenize_emojis_user(s, cfg["MAX_EMOJIS"])
        if len(ems) < cfg["MIN_EMOJIS"]:
            print("[ERROR] This model does not accept single-emoji inputs. Provide 2–6 emojis.")
            continue

        src_ids = [emoji_stoi.get(e, emoji_stoi["<unk>"]) for e in ems]
        src = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)

        ids = beam_generate(model, src, BOS, EOS)
        print("->", decode(ids))

if __name__ == "__main__":
    main()