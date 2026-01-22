import torch
import torch.nn as nn
import torch.nn.functional as F


def make_src_mask(src_ids: torch.Tensor, src_pad_id: int) -> torch.Tensor:
    # True for real tokens, False for PAD
    return (src_ids != src_pad_id)


class Encoder(nn.Module):
    """
    Bidirectional GRU encoder:
      enc_out: [B, S, 2H]
      hidden:  [1, B, H] (projected from bi hidden)
    """
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, pad_id: int = 0, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.hid_proj = nn.Linear(2 * hid_dim, hid_dim)

    def forward(self, src_ids: torch.Tensor, src_lens: torch.Tensor):
        emb = self.dropout(self.embedding(src_ids))  # [B,S,E]

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, hidden = self.gru(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # [B,S,2H]

        # hidden: [2,B,H] -> [B,2H] -> [1,B,H]
        h_fwd = hidden[0]
        h_bwd = hidden[1]
        h_cat = torch.cat([h_fwd, h_bwd], dim=1)  # [B,2H]
        hidden = torch.tanh(self.hid_proj(h_cat)).unsqueeze(0)  # [1,B,H]

        return enc_out, hidden


class DotAttention(nn.Module):
    """
    Dot-product attention with projection of encoder output from 2H -> H
    so dot works with decoder hidden (H).
    """
    def __init__(self, hid_dim: int):
        super().__init__()
        self.enc_proj = nn.Linear(2 * hid_dim, hid_dim, bias=False)

    def forward(self, dec_h: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor):
        """
        dec_h:   [B,H]
        enc_out: [B,S,2H]
        mask:    [B,S] True for tokens, False for PAD
        returns:
          context: [B,2H]
          attn:    [B,S]
        """
        enc_h = self.enc_proj(enc_out)  # [B,S,H]

        # score: [B,S]
        scores = torch.bmm(enc_h, dec_h.unsqueeze(2)).squeeze(2)

        scores = scores.masked_fill(~src_mask, float("-inf"))
        attn = F.softmax(scores, dim=1)  # [B,S]

        context = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)  # [B,2H]
        return context, attn


class Decoder(nn.Module):
    """
    GRU decoder with attention context.
    Input: embedding + context(2H)
    """
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, pad_id: int = 0, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)

        self.attn = DotAttention(hid_dim)

        self.gru = nn.GRU(emb_dim + 2 * hid_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim + 2 * hid_dim, vocab_size)

    def step(self, input_ids: torch.Tensor, hidden: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor):
        """
        input_ids: [B]
        hidden:    [1,B,H]
        enc_out:   [B,S,2H]
        """
        emb = self.dropout(self.embedding(input_ids)).unsqueeze(1)  # [B,1,E]
        dec_h = hidden.squeeze(0)  # [B,H]

        context, _ = self.attn(dec_h, enc_out, src_mask)  # [B,2H]
        context_u = context.unsqueeze(1)                  # [B,1,2H]

        gru_in = torch.cat([emb, context_u], dim=2)       # [B,1,E+2H]
        out, hidden = self.gru(gru_in, hidden)            # out: [B,1,H]

        out_s = out.squeeze(1)                            # [B,H]
        logits = self.fc(torch.cat([out_s, context], dim=1))  # [B,V]
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_pad_id: int, tgt_pad_id: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id

    def forward(self, src_ids, src_lens, tgt_ids, teacher_forcing_ratio=0.5):
        """
        returns logits: [B, T-1, V]
        """
        B, T = tgt_ids.shape
        device = tgt_ids.device

        enc_out, hidden = self.encoder(src_ids, src_lens)
        src_mask = make_src_mask(src_ids, self.src_pad_id)

        inputs = tgt_ids[:, 0]  # SOS
        outputs = []

        for t in range(1, T):
            logits, hidden = self.decoder.step(inputs, hidden, enc_out, src_mask)
            outputs.append(logits.unsqueeze(1))

            use_teacher = (torch.rand(B, device=device) < teacher_forcing_ratio)
            top1 = logits.argmax(1)
            inputs = torch.where(use_teacher, tgt_ids[:, t], top1)

        return torch.cat(outputs, dim=1)

    @torch.no_grad()
    def greedy_decode(self, src_ids, src_lens, sos_id, eos_id, max_len=30):
        device = src_ids.device

        enc_out, hidden = self.encoder(src_ids, src_lens)
        src_mask = make_src_mask(src_ids, self.src_pad_id)

        B = src_ids.size(0)
        out = torch.full((B, max_len), self.tgt_pad_id, dtype=torch.long, device=device)

        inputs = torch.full((B,), sos_id, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(max_len):
            logits, hidden = self.decoder.step(inputs, hidden, enc_out, src_mask)
            next_ids = logits.argmax(1)

            out[:, t] = next_ids
            done = done | (next_ids == eos_id)
            inputs = next_ids

            if done.all():
                break

        return out
