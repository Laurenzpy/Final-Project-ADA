import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)

    def forward(self, src_ids: torch.Tensor, src_lens: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(src_ids)  # [B, S, E]
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)  # [1, B, H]
        return hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, hidden: torch.Tensor):
        emb = self.embedding(input_ids).unsqueeze(1)  # [B, 1, E]
        out, hidden = self.gru(emb, hidden)           # out: [B, 1, H]
        logits = self.fc(out.squeeze(1))              # [B, V]
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, pad_id: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = pad_id

    def forward(self, src_ids, src_lens, tgt_ids, teacher_forcing_ratio: float = 0.5):
        B, T = tgt_ids.shape
        device = tgt_ids.device

        hidden = self.encoder(src_ids, src_lens)
        inputs = tgt_ids[:, 0]  # <SOS>
        outputs = []

        for t in range(1, T):
            logits, hidden = self.decoder(inputs, hidden)
            outputs.append(logits.unsqueeze(1))

            use_teacher = torch.rand(B, device=device) < teacher_forcing_ratio
            top1 = logits.argmax(1)
            inputs = torch.where(use_teacher, tgt_ids[:, t], top1)

        return torch.cat(outputs, dim=1)

    @torch.no_grad()
    def greedy_decode(self, src_ids, src_lens, sos_id: int, eos_id: int, max_len: int = 30):
        self.eval()
        device = src_ids.device
        B = src_ids.size(0)

        hidden = self.encoder(src_ids, src_lens)
        inputs = torch.full((B,), sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        preds = []
        for _ in range(max_len):
            logits, hidden = self.decoder(inputs, hidden)
            next_ids = logits.argmax(1)
            preds.append(next_ids.unsqueeze(1))

            finished = finished | (next_ids == eos_id)
            inputs = next_ids
            if bool(finished.all()):
                break

        return torch.cat(preds, dim=1) if preds else torch.empty((B, 0), dtype=torch.long, device=device)

    @staticmethod
    def _has_repeat_ngram(seq_ids: list[int], next_id: int, n: int) -> bool:
        """
        True if appending next_id creates an n-gram that already appeared in seq_ids.
        seq_ids includes SOS already; we check on the generated sequence including the candidate.
        """
        if n <= 0:
            return False
        cand = seq_ids + [next_id]
        if len(cand) < n * 2:
            return False

        # build last n-gram
        last_ngram = tuple(cand[-n:])
        # scan previous n-grams
        for i in range(len(cand) - n):
            if tuple(cand[i:i+n]) == last_ngram:
                return True
        return False

    @torch.no_grad()
    def beam_decode(
        self,
        src_ids: torch.Tensor,
        src_lens: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int = 30,
        beam_size: int = 5,
        length_penalty: float = 0.7,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
        min_len: int = 3,
    ) -> torch.Tensor:
        """
        Beam search with repetition penalty + n-gram blocking + min_len.
        Returns: [B, <=max_len] without SOS.
        """
        self.eval()
        device = src_ids.device
        B = src_ids.size(0)

        enc_hidden = self.encoder(src_ids, src_lens)  # [1, B, H]

        results = []
        V_pad = self.pad_id

        for b in range(B):
            hidden0 = enc_hidden[:, b:b+1, :].contiguous()  # [1, 1, H]
            beams = [([sos_id], hidden0, 0.0, False)]       # (seq, hidden, logprob, done)

            def norm_score(seq, score):
                L = max(1, len(seq) - 1)  # exclude SOS
                return score / (L ** length_penalty)

            for _step in range(max_len):
                all_candidates = []

                for seq, h, score, done in beams:
                    if done:
                        all_candidates.append((seq, h, score, True))
                        continue

                    inp = torch.tensor([seq[-1]], dtype=torch.long, device=device)
                    logits, h_new = self.decoder(inp, h)   # logits [1,V]
                    logp = F.log_softmax(logits.squeeze(0), dim=-1)  # [V]

                    # never generate PAD
                    logp[V_pad] = -1e9

                    # min_len: don't allow EOS too early
                    if (len(seq) - 1) < min_len:
                        logp[eos_id] = -1e9

                    # repetition penalty: reduce probability of tokens already generated
                    if repetition_penalty and repetition_penalty > 1.0:
                        used = set(seq[1:])  # exclude SOS
                        for tid in used:
                            logp[tid] = logp[tid] / repetition_penalty

                    topk_logp, topk_ids = torch.topk(logp, k=beam_size)

                    for lp, tid in zip(topk_logp.tolist(), topk_ids.tolist()):
                        # n-gram blocking
                        if no_repeat_ngram_size and no_repeat_ngram_size > 0:
                            if self._has_repeat_ngram(seq, tid, no_repeat_ngram_size):
                                continue

                        new_seq = seq + [tid]
                        new_score = score + lp
                        new_done = (tid == eos_id)
                        all_candidates.append((new_seq, h_new, new_score, new_done))

                if not all_candidates:
                    # fallback: keep current beams
                    break

                all_candidates.sort(key=lambda x: norm_score(x[0], x[2]), reverse=True)
                beams = all_candidates[:beam_size]

                if all(done for _, _, _, done in beams):
                    break

            finished = [x for x in beams if x[3]]
            best = max(finished, key=lambda x: norm_score(x[0], x[2])) if finished else beams[0]

            out = best[0][1:]  # drop SOS
            results.append(torch.tensor(out, dtype=torch.long, device=device))

        maxL = max((r.numel() for r in results), default=0)
        out_pad = torch.full((B, maxL), eos_id, dtype=torch.long, device=device)
        for i, r in enumerate(results):
            if r.numel() > 0:
                out_pad[i, : r.numel()] = r
        return out_pad
