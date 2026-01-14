import os
import torch
from torch.utils.data import DataLoader

from seq2seq.vocab import Vocab
from seq2seq.dataset import Emoji2TextDataset, collate_batch
from seq2seq.model import Encoder, Decoder, Seq2Seq

DATA_DIR = "data"
TEST_FILES  = [os.path.join(DATA_DIR, "emoji_dataset_stage6_e2t.csv")]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ckpt = torch.load("artifacts/seq2seq_best.pt", map_location=DEVICE)

    # rebuild vocabs
    src_vocab = Vocab({t: i for i, t in enumerate(ckpt["src_vocab_itos"])}, ckpt["src_vocab_itos"])
    tgt_vocab = Vocab({t: i for i, t in enumerate(ckpt["tgt_vocab_itos"])}, ckpt["tgt_vocab_itos"])

    enc = Encoder(len(src_vocab.itos), emb_dim=ckpt["emb_dim"], hid_dim=ckpt["hid_dim"])
    dec = Decoder(len(tgt_vocab.itos), emb_dim=ckpt["emb_dim"], hid_dim=ckpt["hid_dim"])
    model = Seq2Seq(enc, dec, pad_id=tgt_vocab.pad_id).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # IMPORTANT: emoji -> text
    test_ds = Emoji2TextDataset(TEST_FILES, src_vocab, tgt_vocab, direction="emoji2text")
    test_dl = DataLoader(
        test_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, src_vocab.pad_id, tgt_vocab.pad_id),
    )

    shown = 0
    with torch.no_grad():
        for src_pad, src_lens, tgt_pad, _ in test_dl:
            src_pad, src_lens = src_pad.to(DEVICE), src_lens.to(DEVICE)

            pred_ids = model.beam_decode(
                src_pad, src_lens,
                sos_id=tgt_vocab.sos_id,
                eos_id=tgt_vocab.eos_id,
                max_len=30,
                beam_size=5,
                length_penalty=0.7,
            )



            for i in range(src_pad.size(0)):
                # INPUT (emojis)
                src_tokens = [t for t in src_vocab.decode(src_pad[i].tolist(), stop_at_eos=False) if t != "<PAD>"]
                emoji_in = "".join(src_tokens)

                # TARGET / PRED (text)
                gold_tokens = tgt_vocab.decode(tgt_pad[i].tolist()[1:], stop_at_eos=True)  # skip SOS
                pred_tokens = tgt_vocab.decode(pred_ids[i].tolist(), stop_at_eos=True)

                gold_text = " ".join([t for t in gold_tokens if t not in {"<PAD>", "<SOS>", "<EOS>"}]).strip()
                pred_text = " ".join([t for t in pred_tokens if t not in {"<PAD>", "<SOS>", "<EOS>"}]).strip()

                print("IN :", emoji_in)
                print("GT :", gold_text)
                print("PR :", pred_text)
                print("-" * 60)

                shown += 1
                if shown >= 30:
                    return


if __name__ == "__main__":
    main()
