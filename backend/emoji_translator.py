import re
import random
import pandas as pd
import torch
import regex
import requests
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

VS16 = "\ufe0f"
SKIN_TONES = {chr(cp) for cp in range(0x1F3FB, 0x1F400)}  # 1F3FB..1F3FF

def _clean_text(x):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    s = str(x).strip()
    if not s:
        return None
    if s.lower() in {"nan", "none", "null"}:
        return None
    return s

def _clean_shortcode(sc):
    sc = _clean_text(sc)
    if not sc:
        return None
    sc = sc.strip().strip(":")
    sc = sc.split(",")[0].strip()
    sc = sc.replace("_", " ")
    sc = re.sub(r"\s+", " ", sc).strip()
    return sc.lower() if sc else None

def _normalize_seq_string(s: str) -> str:
    s = str(s)
    s = re.sub(r"\s+", "", s)
    s = s.replace(VS16, "")
    s = "".join(ch for ch in s if ch not in SKIN_TONES)
    return s

def _is_emoji_cluster(cluster: str) -> bool:
    # catches emojis even if not in your metadata dict
    return bool(regex.search(r"\p{Emoji}", cluster))

def _split_graphemes(s: str):
    return [g for g in regex.findall(r"\X", str(s)) if g.strip()]

def _dedupe_preserve(items):
    seen = set()
    out = []
    for it in items:
        it = _clean_text(it)
        if not it:
            continue
        low = it.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(it)
    return out

def _pick_shortest_meaning(meanings, max_chars=140):
    meanings = [m.strip() for m in meanings if _clean_text(m)]
    if not meanings:
        return None
    meanings.sort(key=len)
    best = meanings[0]
    return best[:max_chars].strip()

def _truncate_text(text: str, max_sentences=1, max_words=18) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return text
    # keep first N sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    text = " ".join(sentences[:max_sentences]).strip()

    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(",;:") + "."

    if not text.endswith((".", "!", "?")):
        text += "."
    return text


class EmojiTranslator:
    def __init__(self, ollama_url="http://localhost:11434/api/generate", ollama_model="llama3"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

        self.emoji_dict = {}                 # emoji -> metadata
        self.seq2meanings = defaultdict(list) # exact lookup: seq -> list of meanings

        # For similarity lookup (emoji-side embeddings)
        self.kb_seq = []          # list of clean_seq
        self.kb_meaning = []      # one canonical meaning per seq
        self.kb_desc = []         # emoji description text per seq
        self.kb_desc_embeddings = None

    def load_data(self, metadata_file, stage_files):
        # ---- 1) Metadata ----
        meta_df = pd.read_csv(metadata_file)

        possible_shortcode_cols = [
            "shortcode", "shortcodes", "short_code", "short_code_first",
            "shortcode_first", "shortcode_cldr", "cldr_short_name"
        ]
        shortcode_col = next((c for c in possible_shortcode_cols if c in meta_df.columns), None)

        for _, row in meta_df.iterrows():
            emo = _clean_text(row.get("emoji"))
            if not emo:
                continue
            self.emoji_dict[emo] = {
                "name": _clean_text(row.get("name")) or "something",
                "noun": _clean_text(row.get("sense_noun_first")),
                "verb": _clean_text(row.get("sense_verb_first")),
                "adj":  _clean_text(row.get("sense_adj_first")),
                "shortcode": _clean_shortcode(row.get(shortcode_col)) if shortcode_col else None,
            }

            # also map normalized emoji to same record (fixes ☁ vs ☁️ etc.)
            emo_norm = _normalize_seq_string(emo)
            if emo_norm and emo_norm not in self.emoji_dict:
                self.emoji_dict[emo_norm] = self.emoji_dict[emo]

        # ---- 2) Stage datasets: build exact seq->meanings and a per-seq KB ----
        seq_to_all_meanings = defaultdict(list)

        for f in stage_files:
            df = pd.read_csv(f)
            if "output" not in df.columns or "input" not in df.columns:
                continue
            df = df.dropna(subset=["output", "input"])

            for _, r in df.iterrows():
                out_raw = re.sub(r"\s+", "", str(r["output"]))
                out_norm = _normalize_seq_string(r["output"])
                meaning = _clean_text(r["input"])
                if not meaning:
                    continue

                # exact lookups (raw + normalized)
                self.seq2meanings[out_raw].append(meaning)
                self.seq2meanings[out_norm].append(meaning)

                seq_to_all_meanings[out_norm].append(meaning)

        # Build KB unique sequences for similarity retrieval
        self.kb_seq = []
        self.kb_meaning = []
        self.kb_desc = []

        for seq, meanings in seq_to_all_meanings.items():
            canonical = _pick_shortest_meaning(meanings)
            if not canonical:
                continue
            self.kb_seq.append(seq)
            self.kb_meaning.append(canonical)
            self.kb_desc.append(self._sequence_to_description(seq))

        # Embed emoji-side descriptions for SIM stage
        if self.kb_desc:
            self.kb_desc_embeddings = self.model.encode(self.kb_desc, convert_to_tensor=True)
        else:
            self.kb_desc_embeddings = None

        print(f"[INFO] Exact keys: {len(self.seq2meanings)} | SIM KB seq: {len(self.kb_seq)}")

    def _sequence_to_emojis(self, seq_string: str):
        # split into graphemes from the raw string (no spaces)
        clusters = _split_graphemes(seq_string)
        # keep emoji-like clusters (even if not in dict)
        return [c for c in clusters if _is_emoji_cluster(c)]

    def _emoji_to_buzzword(self, emo: str) -> str:
        # normalize for lookup
        emo_norm = _normalize_seq_string(emo)
        d = self.emoji_dict.get(emo) or self.emoji_dict.get(emo_norm)
        if d:
            return (d.get("shortcode") or d.get("noun") or d.get("name") or emo).strip().lower()
        # if not in metadata, give the emoji itself to the LLM
        return emo

    def _sequence_to_description(self, seq_string: str) -> str:
        emojis = self._sequence_to_emojis(seq_string)
        buzz = [_clean_text(self._emoji_to_buzzword(e)) for e in emojis]
        buzz = [b for b in buzz if b]
        return " ".join(_dedupe_preserve(buzz))

    def _ollama_generate_short(self, buzzwords):
        buzzwords = [bw for bw in buzzwords if _clean_text(bw)]
        buzzwords = _dedupe_preserve(buzzwords)

        if not buzzwords:
            return "I couldn’t interpret that emoji sequence."

        # Keep output short + close to buzzword meaning
        keywords = "; ".join(buzzwords)

        prompt = (
            "You are a careful English writer.\n"
            "Write ONE short sentence that stays close to the literal meaning of the keywords.\n"
            "Constraints:\n"
            "- Max 14 words.\n"
            "- Use the keywords naturally (include as many as possible).\n"
            "- Do not add new details beyond the keywords.\n"
            "- No lists, no bullet points, no hashtags, no emojis.\n"
            "- Do NOT use the word 'nan'.\n"
            f"Keywords: {keywords}\n"
            "Sentence:"
        )

        try:
            r = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.8,
                        "repeat_penalty": 1.15,
                        "num_predict": 40,
                        "seed": random.randint(1, 10_000_000),
                        "stop": ["\n"]  # helps keep it short
                    },
                },
                timeout=30,
            )
            r.raise_for_status()
            out = (r.json().get("response", "") or "").strip()
        except Exception:
            out = ""

        out = out.replace("nan", "").replace("Nan", "").strip()
        out = _truncate_text(out, max_sentences=1, max_words=18)

        # If ollama failed, return a short template (not “unknown sequence”)
        if not out:
            # minimal, still meaningful
            out = _truncate_text(f"{', '.join(buzzwords[:4])}.", max_sentences=1, max_words=12)

        return out

    def translate(self, input_sequence: str) -> str:
        raw = re.sub(r"\s+", "", str(input_sequence))
        norm = _normalize_seq_string(input_sequence)

        # 1) Direct DB match (exact)
        meanings = self.seq2meanings.get(raw) or self.seq2meanings.get(norm)
        if meanings:
            best = _pick_shortest_meaning(meanings, max_chars=140)
            return f"🎯 [DB] {best}"

        # 2) Similar combinations (SIM) using emoji-side description embeddings
        if self.kb_desc_embeddings is not None and self.kb_desc:
            desc = self._sequence_to_description(norm)
            if _clean_text(desc):
                q = self.model.encode(desc, convert_to_tensor=True)
                scores = util.cos_sim(q, self.kb_desc_embeddings)[0]
                best_idx = int(torch.argmax(scores).item())
                if float(scores[best_idx]) > 0.70:  # lower than before because it's emoji-side now
                    return f"💡 [SIM] {self.kb_meaning[best_idx]}"

        # 3) LLM fallback (buzzwords)
        emojis = self._sequence_to_emojis(norm)
        if not emojis:
            # only happens if the input contains no emoji-like characters
            return "❓ Please enter 1–6 emojis."

        buzzwords = [self._emoji_to_buzzword(e) for e in emojis]
        buzzwords = [b for b in buzzwords if _clean_text(b)]
        return f"🦙 [LLM] {self._ollama_generate_short(buzzwords)}"

    # backend/emoji_translator.py

# ... your imports + EmojiTranslator class ...

def build_translator():
    translator = EmojiTranslator(ollama_model="llama3")

    meta = "merged_emoji_sample.csv"
    stages = [f"emoji_dataset_stage{i}.csv" for i in range(1, 6)]  # update if needed

    translator.load_data(meta, stages)
    return translator

