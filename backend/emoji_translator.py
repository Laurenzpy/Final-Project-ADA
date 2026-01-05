import re
import pandas as pd
import torch
import regex
import requests
import random
from sentence_transformers import SentenceTransformer, util

def _clean_text(x):
    """Return None for NaN/None/empty/'nan'/'none' and strip."""
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
    low = s.lower().strip()
    if low in {"nan", "none", "null"}:
        return None
    return s


def _clean_shortcode(sc):
    """
    Normalize shortcode:
      ':face_with_tears_of_joy:' -> 'face with tears of joy'
      'laughing,satisfied' -> 'laughing'
    """
    sc = _clean_text(sc)
    if not sc:
        return None
    sc = sc.strip().strip(":")
    sc = sc.split(",")[0].strip()
    sc = sc.replace("_", " ")
    sc = re.sub(r"\s+", " ", sc).strip()
    return sc.lower() if sc else None


def _short_phrase(x, max_words=3):
    """Keep only short, sentence-friendly phrases."""
    x = _clean_text(x)
    if not x:
        return None
    x = x.strip().rstrip(".")
    x = re.sub(r"\s+", " ", x).strip()
    wc = len(re.findall(r"[A-Za-z']+", x))
    if wc == 0 or wc > max_words:
        return None
    return x.lower()


def _dedupe_preserve(items):
    seen = set()
    out = []
    for it in items:
        it = _clean_text(it)
        if not it:
            continue
        low = it.lower()
        if low in {"nan", "none"}:
            continue
        if low not in seen:
            out.append(it)
            seen.add(low)
    return out

class EmojiTranslator:
    def __init__(self, ollama_url="http://localhost:11434/api/generate", ollama_model="llama3"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.emoji_dict = {}
        self.kb_df = pd.DataFrame()
        self.kb_embeddings = None
        self.use_llm_cache = False  # set True if you want repeatable outputs per buzzword set


        # LLM integration (Ollama / Llama3)
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self._llm_cache = {}  # cache by tuple(buzzwords)

    def load_data(self, metadata_file, stage_files):
        meta_df = pd.read_csv(metadata_file)

        # try common shortcode column names
        possible_shortcode_cols = [
            "shortcode", "shortcodes", "short_code", "short_code_first",
            "shortcode_first", "shortcode_cldr", "cldr_short_name"
        ]
        shortcode_col = None
        for c in possible_shortcode_cols:
            if c in meta_df.columns:
                shortcode_col = c
                break

        for _, row in meta_df.iterrows():
            emoji = _clean_text(row.get("emoji"))
            if not emoji:
                continue

            self.emoji_dict[emoji] = {
                "name": _clean_text(row.get("name")) or "something",
                "adj": _clean_text(row.get("sense_adj_first")),
                "verb": _clean_text(row.get("sense_verb_first")),
                "noun": _clean_text(row.get("sense_noun_first")),
                "shortcode": _clean_shortcode(row.get(shortcode_col)) if shortcode_col else None,
            }

        # Stage datasets: reverse mapping (emoji seq -> english)
        all_data = []
        for f in stage_files:
            try:
                df = pd.read_csv(f)
                df["clean_seq"] = df["output"].astype(str).str.replace(r"\s+", "", regex=True)
                all_data.append(df[["clean_seq", "input"]])
            except Exception:
                continue

        if all_data:
            self.kb_df = pd.concat(all_data, ignore_index=True).drop_duplicates()
            self.kb_embeddings = self.model.encode(self.kb_df["input"].tolist(), convert_to_tensor=True)
        else:
            self.kb_df = pd.DataFrame(columns=["clean_seq", "input"])
            self.kb_embeddings = None

    def _get_emoji_list(self, text):
        """
        Grapheme split to support multi-codepoint emojis (☁️, flags, families, etc.)
        """
        text = str(text)
        clusters = [c for c in regex.findall(r"\X", text) if c.strip()]
        return [c for c in clusters if c in self.emoji_dict]

    def _collect_buzzwords(self, emojis, max_total=10):
        """
        Prefer shortcode, then noun/name. Optionally add short adj/verb.
        Produces compact buzzwords for LLM.
        """
        buzz = []
        for emo in emojis:
            d = self.emoji_dict.get(emo, {})
            if not d:
                continue

            sc = _clean_text(d.get("shortcode"))
            noun = _clean_text(d.get("noun"))
            name = _clean_text(d.get("name"))
            adj = _short_phrase(d.get("adj"), max_words=2)
            verb = _short_phrase(d.get("verb"), max_words=2)

            # Priority: shortcode > noun > name
            base = sc or noun or name
            base = _clean_text(base)
            if base:
                buzz.append(base.lower())

            # Add optional modifiers if they are short and useful
            if adj and adj not in buzz:
                buzz.append(adj)
            if verb and verb not in buzz:
                buzz.append(verb)

            if len(buzz) >= max_total:
                break

        buzz = _dedupe_preserve(buzz)
        return buzz[:max_total]

    def _ollama_generate(self, buzzwords):
        """
        Calls local Ollama (Llama3) to turn buzzwords into 1–2 coherent sentences.
        Produces varied style and length. Avoids the generic 'It relates to ...' pattern.
        """

        buzzwords = [bw for bw in buzzwords if _clean_text(bw)]
        if not buzzwords:
            return "Unknown sequence."

        # Optional caching (OFF by default to allow diversity)
        cache_key = tuple(buzzwords)
        if getattr(self, "use_llm_cache", False) and cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        # A few styles to diversify outputs across calls
        styles = [
            "a casual text message",
            "a short descriptive caption",
            "a tiny story with a human subject (I/we)",
            "a playful tone with mild humor",
            "a neutral informative tone",
            "a motivational tone",
        ]
        style = random.choice(styles)

        keywords = "; ".join(buzzwords)

        def contains_all_keywords(text: str) -> bool:
            low = (text or "").lower()
            return all(bw.lower() in low for bw in buzzwords)

        def bad_generic(text: str) -> bool:
            low = (text or "").lower().strip()
            # block common generic starts
            return low.startswith("it relates to") or low.startswith("this relates to") or "it relates to" in low

        def call_ollama(prompt: str, temperature: float, seed: int):
            try:
                resp = requests.post(
                    self.ollama_url,
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "top_p": 0.9,
                            "repeat_penalty": 1.12,
                            "num_predict": 110,
                            "seed": seed,
                        },
                    },
                    timeout=40,
                )
                resp.raise_for_status()
                out = resp.json().get("response", "")
            except Exception:
                out = ""

            out = re.sub(r"\s+", " ", (out or "")).strip()
            out = out.replace("nan", "").replace("Nan", "").strip()
            return out

        # Prompt: explicitly discourages the generic pattern, encourages coherence and style variation
        base_prompt = (
            "You are a writing assistant.\n"
            f"Write 1–2 coherent English sentences in {style}.\n"
            "You MUST include every keyword (case-insensitive). Use them naturally.\n"
            "Do NOT output a list. Do NOT use bullet points. Do NOT use hashtags.\n"
            "Do NOT start with 'It relates to' and do NOT use the phrase 'it relates to'.\n"
            "Do NOT use the word 'nan'.\n"
            f"Keywords: {keywords}\n"
            "Output:"
        )

        # Try multiple candidates for diversity & higher chance of satisfying constraints
        candidates = []
        for i, temp in enumerate([0.85, 0.95, 0.75]):
            seed = random.randint(1, 10_000_000)
            out = call_ollama(base_prompt, temperature=temp, seed=seed)
            if out:
                candidates.append(out)

        # Pick best candidate: has all keywords, not generic, reasonably sentence-like
        def score(text: str) -> float:
            s = 0.0
            if contains_all_keywords(text):
                s += 5.0
            else:
                # partial credit
                low = text.lower()
                s += sum(1.0 for bw in buzzwords if bw.lower() in low)
            if bad_generic(text):
                s -= 4.0
            # prefer 1–2 sentences
            n_sent = len(re.findall(r"[.!?]", text))
            if 1 <= n_sent <= 2:
                s += 1.5
            # avoid being too short or too long
            n_words = len(text.split())
            if 8 <= n_words <= 35:
                s += 1.0
            return s

        candidates.sort(key=score, reverse=True)
        best = candidates[0] if candidates else ""

        # If best still misses keywords, retry with stricter instructions (no template patch)
        if best and not contains_all_keywords(best):
            strict_prompt = (
                "You are a precise writing assistant.\n"
                "Rewrite the output as 1–2 coherent English sentences.\n"
                "You MUST include EVERY keyword exactly as written (case-insensitive is fine).\n"
                "No lists, no bullets, no hashtags.\n"
                "Do NOT use or include the phrase 'it relates to'.\n"
                "Do NOT use the word 'nan'.\n"
                f"Keywords: {keywords}\n"
                f"Draft: {best}\n"
                "Final:"
            )
            retry = call_ollama(strict_prompt, temperature=0.6, seed=random.randint(1, 10_000_000))
            if retry:
                best = retry

        # Final cleanup / safety
        best = re.sub(r"\s+", " ", (best or "")).strip()
        best = best.replace("nan", "").replace("Nan", "").strip()

        if not best:
            # fallback without the hated pattern
            best = f"I’m thinking about {', '.join(buzzwords[:4])}."

        if not best.endswith((".", "!", "?")):
            best += "."

        if getattr(self, "use_llm_cache", False):
            self._llm_cache[cache_key] = best

        return best


    def translate(self, input_sequence):
        clean_input = str(input_sequence).replace(" ", "").replace("\t", "").replace("\n", "")

        # 1) Exact DB match
        if not self.kb_df.empty:
            exact = self.kb_df[self.kb_df["clean_seq"] == clean_input]
            if not exact.empty:
                meanings = list(pd.unique(exact["input"]))[:3]
                meanings = [m for m in meanings if _clean_text(m)]
                return f"🎯 [DB] {' | '.join(meanings)}"

        # 2) Similarity match (emoji description -> nearest english meaning)
        if self.kb_embeddings is not None and len(self.kb_df) > 0:
            input_emojis = self._get_emoji_list(clean_input)

            # Prefer shortcode in the description signal, then name
            desc_parts = []
            for e in input_emojis:
                d = self.emoji_dict.get(e, {})
                desc_parts.append(d.get("shortcode") or d.get("name") or "")
            input_desc = " ".join([p for p in desc_parts if _clean_text(p)])

            if input_desc:
                input_vec = self.model.encode(input_desc, convert_to_tensor=True)
                scores = util.cos_sim(input_vec, self.kb_embeddings)[0]
                best_idx = torch.argmax(scores).item()
                if scores[best_idx] > 0.80:
                    best = _clean_text(self.kb_df.iloc[best_idx]["input"])
                    if best:
                        return f"💡 [SIM] {best}"

        # 3) LLM fallback: buzzwords -> fluent sentence(s)
        input_emojis = self._get_emoji_list(clean_input)
        buzzwords = self._collect_buzzwords(input_emojis, max_total=10)
        return f"🦙 [LLM] {self._ollama_generate(buzzwords)}"
    

    # backend/emoji_translator.py

# ... your imports + EmojiTranslator class ...

def build_translator():
    translator = EmojiTranslator(ollama_model="llama3")

    meta = "merged_emoji_sample.csv"
    stages = [f"emoji_dataset_stage{i}.csv" for i in range(1, 6)]  # update if needed

    translator.load_data(meta, stages)
    return translator

