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
    if s.strip().lower() in {"nan", "none", "null"}:
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


def _split_graphemes(s: str):
    return [g for g in regex.findall(r"\X", str(s)) if g.strip()]


def _normalize_seq_string(s: str) -> str:
    """Normalization for exact DB matching."""
    s = str(s)
    s = re.sub(r"\s+", "", s)  # remove whitespace
    s = s.replace(VS16, "")    # remove variation selector
    s = "".join(ch for ch in s if ch not in SKIN_TONES)  # remove skin tone modifiers
    return s


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


class EmojiTranslator:
    def __init__(self, ollama_url="http://localhost:11434/api/generate", ollama_model="llama3", verbose=True):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.emoji_dict = {}

        # DB structures
        self.kb_df = pd.DataFrame()
        self.kb_embeddings = None
        self.seq2meanings = defaultdict(list)  # exact-match lookup (raw + normalized)

        # LLM
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.verbose = verbose

    def load_data(self, metadata_file, stage_files):
        # 1) Metadata
        meta_df = pd.read_csv(metadata_file)

        possible_shortcode_cols = [
            "shortcode", "shortcodes", "short_code", "short_code_first",
            "shortcode_first", "shortcode_cldr", "cldr_short_name"
        ]
        shortcode_col = next((c for c in possible_shortcode_cols if c in meta_df.columns), None)

        for _, row in meta_df.iterrows():
            emoji = _clean_text(row.get("emoji"))
            if not emoji:
                continue
            self.emoji_dict[emoji] = {
                "name": _clean_text(row.get("name")) or "something",
                "noun": _clean_text(row.get("sense_noun_first")),
                "verb": _clean_text(row.get("sense_verb_first")),
                "adj":  _clean_text(row.get("sense_adj_first")),
                "shortcode": _clean_shortcode(row.get(shortcode_col)) if shortcode_col else None,
            }

        # 2) Stage datasets
        all_rows = []
        loaded_files = 0
        total_rows = 0

        for f in stage_files:
            try:
                df = pd.read_csv(f)
                if "output" not in df.columns or "input" not in df.columns:
                    if self.verbose:
                        print(f"[WARN] {f} missing 'input'/'output' columns. Found: {list(df.columns)}")
                    continue

                df = df[["output", "input"]].dropna()
                total_rows += len(df)
                loaded_files += 1

                # build exact lookup keys (raw and normalized)
                for _, r in df.iterrows():
                    out_seq_raw = re.sub(r"\s+", "", str(r["output"]))
                    out_seq_norm = _normalize_seq_string(r["output"])
                    meaning = _clean_text(r["input"])
                    if not meaning:
                        continue
                    self.seq2meanings[out_seq_raw].append(meaning)
                    self.seq2meanings[out_seq_norm].append(meaning)

                # also store for similarity model
                df["clean_seq"] = df["output"].astype(str).str.replace(r"\s+", "", regex=True)
                all_rows.append(df[["clean_seq", "input"]])

            except FileNotFoundError:
                if self.verbose:
                    print(f"[WARN] Stage file not found: {f}")
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Failed to load {f}: {e}")

        if self.verbose:
            print(f"[INFO] Loaded {loaded_files}/{len(stage_files)} stage files, rows={total_rows}, unique seq keys={len(self.seq2meanings)}")

        if all_rows:
            self.kb_df = pd.concat(all_rows, ignore_index=True).drop_duplicates()
            self.kb_embeddings = self.model.encode(self.kb_df["input"].tolist(), convert_to_tensor=True)
        else:
            self.kb_df = pd.DataFrame(columns=["clean_seq", "input"])
            self.kb_embeddings = None

    def _get_emoji_list(self, text):
        clusters = _split_graphemes(text)
        return [c for c in clusters if c in self.emoji_dict]

    def _collect_buzzwords(self, emojis, max_total=10):
        buzz = []
        for emo in emojis:
            d = self.emoji_dict.get(emo, {})
            if not d:
                continue
            # prefer shortcode -> noun -> name
            base = d.get("shortcode") or d.get("noun") or d.get("name")
            base = _clean_text(base)
            if base:
                buzz.append(base.lower())

            # add short adj/verb if present and short-ish
            adj = _clean_text(d.get("adj"))
            if adj:
                adj = adj.strip().rstrip(".").lower()
                if 1 <= len(adj.split()) <= 2:
                    buzz.append(adj)

            verb = _clean_text(d.get("verb"))
            if verb:
                verb = verb.strip().rstrip(".").lower()
                if 1 <= len(verb.split()) <= 2:
                    buzz.append(verb)

            if len(buzz) >= max_total:
                break

        return _dedupe_preserve(buzz)[:max_total]

    def _ollama_generate(self, buzzwords):
        """Generate varied 1–2 sentences using all buzzwords. No forced 'I'm thinking about'."""
        buzzwords = [bw for bw in buzzwords if _clean_text(bw)]
        if not buzzwords:
            return "Unknown sequence."

        styles = [
            "a casual text message",
            "a short caption",
            "a mini story with 'I' or 'we'",
            "a playful tone",
            "a neutral informative tone",
            "a slightly dramatic tone",
        ]
        style = random.choice(styles)

        keywords = "; ".join(buzzwords)

        prompt = (
            "You are a skilled English writer.\n"
            f"Write 1–2 coherent sentences in {style}.\n"
            "You MUST naturally include EVERY keyword (case-insensitive is fine).\n"
            "Do NOT output a list, bullets, or hashtags.\n"
            "Avoid generic openers like 'I'm thinking about' or 'It relates to'.\n"
            "Do NOT use the word 'nan'.\n"
            f"Keywords: {keywords}\n"
            "Output:"
        )

        try:
            resp = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.9,
                        "top_p": 0.9,
                        "repeat_penalty": 1.12,
                        "num_predict": 120,
                        "seed": random.randint(1, 10_000_000),
                    },
                },
                timeout=40,
            )
            resp.raise_for_status()
            out = (resp.json().get("response", "") or "").strip()
        except Exception as e:
            # If Ollama is not reachable, return a varied template (and make it obvious)
            if self.verbose:
                print(f"[WARN] Ollama call failed: {e}")
            templates = [
                "What a vibe: {k}.",
                "Big mood: {k}.",
                "Honestly, {k} sounds like my day.",
                "This feels like {k}.",
                "I’d sum it up as {k}.",
            ]
            k = ", ".join(buzzwords[:4])
            return templates[random.randrange(len(templates))].format(k=k)

        out = re.sub(r"\s+", " ", out).strip()
        out = out.replace("nan", "").replace("Nan", "").strip()
        if not out.endswith((".", "!", "?")):
            out += "."

        # If model forgot keywords, do ONE rewrite pass (no templated “it relates to”)
        low = out.lower()
        missing = [bw for bw in buzzwords if bw.lower() not in low]
        if missing:
            rewrite = (
                "Rewrite as 1–2 coherent English sentences.\n"
                "You MUST include EVERY keyword naturally.\n"
                "No lists. No generic openers ('I'm thinking about', 'It relates to').\n"
                f"Keywords: {keywords}\n"
                f"Draft: {out}\n"
                "Final:"
            )
            try:
                resp2 = requests.post(
                    self.ollama_url,
                    json={
                        "model": self.ollama_model,
                        "prompt": rewrite,
                        "stream": False,
                        "options": {
                            "temperature": 0.6,
                            "top_p": 0.9,
                            "repeat_penalty": 1.12,
                            "num_predict": 120,
                            "seed": random.randint(1, 10_000_000),
                        },
                    },
                    timeout=40,
                )
                resp2.raise_for_status()
                out2 = (resp2.json().get("response", "") or "").strip()
                out2 = re.sub(r"\s+", " ", out2).strip()
                out2 = out2.replace("nan", "").replace("Nan", "").strip()
                if out2:
                    if not out2.endswith((".", "!", "?")):
                        out2 += "."
                    out = out2
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Ollama rewrite failed: {e}")

        return out

    def translate(self, input_sequence):
        # normalize for DB lookups
        raw = re.sub(r"\s+", "", str(input_sequence))
        norm = _normalize_seq_string(input_sequence)

        # 1) Exact DB hit (raw OR normalized)
        meanings = self.seq2meanings.get(raw) or self.seq2meanings.get(norm)
        if meanings:
            uniq = _dedupe_preserve(meanings)[:3]
            return f"🎯 [DB] {' | '.join(uniq)}"

        # 2) Similarity (only if KB exists)
        if self.kb_embeddings is not None and len(self.kb_df) > 0:
            emojis = self._get_emoji_list(norm)
            desc = " ".join(
                _dedupe_preserve([
                    self.emoji_dict[e].get("shortcode") or self.emoji_dict[e].get("name")
                    for e in emojis
                ])
            )
            if _clean_text(desc):
                vec = self.model.encode(desc, convert_to_tensor=True)
                scores = util.cos_sim(vec, self.kb_embeddings)[0]
                best_idx = int(torch.argmax(scores).item())
                if float(scores[best_idx]) > 0.80:
                    best = _clean_text(self.kb_df.iloc[best_idx]["input"])
                    if best:
                        return f"💡 [SIM] {best}"

        # 3) LLM fallback (only if we have emoji buzzwords)
        emojis = self._get_emoji_list(norm)
        buzz = self._collect_buzzwords(emojis)
        if not buzz:
            return "❓ [UNK] Unknown sequence."

        return f"🦙 [LLM] {self._ollama_generate(buzz)}"


# --- EXTENDED TEST SUITE ---
def run_extended_test(translator):
    test_cases = [
    # Length 1
    "🪨",   # rock
    "🧯",   # fire extinguisher

    # Length 2
    "🚀🧠", # rocket + brain
    "🧯🧪", # fire extinguisher + test tube

    # Length 3
    "🧑‍💻☕😴",  # coder + coffee + sleepy
    "🐙📚🧠",    # octopus + books + brain

    # Length 4
    "🎻🧠✨📚",  # violin + brain + sparkles + books
    "🦾🤖🔋⚡",  # robotic arm + robot + battery + zap

    # Length 5
    "🧑‍💻☕🐛🔧✅",  # debugging success
    "🎥🍿😱🙈🔦",    # scary movie night

    # Length 6
    "🧑‍💻🧠💡🐛🔧✅",  # think -> idea -> debug -> fix -> done
    "🎒📚📝⏰😩☕",      # school + deadline stress + coffee
     ]

    print(f"{'INPUT':<15} | {'RESULT'}")
    print("-" * 80)
    for case in test_cases:
        result = translator.translate(case)
        print(f"{case:<15} | {result}")


if __name__ == "__main__":
    translator = EmojiTranslator()
    meta = "merged_emoji_sample.csv"
    stages = [f"emoji_dataset_stage{i}.csv" for i in range(1, 7)]

    try:
        translator.load_data(meta, stages)
        run_extended_test(translator)
    except FileNotFoundError as e:
        print(f"Fehler: Bitte stelle sicher, dass alle Dateien im Ordner liegen. ({e})")
