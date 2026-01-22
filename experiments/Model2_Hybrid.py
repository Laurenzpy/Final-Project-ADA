import re
import pandas as pd
import regex
from collections import defaultdict, Counter

# =========================
# CONFIG: your files
# =========================
MERGED_PARQUET_PATH = "merged_emoji_df.parquet"

STAGE_PATHS = [
    "emoji_dataset_stage1.csv",
    "emoji_dataset_stage2.csv",
    "emoji_dataset_stage3.csv",
    "emoji_dataset_stage4.csv",
    "emoji_dataset_stage5.csv",
    "emoji_dataset_stage6.csv",
]

# =========================
# Utilities (your helpers)
# =========================
def clean_token(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    x = str(x).strip()
    return x if x else None

def pick_first(*vals):
    for v in vals:
        v = clean_token(v)
        if v:
            return v
    return None

def a_or_an(word):
    if not word:
        return "a"
    return "an" if re.match(r"^[aeiou]", word.lower()) else "a"

def _strip_trailing_punct(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[.!\s]+$", "", s)
    return s

def _word_count(s: str) -> int:
    return len(re.findall(r"[A-Za-z']+", s))

def _looks_like_definition(s: str) -> bool:
    s_low = s.lower()
    return (
        _word_count(s) >= 6
        or " or " in s_low
        or ";" in s_low
        or s_low.startswith("to ")
        or s_low.startswith("part of the body")
        or s_low.startswith("the emotion of")
        or s_low.startswith("consider in detail")
        or s_low.startswith("change one's")
        or s_low.startswith("change location")
        or s_low.startswith("operate (")
    )

def _compact_adj(adj: str | None) -> str | None:
    if not adj:
        return None
    adj = _strip_trailing_punct(str(adj))
    if not adj or _looks_like_definition(adj):
        low = adj.lower()
        if "joy" in low or "happiness" in low or "mirth" in low:
            return "happy"
        if "wink" in low:
            return "winking"
        if "top and bottom are reversed" in low or "reversed" in low:
            return "upside-down"
        if "cute" in low:
            return "cute"
        return None

    if _word_count(adj) <= 2:
        return adj.lower()

    first = adj.split(",")[0].strip()
    if _word_count(first) <= 3:
        return first.lower()

    return None

def _compact_verb(v: str | None) -> str | None:
    if not v:
        return None
    v = _strip_trailing_punct(str(v))
    low = v.lower()

    mappings = [
        ("take in solid food", "eat"),
        ("produce laughter", "laugh"),
        ("change one's facial expression", "smile"),
        ("to feel an affection", "love"),
        ("operate (a wheeled motorized vehicle", "drive"),
        ("communicate electronically", "message"),
        ("to take a photograph", "take a photo"),
        ("to contact by telephone", "call"),
        ("travel through the air", "fly"),
        ("to ride a bicycle", "bike"),
        ("be dressed in", "wear"),
        ("hear with intention", "listen"),
        ("to perform in (a sport", "play"),
        ("to express something by a gesture", "gesture"),
        ("change location", "go"),
    ]
    for prefix, simple in mappings:
        if low.startswith(prefix):
            return simple

    if _word_count(v) <= 2 and re.match(r"^[a-zA-Z\s'-]+$", v):
        return low

    return None

def _clean_shortcode(sc: str | None) -> str | None:
    if not sc:
        return None
    sc = str(sc).strip()
    if not sc:
        return None
    sc = sc.split(",")[0].strip()
    sc = sc.replace("_", " ")
    return sc.lower() if sc else None

def _articleize(phrase: str) -> str:
    phrase = phrase.strip()
    if not phrase:
        return phrase
    if re.match(r"^(a|an|the)\s+", phrase.lower()):
        return phrase
    return f"{a_or_an(phrase)} {phrase}"

def _join_with_and(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) == 0:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"

def emoji_info_lookup(df):
    cols = ["noun", "verb", "adjective", "name", "definition", "shortcode"]
    lookup = {}
    for _, r in df.iterrows():
        e = str(r["emoji"])
        lookup[e] = {c: clean_token(r[c]) if c in df.columns else None for c in cols}
    return lookup

def interpret_sequence(emojis, lookup):
    """
    Your definition-based interpreter (kept in this script).
    """
    infos = [lookup.get(e, {}) for e in emojis]

    phrases = []
    verbs = []

    for info in infos:
        noun = pick_first(info.get("noun"), info.get("name"))
        name = clean_token(info.get("name"))
        sc   = _clean_shortcode(info.get("shortcode"))
        adj  = _compact_adj(info.get("adjective"))

        base = pick_first(noun, name, sc, info.get("definition"))
        if not base:
            continue
        base = _strip_trailing_punct(str(base)).lower()

        if _looks_like_definition(base):
            base = pick_first(name, sc) or base
            base = _strip_trailing_punct(str(base)).lower()

        if adj and adj not in base:
            base = f"{adj} {base}"

        phrases.append(_articleize(base))

        v = _compact_verb(info.get("verb"))
        if v:
            verbs.append(v)

    seen = set()
    phrases_uniq = []
    for p in phrases:
        if p not in seen:
            phrases_uniq.append(p)
            seen.add(p)

    main_verb = verbs[0] if verbs else None

    if main_verb and phrases_uniq:
        obj = phrases_uniq[0]
        extras = phrases_uniq[1:]
        if extras:
            return f"I {main_verb} {obj} with {_join_with_and(extras)}."
        return f"I {main_verb} {obj}."

    if len(phrases_uniq) == 1:
        return f"It's {phrases_uniq[0]}."
    if len(phrases_uniq) > 1:
        return f"It shows {_join_with_and(phrases_uniq)}."

    names = [pick_first(i.get("shortcode"), i.get("name"), i.get("definition")) for i in infos]
    names = [n for n in names if n]
    return (" ".join(names) + ".") if names else "."

# =========================
# NEW: add knowledge from stage CSVs
# =========================
VS16 = "\ufe0f"
SKIN_TONES = {chr(cp) for cp in range(0x1F3FB, 0x1F400)}  # 1F3FB..1F3FF

def split_emoji_sequence(s: str):
    s = str(s).strip()
    if not s:
        return []
    if any(ch.isspace() for ch in s):
        return [t for t in s.split() if t]
    return [c for c in regex.findall(r"\X", s) if c.strip()]

def normalize_emoji_token(tok: str, strip_tone=True, strip_vs16=True) -> str:
    if tok is None:
        return ""
    t = str(tok)
    if strip_vs16:
        t = t.replace(VS16, "")
    if strip_tone:
        t = "".join(ch for ch in t if ch not in SKIN_TONES)
    return t.strip()

def normalize_emoji_sequence(tokens, strip_tone=True, strip_vs16=True):
    return tuple(
        normalize_emoji_token(t, strip_tone=strip_tone, strip_vs16=strip_vs16)
        for t in tokens
        if t
    )

def normalize_english(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\r\n.")
    return s

def canonical_english(variants):
    """
    Choose a stable meaning for a sequence:
    vote by frequency, tie-break by shorter.
    """
    normed = [normalize_english(v).lower() for v in variants if isinstance(v, str) and v.strip()]
    if not normed:
        return ""
    counts = Counter(normed)
    best_norm, _ = max(counts.items(), key=lambda kv: (kv[1], -len(kv[0])))
    originals = [normalize_english(v) for v in variants if normalize_english(v).lower() == best_norm]
    originals.sort(key=len)
    return originals[0] if originals else best_norm

def multiset_jaccard(a_tokens, b_tokens):
    ca, cb = Counter(a_tokens), Counter(b_tokens)
    inter = sum(min(ca[k], cb[k]) for k in ca.keys() | cb.keys())
    union = sum(max(ca[k], cb[k]) for k in ca.keys() | cb.keys())
    return (inter / union) if union else 0.0

class EmojiSequenceMemory:
    def __init__(self):
        self.map = defaultdict(list)
        self.keys = []

    def add(self, seq_tokens, english_text):
        key = tuple(seq_tokens)
        if english_text:
            self.map[key].append(english_text)

    def finalize(self):
        self.keys = list(self.map.keys())

    @classmethod
    def from_stage_csvs(cls, paths, min_len=2, max_len=6):
        mem = cls()
        for p in paths:
            df = pd.read_csv(p)

            # Expect columns: input (english), output (emoji seq).
            # If yours differ, edit these two lines.
            in_col = "input"
            out_col = "output"

            for _, r in df.iterrows():
                eng = r.get(in_col, None)
                emo = r.get(out_col, None)
                if pd.isna(eng) or pd.isna(emo):
                    continue

                emo_tokens = split_emoji_sequence(emo)
                if not (min_len <= len(emo_tokens) <= max_len):
                    continue

                eng_text = normalize_english(eng)

                # store (a) strip VS16 only, keep tone
                k1 = normalize_emoji_sequence(emo_tokens, strip_tone=False, strip_vs16=True)
                # store (b) strip tone + VS16
                k2 = normalize_emoji_sequence(emo_tokens, strip_tone=True, strip_vs16=True)

                mem.add(k1, eng_text)
                mem.add(k2, eng_text)

        mem.finalize()
        return mem

    def lookup_exact(self, seq_tokens):
        key = tuple(seq_tokens)
        if key in self.map:
            return canonical_english(self.map[key]), 1.0
        return None, 0.0

    def retrieve_soft(self, seq_tokens, max_candidates=5000):
        if not self.keys:
            return None, 0.0

        q = tuple(seq_tokens)
        qlen = len(q)

        # restrict by length for speed
        candidates = [k for k in self.keys if len(k) == qlen]
        if not candidates:
            candidates = [k for k in self.keys if abs(len(k) - qlen) <= 1]

        if len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]

        best_k = None
        best_s = 0.0
        for k in candidates:
            s = multiset_jaccard(q, k)
            s -= 0.05 * abs(len(k) - qlen)  # mild penalty if length differs
            if s > best_s:
                best_s = s
                best_k = k

        if best_k is None:
            return None, 0.0

        return canonical_english(self.map[best_k]), float(best_s)

def interpret_sequence_hybrid(emojis, lookup, seq_memory, soft_threshold=0.70):
    """
    1 emoji  -> your definition interpreter
    2-6      -> sequence memory exact -> sequence memory soft -> fallback to definition interpreter
    """
    raw = list(emojis)

    # normalize query tokens two ways
    q1 = normalize_emoji_sequence(raw, strip_tone=False, strip_vs16=True)
    q2 = normalize_emoji_sequence(raw, strip_tone=True,  strip_vs16=True)

    # single emoji: definition-based is usually best
    if len(raw) <= 1:
        return interpret_sequence(raw, lookup)

    # exact lookup
    for q in (q1, q2):
        ans, score = seq_memory.lookup_exact(q)
        if ans:
            return ans if ans.endswith((".", "!", "?")) else ans + "."

    # soft lookup
    best_ans, best_score = None, 0.0
    for q in (q1, q2):
        ans, score = seq_memory.retrieve_soft(q)
        if ans and score > best_score:
            best_ans, best_score = ans, score

    if best_ans and best_score >= soft_threshold:
        return best_ans if best_ans.endswith((".", "!", "?")) else best_ans + "."

    # fallback: definition-based composition
    return interpret_sequence(raw, lookup)

# =========================
# MAIN: load + run examples
# =========================
if __name__ == "__main__":
    # Load emoji metadata
    merged_emoji_df = pd.read_parquet(MERGED_PARQUET_PATH)
    print("merged_emoji_df shape:", merged_emoji_df.shape)

    lookup = emoji_info_lookup(merged_emoji_df)

    # Build knowledge base from the other CSV files (stages)
    seq_memory = EmojiSequenceMemory.from_stage_csvs(STAGE_PATHS, min_len=2, max_len=6)
    print("sequence memory size (keys):", len(seq_memory.keys))

    # Examples
    tests = [
        ["🍕"],
        ["🍕", "❤️"],
        ["😂", "🔥"],
        ["🏃", "🌧️"],
        ["🤥", "👅", "🧠"],
        ["✈️", "🧳", "🏝️", "😴"],
    ]

    for t in tests:
        print(t, "->", interpret_sequence_hybrid(t, lookup, seq_memory))
