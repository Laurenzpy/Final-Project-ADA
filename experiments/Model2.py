import re
import pandas as pd

# Load the merged emoji dataframe from a parquet file
merged_emoji_df = pd.read_parquet("merged_emoji_df.parquet")
print(merged_emoji_df.shape)

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
    """
    Heuristic: your sample 'sense_*' fields are often full definitions.
    """
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
        # map common definition-y patterns to a simple adjective
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

    # if it's already short, keep it
    if _word_count(adj) <= 2:
        return adj.lower()

    # try to keep a short first chunk
    first = adj.split(",")[0].strip()
    if _word_count(first) <= 3:
        return first.lower()

    return None

def _compact_verb(v: str | None) -> str | None:
    if not v:
        return None
    v = _strip_trailing_punct(str(v))
    low = v.lower()

    # Turn common definition-y verb senses into simple verbs
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

    # If it’s short already, keep it
    if _word_count(v) <= 2 and re.match(r"^[a-zA-Z\s'-]+$", v):
        return low

    return None

def _clean_shortcode(sc: str | None) -> str | None:
    if not sc:
        return None
    sc = str(sc).strip()
    if not sc:
        return None
    # handle "laughing,satisfied" -> "laughing"
    sc = sc.split(",")[0].strip()
    sc = sc.replace("_", " ")
    return sc.lower() if sc else None

def _articleize(phrase: str) -> str:
    phrase = phrase.strip()
    if not phrase:
        return phrase
    # don't double-article
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



def simple_present(verb):
    # Keep it very simple: "I <verb>" (base form)
    # If verb contains spaces ("fall in love"), keep it.
    return verb

def emoji_info_lookup(df):
    # emoji -> dict of fields
    cols = ["noun", "verb", "adjective", "name", "definition", "shortcode"]
    lookup = {}
    for _, r in df.iterrows():
        e = str(r["emoji"])
        lookup[e] = {c: clean_token(r[c]) if c in df.columns else None for c in cols}
    return lookup

def interpret_sequence(emojis, lookup):
    """
    Uses ALL emojis by turning each into a clean noun-phrase,
    then optionally uses a verb to form SVO.
    """
    infos = [lookup.get(e, {}) for e in emojis]

    # Build a nice phrase for each emoji (in order)
    phrases = []
    verbs = []

    for info in infos:
        noun = pick_first(info.get("noun"), info.get("name"))
        name = clean_token(info.get("name"))
        sc   = _clean_shortcode(info.get("shortcode"))
        adj  = _compact_adj(info.get("adjective"))

        # Prefer name if noun looks like a long definition
        base = pick_first(noun, name, sc, info.get("definition"))
        if not base:
            continue
        base = _strip_trailing_punct(str(base)).lower()

        if _looks_like_definition(base):
            base = pick_first(name, sc) or base
            base = _strip_trailing_punct(str(base)).lower()

        # Attach adjective if it’s clean and not already included
        if adj and adj not in base:
            base = f"{adj} {base}"

        phrases.append(_articleize(base))

        v = _compact_verb(info.get("verb"))
        if v:
            verbs.append(v)

    # De-dupe phrases while preserving order
    seen = set()
    phrases_uniq = []
    for p in phrases:
        if p not in seen:
            phrases_uniq.append(p)
            seen.add(p)

    # Pick a main verb if any
    main_verb = verbs[0] if verbs else None

    # If we have a verb, use SVO and include the rest with "with ..."
    if main_verb and phrases_uniq:
        obj = phrases_uniq[0]
        extras = phrases_uniq[1:]

        if extras:
            return f"I {main_verb} {obj} with {_join_with_and(extras)}."
        return f"I {main_verb} {obj}."

    # No verb: describe the set of emojis cleanly
    if len(phrases_uniq) == 1:
        return f"It's {phrases_uniq[0]}."
    if len(phrases_uniq) > 1:
        return f"It shows {_join_with_and(phrases_uniq)}."

    # Fallback
    names = [pick_first(i.get("shortcode"), i.get("name"), i.get("definition")) for i in infos]
    names = [n for n in names if n]
    return " ".join(names) + "."


# -------- Example usage --------
# merged_emoji_df should have columns: emoji, noun, verb, adjective, name, definition, shortcode
lookup = emoji_info_lookup(merged_emoji_df)

print(interpret_sequence(["🍕"], lookup))
print(interpret_sequence(["🍕","❤️"], lookup))
print(interpret_sequence(["😂","🔥"], lookup))
print(interpret_sequence(["🏃","🌧️"], lookup))
print(interpret_sequence(["🤥","👅", "🧠"], lookup))