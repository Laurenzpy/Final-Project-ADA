import re
import regex  # wichtig für Emoji-Grapheme
import unicodedata


def tokenize_emojis(s: str) -> list[str]:
    """
    Zerlegt eine Emoji-Sequenz in echte Emoji-Tokens (Grapheme Cluster).
    Beispiel: '👨‍👩‍👧‍👦🔥' -> ['👨‍👩‍👧‍👦', '🔥']
    """
    if s is None:
        return []
    s = s.strip()
    tokens = regex.findall(r"\X", s)
    return [t for t in tokens if not t.isspace()]


_whitespace = re.compile(r"\s+")


def tokenize_text(s: str) -> list[str]:
    """
    Einfache Wort-Tokenisierung für kurze Textphrasen.
    """
    if s is None:
        return []
    s = s.lower().strip()
    s = _whitespace.sub(" ", s)
    return s.split(" ") if s else []


# -----------------------------
# NEW: Emoji -> Text (T5-safe)
# -----------------------------

# Characters we want to ignore when creating readable emoji names
_IGNORE_CODEPOINTS = {
    0x200D,  # ZERO WIDTH JOINER
    0xFE0F,  # VARIATION SELECTOR-16
    0xFE0E,  # VARIATION SELECTOR-15
}


def _safe_unicode_name(ch: str) -> str:
    """
    Returns a lowercase, underscore-separated unicode name, or a hex fallback.
    """
    try:
        name = unicodedata.name(ch)
        name = name.lower().replace(" ", "_").replace("-", "_")
        return name
    except ValueError:
        # no unicode name (rare) -> fallback
        return f"u{ord(ch):04x}"


def _emoji_grapheme_to_alias(grapheme: str) -> str:
    """
    Converts one grapheme cluster into a stable, readable alias token.
    Examples:
      '🍕' -> 'emoji_slice_of_pizza'
      '❤️' -> 'emoji_heavy_black_heart' (variation selector ignored)
      '👨‍👩‍👧‍👦' -> 'emoji_man_woman_girl_boy' (joined names)
    """
    parts = []
    for ch in grapheme:
        cp = ord(ch)
        if cp in _IGNORE_CODEPOINTS:
            continue

        # also ignore most combining marks
        if unicodedata.category(ch).startswith("M"):
            continue

        parts.append(_safe_unicode_name(ch))

    # If everything got ignored, keep a stable fallback
    if not parts:
        return "emoji_unknown"

    # Clean up some very noisy tokens
    # (optional: you can expand this list if you see garbage in outputs)
    noisy = {
        "zero_width_joiner",
        "variation_selector_16",
        "variation_selector_15",
    }
    parts = [p for p in parts if p not in noisy]

    alias = "emoji_" + "_".join(parts)
    # normalize repeated underscores
    alias = re.sub(r"_+", "_", alias).strip("_")
    return alias


def emojis_to_text_aliases(s: str) -> str:
    """
    MAIN FUNCTION FOR T5:
    Converts a string containing emojis into a whitespace-separated alias text.

    Input:
      '🍕 ❤️'
    Output:
      'emoji_slice_of_pizza emoji_heavy_black_heart'

    Why:
      T5Tokenizer does not have emoji chars in vocab -> they become <unk>.
      Text aliases are tokenizable by SentencePiece -> no <unk> collapse.
    """
    if s is None:
        return ""

    s = s.strip()
    if not s:
        return ""

    graphemes = regex.findall(r"\X", s)

    out_tokens = []
    for g in graphemes:
        if g.isspace():
            continue

        # If it's a normal word/number/punctuation (non-emoji), keep it as text
        # Heuristic: if all chars are ASCII and not control, keep as-is
        if all(ord(ch) < 128 and not unicodedata.category(ch).startswith("C") for ch in g):
            # split on whitespace later anyway
            out_tokens.append(g.lower())
            continue

        # Otherwise treat it as emoji grapheme cluster -> alias
        out_tokens.append(_emoji_grapheme_to_alias(g))

    return " ".join(out_tokens)