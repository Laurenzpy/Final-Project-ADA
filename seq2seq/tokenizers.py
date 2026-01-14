import re
import regex  # wichtig für Emoji-Grapheme

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
