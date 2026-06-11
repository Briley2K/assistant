"""
Spoken-phrase normalization and matching, shared by the wake-word listener
and the sleep/wake command handling.
"""
import re
import difflib


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return " ".join(re.sub(r"[^a-z0-9\s]", "", text.lower()).split())


def contains(text: str, phrase: str) -> bool:
    """True if the phrase appears in the text as a whole-word sequence."""
    tw, pw = normalize(text).split(), normalize(phrase).split()
    if not pw:
        return False
    return any(tw[i:i + len(pw)] == pw for i in range(len(tw) - len(pw) + 1))


def similarity(text: str, phrase: str) -> float:
    """Fuzzy match ratio (0–1) between normalized strings."""
    return difflib.SequenceMatcher(None, normalize(text), normalize(phrase)).ratio()
