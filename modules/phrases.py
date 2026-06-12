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


def matches(text: str, phrase: str, threshold: float = 0.7) -> bool:
    """Robust spoken-command match, tolerant of the mishearings Whisper makes on
    short commands (e.g. "cleo" → "clio"/"leo") and of surrounding filler words.

    True if the phrase appears verbatim, or if any window of the text roughly the
    length of the phrase is fuzzily similar to it above `threshold`. Use a higher
    threshold (≈0.8) where a false match is costly (e.g. going to sleep mid-chat)
    and a lower one (≈0.6) where missing the command is worse (e.g. waking up)."""
    if contains(text, phrase):
        return True
    tw, pw = normalize(text).split(), normalize(phrase).split()
    if not pw or not tw:
        return False
    target = " ".join(pw)
    n = len(pw)
    best = 0.0
    # Slide windows of n-1..n+1 words so a dropped or inserted word still matches.
    for size in {max(1, n - 1), n, n + 1}:
        for i in range(len(tw) - size + 1):
            window = " ".join(tw[i:i + size])
            best = max(best, difflib.SequenceMatcher(None, window, target).ratio())
    return best >= threshold
