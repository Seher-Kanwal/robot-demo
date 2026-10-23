"""
Turns whatever the user said or typed in answer to "What is this called?" into a clean
object name -- "Um, it's a TV remote." -> "TV remote". Shared by the speech and the typed
path, so both end up storing the same name for the same answer.
"""
import re

MAX_NAME_LEN = 40

# Spoken hesitations in front of the actual answer ("um, ...", "okay so ...").
_LEADING_FILLERS = re.compile(r"^(?:(?:um+|uh+|erm*|hmm+|so|well|okay|ok|oh)[,\s]+)+", re.I)
# Sentence frames around the name ("it's a ...", "this is called ...").
_LEADING_PHRASES = re.compile(
    r"^(?:(?:i think|i guess|maybe|probably)\s+)?"
    # longer frames first: "this is called" must win over "this is"
    r"(?:(?:it'?s|it is|this is|that'?s|that is) called|we call (?:it|this|that)|call it|"
    r"the name is|its name is|it'?s named|"
    r"it'?s|it is|this is|that'?s|that is|these are|those are|they'?re|they are|this one is)(?:\s+|$)",
    re.I,
)
_ARTICLES = re.compile(r"^(?:a|an|the|my|our|your|some)(?:\s+|$)", re.I)
_EDGE_PUNCT = " \t.,!?;:\"'"


def normalize_name(text, max_len=MAX_NAME_LEN):
    """Strips fillers, sentence frames, articles and edge punctuation; keeps the user's
    casing ("TV remote" stays "TV remote"). Returns "" if nothing name-like is left."""
    s = (text or "").replace("\u2019", "'").replace("\u2018", "'")  # curly apostrophes from speech/phones
    s = re.sub(r"\s+", " ", s).strip(_EDGE_PUNCT)
    prev = None
    while s != prev:  # frames can stack: "um, I think it's called a remote"
        prev = s
        s = _LEADING_FILLERS.sub("", s)
        s = _LEADING_PHRASES.sub("", s)
        s = _ARTICLES.sub("", s)
        s = s.strip(_EDGE_PUNCT)
    return s[:max_len].strip(_EDGE_PUNCT)


def name_key(name):
    """Case-insensitive identity for a name, so "TV remote" and "tv remote" merge."""
    return re.sub(r"\s+", " ", (name or "").strip()).casefold()
