"""Pick the support sentences that ground an explanation or a stimulus excerpt."""

from __future__ import annotations

import re

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_WORD = re.compile(r"[a-z0-9]+")
STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "by",
        "with",
        "as",
        "that",
        "this",
        "which",
        "what",
        "who",
        "whom",
        "how",
        "why",
        "when",
        "where",
        "do",
        "does",
        "did",
        "it",
        "its",
        "from",
        "into",
        "can",
        "could",
        "would",
        "will",
        "may",
        "might",
        "than",
        "then",
        "there",
        "their",
        "they",
        "these",
        "those",
        "called",
        "known",
    ]
)
MAX_PASSAGE_CHARS = 1200


def sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT.split(text or "") if s.strip()]
    return [p for p in parts if len(p) > 15]


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in STOP and len(w) > 2}


def score_sentence(sentence: str, question: str, answer: str) -> float:
    words = _content_words(sentence)
    q = _content_words(question)
    a = _content_words(answer)
    overlap = len(words & q) / (len(q) or 1)
    has_answer = 1.0 if answer.lower() in sentence.lower() else (len(words & a) / (len(a) or 1)) * 0.6
    return 1.5 * has_answer + overlap


def best_sentences(support: str, question: str, answer: str, k: int = 2) -> list[int]:
    sents = sentences(support)
    if not sents:
        return []
    ranked = sorted(range(len(sents)), key=lambda i: -score_sentence(sents[i], question, answer))
    return sorted(ranked[:k])


def explanation(support: str, question: str, answer: str, letter: str) -> str | None:
    sents = sentences(support)
    idx = best_sentences(support, question, answer, k=2)
    if not idx:
        return None
    # Keep only the second sentence if it adds something related to the question.
    chosen = [sents[idx[0]]] + [sents[i] for i in idx[1:] if score_sentence(sents[i], question, answer) > 0.3]
    body = " ".join(chosen)
    return f"{letter} is correct. {body}"


def stimulus_excerpt(support: str, question: str, answer: str, max_sentences: int = 3) -> str | None:
    sents = sentences(support)
    idx = best_sentences(support, question, answer, k=1)
    if not idx:
        return None
    center = idx[0]
    lo = max(0, center - 1)
    window = sents[lo : lo + max_sentences]
    text = " ".join(window)
    return text[:MAX_PASSAGE_CHARS]


def trim_passage(support: str, question: str, answer: str) -> str:
    """Passage given to the generator: whole support if short, else a window around the key sentence."""
    if len(support) <= MAX_PASSAGE_CHARS:
        return support
    sents = sentences(support)
    idx = best_sentences(support, question, answer, k=1)
    if not idx:
        return support[:MAX_PASSAGE_CHARS]
    out: list[str] = []
    start = max(0, idx[0] - 2)
    for s in sents[start:]:
        if sum(len(x) + 1 for x in out) + len(s) > MAX_PASSAGE_CHARS:
            break
        out.append(s)
    return " ".join(out) or support[:MAX_PASSAGE_CHARS]
