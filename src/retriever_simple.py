# src/retriever_simple.py
import re
from typing import List, Tuple
from collections import Counter

from .models import Chunk

# bardzo prosty zestaw stopwords PL/EN – tylko do przycięcia szumu
STOPWORDS = {
    "i", "oraz", "a", "w", "we", "z", "za", "do", "na", "o", "od", "u",
    "jest", "są", "to", "że", "czy", "jak", "dla", "lub",
    "the", "and", "or", "a", "an", "in", "of", "on", "at", "to", "for",
}


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"\w+", text.lower())
    # odfiltruj stopwords i bardzo krótkie tokeny
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def _score_chunk(query_tokens: List[str], chunk: Chunk, variant: str) -> float:
    """
    Lekko lepszy score:
    - liczymy overlap słów z treścią + tytułem,
    - słowa z tytułu są nieco ważniejsze,
    - wynik normalizujemy przez długość chunku.
    """
    text = chunk.get_text(variant=variant)
    content_tokens = _tokenize(text)
    title_tokens = _tokenize(chunk.title)

    if not content_tokens:
        return 0.0

    content_counter = Counter(content_tokens)
    title_counter = Counter(title_tokens)

    score = 0.0
    for t in query_tokens:
        if t in content_counter:
            score += 1.0
        if t in title_counter:
            score += 1.5  # tytuł trochę ważniejszy

    # prosta normalizacja: im dłuższy chunk, tym trochę mniejszy score
    norm = len(set(content_tokens)) ** 0.5
    if norm > 0:
        score = score / norm

    return score


def retrieve_top_k(
    chunks: List[Chunk],
    query: str,
    k: int = 5,
    variant: str = "BPLUS",
) -> List[Tuple[Chunk, float]]:
    """
    Zwraca k najlepszych chunków z prostym, ale sensowniejszym scorem.
    """
    query_tokens = _tokenize(query)
    scored: List[Tuple[Chunk, float]] = []

    for ch in chunks:
        s = _score_chunk(query_tokens, ch, variant=variant)
        if s > 0:
            scored.append((ch, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
