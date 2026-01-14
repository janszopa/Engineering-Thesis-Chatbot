# src/retrievers.py
import re
from typing import List, Tuple, Dict
from collections import Counter

from .models import Chunk

# --- LEXICAL RETRIEVER (R1) ---

STOPWORDS = {
    "i", "oraz", "a", "w", "we", "z", "za", "do", "na", "o", "od", "u",
    "jest", "są", "to", "że", "czy", "jak", "dla", "lub",
    "the", "and", "or", "a", "an", "in", "of", "on", "at", "to", "for",
}


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


class LexicalRetriever:
    """
    Prosty lexical retriever oparty o overlap słów (R1).
    """
    def __init__(self, chunks: List[Chunk], variant: str = "BPLUS") -> None:
        self.chunks = chunks
        self.variant = variant.upper()

    def _score_chunk(self, query_tokens: List[str], chunk: Chunk) -> float:
        text = chunk.get_text(variant=self.variant)
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

        norm = len(set(content_tokens)) ** 0.5
        if norm > 0:
            score = score / norm

        return score

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        query_tokens = _tokenize(query)
        scored: List[Tuple[Chunk, float]] = []

        for ch in self.chunks:
            s = self._score_chunk(query_tokens, ch)
            if s > 0:
                scored.append((ch, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


# --- SEMANTIC RETRIEVER (R2) ---

import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path


class SemanticRetriever:
    """
    Semantic retriever oparty o embeddings BAAI/bge-m3 + ChromaDB (R2).

    Zakładamy, że indeks został wcześniej zbudowany przez semantic_index.py
    do kolekcji 'chunks_<variant.lower()>'.
    """
    def __init__(
        self,
        chunks: List[Chunk],
        variant: str = "BPLUS",
        persist_dir: str = "chroma_db",
    ) -> None:
        self.variant = variant.upper()
        self.chunks_by_id: Dict[str, Chunk] = {ch.id: ch for ch in chunks}

        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(persist_path))

        self.collection_name = f"chunks_{self.variant.lower()}"
        self.collection = self.client.get_or_create_collection(self.collection_name)

        # model embeddingowy
        self.embedder = SentenceTransformer("BAAI/bge-m3")

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        query_emb = self.embedder.encode([query])[0]

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=k,
        )

        ids_lists = results.get("ids", [[]])
        distances_lists = results.get("distances", [[]])
        documents_lists = results.get("documents", [[]])
        metadatas_lists = results.get("metadatas", [[]])

        ids = ids_lists[0] if ids_lists else []
        distances = distances_lists[0] if distances_lists else []
        documents = documents_lists[0] if documents_lists else []
        metadatas = metadatas_lists[0] if metadatas_lists else []

        out: List[Tuple[Chunk, float]] = []

        for idx, doc_id in enumerate(ids):
            # Chroma zwraca id jako string
            if isinstance(doc_id, list):
                doc_id = doc_id[0]

            # zamiana distance -> score (im mniejsza odległość, tym większy score)
            if distances and idx < len(distances):
                dist = distances[idx]
                score = 1.0 / (1.0 + dist)  # prosty, monotoniczny mapping
            else:
                score = 1.0

            ch = self.chunks_by_id.get(doc_id)
            if ch is None:
                # fallback: budujemy minimalny Chunk z metadata i documents
                meta = metadatas[idx] if metadatas and idx < len(metadatas) else {}
                lecture = meta.get("lecture", "")
                slide = meta.get("slide", -1)
                title = meta.get("title", "")
                text = documents[idx] if documents and idx < len(documents) else ""
                ch = Chunk(
                    id=str(doc_id),
                    lecture=lecture,
                    slide=slide,
                    title=title,
                    text=text,
                    enriched_text=None,
                )
            out.append((ch, score))

        return out


# --- FABRYKA RETRIEVERÓW ---

def make_retriever(
    name: str,
    chunks: List[Chunk],
    variant: str = "BPLUS",
    persist_dir: str = "chroma_db",
):
    """
    name: 'lexical' | 'semantic'
    """
    name_low = name.lower()
    if name_low == "lexical":
        return LexicalRetriever(chunks, variant=variant)
    elif name_low == "semantic":
        return SemanticRetriever(chunks, variant=variant, persist_dir=persist_dir)
    else:
        raise ValueError(f"Nieznany retriever: {name}")
