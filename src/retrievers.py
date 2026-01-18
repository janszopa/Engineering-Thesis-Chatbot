# src/retrievers.py
import re
import numpy as np
import chromadb

from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter

from sentence_transformers import SentenceTransformer
from .models import Chunk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- LEXICAL RETRIEVER (R1) – TF-IDF ---

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
    Lexical retriever oparty o TF-IDF + cosine similarity.
    """
    def __init__(self, chunks: List[Chunk], variant: str = "BPLUS") -> None:
        self.chunks = chunks
        self.variant = variant.upper()

        # Teksty dokumentów (chunków)
        self.docs: List[str] = [ch.get_text(variant=self.variant) for ch in self.chunks]
        # Mapowanie id -> indeks w macierzy TF-IDF
        self.id_to_idx: Dict[str, int] = {ch.id: i for i, ch in enumerate(self.chunks)}

        # TF-IDF (możesz dopasować parametry)
        self.vectorizer = TfidfVectorizer(
            max_df=0.8,
            min_df=1,
            ngram_range=(1, 2),      # jedno- i dwuwyrazowe n-gramy
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.docs)

    def _scores_for_all(self, query: str) -> np.ndarray:
        """Cosine similarity query vs wszystkie dokumenty."""
        q_vec = self.vectorizer.transform([query])
        # linear_kernel = cos similarity dla znormalizowanych TF-IDF
        scores = linear_kernel(q_vec, self.tfidf_matrix).flatten()
        return scores

    def _scores_for_subset(self, query: str, subset_chunks: List[Chunk]) -> Dict[str, float]:
        """
        Cosine similarity query vs wybrany podzbiór chunków.
        Zwraca dict: chunk.id -> score.
        """
        q_vec = self.vectorizer.transform([query])
        # indeksy w macierzy TF-IDF dla tych chunków
        indices = [self.id_to_idx[ch.id] for ch in subset_chunks if ch.id in self.id_to_idx]
        if not indices:
            return {}

        sub_matrix = self.tfidf_matrix[indices]
        sims = linear_kernel(q_vec, sub_matrix).flatten()

        id_to_score: Dict[str, float] = {}
        for idx_local, global_idx in enumerate(indices):
            ch_id = self.chunks[global_idx].id
            id_to_score[ch_id] = float(sims[idx_local])
        return id_to_score

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        scores = self._scores_for_all(query)
        if len(scores) == 0:
            return []

        # top-k indeksów
        k = min(k, len(scores))
        top_indices = np.argpartition(-scores, k - 1)[:k]
        # posortuj malejąco po score
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results: List[Tuple[Chunk, float]] = []
        for idx in top_indices:
            ch = self.chunks[idx]
            s = float(scores[idx])
            if s <= 0:
                continue
            results.append((ch, s))
        return results


# --- SEMANTIC RETRIEVER (R2) ---

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

        ids = ids_lists[0] if ids_lists else []
        distances = distances_lists[0] if distances_lists else []

        out: List[Tuple[Chunk, float]] = []

        for idx, doc_id in enumerate(ids):
            if isinstance(doc_id, list):
                doc_id = doc_id[0]

            if distances and idx < len(distances):
                dist = distances[idx]
                score = 1.0 / (1.0 + dist)  # im mniejsza odległość, tym większy score
            else:
                score = 1.0

            ch = self.chunks_by_id.get(doc_id)
            if ch is None:
                continue
            out.append((ch, float(score)))

        return out


# --- HYBRID RETRIEVER (R3) – semantic + lexical reranking ---

class HybridRetriever:
    """
    Hybrid retriever (R3):
    1. SemanticRetriever wybiera top_N kandydatów.
    2. LexicalRetriever liczy TF-IDF score dla tych kandydatów.
    3. Łączymy score: alpha * semantic + (1-alpha) * lexical.
    """
    def __init__(
        self,
        chunks: List[Chunk],
        variant: str = "BPLUS",
        persist_dir: str = "chroma_db",
        candidate_k: int = 30,
        alpha: float = 0.7,   # waga semantic
    ) -> None:
        self.variant = variant.upper()
        self.candidate_k = candidate_k
        self.alpha = alpha

        # oddzielne instancje retrieverów
        self.semantic = SemanticRetriever(
            chunks=chunks,
            variant=variant,
            persist_dir=persist_dir,
        )
        self.lexical = LexicalRetriever(
            chunks=chunks,
            variant=variant,
        )

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        # 1) semantic recall
        sem_results = self.semantic.retrieve(query, k=self.candidate_k)
        if not sem_results:
            return []

        candidate_chunks = [ch for (ch, _s) in sem_results]
        # semantic scores w dict
        sem_scores = {ch.id: s for (ch, s) in sem_results}

        # 2) lexical scores tylko dla kandydatów
        lex_scores = self.lexical._scores_for_subset(query, candidate_chunks)

        # 3) połączenie score'ów
        combined: List[Tuple[Chunk, float]] = []
        for ch in candidate_chunks:
            s_sem = sem_scores.get(ch.id, 0.0)
            s_lex = lex_scores.get(ch.id, 0.0)
            score = self.alpha * s_sem + (1.0 - self.alpha) * s_lex
            combined.append((ch, score))

        # sort malejąco i wybierz top k
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:k]


# --- FABRYKA RETRIEVERÓW ---

def make_retriever(
    name: str,
    chunks: List[Chunk],
    variant: str = "BPLUS",
    persist_dir: str = "chroma_db",
):
    """
    name: 'lexical' | 'semantic' | 'hybrid'
    """
    name_low = name.lower()
    if name_low == "lexical":
        return LexicalRetriever(chunks, variant=variant)
    elif name_low == "semantic":
        return SemanticRetriever(chunks, variant=variant, persist_dir=persist_dir)
    elif name_low == "hybrid":
        return HybridRetriever(chunks, variant=variant, persist_dir=persist_dir)
    else:
        raise ValueError(f"Nieznany retriever: {name}")