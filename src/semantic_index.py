# src/semantic_index.py
import argparse
from pathlib import Path
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

from .rag_loader import load_chunks
from .models import Chunk


def build_semantic_index(
    jsonl_path: str,
    variant: str = "BPLUS",
    persist_dir: str = "chroma_db",
) -> None:
    variant = variant.upper()
    print(f"[INFO] Buduję indeks semantyczny dla wariantu: {variant}")

    chunks: List[Chunk] = load_chunks(jsonl_path)
    print(f"[INFO] Załadowano {len(chunks)} chunków z {jsonl_path}")

    # przygotuj Chroma
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_path))

    collection_name = f"chunks_{variant.lower()}"

    # jeśli kolekcja istnieje, usuń i utwórz od nowa
    try:
        client.delete_collection(collection_name)
        print(f"[INFO] Usunięto istniejącą kolekcję '{collection_name}'")
    except Exception:
        pass

    collection = client.create_collection(collection_name)
    print(f"[INFO] Utworzono kolekcję '{collection_name}' w {persist_dir}")

    # model embeddingowy
    embedder = SentenceTransformer("BAAI/bge-m3")

    # przygotuj dane do embedowania
    ids = []
    texts = []
    metadatas = []

    for ch in chunks:
        text = ch.get_text(variant=variant)
        if not text:
            continue
        ids.append(ch.id)
        texts.append(text)
        metadatas.append({
            "lecture": ch.lecture,
            "slide": ch.slide,
            "title": ch.title,
        })

    print(f"[INFO] Embedduję {len(texts)} fragmentów...")
    embeddings = embedder.encode(texts, show_progress_bar=True)

    print(f"[INFO] Zapisuję embeddingi do Chroma...")
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        documents=texts,
    )

    print("[INFO] Indeks semantyczny zbudowany i zapisany.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Budowa indeksu semantycznego RAG")
    parser.add_argument(
        "--variant",
        choices=["A", "BPLUS"],
        default="BPLUS",
        help="Wariant korpusu: A = text, BPLUS = enriched_text",
    )
    parser.add_argument(
        "--jsonl-path",
        default="processed/Chunks_Bplus_full_enriched.jsonl",
        help="Ścieżka do pliku JSONL z chunkami",
    )
    parser.add_argument(
        "--persist-dir",
        default="chroma_db",
        help="Katalog na dane ChromaDB",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    build_semantic_index(
        jsonl_path=args.jsonl_path,
        variant=args.variant,
        persist_dir=args.persist_dir,
    )


if __name__ == "__main__":
    main()
