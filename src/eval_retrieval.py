# src/eval_retrieval.py
import argparse
import json
from pathlib import Path
from typing import List, Dict, Set

from .rag_loader import load_chunks
from .retrievers import make_retriever
from .models import Chunk


def load_gold(path: str) -> List[Dict]:
    items = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            items.append(obj)
    return items


def eval_retrieval(
    chunks: List[Chunk],
    gold: List[Dict],
    variant: str,
    retriever_name: str,
    k: int = 5,
    persist_dir: str = "chroma_db",
) -> None:
    variant = variant.upper()
    retriever_name = retriever_name.lower()

    retriever = make_retriever(
        name=retriever_name,
        chunks=chunks,
        variant=variant,
        persist_dir=persist_dir,
    )

    total = len(gold)
    hits = 0
    mrr_sum = 0.0

    print(f"[INFO] Start ewaluacji – variant={variant}, retriever={retriever_name}, k={k}")
    print(f"[INFO] Liczba pytań testowych: {total}")

    for item in gold:
        qid = item.get("id", "")
        question = item["question"]
        relevant_ids: Set[str] = set(item.get("relevant_ids", []))

        results = retriever.retrieve(question, k=k)
        retrieved_ids = [ch.id for (ch, _score) in results]

        hit = False
        rank = None
        for idx, cid in enumerate(retrieved_ids, start=1):
            if cid in relevant_ids:
                hit = True
                rank = idx
                break

        if hit:
            hits += 1
            mrr_sum += 1.0 / rank

        print(f"\n[Q {qid}] {question}")
        print(f"  relevant_ids: {sorted(relevant_ids)}")
        print(f"  retrieved_ids (top {k}): {retrieved_ids}")
        print(f"  HIT: {hit} (rank={rank})")

    hit_rate = hits / total if total > 0 else 0.0
    mrr = mrr_sum / total if total > 0 else 0.0

    print("\n=== PODSUMOWANIE ===")
    print(f"Hit@{k}: {hit_rate:.3f}  ({hits}/{total})")
    print(f"MRR@{k}: {mrr:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ewaluacja retrievalu RAG")
    parser.add_argument(
        "--variant",
        choices=["A", "BPLUS"],
        default="BPLUS",
        help="Wariant korpusu: A = text, BPLUS = enriched_text",
    )
    parser.add_argument(
        "--retriever",
        choices=["lexical", "semantic"],
        default="lexical",
        help="Rodzaj retrievera: lexical lub semantic",
    )
    parser.add_argument(
        "--gold-path",
        default="tests/retrieval_gold.jsonl",
        help="Ścieżka do pliku JSONL z pytaniami testowymi",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k w retrievalu",
    )
    parser.add_argument(
        "--persist-dir",
        default="chroma_db",
        help="Katalog na dane ChromaDB (dla semantic)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    variant = args.variant.upper()
    retriever_name = args.retriever.lower()

    jsonl_path = Path("data/Chunks_Bplus_full_enriched.jsonl")
    print(f"[INFO] Wczytuję korpus z {jsonl_path} ...")
    chunks = load_chunks(str(jsonl_path))
    print(f"[INFO] Załadowano {len(chunks)} chunków.")

    gold = load_gold(args.gold_path)

    eval_retrieval(
        chunks=chunks,
        gold=gold,
        variant=variant,
        retriever_name=retriever_name,
        k=args.k,
        persist_dir=args.persist_dir,
    )


if __name__ == "__main__":
    main()
