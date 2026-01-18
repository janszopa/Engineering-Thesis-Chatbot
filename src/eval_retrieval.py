# src/eval_retrieval.py
import argparse
import json
import csv
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
    csv_out: str | None = None,
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

    rows_for_csv: List[Dict] = []

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

        rows_for_csv.append({
            "variant": variant,
            "retriever": retriever_name,
            "k": k,
            "question_id": qid,
            "question": question,
            "relevant_ids": "|".join(sorted(relevant_ids)),
            "retrieved_ids": "|".join(retrieved_ids),
            "hit": int(hit),
            "rank": rank if rank is not None else "",
        })

    hit_rate = hits / total if total > 0 else 0.0
    mrr = mrr_sum / total if total > 0 else 0.0

    print("\n=== PODSUMOWANIE ===")
    print(f"Hit@{k}: {hit_rate:.3f}  ({hits}/{total})")
    print(f"MRR@{k}: {mrr:.3f}")

    if csv_out is not None:
        out_path = Path(csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "variant",
                "retriever",
                "k",
                "question_id",
                "question",
                "relevant_ids",
                "retrieved_ids",
                "hit",
                "rank",
            ])
            for row in rows_for_csv:
                writer.writerow([
                    row["variant"],
                    row["retriever"],
                    row["k"],
                    row["question_id"],
                    row["question"],
                    row["relevant_ids"],
                    row["retrieved_ids"],
                    row["hit"],
                    row["rank"],
                ])
        print(f"[INFO] Zapisano szczegółowe wyniki do {out_path}")
        

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
        choices=["lexical", "semantic", "hybrid"],
        default="lexical",
        help="Rodzaj retrievera: lexical, semantic lub hybrid",
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
        help="Katalog na dane ChromaDB (dla semantic/hybrid)",
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Jeśli podane, ścieżka do pliku CSV z wynikami",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    variant = args.variant.upper()
    retriever_name = args.retriever.lower()

    jsonl_path = Path("processed/Chunks_Bplus_full_enriched.jsonl")
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
        csv_out=args.csv_out,
    )


if __name__ == "__main__":
    main()
