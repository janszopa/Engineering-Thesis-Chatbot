# src/chat_rag.py
from pathlib import Path
import argparse

from .rag_loader import load_chunks
from .llm_client_ollama import LLMClientOllama
from .prompt_builder import build_rag_messages
from .retrievers import make_retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chatbot RAG – tutor z przedmiotu")
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
        "--persist-dir",
        default="chroma_db",
        help="Katalog na dane ChromaDB (dla semantic)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    variant = args.variant.upper()
    retriever_name = args.retriever.lower()

    print(f"[INFO] Wariant korpusu: {variant}")
    print(f"[INFO] Retriever: {retriever_name}")

    jsonl_path = Path("processed/Chunks_Bplus_full_enriched.jsonl")
    print(f"[INFO] Wczytuję korpus z {jsonl_path} ...")
    chunks = load_chunks(str(jsonl_path))
    print(f"[INFO] Załadowano {len(chunks)} chunków.")

    # wybierz retriever
    retriever = make_retriever(
        name=retriever_name,
        chunks=chunks,
        variant=variant,
        persist_dir=args.persist_dir,
    )

    llm = LLMClientOllama(model="llama3.1")  # lub inny model w Ollamie

    print("=== Chatbot RAG – tutor z przedmiotu ===")
    print("Napisz pytanie (albo 'quit' żeby wyjść).")

    while True:
        try:
            question = input("\nTy: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Kończę.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("[INFO] Kończę.")
            break

        # RAG: wybierz najistotniejsze fragmenty
        top = retriever.retrieve(question, k=5)
        if not top:
            print("Bot: Nie znalazłem żadnych powiązanych fragmentów w materiałach.")
            continue

        # LOGOWANIE – pokaż, na czym bot bazuje
        print("\n[DEBUG] Użyte fragmenty:")
        for ch, score in top:
            print(f"  - {ch.id} (slajd {ch.slide}, score={score:.3f}): {ch.title}")

        context_chunks = [ch for (ch, _score) in top]
        messages = build_rag_messages(question, context_chunks, variant=variant)
        answer = llm.chat(messages, temperature=0.1, max_tokens=512)

        print("\nBot:", answer)


if __name__ == "__main__":
    main()
