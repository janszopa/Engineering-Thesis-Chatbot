# src/prompt_builder.py
from typing import List, Dict
from .models import Chunk


def build_rag_messages(
    question: str,
    context_chunks: List[Chunk],
    variant: str = "BPLUS",
) -> List[Dict]:
    """
    Buduje listę messages dla /api/chat w Ollamie.
    """
    context_parts = []
    for ch in context_chunks:
        header = f"[{ch.lecture}, slajd {ch.slide}: {ch.title}]"
        text = ch.get_text(variant=variant)
        context_parts.append(f"{header}\n{text}")
    context_text = "\n\n".join(context_parts)

    system_msg = {
        "role": "system",
        "content": (
            "Jesteś tutorem z przedmiotu „Zarządzanie projektami, wiedzą i pracą zespołową”. "
            "Odpowiadasz krótko, rzeczowo i po polsku. "
            "Korzystaj WYŁĄCZNIE z materiałów z kursu podanych poniżej. "
            "Jeśli czegoś nie ma w materiałach, powiedz wprost, że nie wiesz i nie zgaduj."
        ),
    }

    user_content = (
        "Poniżej masz fragmenty materiałów z kursu.\n\n"
        "== MATERIAŁY Z KURSU ==\n"
        f"{context_text}\n\n"
        "== ZADANIE ==\n"
        "Na podstawie powyższych materiałów odpowiedz na pytanie użytkownika. "
        "Odpowiedź ma być zwięzła, ale zrozumiała.\n\n"
        f"Pytanie użytkownika: {question}"
    )

    user_msg = {
        "role": "user",
        "content": user_content,
    }

    return [system_msg, user_msg]
