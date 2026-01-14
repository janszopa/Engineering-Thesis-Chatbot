# src/rag_loader.py
import json
from pathlib import Path
from typing import List
from .models import Chunk


def load_chunks(jsonl_path: str) -> List[Chunk]:
    """
    Wczytuje korpus z JSONL.
    Zakładamy pola: id, lecture, slide, title, text, enriched_text.
    """

    chunks: List[Chunk] = []
    path = Path(jsonl_path)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            base_text = (obj.get("text") or "").strip()
            enriched = obj.get("enriched_text")
            if enriched is not None:
                enriched = enriched.strip()

            chunk = Chunk(
                id=obj["id"],
                lecture=obj.get("lecture", ""),
                slide=obj.get("slide", -1),
                title=obj.get("title", ""),
                text=base_text,
                enriched_text=enriched,
            )
            chunks.append(chunk)

    return chunks
