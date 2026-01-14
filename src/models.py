# src/models.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class Chunk:
    id: str
    lecture: str
    slide: int
    title: str
    text: str                 # wariant A – oryginalny text
    enriched_text: Optional[str] = None  # wariant B+ – enriched_text

    def get_text(self, variant: str = "BPLUS") -> str:
        """
        Zwraca tekst chunku w zależności od wariantu:
        - A      -> text
        - BPLUS  -> enriched_text (jeśli jest), inaczej text
        """
        v = variant.upper()
        if v == "BPLUS":
            return (self.enriched_text or self.text).strip()
        return self.text.strip()
