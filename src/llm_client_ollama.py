# src/llm_client_ollama.py
import requests
from typing import List, Dict


class LLMClientOllama:
    """
    Prosty klient do Ollamy używający /api/chat.
    """
    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434"
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: List[Dict], temperature: float = 0.2, max_tokens: int = 512) -> str:
        """
        messages: lista {"role": "system"|"user"|"assistant", "content": "..."}
        Zwraca treść odpowiedzi (string).
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        # Ollama zwraca jedno pole "message"
        return data["message"]["content"]
