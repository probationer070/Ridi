from abc import ABC, abstractmethod
from typing import Iterator, List, Dict


class LLMBackend(ABC):
    @abstractmethod
    def generate_stream(self, messages: List[Dict]) -> Iterator[str]: ...


class LocalLlamaBackend(LLMBackend):
    """Delegates to the existing ModelChat wrapper for llama-cpp."""

    def __init__(self, model_chat):
        self.model_chat = model_chat

    def generate_stream(self, messages: List[Dict]) -> Iterator[str]:
        return self.model_chat.generate_stream(messages)


class GeminiBackend(LLMBackend):
    """Calls the Google Gemini API and yields the full response as one chunk."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model

    def generate_stream(self, messages: List[Dict]) -> Iterator[str]:
        from google import genai

        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"[System]: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"User: {content}")

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model, contents="\n".join(parts)
        )
        yield response.text
