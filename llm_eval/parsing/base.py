from abc import ABC, abstractmethod


class BaseAnswerParser(ABC):
    @abstractmethod
    def parse(self, text: str) -> str | None:
        """Extract the predicted answer from LLM output. Returns None if parsing fails."""
        ...
