from abc import ABC, abstractmethod
from llm_eval.datasets.base import Sample


class BasePromptBuilder(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def build(self, sample: Sample) -> list[dict[str, str]]:
        ...
