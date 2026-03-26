from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Sample:
    """Universal evaluation item. Every dataset must produce these."""

    id: str  # e.g. "causal_reasoning/42"
    question: str  # Raw question text (may contain <ts><ts/> placeholder)
    answer: str  # Gold answer: "A", "B", "C", or "D"
    timeseries: list[list[float]]  # List of channels, each a list of floats
    choices: dict[str, str] | None  # {"A": "...", "B": "..."} or None if embedded in question
    metadata: dict = field(default_factory=dict)  # domain, task_name, etc.


class BaseDataset(ABC):
    """Abstract base for any evaluation dataset."""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def load(self, tasks: list[str] | None = None) -> Iterator[Sample]:
        """Yield Sample objects. If tasks is None, yield all."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Return dataset identifier string."""
        ...
