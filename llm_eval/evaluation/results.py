import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ResultRecord:
    sample_id: str
    model: str
    gold_answer: str
    predicted_answer: str | None
    raw_output: str
    correct: bool
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    metadata: dict


class ResultStore:
    """Append-only JSONL file for results. Supports resume by tracking completed IDs."""

    def __init__(self, output_path: str | Path):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.completed_ids: set[str] = set()
        self._load_existing()

    def _load_existing(self):
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        self.completed_ids.add(rec["sample_id"])

    def is_done(self, sample_id: str) -> bool:
        return sample_id in self.completed_ids

    def save(self, record: ResultRecord):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        self.completed_ids.add(record.sample_id)

    def load_all(self) -> list[ResultRecord]:
        records = []
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        d = json.loads(line)
                        records.append(ResultRecord(**d))
        return records
