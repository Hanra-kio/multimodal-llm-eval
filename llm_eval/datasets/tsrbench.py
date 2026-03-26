import json
from pathlib import Path
from typing import Iterator

from .base import BaseDataset, Sample

TASK_REGISTRY: dict[str, str] = {
    "perception": "perception/perception.jsonl",
    "qualitative_decision": "decision/qualitative_decision.jsonl",
    "quantitative_decision": "decision/quantitative_decision.jsonl",
    "event_prediction": "prediction/event_prediction.jsonl",
    "time_series_forecasting": "prediction/time_series_forecasting.jsonl",
    "abductive_reasoning": "reasoning/abductive_reasoning.jsonl",
    "causal_reasoning": "reasoning/causal_reasoning.jsonl",
    "deductive_reasoning": "reasoning/deductive_reasoning.jsonl",
    "etiological_reasoning": "reasoning/etiological_reasoning.jsonl",
    "inductive_reasoning": "reasoning/inductive_reasoning.jsonl",
    "numerical_reasoning": "reasoning/numerical_reasoning.jsonl",
    "temporal_relation_reasoning": "reasoning/temporal_relation_reasoning.jsonl",
}


class TSRBenchDataset(BaseDataset):
    def name(self) -> str:
        return "tsrbench"

    def load(self, tasks: list[str] | None = None) -> Iterator[Sample]:
        root = Path(self.config["data_dir"])
        selected = tasks or list(TASK_REGISTRY.keys())
        for task in selected:
            if task not in TASK_REGISTRY:
                raise ValueError(
                    f"Unknown task '{task}'. Available: {list(TASK_REGISTRY.keys())}"
                )
            path = root / TASK_REGISTRY[task]
            with open(path, encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    raw = json.loads(line)
                    yield self._parse_sample(task, idx, raw)

    def _parse_sample(self, task: str, idx: int, raw: dict) -> Sample:
        sample_id = f"{task}/{idx}"

        # unique structure
        if task == "abductive_reasoning":
            return self._parse_abductive(sample_id, raw)

        question = raw["question"]
        answer = raw["answer"]
        timeseries = raw["timeseries"]
        choices = _normalize_choices(raw.get("choices"))

        metadata: dict = {"task_name": task}
        for key in ("domain", "category", "task", "type", "name_of_series"):
            if key in raw:
                metadata[key] = raw[key]

        return Sample(
            id=sample_id,
            question=question,
            answer=answer,
            timeseries=timeseries,
            choices=choices,
            metadata=metadata,
        )

    def _parse_abductive(self, sample_id: str, raw: dict) -> Sample:
        mcq = raw["multiple_choice_question"]

        nts = raw["numerical_time_series"]
        ts_channels: list[list[float]] = []
        series_names: list[str] = []
        for series_name, ts_data in nts.items():
            series_names.append(series_name)
            ts_channels.append(ts_data["history"] + ts_data["future"])

        choices_list = mcq.get("choices", [])
        choices_dict = {chr(65 + i): str(c) for i, c in enumerate(choices_list)}

        return Sample(
            id=sample_id,
            question=mcq["question"],
            answer=mcq["answer"],
            timeseries=ts_channels,
            choices=choices_dict,
            metadata={
                "task_name": "abductive_reasoning",
                "name_of_series": series_names,
                "context": raw.get("context"),
                "game_info": raw.get("game_info"),
            },
        )


def _normalize_choices(choices_raw) -> dict[str, str] | None:
    """Convert choices to {"A": "text", ...} regardless of input format."""
    if choices_raw is None:
        return None
    if isinstance(choices_raw, dict):
        return {k: str(v) for k, v in choices_raw.items()}
    if isinstance(choices_raw, list):
        return {chr(65 + i): str(v) for i, v in enumerate(choices_raw)}
    return None
