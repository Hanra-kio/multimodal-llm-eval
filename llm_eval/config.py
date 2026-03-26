from dataclasses import dataclass, field
import yaml

@dataclass
class DatasetConfig:
    name: str
    tasks: list[str] | None = None
    params: dict = field(default_factory=dict)

@dataclass
class OutputConfig:
    results_path: str
    metrics_path: str

@dataclass
class ExperimentConfig:
    model: dict
    dataset: DatasetConfig
    prompt: dict
    output: OutputConfig

def load_config(path: str) -> ExperimentConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return ExperimentConfig(
        model=raw["model"],
        dataset=DatasetConfig(
            name=raw["dataset"]["name"],
            tasks=raw["dataset"].get("tasks"),
            params=raw["dataset"].get("params", {}),
        ),
        prompt=raw.get("prompt", {}),
        output=OutputConfig(**raw["output"]),
    )
