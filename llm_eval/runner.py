import asyncio
import json
import logging
from pathlib import Path

from llm_eval.config import ExperimentConfig
from llm_eval.datasets.base import Sample
from llm_eval.evaluation.metrics import compute_metrics
from llm_eval.evaluation.results import ResultRecord, ResultStore
from llm_eval.datasets import REGISTRY as DATASET_REGISTRY
from llm_eval.inference.client import LLMClient
from llm_eval.parsing import REGISTRY as PARSER_REGISTRY
from llm_eval.prompts import REGISTRY as PROMPT_REGISTRY

logger = logging.getLogger(__name__)


def _get_class(registry: dict, kind: str, name: str):
    if name not in registry:
        raise ValueError(f"Unknown {kind} '{name}'. Available: {list(registry.keys())}")
    return registry[name]


class Runner:
    def __init__(self, config: ExperimentConfig):
        self.config = config

        dataset_cls = _get_class(DATASET_REGISTRY, "dataset", config.dataset.name)
        self.dataset = dataset_cls(config.dataset.params)

        builder_name = config.prompt.get("builder", config.dataset.name)
        builder_cls = _get_class(PROMPT_REGISTRY, "prompt builder", builder_name)
        self.prompt_builder = builder_cls(config.prompt)

        parser_name = config.prompt.get("parser", config.dataset.name)
        parser_cls = _get_class(PARSER_REGISTRY, "parser", parser_name)
        self.parser = parser_cls()

        self.client = LLMClient(config.model)
        self.results = ResultStore(config.output.results_path)

    async def run(self):
        samples = list(self.dataset.load(self.config.dataset.tasks))
        logger.info(f"Loaded {len(samples)} samples from {self.dataset.name()}")

        # Resume: skip completed samples
        pending = [s for s in samples if not self.results.is_done(s.id)]
        skipped = len(samples) - len(pending)
        if skipped:
            logger.info(f"Resuming: skipping {skipped} completed, {len(pending)} pending")
        else:
            logger.info(f"Processing {len(pending)} samples")

        if not pending:
            logger.info("All samples already completed.")
        else:
            # Run all pending samples concurrently (bounded by LLMClient semaphore)
            tasks = [self._process_sample(s) for s in pending]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log any unhandled exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Unhandled error for {pending[i].id}: {result}")

        # Compute and save metrics
        all_records = self.results.load_all()
        if all_records:
            metrics = compute_metrics(all_records)
            self._save_metrics(metrics)
            self._print_metrics(metrics)
        else:
            logger.warning("No results to compute metrics from.")

    async def _process_sample(self, sample: Sample):
        try:
            messages = self.prompt_builder.build(sample)
            response = await self.client.complete(messages)
            parsed = self.parser.parse(response.text)
            correct = parsed == sample.answer

            record = ResultRecord(
                sample_id=sample.id,
                model=response.model,
                gold_answer=sample.answer,
                predicted_answer=parsed,
                raw_output=response.text,
                correct=correct,
                latency_s=response.latency_s,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                metadata=sample.metadata,
            )
            self.results.save(record)
            logger.debug(
                f"{sample.id}: predicted={parsed} gold={sample.answer} correct={correct}"
            )
        except Exception as e:
            logger.error(f"Failed to process {sample.id}: {e}")
            raise

    def _save_metrics(self, metrics: dict):
        metrics_path = Path(self.config.output.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"Metrics saved to {metrics_path}")

    @staticmethod
    def _print_metrics(metrics: dict):
        print(f"\n{'='*60}")
        # Print top-level metrics (exclude per_task)
        for key, val in metrics.items():
            if key == "per_task":
                continue
            if isinstance(val, float):
                print(f"{key}: {val:.4f}")
            else:
                print(f"{key}: {val}")
        print(f"{'='*60}")
        # Print per-task breakdown
        if "per_task" in metrics:
            for task, d in metrics["per_task"].items():
                parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in d.items() if k != "subtasks"]
                print(f"  {task:<35} {', '.join(parts)}")
                if "subtasks" in d:
                    for sub, sd in d["subtasks"].items():
                        sub_parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                     for k, v in sd.items()]
                        print(f"    {sub:<33} {', '.join(sub_parts)}")
        print(f"{'='*60}\n")
