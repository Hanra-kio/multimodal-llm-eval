from collections import defaultdict

from .results import ResultRecord


def compute_metrics(records: list[ResultRecord]) -> dict:
    """Compute accuracy overall, per-task, and per-subtask (via metadata 'category')."""
    overall_correct = 0
    overall_total = 0
    by_task: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    by_subtask: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"correct": 0, "total": 0})
    )

    for r in records:
        overall_total += 1
        if r.correct:
            overall_correct += 1

        task = r.metadata.get("task_name", "unknown")
        by_task[task]["total"] += 1
        if r.correct:
            by_task[task]["correct"] += 1

        category = r.metadata.get("category")
        if category:
            by_subtask[task][category]["total"] += 1
            if r.correct:
                by_subtask[task][category]["correct"] += 1

    per_task = {}
    for task, d in sorted(by_task.items()):
        task_result = {
            "accuracy": d["correct"] / d["total"] if d["total"] else 0,
            "correct": d["correct"],
            "total": d["total"],
        }
        if task in by_subtask:
            task_result["subtasks"] = {
                sub: {
                    "accuracy": sd["correct"] / sd["total"] if sd["total"] else 0,
                    "correct": sd["correct"],
                    "total": sd["total"],
                }
                for sub, sd in sorted(by_subtask[task].items())
            }
        per_task[task] = task_result

    return {
        "overall_accuracy": overall_correct / overall_total if overall_total else 0,
        "total_samples": overall_total,
        "correct_samples": overall_correct,
        "per_task": per_task,
    }
