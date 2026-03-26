from llm_eval.datasets.base import Sample
from .base import BasePromptBuilder


class TSRBenchPromptBuilder(BasePromptBuilder):
    """Prompt builder following TSRBench official code.

    Reference: https://github.com/tianyi-lab/TSRBench/blob/main/inference/text_gpt/text_inference.py
    """

    def build(self, sample: Sample) -> list[dict[str, str]]:
        task = sample.metadata.get("task_name", "")

        if task == "abductive_reasoning":
            prompt = self._build_abductive(sample)
        else:
            prompt = self._build_standard(sample)

        return [
            {"role": "user", "content": prompt},
        ]

    def _build_standard(self, sample: Sample) -> str:
        question = sample.question
        names = sample.metadata.get("name_of_series")

        # Append timeseries prompt: " Here are the time series '{name}': <ts><ts/>. "
        ts_prompt = " Here are the time series"
        for i in range(len(sample.timeseries)):
            label = names[i] if names and i < len(names) else f"Series {i + 1}"
            ts_prompt += f" '{label}': <ts><ts/>. "
        question += ts_prompt

        # Replace each <ts><ts/> with actual values (one per series)
        parts = question.split("<ts><ts/>")
        prompt = parts[0]
        for ts_idx in range(len(sample.timeseries)):
            channel = sample.timeseries[ts_idx]
            if channel and not isinstance(channel[0], float):
                cur_ts = ",".join(f"{v}" for v in channel[::6])
            else:
                cur_ts = ",".join(f"{v:.2f}" for v in channel[::4])
            prompt += cur_ts + parts[ts_idx + 1]

        # Append choices
        choice_text = "\n"
        choices = sample.choices
        if choices is not None:
            for key in sorted(choices.keys()):
                choice_text += f"{key}. {choices[key]}\n"

        prompt += " Select from the options below:" + choice_text
        prompt += "Output your reasoning and answer in JSON format."

        return prompt

    def _build_abductive(self, sample: Sample) -> str:
        """Build prompt for abductive reasoning following official code.

        Reference: text_inference_abductive.py
        Uses event history/future and win probability table format.
        """
        context = sample.metadata.get("context", {})
        history_events = context.get("history_events", [])
        history_times = context.get("history_times", [])
        future_events = context.get("future_events", [])
        future_times = context.get("future_times", [])

        # Past events
        past_text = "Past Events (History):\n"
        for t, event in zip(history_times, history_events):
            past_text += f"- {t}: {event}\n"

        # Future events
        future_text = "Future Events:\n"
        for t, event in zip(future_times, future_events):
            future_text += f"- {t}: {event}\n"

        # Win probability time series table
        names = sample.metadata.get("name_of_series", [])
        # Find wp channels (wp_Team A, wp_Team B)
        wp_a = wp_b = None
        all_times = history_times + future_times
        for i, name in enumerate(names):
            if "wp_Team A" in name:
                wp_a = sample.timeseries[i]
            elif "wp_Team B" in name:
                wp_b = sample.timeseries[i]

        ts_text = "\nTime Series Data (Win Probability):\n"
        ts_text += "Time | Team A Win Prob | Team B Win Prob\n"
        ts_text += "-" * 60 + "\n"
        if wp_a and wp_b:
            for i, t in enumerate(all_times):
                if i < len(wp_a) and i < len(wp_b):
                    ts_text += f"{t} | {wp_a[i]:.3f} | {wp_b[i]:.3f}\n"

        # Choices
        choice_text = "\nOptions:\n"
        if sample.choices:
            for key in sorted(sample.choices.keys()):
                choice_text += f"{key}. {sample.choices[key]}\n"

        prompt = (
            "Given a sequence of past events, future events, and corresponding time series data "
            "from a game, determine the most plausible event that occurred in between to link them.\n\n"
            "--- CONTEXT ---\n"
            f"{past_text.strip()}\n"
            "\n... [A CRITICAL EVENT HAPPENED HERE] ...\n\n"
            f"{future_text.strip()}\n\n"
            f"{ts_text.strip()}\n\n"
            "--- TASK ---\n"
            f"{sample.question}\n"
            f"{choice_text.strip()}\n\n"
            'Based on the context, events, and time series data, what is the most likely event '
            'that happened? Please respond with a JSON object containing your answer: {"answer": "X"}.'
        )

        return prompt
