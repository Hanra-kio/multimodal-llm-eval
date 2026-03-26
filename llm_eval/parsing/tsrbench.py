import json
import re

from .base import BaseAnswerParser


class TSRBenchAnswerParser(BaseAnswerParser):
    """
    Answer parser following TSRBench official evaluation code.
    Reference: https://github.com/tianyi-lab/TSRBench/blob/main/evaluation/evaluate.py
    Expects JSON response with "answer" field, e.g. {"reasoning": "...", "answer": "A"}
    """

    def parse(self, text: str) -> str | None:
        if not text:
            return None

        # Try JSON parsing (official format)
        try:
            obj = json.loads(text.strip())
            if "answer" in obj:
                answer = obj["answer"]
                answer = re.sub(r"[,$()*\s]", "", answer).strip()
                if answer:
                    return answer[0].upper()
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: some models wrap JSON in markdown code blocks
        cleaned = text.replace("```json", "").replace("```", "").strip()
        try:
            obj = json.loads(cleaned)
            if "answer" in obj:
                answer = obj["answer"]
                answer = re.sub(r"[,$()*\s]", "", answer).strip()
                if answer:
                    return answer[0].upper()
        except (json.JSONDecodeError, TypeError):
            pass

        return None
