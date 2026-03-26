import asyncio
import logging
import time
from dataclasses import dataclass

import litellm

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


@dataclass
class LLMResponse:
    text: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    model: str


class LLMClient:
    def __init__(self, config: dict):
        self.model = config["model"]
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 256)
        self.semaphore = asyncio.Semaphore(config.get("max_concurrent", 10))
        self.timeout = config.get("timeout", 120)
        self.retries = config.get("retries", 3)
        self.extra_kwargs = config.get("litellm_kwargs", {})

    async def complete(self, messages: list[dict]) -> LLMResponse:
        async with self.semaphore:
            start = time.monotonic()
            last_exc: Exception | None = None

            for attempt in range(self.retries):
                try:
                    resp = await litellm.acompletion(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        timeout=self.timeout,
                        **self.extra_kwargs,
                    )
                    elapsed = time.monotonic() - start
                    usage = resp.usage
                    return LLMResponse(
                        text=resp.choices[0].message.content or "",
                        latency_s=elapsed,
                        prompt_tokens=usage.prompt_tokens if usage else 0,
                        completion_tokens=usage.completion_tokens if usage else 0,
                        model=self.model,
                    )
                except Exception as e:
                    last_exc = e
                    wait = 2**attempt
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.retries} failed: {e}. "
                        f"Retrying in {wait}s..."
                    )
                    await asyncio.sleep(wait)

            raise last_exc  # type: ignore[misc]
