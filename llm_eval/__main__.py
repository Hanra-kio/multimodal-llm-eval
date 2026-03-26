import argparse
import asyncio
import logging

from llm_eval.config import load_config
from llm_eval.runner import Runner


def main():
    parser = argparse.ArgumentParser(
        description="LLM Time Series Reasoning Benchmark Evaluation"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)
    runner = Runner(config)
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
