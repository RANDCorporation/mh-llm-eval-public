# Copyright (C) 2026 RAND Corporation. MIT License — see LICENSE file.
"""Task 1: Generate LLM responses for the mental health safety evaluation.

Reads prompts from prompts.yaml, submits each prompt to each model for each
iteration, and writes results to a parquet file. Supports resuming interrupted runs.

Usage:
    python scripts/generate.py --models gpt-4o-mini --run-id my_run_01
    python scripts/generate.py --models gpt-4o-mini claude-3-haiku --run-id my_run_01 --iterations 100
    python scripts/generate.py --models gpt-4o-mini --run-id my_run_01 --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import yaml
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(override=True)


class _TqdmHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.root.setLevel(logging.INFO)
logging.root.addHandler(_handler)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
PROMPTS_FILE = REPO_ROOT / "prompts.yaml"
OUTPUT_DIR = REPO_ROOT / "output" / "responses"


def load_prompts(path: Path) -> list:
    with open(path) as f:
        config = yaml.safe_load(f)
    return config.get("prompts", [])


def main():
    parser = argparse.ArgumentParser(description="Task 1: Generate LLM responses")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model IDs from models.yaml")
    parser.add_argument("--run-id", default=None,
                        help="Run identifier (auto-generated if not set)")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of iterations per prompt (default: 1)")
    parser.add_argument("--prompts", nargs="+", default=None,
                        help="Subset of prompt IDs to run (default: all)")
    parser.add_argument("--config-dir", default=".",
                        help="Directory containing models.yaml (default: repo root)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (default: read from judge_config.yaml or 0.7)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print execution plan without calling any APIs")
    args = parser.parse_args()

    if not args.run_id:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Read generate_temperature from judge_config.yaml if not overridden on CLI
    temperature = args.temperature
    if temperature is None:
        try:
            with open("judge_config.yaml") as f:
                temperature = yaml.safe_load(f).get("generate_temperature")
        except Exception:
            pass
    if temperature is None:
        temperature = 0.7

    prompts = load_prompts(PROMPTS_FILE)
    if args.prompts:
        prompts = [p for p in prompts if p.get("id") in args.prompts]
    logger.info(f"Loaded {len(prompts)} prompts from {PROMPTS_FILE}")

    with open(Path(args.config_dir) / "models.yaml") as f:
        all_models = yaml.safe_load(f).get("models", [])
    models = [m for m in all_models if m["id"] in args.models]
    missing = set(args.models) - {m["id"] for m in models}
    if missing:
        logger.warning(f"Models not found in models.yaml: {missing}")
    if not models:
        logger.error("No valid models found. Exiting.")
        sys.exit(1)

    iter_range = range(1, args.iterations + 1)
    total_calls = len(models) * len(prompts) * len(iter_range)

    logger.info("=" * 60)
    logger.info(f"Run ID:      {args.run_id}")
    logger.info(f"Models:      {[m['id'] for m in models]}")
    logger.info(f"Prompts:     {len(prompts)}")
    logger.info(f"Iterations:  {args.iterations}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Total calls: {total_calls}")
    logger.info(f"Output:      {OUTPUT_DIR / args.run_id}.parquet")

    if args.dry_run:
        logger.info("[DRY RUN] No API calls will be made.")
        for m in models:
            rpm = m.get("rate_limit_rpm", 60)
            calls = len(prompts) * args.iterations
            mins = round(calls / rpm, 1)
            logger.info(
                f"  {m['id']} ({m.get('provider', '?')}) — "
                f"{calls} calls @ {rpm} RPM ≈ {mins} min"
            )
        return
    logger.info("=" * 60)

    from src.engine.api_runner import Runner

    runner = Runner(config_dir=args.config_dir)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{args.run_id}.parquet"

    existing_dfs = []
    completed_prompts: dict = {}
    if output_path.exists():
        existing_df = pd.read_parquet(output_path)
        clean_df = existing_df[
            existing_df["response_text"].notna()
            & (existing_df["response_text"].str.strip() != "")
        ]
        dropped = len(existing_df) - len(clean_df)
        if dropped:
            logger.info(f"Dropped {dropped} empty-response rows (will rerun)")
        existing_dfs.append(clean_df)
        for (mid, it), grp in clean_df.groupby(["model_id", "iteration"]):
            completed_prompts.setdefault(mid, {})[it] = set(grp["prompt_id"])
        for mid, iters in completed_prompts.items():
            total_done = sum(len(ps) for ps in iters.values())
            logger.info(f"  {mid}: {total_done} completed — resuming")

    all_prompt_ids = [p.get("id") for p in prompts]

    for i in iter_range:
        prompt_ids_per_model: dict = {}
        for mid in args.models:
            done = completed_prompts.get(mid, {}).get(i, set())
            todo = [pid for pid in all_prompt_ids if pid not in done]
            prompt_ids_per_model[mid] = todo

        models_todo = [m for m in args.models if len(prompt_ids_per_model[m]) > 0]
        if not models_todo:
            logger.info(f"Iteration {i}/{args.iterations}: all models complete, skipping")
            continue

        iter_df = runner.execute(
            model_ids=models_todo,
            prompts=prompts,
            run_id=args.run_id,
            iteration=i,
            temperature=temperature,
            prompt_ids_per_model={
                mid: set(prompt_ids_per_model[mid]) for mid in models_todo
            },
        )

        existing_dfs.append(iter_df)
        pd.concat(existing_dfs, ignore_index=True).to_parquet(output_path, index=False)

        errors = iter_df["error"].notna().sum()
        logger.info(
            f"Iteration {i}/{args.iterations}: {len(iter_df)} responses, {errors} errors"
        )

    logger.info(f"Done. Run ID for next step: --run-id {args.run_id}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
