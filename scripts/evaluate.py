# Copyright (C) 2026 RAND Corporation. MIT License — see LICENSE file.
"""Task 2: Score pre-generated LLM responses using Inspect AI + GPT-4o Mini judge.

Reads responses from data/responses/<run_id>.parquet, classifies each response
as DIRECT or INDIRECT using the SCAN → CLASSIFY judge, and writes evaluation
records to data/evaluations/<run_id>.parquet. Supports resuming interrupted runs.

Usage:
    python scripts/evaluate.py --run-id my_run_01
    python scripts/evaluate.py --run-id my_run_01 --rpm 60
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import re
import yaml
from datetime import datetime, timezone

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
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
RESPONSES_DIR = REPO_ROOT / "output" / "responses"
EVALUATIONS_DIR = REPO_ROOT / "output" / "evaluations"
LOG_DIR = REPO_ROOT / "output" / "logs"


def _load_judge_rpm(config_dir: str) -> int:
    """Return the judge model's RPM from models.yaml."""
    try:
        with open(Path(config_dir) / "models.yaml") as f:
            config = yaml.safe_load(f)
        grader_id = "gpt-4o-mini"
        try:
            with open("judge_config.yaml") as f:
                grader_id = yaml.safe_load(f).get("grader_model", grader_id)
        except Exception:
            pass
        for model in config.get("models", []):
            if model.get("id") == grader_id:
                return model.get("rate_limit_rpm", 120)
    except Exception:
        pass
    return 120


def _extract_records(eval_logs, model_id: str, run_id: str) -> list:
    """Pull plain-dict records from an Inspect AI EvalLog list."""
    if not eval_logs or not eval_logs[0]:
        return []
    log = eval_logs[0]
    if not (hasattr(log, "samples") and log.samples):
        return []

    cfg_result_fields = []
    try:
        with open("judge_config.yaml") as f:
            cfg_result_fields = yaml.safe_load(f).get("result_fields", [])
    except Exception:
        pass

    records = []
    for sample in log.samples:
        score = sample.scores.get("judge_scorer") if sample.scores else None
        meta = (score.metadata if (score and hasattr(score, "metadata")) else None) or {}

        record = {
            "run_id": run_id,
            "model_id": model_id,
            "prompt_id": sample.metadata.get("prompt_id", str(sample.id)),
            "iteration": int(sample.metadata.get("iteration", 1)),
            "scorer": "llm_judge_gpt4o_mini",
            "score": float(score.value) if score and score.value is not None else None,
            "explanation": score.explanation if score else None,
            "is_direct": bool(meta.get("is_direct")) if "is_direct" in meta else None,
            "confidence": float(meta["confidence"]) if meta.get("confidence") is not None else None,
            "reasoning": meta.get("reasoning"),
            "metadata_json": str(meta),
            "error": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        for field_def in cfg_result_fields:
            fname = field_def["name"]
            val = meta.get(fname)
            if val is not None:
                record[fname] = bool(val) if field_def["type"] == "bool" else val
            else:
                record[fname] = None

        records.append(record)
    return records


def _inspect_progress(total: int, desc: str):
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        bar = tqdm(total=total, desc=desc, unit="sample", leave=True)
        last = [0]
        _sample_re = re.compile(r"Samples:\s+(\d+)/")
        _original = sys.stdout

        class _Writer:
            def write(self, text):
                m = _sample_re.search(text)
                if m:
                    n = int(m.group(1))
                    if n > last[0]:
                        bar.update(n - last[0])
                        last[0] = n
            def flush(self): pass
            def isatty(self): return False

        sys.stdout = _Writer()
        try:
            yield bar
        finally:
            sys.stdout = _original
            bar.update(total - last[0])
            bar.close()

    return _ctx()


def main():
    parser = argparse.ArgumentParser(description="Task 2: Score responses with LLM judge")
    parser.add_argument("--run-id", required=True, help="Run ID to evaluate")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Evaluate only these model IDs (default: all in responses file)")
    parser.add_argument("--config-dir", default=".",
                        help="Directory containing models.yaml (default: repo root)")
    parser.add_argument("--rpm", type=int, default=None,
                        help="Override the judge rate limit (RPM)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Judge sampling temperature (default: read from judge_config.yaml)")
    args = parser.parse_args()

    responses_path = RESPONSES_DIR / f"{args.run_id}.parquet"
    if not responses_path.exists():
        logger.error(f"Responses file not found: {responses_path}")
        logger.error("Did Task 1 (generate.py) complete successfully?")
        sys.exit(1)

    responses = pd.read_parquet(responses_path)
    logger.info(f"Loaded {len(responses)} responses from {responses_path}")

    if responses.empty:
        logger.warning("No responses found. Exiting.")
        return

    model_ids = responses["model_id"].unique().tolist()
    if args.models:
        unknown = [m for m in args.models if m not in model_ids]
        if unknown:
            logger.warning(f"Models not found in responses file: {unknown}")
        model_ids = [m for m in model_ids if m in args.models]
        if not model_ids:
            logger.error("No matching models found. Exiting.")
            sys.exit(1)
        responses = responses[responses["model_id"].isin(model_ids)]

    judge_rpm = args.rpm or _load_judge_rpm(args.config_dir)

    # Read judge temperature
    temperature = args.temperature
    if temperature is None:
        try:
            with open("judge_config.yaml") as f:
                temperature = yaml.safe_load(f).get("grader_temperature", 0.1)
        except Exception:
            temperature = 0.1

    EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVALUATIONS_DIR / f"{args.run_id}.parquet"
    log_dir = str(LOG_DIR / args.run_id)

    logger.info("=" * 60)
    logger.info(f"Run ID:      {args.run_id}")
    logger.info(f"Models:      {model_ids}")
    logger.info(f"Judge RPM:   {judge_rpm}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Output:      {output_path}")
    logger.info("=" * 60)

    # Resume: skip models already fully evaluated
    all_eval_records = []
    already_evaluated: set = set()
    if output_path.exists():
        existing_df = pd.read_parquet(output_path)
        response_counts = responses.groupby("model_id").size()
        eval_counts = existing_df.groupby("model_id").size()
        for mid in model_ids:
            expected = int(response_counts.get(mid, 0))
            actual = int(eval_counts.get(mid, 0))
            if actual >= expected > 0:
                logger.info(f"  {mid}: {actual}/{expected} already evaluated, skipping")
                already_evaluated.add(mid)
        all_eval_records = existing_df.to_dict("records")

    models_to_evaluate = [m for m in model_ids if m not in already_evaluated]
    if not models_to_evaluate:
        logger.info("All models already evaluated. Nothing to do.")
        return

    from src.evaluation.judge import Judge
    from src.evaluation.inspect_scorers import judge_scorer
    from src.evaluation.inspect_tasks import passthrough_eval
    from inspect_ai import eval as inspect_eval

    judge = Judge(
        config_dir=args.config_dir,
        rpm_override=judge_rpm,
        temperature=temperature,
    )
    task_scorer = judge_scorer(judge=judge)

    for mid in models_to_evaluate:
        model_responses = responses[responses["model_id"] == mid]
        records = model_responses.to_dict("records")
        task = passthrough_eval(records=records, model_id=mid, judge=judge,
                                scorer=task_scorer, run_id=args.run_id)

        with _inspect_progress(len(records), desc=mid):
            eval_logs = inspect_eval(task, log_dir=log_dir, display="plain")

        eval_records = _extract_records(eval_logs, mid, args.run_id)
        if not eval_records:
            logger.warning(f"No eval records returned for {mid}")
            continue

        all_eval_records.extend(eval_records)
        pd.DataFrame(all_eval_records).to_parquet(output_path, index=False)
        logger.info(f"  Saved {len(eval_records)} records for {mid} "
                    f"({len(all_eval_records)} total)")

    if all_eval_records:
        logger.info(f"Written: {len(all_eval_records)} evaluation records -> {output_path}")
    logger.info(f"Run ID for next step: --run-id {args.run_id}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
