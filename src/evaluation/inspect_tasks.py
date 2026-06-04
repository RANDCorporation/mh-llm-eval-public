# Copyright (C) 2026 RAND Corporation. MIT License — see LICENSE file.
"""Inspect AI task definition for scoring pre-generated responses."""

from typing import List, Optional
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.solver import solver, TaskState, Generate

from src.evaluation.judge import Judge
from src.evaluation.inspect_scorers import judge_scorer


@solver
def passthrough_metadata(response_key: str = "model_response"):
    """Copy a stored response from sample metadata into the task state output.

    Used by passthrough_eval to score pre-generated responses without
    re-invoking the model under test.
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        stored = (state.metadata or {}).get(response_key, "")
        state.output.completion = stored
        return state
    return solve


@task
def passthrough_eval(
    records: List[dict],
    model_id: str,
    run_id: Optional[str] = None,
    judge: Optional[Judge] = None,
    scorer=None,
):
    """
    Score pre-generated LLM responses without re-invoking the model.

    Each record's response_text is copied into the task state via
    passthrough_metadata, then scored with the LLM judge.

    Args:
        records: Rows from the responses parquet. Expected keys:
                 prompt_text, response_text, prompt_id, risk_category, iteration
        model_id: Model ID these responses belong to (used for task naming)
        run_id: Experiment run identifier
        judge: Judge instance (created by evaluate.py)
        scorer: Optional scorer override (defaults to judge_scorer(judge))
    """
    samples = []
    for i, record in enumerate(records):
        prompt_id = record.get("prompt_id", str(i))
        iteration = record.get("iteration", 1)
        samples.append(Sample(
            input=record.get("prompt_text", ""),
            target="",
            id=f"{prompt_id}_iter{iteration}",
            metadata={
                "model_response": record.get("response_text", ""),
                "model_id": model_id,
                "prompt_id": prompt_id,
                "risk_category": record.get("risk_category", "medium"),
                "domain": record.get("domain", "mental_health"),
                "run_id": run_id or record.get("run_id", ""),
                "iteration": iteration,
            },
        ))

    _scorer = scorer if scorer is not None else judge_scorer(judge=judge)
    return Task(
        dataset=MemoryDataset(samples),
        solver=[passthrough_metadata("model_response")],
        scorer=_scorer,
        name=model_id,
    )
