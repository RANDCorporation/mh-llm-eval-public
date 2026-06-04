# Copyright (C) 2026 RAND Corporation. MIT License — see LICENSE file.
"""Sequential execution engine for LLM response generation.

Iterates over all model × prompt pairs, sleeping between calls to respect
per-model RPM limits. For faster runs, the inner loop can be parallelised
with a ThreadPoolExecutor — see README for details.

Usage:
    runner = Runner(config_dir=".")
    df = runner.execute(
        model_ids=["gpt-4o-mini"],
        run_id="my_run_01",
        iteration=1,
        prompts=my_prompts,
    )
    df.to_parquet("output/responses/my_run_01.parquet", index=False)
"""

import json
import logging
import time
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

_RETRYABLE_PATTERNS = (
    "429", "rate limit", "throttl", "resource_exhausted",
    "too many requests", "503", "502", "500",
    "timeout", "timed out", "connection reset", "service unavailable",
)


def _is_retryable(exc: Exception) -> bool:
    return any(p in str(exc).lower() for p in _RETRYABLE_PATTERNS)


def _retry_with_backoff(fn: Any, *args: Any, max_retries: int = 3, base_delay: float = 5.0, label: str = "", **kwargs: Any) -> Any:
    import random
    tag = f" [{label}]" if label else ""
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == max_retries or not _is_retryable(exc):
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(
                "Retryable error%s (attempt %d/%d): %s — retrying in %.1fs",
                tag, attempt + 1, max_retries, exc, delay,
            )
            time.sleep(delay)


RESPONSE_COLUMNS = [
    "run_id", "iteration", "model_id", "provider",
    "prompt_id", "prompt_text", "risk_category", "domain",
    "response_text", "input_tokens", "output_tokens",
    "latency_ms", "cost_usd", "error", "timestamp",
]


def _compute_cost(input_tokens: int, output_tokens: int, pricing: dict) -> float:
    return (
        input_tokens * pricing.get("input_per_1m", 0.0)
        + output_tokens * pricing.get("output_per_1m", 0.0)
    ) / 1_000_000


# ---------------------------------------------------------------------------
# One call function per provider
# ---------------------------------------------------------------------------

def _call_azure_openai(row: dict) -> dict:
    import os, time as _time
    from openai import AzureOpenAI

    cfg = json.loads(row["model_config_json"])
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        max_retries=0,
    )
    start = _time.time()
    try:
        kwargs = {}
        if cfg.get("api_compat", {}).get("use_max_completion_tokens"):
            kwargs["max_completion_tokens"] = cfg.get("max_tokens", 2048)
        else:
            kwargs["max_tokens"] = cfg.get("max_tokens", 2048)
        if not cfg.get("api_compat", {}).get("no_temperature"):
            kwargs["temperature"] = cfg.get("temperature", 0.7)
        resp = _retry_with_backoff(
            client.chat.completions.create,
            model=cfg["model_id"],
            messages=[{"role": "user", "content": row["prompt_text"]}],
            label=cfg["model_id"], **kwargs,
        )
        text = resp.choices[0].message.content or ""
        in_t = resp.usage.prompt_tokens if resp.usage else 0
        out_t = resp.usage.completion_tokens if resp.usage else 0
        return {**row, "response_text": text, "input_tokens": in_t, "output_tokens": out_t,
                "latency_ms": (_time.time() - start) * 1000,
                "cost_usd": _compute_cost(in_t, out_t, cfg.get("pricing", {})),
                "error": None, "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        return {**row, "response_text": "", "input_tokens": 0, "output_tokens": 0,
                "latency_ms": (_time.time() - start) * 1000, "cost_usd": 0.0,
                "error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}


def _call_aws_bedrock(row: dict) -> dict:
    import os, json as _json, time as _time
    import boto3  # type: ignore[import-untyped]

    cfg = _json.loads(row["model_config_json"])
    model_id = cfg["model_id"]
    is_claude = model_id.startswith(("anthropic.", "us.anthropic."))
    is_llama = model_id.startswith(("meta.", "us.meta."))

    client = boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    )
    start = _time.time()
    try:
        if is_claude:
            body = _json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": cfg.get("max_tokens", 2048),
                "messages": [{"role": "user", "content": row["prompt_text"]}],
                "temperature": cfg.get("temperature", 0.7),
            })
        elif is_llama:
            prompt = row["prompt_text"]
            body = _json.dumps({
                "prompt": (f"<|start_header_id|>user<|end_header_id|>\n\n"
                           f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
                "max_gen_len": cfg.get("max_tokens", 2048),
                "temperature": cfg.get("temperature", 0.7),
            })
        else:
            raise ValueError(f"Unsupported Bedrock model: {model_id}")

        resp = _retry_with_backoff(client.invoke_model, modelId=model_id, body=body, label=model_id)
        result = _json.loads(resp["body"].read())
        latency_ms = (_time.time() - start) * 1000

        if is_claude:
            text = result["content"][0]["text"]
            in_t, out_t = result["usage"]["input_tokens"], result["usage"]["output_tokens"]
        else:
            text = result["generation"]
            in_t = result.get("prompt_token_count", 0)
            out_t = result.get("generation_token_count", 0)

        return {**row, "response_text": text, "input_tokens": in_t, "output_tokens": out_t,
                "latency_ms": latency_ms,
                "cost_usd": _compute_cost(in_t, out_t, cfg.get("pricing", {})),
                "error": None, "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        return {**row, "response_text": "", "input_tokens": 0, "output_tokens": 0,
                "latency_ms": (_time.time() - start) * 1000, "cost_usd": 0.0,
                "error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}


def _call_google_gemini(row: dict) -> dict:
    import os, time as _time
    from google import genai  # type: ignore[import-untyped]
    from google.genai import types  # type: ignore[import-untyped]

    cfg = json.loads(row["model_config_json"])
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    start = _time.time()
    try:
        resp = _retry_with_backoff(
            client.models.generate_content,
            model=cfg["model_id"],
            contents=[row["prompt_text"]],
            config=types.GenerateContentConfig(
                max_output_tokens=cfg.get("max_tokens", 2048),
                temperature=cfg.get("temperature", 0.7),
            ),
            label=cfg["model_id"],
        )
        text = resp.text or ""
        meta = resp.usage_metadata
        in_t = (meta.prompt_token_count or 0) if meta else 0
        out_t = (meta.candidates_token_count or 0) if meta else 0
        return {**row, "response_text": text, "input_tokens": in_t, "output_tokens": out_t,
                "latency_ms": (_time.time() - start) * 1000,
                "cost_usd": _compute_cost(in_t, out_t, cfg.get("pricing", {})),
                "error": None, "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        return {**row, "response_text": "", "input_tokens": 0, "output_tokens": 0,
                "latency_ms": (_time.time() - start) * 1000, "cost_usd": 0.0,
                "error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}


_PROVIDER_CALL = {
    "azure_openai":  _call_azure_openai,
    "aws_bedrock":   _call_aws_bedrock,
    "google_gemini": _call_google_gemini,
}


class Runner:
    """Sequential LLM API runner.

    Iterates over model × prompt pairs in order, sleeping between calls
    to stay within each model's RPM limit.

    Example:
        runner = Runner(config_dir=".")
        df = runner.execute(model_ids=["gpt-4o-mini"], run_id="test_01",
                            iteration=1, prompts=prompts)
    """

    def __init__(self, config_dir: str = "."):
        self.config_dir = Path(config_dir)
        with open(self.config_dir / "models.yaml") as f:
            self._models: List[dict] = yaml.safe_load(f).get("models", [])

    def execute(
        self,
        model_ids: List[str],
        run_id: str,
        iteration: int = 1,
        prompts: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
        prompt_ids_per_model: Optional[dict] = None,
    ) -> pd.DataFrame:
        models = [m for m in self._models if m["id"] in model_ids]
        if not models:
            raise ValueError(f"No models found for IDs: {model_ids}. Check models.yaml.")
        if not prompts:
            raise ValueError("prompts must be provided.")

        results = []
        total = sum(
            len([p for p in prompts if p.get("id") in prompt_ids_per_model.get(m["id"], set())])
            if prompt_ids_per_model else len(prompts)
            for m in models
        )

        with tqdm(total=total, unit="prompt", leave=True) as pbar:
            for model in models:
                call_fn = _PROVIDER_CALL.get(model["provider"])
                if call_fn is None:
                    raise ValueError(f"Unsupported provider '{model['provider']}' for model '{model['id']}'")

                rpm = model.get("rate_limit_rpm", 60)
                interval = 60.0 / rpm if rpm > 0 else 0.0

                model_config_json = json.dumps({
                    "model_id":    model.get("model_id", model["id"]),
                    "pricing":     model.get("pricing", {}),
                    "api_compat":  model.get("api_compat", {}),
                    "temperature": temperature if temperature is not None else model.get("temperature", 0.7),
                    "max_tokens":  model.get("max_tokens", 2048),
                })

                model_prompts = prompts
                if prompt_ids_per_model is not None:
                    todo = prompt_ids_per_model.get(model["id"], set())
                    model_prompts = [p for p in prompts if p.get("id") in todo]

                for i, prompt in enumerate(model_prompts):
                    if i > 0:
                        time.sleep(interval)

                    row = {
                        "run_id":            run_id,
                        "iteration":         iteration,
                        "model_id":          model["id"],
                        "provider":          model["provider"],
                        "model_config_json": model_config_json,
                        "prompt_id":         prompt.get("id", ""),
                        "prompt_text":       prompt.get("text", ""),
                        "risk_category":     prompt.get("risk_category", "medium"),
                        "domain":            prompt.get("domain", "mental_health"),
                    }
                    result = call_fn(row)
                    results.append({k: result.get(k) for k in RESPONSE_COLUMNS})
                    pbar.update(1)

        return pd.DataFrame(results, columns=RESPONSE_COLUMNS)
