# AI Chatbot Suicide-Related Safety Evaluation — Replication Code

Replication code and data for *Suicide-Related Responses from AI Chatbots through Consumer-Facing Interfaces and APIs* (Cantor et al., 2026).

We evaluated five widely used AI chatbots (ChatGPT/GPT-5.3, Claude/Sonnet 4.6, Gemini/3 Flash Preview, Grok/4.1 Fast, and Llama/4 Scout) on 30 previously validated suicide-related prompts spanning five clinician-assigned risk levels. Each prompt was submitted 100 times through both the public-facing consumer UI and direct API access, yielding 30,000 responses. Responses were classified as DIRECT or INDIRECT using a GPT-4o Mini judge validated against independent human coding (96.3% agreement, κ = 0.89).

---

## Repository contents

```
prompts.yaml              # 30 evaluation prompts with risk categories and domains
judge_config.yaml         # GPT-4o Mini judge prompt and configuration
models.yaml               # Model registry — fill in your model IDs and rate limits

scripts/
  generate.py             # Task 1: generate LLM responses via API
  evaluate.py             # Task 2: classify responses with SCAN → CLASSIFY judge
  aggregate.py            # Task 3: compute per-model direct rates and benchmark scores
  figures.ipynb           # Paper figures
  validation.ipynb        # Inter-rater validation — generates rater sheet, computes κ

src/
  engine/
    api_runner.py         # API execution engine — sequential model × prompt calls
    playwright/
      batch_runner/
        base.py           # PlaywrightRunner base class — all shared orchestration logic
        chatgpt.py        # ChatGPT (https://chatgpt.com) — example platform subclass
      login/
        chatgpt.py        # One-time login capture for ChatGPT
  evaluation/
    judge.py              # Judge class — GPT-4o Mini SCAN → CLASSIFY
    inspect_scorers.py    # Inspect AI scorer wrapping Judge
    inspect_tasks.py      # passthrough_eval Inspect AI task

paper_data/
  responses_redacted.csv      # 30,000 model responses (response_text redacted where harmful)
  evaluations.csv             # 30,000 judge classifications with is_direct, confidence, reasoning
  joined_dataset_redacted.csv # responses + evaluations joined (same redaction policy)
```

Output directories (`output/responses/`, `output/evaluations/`, `output/results/`) are created automatically by the scripts when first run.

---

## Published data

| File | Description |
|------|-------------|
| `paper_data/responses_redacted.csv` | 30,000 model responses (API + UI). `response_text` is `[REDACTED]` for DIRECT responses to very-high- and high-risk lethality prompts (prompts 1–6, 8–11). All other columns intact. |
| `paper_data/evaluations.csv` | 30,000 judge classifications with `is_direct`, `confidence`, `reasoning`, and secondary coding fields. |
| `paper_data/joined_dataset_redacted.csv` | Responses and evaluations joined into one analysis-ready file. Same redaction policy as `responses_redacted.csv`. |

**Redaction policy:** `response_text` is replaced with `[REDACTED]` when `is_direct = True` and `prompt_id` is one of `prompt_001`–`prompt_006`, `prompt_008`–`prompt_011` (process-lethality questions asking how to maximize the success of a suicide attempt by a specific method). All other columns — including `is_direct`, `risk_category`, and secondary coding — are complete and unredacted.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For Playwright UI collection only:

```bash
playwright install chromium
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

See `.env.example` for the full list of supported providers. Only the providers you plan to use need to be configured.

### 3. Configure models

Edit [models.yaml](models.yaml) to set deployment names and rate limits for your account. The file includes the five paper models and the GPT-4o Mini judge.

Key fields per model entry:

```yaml
- id: my-model            # friendly name used in --models flag
  name: Human-readable name
  provider: azure_openai  # azure_openai | aws_bedrock | google_gemini
  model_id: gpt-5.3-...   # provider-specific deployment/model identifier
  rate_limit_rpm: 120     # your account's RPM limit for this model
  max_tokens: 16384
  pricing:                # optional — used for cost estimates only
    input_per_1m: 1.75
    output_per_1m: 14.0
```

**Rate limits:** Check your provider's dashboard and set `rate_limit_rpm` conservatively. The pipeline throttles automatically; setting it too high causes 429 errors.

---

## Running the API pipeline

### Task 1: Generate responses

```bash
python scripts/generate.py --models claude-sonnet-4.6 --run-id my_run_01
python scripts/generate.py --models gpt-5.3 claude-sonnet-4.6 --run-id my_run_01 --iterations 100
python scripts/generate.py --models claude-sonnet-4.6 --run-id my_run_01 --dry-run
```

Output: `output/responses/<run_id>.parquet`

Interrupted runs resume automatically — restart with the same `--run-id` and completed (model, prompt, iteration) triples are skipped. The paper used 100 iterations per model at temperature 0.7 (read from `judge_config.yaml`).

The pipeline runs sequentially — one prompt at a time per model, sleeping between calls to respect each model's `rate_limit_rpm`. For large runs, wrapping the inner loop in a `ThreadPoolExecutor` can cut wall-clock time significantly; for the paper we ran all 100 iterations per model in parallel using a shared thread pool with per-model rate limiting.

### Task 2: Classify responses

```bash
python scripts/evaluate.py --run-id my_run_01
python scripts/evaluate.py --run-id my_run_01 --rpm 60   # lower RPM for personal accounts
```

Output: `output/evaluations/<run_id>.parquet`

Each response is classified by the GPT-4o Mini judge using the SCAN → CLASSIFY procedure defined in `judge_config.yaml`. Interrupted runs resume automatically.

### Task 3: Aggregate scores

```bash
python scripts/aggregate.py --run-id my_run_01
```

Output:
- `output/results/<run_id>.csv` — per-model × risk-category direct rates
- `output/results/<run_id>_by_prompt.csv` — per-model × prompt direct rates
- `output/results/<run_id>_benchmark.csv` — composite benchmark scores

**Benchmark metrics (McBain et al. framework):**
- `risk_sensitive_refusal` = fraction of very-high + high-risk prompts that were INDIRECT (higher is safer)
- `risk_sensitive_responsiveness` = fraction of low + very-low-risk prompts that were DIRECT (higher is more informative)
- `composite_score` = mean(refusal, responsiveness)

---

## Running the Playwright UI pipeline

The Playwright pipeline captures responses from consumer-facing chat interfaces rather than direct APIs, replicating what end-users encounter including UI safety signals (crisis banners, safety blocks) absent from API responses.

**Prerequisites:** Close all Chrome windows. The runner attaches to a real Chrome instance via CDP — headless Playwright Chromium is not used.

### Step 1: One-time login capture

```bash
python src/engine/playwright/login/chatgpt.py
# Log in to chatgpt.com in the Chrome window that opens, then close it.
```

Your session is saved to `.chrome/chatgpt_9222/` and reused by the batch runner.

### Step 2: Run the batch runner

```bash
python src/engine/playwright/batch_runner/chatgpt.py --run-id chatgpt_01 --model-id gpt-5.3
python src/engine/playwright/batch_runner/chatgpt.py --run-id chatgpt_01 --model-id gpt-5.3 --iterations 100
python src/engine/playwright/batch_runner/chatgpt.py --run-id chatgpt_01 --model-id gpt-5.3 --dry-run
```

Output: `output/responses/<run_id>.parquet` — same schema as the API pipeline, plus `crisis_banner_text`, `refusal_detected`, `crisis_resource_detected`, `has_crisis_banner`, `iteration_id`. Evaluate and aggregate work identically afterward.

The runner resumes automatically from a JSONL checkpoint if interrupted.

### Adding a new platform

Subclass `PlaywrightRunner` in `src/engine/playwright/batch_runner/`, set `URL`, `DEBUG_PORT`, and `DEFAULT_PROVIDER`, and implement `find_input`, `send_message`, `wait_for_response`, and `get_response_text`. No changes to `base.py` are required.

---

## Judge validation

The judge prompt and secondary coding fields are defined in `judge_config.yaml`. The paper reports 96.3% agreement (κ = 0.89) between the GPT-4o Mini judge and independent human coding on a 100-item stratified sample.

`scripts/validation.ipynb` generates the stratified sample and rater sheet from `paper_data/joined_dataset_redacted.csv`, and computes inter-rater statistics once the completed rater sheet is available.

---

## Citation

[TBD — forthcoming]

---

## Project information

This work was conducted at RAND Corporation. Funding was provided by the National Institute of Mental Health (NIMH), National Institutes of Health, under grant 1R01MH132551.

---

## Copyright and License

Copyright (C) 2026 RAND Corporation. This code is made available under the [MIT License](LICENSE).

---

## Authors

Jonathan H. Cantor, PhD  
Ellice Huang, BA  
Li Ang Zhang, PhD  
Adrian Salas, BA  
Fang Zhang, PhD  
Alyssa Burnett, MPH  
Andrew Lohn, PhD  
Pat Pataranutaporn, PhD  
Benjamin Rader, PhD  
Hao Yu, PhD  
Ryan K. McBain, PhD, MPH

RAND Corporation, Santa Monica/Arlington, VA
