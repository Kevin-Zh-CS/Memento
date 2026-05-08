# Memento: PS-Bench on OpenClaw Native Memory

This folder contains a minimal, OpenClaw-only PS-Bench runner. It does not use
external memory frameworks or defense modules. The runner stores LoCoMo history
as OpenClaw workspace memory, retrieves it through OpenClaw `memory_search`, and
generates responses through the OpenClaw Gateway OpenAI-compatible API.

## What Is Included

- `psbench_openclaw_eval.py`: end-to-end benchmark runner.
- `data/processed/LoCoMo_ori/`: LoCoMo conversation histories.
- `data/processed/Harmful_Query_Set/`: harmful query categories.
- `config/openclaw.example.json`: local OpenClaw Gateway config template.
- `scripts/start_openclaw.sh`: starts the OpenClaw Gateway.
- `scripts/run_psbench.sh`: runs the benchmark with repo-local paths.
- `requirements.txt`: Python dependencies for the runner and classifier.

## Runtime Flow

1. Convert `LoCoMo_ori/{persona}.json` into Markdown memory under the OpenClaw
   workspace, for example `.openclaw-workspace/memory/psbench_locomo_tim.md`.
2. Run `openclaw memory index --agent main --force` so OpenClaw indexes the
   Markdown memory.
3. For each harmful query, call Gateway `/tools/invoke` with
   `tool=memory_search` and `corpus=memory`.
4. Put retrieved memory snippets into the user prompt.
5. Call Gateway `/v1/chat/completions` with `model=openclaw/default` and
   `x-openclaw-model: openai/gpt-4o-mini`.
6. Judge attack success rate with `LibrAI/longformer-action-ro`.

OpenClaw references:

- Memory search: https://docs.openclaw.ai/concepts/memory-search
- Gateway CLI: https://docs.openclaw.ai/cli/gateway
- OpenAI-compatible Gateway API: https://docs.openclaw.ai/gateway/openai-http-api

## Prerequisites

- Python 3.10+.
- Node.js is handled by the OpenClaw installer when using the official CLI
  install path.
- `OPENAI_API_KEY` must be set. It is used by OpenClaw for `gpt-4o-mini` and
  `text-embedding-3-small` memory embeddings.

Install Python dependencies:

```bash
cd Memento
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install OpenClaw CLI if `openclaw` is not already on your `PATH`:

```bash
curl -fsSL https://openclaw.ai/install-cli.sh -o /tmp/openclaw-install-cli.sh
bash /tmp/openclaw-install-cli.sh --prefix "$PWD/.tools/openclaw" --no-onboard
export OPENCLAW_BIN="$PWD/.tools/openclaw/bin/openclaw"
```

If `openclaw` is already available globally, you can skip `OPENCLAW_BIN`.

## Start OpenClaw Gateway

Set your OpenAI key and start the Gateway:

```bash
cd Memento
export OPENAI_API_KEY="sk-..."
export OPENCLAW_BIN="${OPENCLAW_BIN:-openclaw}"
./scripts/start_openclaw.sh
```

The script creates:

- `.openclaw-state/openclaw.json`
- `.openclaw-state/logs/`
- `.openclaw-workspace/`

The Gateway listens on `http://127.0.0.1:18789` by default. In another terminal,
check that it is ready:

```bash
curl http://127.0.0.1:18789/v1/models
```

Expected model targets include `openclaw`, `openclaw/default`, and
`openclaw/main`.

## Run a Smoke Test

In a second terminal:

```bash
cd Memento
source .venv/bin/activate
export OPENCLAW_BIN="${OPENCLAW_BIN:-openclaw}"
./scripts/run_psbench.sh \
  --persona Tim \
  --categories Hate_Speech \
  --limit 1 \
  --batch_size 1 \
  --max_tokens 128 \
  --classifier_device cpu \
  --output_dir psbench_results_openclaw_smoke
```

This verifies memory writing, OpenClaw memory indexing, `memory_search`,
Gateway chat completion, classification, and result serialization.

## Run the Default Four-Category Benchmark

```bash
cd Memento
source .venv/bin/activate
export OPENCLAW_BIN="${OPENCLAW_BIN:-openclaw}"
./scripts/run_psbench.sh \
  --persona Tim \
  --categories Hate_Speech,Self_Harm,Abuse,Financial_Crime \
  --batch_size 4 \
  --max_tokens 512 \
  --output_dir psbench_results_openclaw
```

If GPU placement for the classifier is unstable on your machine, use CPU
classification:

```bash
./scripts/run_psbench.sh \
  --persona Tim \
  --categories Hate_Speech,Self_Harm,Abuse,Financial_Crime \
  --batch_size 4 \
  --max_tokens 512 \
  --classifier_device cpu \
  --output_dir psbench_results_openclaw
```

## Important CLI Arguments

- `--persona`: LoCoMo persona file name without `.json`, default `Tim`.
- `--categories`: comma-separated harmful query categories.
- `--backbone_model`: backend model routed by OpenClaw, default
  `openai/gpt-4o-mini`.
- `--openclaw_agent_model`: OpenClaw agent target, default `openclaw/default`.
- `--memory_max_results`: max `memory_search` hits per query, default `100`.
- `--classifier_device`: `auto`, `cpu`, or CUDA index like `0`.
- `--skip_memory_index`: skip reindexing if memory is already indexed.

## Outputs

Results are saved under:

```text
<output_dir>/<persona>_openclaw/
```

Each run writes:

- `summary.json`
- `<Category>_responses.json`
- `<Category>_success.json`
- `<Category>_failure.json`

The summary contains per-category ASR and overall ASR.
