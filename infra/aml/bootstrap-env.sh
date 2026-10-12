#!/usr/bin/env bash
# Run on the Azure ML GPU compute instance from the repo root:
#   bash infra/aml/bootstrap-env.sh
# Creates the local .venv with the GPU stack, pre-downloads gpt-oss-20b and runs preflight.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/mnt/hf-cache}"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

[ -f .env ] || cp .env.example .env

uv sync --extra local --extra aml

# Entra auth for the Foundry endpoints: prefer the compute instance's managed identity.
az login --identity --allow-no-subscriptions >/dev/null 2>&1 \
  || echo "Managed identity login failed; run 'az login --use-device-code' instead."

uv run python -c "import os; from huggingface_hub import snapshot_download; \
print(snapshot_download(os.environ.get('OSS_MODEL_ID', 'openai/gpt-oss-20b'), \
ignore_patterns=['original/*', 'metal/*']))"

uv run python scripts/run_preflight.py
