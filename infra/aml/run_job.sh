#!/usr/bin/env bash
# Azure ML job entry point for benchmark runs on the GPU cluster.
#
#   bash infra/aml/run_job.sh <output_dir> -- <python script> <args>
#
# 1. Bootstraps Python 3.12 + the uv.lock-pinned requirements (infra/aml/env/requirements.txt)
#    with uv (no custom image build needed).
# 2. Works on local disk and syncs <output_dir> (blob-backed rw_mount, fixed per run unit)
#    every few minutes and on exit, so a preempted (low-priority) or resubmitted job resumes.
set -euo pipefail

OUT="$1"; shift
[ "${1:-}" = "--" ] && shift

WORK="${PWD}/.work"
mkdir -p "${WORK}"

echo "== bootstrap python env"
t0=$(date +%s)
if ! command -v uv >/dev/null; then
  command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl ca-certificates)
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${WORK}/bin" sh
  export PATH="${WORK}/bin:${PATH}"
fi
export UV_PYTHON_INSTALL_DIR="${WORK}/python" UV_CACHE_DIR="${WORK}/uv-cache" UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=300
uv venv "${WORK}/venv" --python 3.12 -q
uv pip install --python "${WORK}/venv/bin/python" -q -r infra/aml/env/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
export PATH="${WORK}/venv/bin:${PATH}"
echo "bootstrap took $(( $(date +%s) - t0 ))s"

export PYTHONPATH="${PWD}/src"
export ARTIFACTS_DIR="${WORK}/artifacts"
export CACHE_DIR="${WORK}/cache"
export DATA_DIR="${WORK}/data"
export HF_HOME="${WORK}/hf"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${ARTIFACTS_DIR}" "${CACHE_DIR}" "${DATA_DIR}" "${HF_HOME}"

echo "== resume from ${OUT}"
if [ -d "${OUT}/runs" ]; then cp -r "${OUT}/." "${ARTIFACTS_DIR}/"; fi
find "${ARTIFACTS_DIR}" -name .lock -delete  # AML guarantees one job per output path

sync_out() { mkdir -p "${OUT}" && cp -r "${ARTIFACTS_DIR}/." "${OUT}/" || true; }
( while true; do sleep "${SYNC_EVERY_S:-180}"; sync_out; done ) &
SYNC_PID=$!
trap 'kill ${SYNC_PID} 2>/dev/null || true; sync_out' EXIT

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), 'gpus', torch.cuda.device_count(), 'transformers', transformers.__version__)"

python "$@"
