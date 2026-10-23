#!/usr/bin/env bash
# Azure ML job entry point for benchmark runs on the GPU cluster.
#
#   bash infra/aml/run_job.sh <output_dir> -- <python script> <args>
#
# 1. Bootstraps Python 3.12 + the uv.lock-pinned requirements (infra/aml/env/requirements.txt)
#    with uv (no custom image build needed).
# 2. Works on local disk and syncs <output_dir> (blob-backed rw_mount, fixed per run unit)
#    every few minutes and on exit, so a preempted (low-priority) or resubmitted job resumes.
# 3. On exit, attaches a compact summary (scripts/job_summary.py), the exit code and the log
#    tail to the job as MLflow tags (infra/aml/report_tags.py), readable through the workspace
#    API without storage access (scripts/aml_summaries.py). Best effort; never fails the job.
set -euo pipefail

OUT="$1"; shift
[ "${1:-}" = "--" ] && shift

WORK="${PWD}/.work"
mkdir -p "${WORK}"
LOG="${WORK}/job.log"
exec > >(tee -a "${LOG}") 2>&1

SYNC_PID=""
sync_out() { [ -n "${ARTIFACTS_DIR:-}" ] && mkdir -p "${OUT}" && cp -r "${ARTIFACTS_DIR}/." "${OUT}/" || true; }
finish() {
  rc=$?
  set +e
  [ -n "${SYNC_PID}" ] && kill "${SYNC_PID}" 2>/dev/null
  sync_out
  echo "== exit code ${rc}"
  if [ -x "${WORK}/venv/bin/python" ] && [ -n "${ARTIFACTS_DIR:-}" ]; then
    "${WORK}/venv/bin/python" scripts/job_summary.py --out "${WORK}/summary.json"
  fi
  if [ -x "${WORK}/report-venv/bin/python" ]; then
    "${WORK}/report-venv/bin/python" infra/aml/report_tags.py --exit-code "${rc}" \
      --summary "${WORK}/summary.json" --log "${LOG}"
  fi
  exit "${rc}"
}
trap finish EXIT

echo "== bootstrap python env"
t0=$(date +%s)
if ! command -v uv >/dev/null; then
  command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl ca-certificates)
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${WORK}/bin" sh
  export PATH="${WORK}/bin:${PATH}"
fi
export UV_PYTHON_INSTALL_DIR="${WORK}/python" UV_CACHE_DIR="${WORK}/uv-cache" UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=300
# Reporting venv (kept apart so it cannot change the pinned job environment).
( uv venv "${WORK}/report-venv" --python 3.12 -q && \
  uv pip install --python "${WORK}/report-venv/bin/python" -q mlflow-skinny azureml-mlflow ) \
  || echo "[report] reporting venv unavailable; tags will not be set"
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
export PYTHONUNBUFFERED=1
mkdir -p "${ARTIFACTS_DIR}" "${CACHE_DIR}" "${DATA_DIR}" "${HF_HOME}"

echo "== resume from ${OUT}"
if [ -n "$(ls -A "${OUT}" 2>/dev/null)" ]; then cp -r "${OUT}/." "${ARTIFACTS_DIR}/"; fi
find "${ARTIFACTS_DIR}" -name .lock -delete  # AML guarantees one job per output path

( while true; do sleep "${SYNC_EVERY_S:-180}"; sync_out; done ) &
SYNC_PID=$!

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), 'gpus', torch.cuda.device_count(), 'transformers', transformers.__version__)"

python "$@"
