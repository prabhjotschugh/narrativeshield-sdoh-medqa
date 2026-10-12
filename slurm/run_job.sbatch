#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# NarrativeShield — single-job SLURM template (one model x one condition)
# ═══════════════════════════════════════════════════════════════════════════
# Not run directly: submit_all.sh calls `sbatch` on this file once per
# (model, condition) pair, passing MODEL_KEY / SCRIPT / ABLATION and the
# walltime on the command line. Every run_*.py is resume-safe, so a job that
# times out or crashes just needs to be submitted again.
#
# Layout on HPC (flat, everything in one folder):
#   /path/to/scratch/narrative_shield/
# ═══════════════════════════════════════════════════════════════════════════
#SBATCH -J narrativeshield
#SBATCH -o narrativeshield.o%j
#SBATCH -e narrativeshield.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00


# NOTE: no `set -u` / `set -e` here on purpose. ~/.bashrc and `conda activate`
# reference unset variables and return non-zero codes, and strict mode makes
# the job die before any Python runs.

echo "================================================================"
echo "  Job ID    : ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "  Script    : ${SCRIPT:?SCRIPT not set}"
echo "  Model     : ${MODEL_KEY:?MODEL_KEY not set}"
echo "  Ablation  : ${ABLATION:-n/a}"
echo "  Started   : $(date)"
echo "================================================================"

source ~/.bashrc

# `conda` is not available inside batch jobs on this system (~/.bashrc does not
# set it up non-interactively), so do not use `conda activate`. Put the env's
# own bin/ first on PATH instead - same effect, no conda needed.
ENV_DIR="${WORK:-/path/to/work}/envs/gpu_env"
if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    echo "ERROR: python not found at ${ENV_DIR}/bin/python" >&2
    exit 1
fi
export PATH="${ENV_DIR}/bin:${PATH}"
echo "  Python    : $(which python)"

SCRATCH_BASE="${SCRATCH_BASE:-/path/to/scratch}"
REPO_ROOT="${REPO_ROOT:-${SCRATCH_BASE}/narrative_shield}"

# Model weights + HF cache stay on scratch (the run_*.py files also set this).
export HF_HOME="${SCRATCH_BASE}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# Triton JIT must use gcc, not the cluster's default nvc (rejects -Wno-psabi).
export CC="$(which gcc)"
export CXX="$(which g++)"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. Add 'export HF_TOKEN=...' to ~/.bashrc." >&2
    exit 1
fi

cd "${REPO_ROOT}" || { echo "ERROR: cannot cd to ${REPO_ROOT}" >&2; exit 1; }

CMD="${ENV_DIR}/bin/python ${SCRIPT} --model ${MODEL_KEY}"
if [[ ( "${SCRIPT}" == "run_ns.py" || "${SCRIPT}" == "rerun_ns_stem.py" || "${SCRIPT}" == "run_rewrite.py" ) && -n "${ABLATION:-}" ]]; then
    CMD="${CMD} --ablation ${ABLATION}"
fi

echo "Running: ${CMD}"
${CMD}
STATUS=$?

echo "================================================================"
echo "  Finished: $(date)   exit status: ${STATUS}"
echo "================================================================"
exit ${STATUS}
