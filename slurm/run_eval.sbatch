#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# NarrativeShield - run ALL evaluation scripts once inference has finished.
#   cd /path/to/scratch/narrative_shield
#   sbatch run_eval.sbatch
# Order matters: eval_b1 first (B2/B3/B4 build delta tables against it).
# Needs the packages in requirements.txt (sentence-transformers, statsmodels,
# scikit-learn) inside the gpu_env environment.
# ═══════════════════════════════════════════════════════════════════════════
#SBATCH -J ns_eval
#SBATCH -o ns_eval.o%j
#SBATCH -e ns_eval.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00

source ~/.bashrc
ENV_DIR="${WORK:-/path/to/work}/envs/gpu_env"
export PATH="${ENV_DIR}/bin:${PATH}"

SCRATCH_BASE="/path/to/scratch"
export HF_HOME="${SCRATCH_BASE}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export CC="$(which gcc)"; export CXX="$(which g++)"

# result folders (each eval script writes its CSVs into its own folder)
export RESULT_DIR="${SCRATCH_BASE}/b1_results"      # eval_b1
export RESULT_DIR_B2="${SCRATCH_BASE}/b2_results"
export RESULT_DIR_B3="${SCRATCH_BASE}/b3_results"
export RESULT_DIR_B4="${SCRATCH_BASE}/b4_results"

cd "${SCRATCH_BASE}/narrative_shield" || exit 1
echo "python: $(which python)"
python -c "import sentence_transformers, statsmodels, sklearn; print('eval packages OK')" || { echo "Missing packages: pip install sentence-transformers statsmodels scikit-learn"; exit 1; }

for s in eval_b1.py eval_b2.py eval_b3.py eval_b4.py eval_transitions.py; do
    echo "================ $s  $(date)"
    python "$s" || echo "!! $s failed (continuing)"
done

# eval_ns.py reads/writes under ns_results by default
export RESULT_DIR="${SCRATCH_BASE}/ns_results"
echo "================ eval_ns.py  $(date)"
python eval_ns.py || echo "!! eval_ns.py failed"

echo "Done $(date). CSVs are in each *_results folder (b1_evaluation_primary.csv, ns_evaluation_combined.csv, transition_tables.csv, ...)"
