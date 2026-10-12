#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# NarrativeShield — submit the (model x condition) job matrix to SLURM
# ═══════════════════════════════════════════════════════════════════════════
# Run from the HPC login node, inside the flat project folder:
#   cd /path/to/scratch/narrative_shield
#   bash submit_all.sh --dry-run            # show what would be submitted
#   bash submit_all.sh --condition B1       # one wave at a time (recommended)
#   bash submit_all.sh                      # everything
#
# One job = one model x one condition, on one GH node. job_matrix.py defines
# the 87 jobs (B1 12, B2 13, B3 13, B4 13, NS 36).
#
# Safe to re-run any number of times: a job whose output file already has all
# questions is skipped, and a partly-finished one resumes where it stopped.
# ═══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH_BASE="${SCRATCH_BASE:-/path/to/scratch}"
N_QUESTIONS="${N_QUESTIONS:-1000}"
MATRIX_CSV="${SCRIPT_DIR}/job_matrix_generated.csv"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

DRY_RUN=false
CONDITION_FILTER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --condition) CONDITION_FILTER="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# job_matrix.py is dependency-free, so no conda env is needed on the login node.
python "${SCRIPT_DIR}/job_matrix.py" --csv "${MATRIX_CSV}" || { echo "job_matrix.py failed"; exit 1; }

# Where each condition writes its results (must match OUTPUT_DIR in run_*.py).
output_file_for() {
    local condition="$1" model="$2" ablation="$3"
    case "${condition}" in
        B1) echo "${SCRATCH_BASE}/b1_results/results_B1_${model}.jsonl" ;;
        B2) echo "${SCRATCH_BASE}/b2_results/results_B2_${model}.jsonl" ;;
        B3) echo "${SCRATCH_BASE}/b3_results/results_B3_${model}.jsonl" ;;
        B4) echo "${SCRATCH_BASE}/b4_results/results_B4_${model}.jsonl" ;;
        NS-*) echo "${SCRATCH_BASE}/ns_results/results_NS_${ablation}_${model}.jsonl" ;;
    esac
}

# The cluster allows at most 40 jobs per user in the queue (QOSMaxSubmitJobPerUserLimit).
# Submit until the queue is full, then stop; re-run later to submit the rest.
MAX_QUEUE="${MAX_QUEUE:-40}"

# Jobs of mine already queued/running, by full job name (avoids duplicates).
declare -A IN_QUEUE
N_IN_QUEUE=0
while read -r qname; do
    [[ -z "${qname}" ]] && continue
    IN_QUEUE["${qname}"]=1
    N_IN_QUEUE=$((N_IN_QUEUE + 1))
done < <(squeue -u "${USER}" -h -o "%.100j" 2>/dev/null | tr -d ' ')
echo "Jobs currently in your queue: ${N_IN_QUEUE}  (limit ${MAX_QUEUE})"

N_SUBMITTED=0; N_DONE=0; N_FILTERED=0; N_FAILED=0; N_ALREADY_QUEUED=0; N_WAITING=0

while IFS=, read -r condition script model_key ablation walltime; do
    walltime="$(printf "%s" "${walltime}" | tr -d "[:space:]")"
    if [[ -n "${CONDITION_FILTER}" && "${condition}" != "${CONDITION_FILTER}"* ]]; then
        N_FILTERED=$((N_FILTERED + 1)); continue
    fi

    JOB_NAME="${condition}_${model_key}"

    out_file="$(output_file_for "${condition}" "${model_key}" "${ablation}")"
    if [[ -f "${out_file}" ]]; then
        n_lines=$(wc -l < "${out_file}")
        if [[ "${n_lines}" -ge "${N_QUESTIONS}" ]]; then
            echo "[complete]  ${JOB_NAME} (${n_lines} rows)"
            N_DONE=$((N_DONE + 1)); continue
        fi
    fi

    if [[ -n "${IN_QUEUE[${JOB_NAME}]}" ]]; then
        echo "[in queue]  ${JOB_NAME}"
        N_ALREADY_QUEUED=$((N_ALREADY_QUEUED + 1)); continue
    fi

    if [[ "${N_IN_QUEUE}" -ge "${MAX_QUEUE}" ]]; then
        N_WAITING=$((N_WAITING + 1)); continue
    fi

    EXPORT_VARS="ALL,MODEL_KEY=${model_key},SCRIPT=${script}"
    [[ -n "${ablation}" ]] && EXPORT_VARS="${EXPORT_VARS},ABLATION=${ablation}"

    CMD=(sbatch --job-name="${JOB_NAME}" --time="${walltime}"
         --output="${LOG_DIR}/${JOB_NAME}.o%j" --error="${LOG_DIR}/${JOB_NAME}.e%j"
         --export="${EXPORT_VARS}" "${SCRIPT_DIR}/run_job.sbatch")

    if ${DRY_RUN}; then
        echo "[dry-run] ${JOB_NAME}  walltime=${walltime}"
    else
        echo "Submitting ${JOB_NAME}  walltime=${walltime}"
        if ! "${CMD[@]}"; then
            echo "  !! sbatch FAILED for ${JOB_NAME}"
            N_FAILED=$((N_FAILED + 1))
            if [[ "${N_FAILED}" -ge 3 ]]; then
                echo "3 sbatch failures - stopping. Fix the error above and re-run."
                exit 1
            fi
            continue
        fi
    fi
    IN_QUEUE["${JOB_NAME}"]=1
    N_IN_QUEUE=$((N_IN_QUEUE + 1))
    N_SUBMITTED=$((N_SUBMITTED + 1))
done < <(tail -n +2 "${MATRIX_CSV}")

echo
echo "Submitted now: ${N_SUBMITTED}   failed: ${N_FAILED}"
echo "Already in queue: ${N_ALREADY_QUEUED}   complete: ${N_DONE}   filtered out: ${N_FILTERED}"
if [[ "${N_WAITING}" -gt 0 ]]; then
    echo "Queue is full (${MAX_QUEUE}): ${N_WAITING} job(s) still waiting to be submitted."
    echo "Re-run 'bash submit_all.sh' after some jobs finish to submit the next ones."
fi
echo "Logs: ${LOG_DIR}      Monitor: squeue -u \$USER"
