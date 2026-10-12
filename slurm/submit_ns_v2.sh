#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Submit the NarrativeShield v2 reruns (rerun_ns_stem.py): reasoning step
# without the leaked vignette. One job = one model x one ablation.
#   cd /path/to/scratch/narrative_shield
#   bash submit_ns_v2.sh --dry-run
#   bash submit_ns_v2.sh agent1only          # main method first (12 jobs)
#   bash submit_ns_v2.sh notool full         # then the ablations (24 jobs)
#   bash submit_ns_v2.sh rewrite             # Rewrite-then-Reason (run_rewrite.py)
#   bash submit_ns_v2.sh storyform           # Story + Form (run_rewrite.py)
# Safe to re-run: finished jobs are skipped, partial ones resume.
# ═══════════════════════════════════════════════════════════════════════════
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH_BASE="${SCRATCH_BASE:-/path/to/scratch}"
NEW_DIR="${SCRATCH_BASE}/ns_results_v2"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}" "${NEW_DIR}"

DRY_RUN=false
ABLATIONS=()
for a in "$@"; do
    if [[ "$a" == "--dry-run" ]]; then DRY_RUN=true; else ABLATIONS+=("$a"); fi
done
[[ ${#ABLATIONS[@]} -eq 0 ]] && ABLATIONS=(agent1only notool full)

MODELS=(llama32_3b medgemma_4b phi4_mini qwen25_7b mistral7b qwen3_8b ministral8b llama31_8b openbiollm_8b gemma3_12b qwen25_14b mistral_small_24b)
# agent1only only regenerates a one-letter answer; the others regenerate Agent 2.
declare -A WALL=([agent1only]=06:00:00 [notool]=24:00:00 [full]=24:00:00 [rewrite]=48:00:00 [storyform]=24:00:00)
MAX_QUEUE="${MAX_QUEUE:-40}"

queued=$(squeue -u "$USER" -h -o "%j" 2>/dev/null)
n_queue=$(echo "${queued}" | grep -c . )
for abl in "${ABLATIONS[@]}"; do
    for m in "${MODELS[@]}"; do
        name="nsv2_${abl}_${m}"
        out="${NEW_DIR}/results_NS_${abl}_${m}.jsonl"
        if [[ -f "${out}" && $(wc -l < "${out}") -ge 1000 ]]; then echo "done     ${name}"; continue; fi
        if echo "${queued}" | grep -qx "${name}"; then echo "queued   ${name}"; continue; fi
        if (( n_queue >= MAX_QUEUE )); then echo "queue full (${MAX_QUEUE}); re-run later for the rest"; exit 0; fi
        cmd=(sbatch -J "${name}" -t "${WALL[$abl]}" -o "${LOG_DIR}/${name}.o%j" -e "${LOG_DIR}/${name}.e%j"
             --export=ALL,SCRIPT=$( [[ "$abl" == rewrite || "$abl" == storyform ]] && echo run_rewrite.py || echo rerun_ns_stem.py ),MODEL_KEY=${m},ABLATION=${abl} "${SCRIPT_DIR}/run_job.sbatch")
        if $DRY_RUN; then echo "would run: ${cmd[*]}"; else "${cmd[@]}" && n_queue=$((n_queue + 1)); fi
    done
done
