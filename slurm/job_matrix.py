"""
NarrativeShield — SLURM Job Matrix Generator
================================================================
Generates the full (model x condition) job matrix as a flat list, so
submit_all.sh can submit one sbatch call per row, each requesting exactly
1 GPU node. This is the single source of truth for "which jobs exist" —
both submit_all.sh and any manual re-submission should read from here
rather than hand-listing model/condition combinations, to avoid the same
kind of drift that caused the MODEL_KEYS bug in the original eval scripts.

Conditions
----------
  B1  — run_b1.py --model <key>, one job per model (12 models). run_b1.py's
        --model flag defaults to "all" for backward-compatible local use,
        but every job this matrix generates passes an explicit single key.
  B2  — run_b2.py, one job per model, PLUS the base-model stress test
        (13 models — run_b2.py/run_b3.py are the only scripts that get
        the llama31_8b_base condition; run_ns.py does not, since
        NarrativeShield's 3-agent pipeline requires instruction-following)
  B3  — run_b3.py, same 13-model set as B2
  B4  — run_b4.py (chain-of-thought + explicit register-neutrality
        instruction combined — see run_b4.py docstring), same 13-model set
        as B2/B3
  NS  — run_ns.py, one job per model PER ABLATION (12 models x 3 ablations
        = 36 jobs) — no base-model entry (see above)

Total: 12 (B1) + 13 (B2) + 13 (B3) + 13 (B4) + 36 (NS) = 87 jobs.

Walltime is sized per model scale — larger models need more time per
question for both generation and (for NS) the 3-agent pipeline's multiple
forward passes. These are starting estimates; adjust WALLTIME_BY_SCALE
after the first few jobs complete and you have real per-model throughput
numbers logged in the .out files.

Usage
-----
  python job_matrix.py                  # print the full matrix as a table
  python job_matrix.py --csv jobs.csv   # write to CSV for submit_all.sh to read
"""

import argparse
import csv
import sys
from pathlib import Path



def _load_identity() -> dict:
    """Read MODEL_KEYS / BASE_MODEL_KEY straight out of metrics_common.py
    without importing it (importing needs numpy/statsmodels/sklearn, which are
    not available outside the conda env, e.g. on the login node). Still a
    single source of truth - no second copy of the model list."""
    import ast
    src = (Path(__file__).resolve().parent / "metrics_common.py").read_text(encoding="utf-8")
    wanted, found = ("MODEL_KEYS", "BASE_MODEL_KEY"), {}
    for node in ast.parse(src).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in wanted):
            found[node.targets[0].id] = ast.literal_eval(node.value)
    missing = [w for w in wanted if w not in found]
    if missing:
        raise SystemExit(f"metrics_common.py is missing: {missing}")
    return found


_ID = _load_identity()
MODEL_KEYS = _ID["MODEL_KEYS"]

# Approximate parameter count per model key, used only to size walltime
# buckets — not used for anything else. Keep in sync with metrics_common.MODEL_KEYS.
MODEL_SCALE_B = {
    "llama32_3b": 3, "medgemma_4b": 4, "phi4_mini": 4,
    "qwen25_7b": 7, "mistral7b": 7, "qwen3_8b": 8, "ministral8b": 8,
    "llama31_8b": 8, "openbiollm_8b": 8, "llama31_8b_base": 8,
    "gemma3_12b": 12, "qwen25_14b": 14, "mistral_small_24b": 24,
}

# (max_scale_inclusive, walltime) — first matching bucket wins. NS jobs get
# a multiplier since the 3-agent pipeline does up to 3x the forward passes
# of a single-call condition (Agent 1 + Agent 2 + occasional Agent 3).
# Walltime by model size (billions of params). Estimates, not measurements:
#   B1/B3  : 128 new tokens, batched   -> short
#   B2/B4  : 512 new tokens (CoT), batched -> a few times longer
#   NS     : 2-3 sequential, UN-batched generations per persona x 3000
#            personas -> by far the slowest (roughly 15-20h for an 8B model)
# Every run_*.py is resume-safe, so if a job hits its limit just re-run
# submit_all.sh and it continues. Values above the partition's max walltime
# (48h assumed for HPC `gh`) will be rejected by sbatch - lower them if so.
WALLTIME_SHORT = [(30, "48:00:00")]   # B1, B3
WALLTIME_COT   = [(30, "48:00:00")]   # B2, B4
WALLTIME_NS    = [(30, "48:00:00")]   # NS (all ablations)

NS_ABLATIONS = ["full", "notool", "agent1only"]
BASE_MODEL_KEY = _ID["BASE_MODEL_KEY"]  # "llama31_8b_base" — B2/B3 only


def _walltime_for(model_key: str, table: list) -> str:
    scale = MODEL_SCALE_B.get(model_key, 8)
    for max_scale, wt in table:
        if scale <= max_scale:
            return wt
    return table[-1][1]


def build_matrix() -> list:
    rows = []

    # B1 — one job per model, no base-model condition (B1 is a chat-prompt
    # condition; run_b1.py's original design runs all models in one script
    # with no --model flag, so these may be submitted as a single job if
    # run_b1.py hasn't been split to accept --model — check before submitting).
    for m in MODEL_KEYS:
        rows.append({
            "script": "run_b1.py", "model_key": m, "ablation": "",
            "walltime": _walltime_for(m, WALLTIME_SHORT),
            "condition": "B1",
        })

    # B2 / B3 / B4 — 12 models + the base-model stress test
    for script, cond in [("run_b2.py", "B2"), ("run_b3.py", "B3"), ("run_b4.py", "B4")]:
        for m in MODEL_KEYS + [BASE_MODEL_KEY]:
            rows.append({
                "script": script, "model_key": m, "ablation": "",
                "walltime": _walltime_for(m, WALLTIME_SHORT if cond == "B3" else WALLTIME_COT),
                "condition": cond,
            })

    # NS — 12 models x 3 ablations, no base-model entry
    for m in MODEL_KEYS:
        for ablation in NS_ABLATIONS:
            rows.append({
                "script": "run_ns.py", "model_key": m, "ablation": ablation,
                "walltime": _walltime_for(m, WALLTIME_NS),
                "condition": f"NS-{ablation}",
            })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Write the matrix to this CSV path instead of printing.")
    args = parser.parse_args()

    rows = build_matrix()

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["condition", "script", "model_key", "ablation", "walltime"],
                                lineterminator=chr(10))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} jobs -> {args.csv}")
    else:
        print(f"{'idx':>4}  {'condition':<12} {'script':<12} {'model_key':<20} {'ablation':<12} {'walltime':<10}")
        print("-" * 78)
        for i, r in enumerate(rows):
            print(f"{i:>4}  {r['condition']:<12} {r['script']:<12} {r['model_key']:<20} "
                  f"{r['ablation']:<12} {r['walltime']:<10}")
        print(f"\nTotal jobs: {len(rows)}")


if __name__ == "__main__":
    main()
