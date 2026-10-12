"""
NarrativeShield - progress report for the 87-job matrix.

Usage (login node, from the project folder, no conda env needed):
    python status.py

For every (model x condition) job it looks at two things:
  - the result file: 1000 rows = DONE
  - your SLURM queue: RUNNING / PENDING
Anything else is "TO SUBMIT" (never submitted, or failed and not resubmitted).
A failed job that was resubmitted shows up as RUNNING/PENDING, not as failed.
"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from job_matrix import build_matrix  # noqa: E402  (stdlib-only, reads model list from metrics_common.py)

SCRATCH = os.environ.get("SCRATCH_BASE", "/path/to/scratch")
N_QUESTIONS = int(os.environ.get("N_QUESTIONS", "1000"))


def result_file(job):
    cond, model, abl = job["condition"], job["model_key"], job["ablation"]
    if cond.startswith("NS-"):
        return "%s/ns_results/results_NS_%s_%s.jsonl" % (SCRATCH, abl, model)
    return "%s/%s_results/results_%s_%s.jsonl" % (SCRATCH, cond.lower(), cond, model)


def count_rows(path):
    if not os.path.isfile(path):
        return 0
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def queue_states():
    user = os.environ.get("USER", "")
    try:
        out = subprocess.run(
            ["squeue", "-u", user, "-h", "-o", "%.100j|%T"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True,
        ).stdout
    except Exception:
        return {}
    states = {}
    for line in out.splitlines():
        if "|" in line:
            name, state = line.rsplit("|", 1)
            states[name.strip()] = state.strip()
    return states


def main():
    jobs = build_matrix()
    queue = queue_states()

    groups = {"DONE": [], "RUNNING": [], "PENDING": [], "TO SUBMIT": []}
    per_cond = {}

    for job in jobs:
        name = "%s_%s" % (job["condition"], job["model_key"])
        rows = count_rows(result_file(job))
        state = queue.get(name)

        if rows >= N_QUESTIONS:
            group = "DONE"
        elif state == "PENDING":
            group = "PENDING"
        elif state is not None:          # RUNNING, COMPLETING, CONFIGURING, ...
            group = "RUNNING"
        else:
            group = "TO SUBMIT"

        groups[group].append((name, rows))
        c = per_cond.setdefault(job["condition"], {"DONE": 0, "total": 0})
        c["total"] += 1
        if group == "DONE":
            c["DONE"] += 1

    total = len(jobs)
    print("=" * 60)
    print("  DONE       : %3d / %d" % (len(groups["DONE"]), total))
    print("  RUNNING    : %3d" % len(groups["RUNNING"]))
    print("  PENDING    : %3d" % len(groups["PENDING"]))
    print("  TO SUBMIT  : %3d   (not in queue, not complete)" % len(groups["TO SUBMIT"]))
    print("=" * 60)

    print("\nBy condition (done / total):")
    for cond in sorted(per_cond):
        c = per_cond[cond]
        print("  %-14s %2d / %2d" % (cond, c["DONE"], c["total"]))

    for label in ("RUNNING", "PENDING", "TO SUBMIT"):
        if groups[label]:
            print("\n%s:" % label)
            for name, rows in groups[label]:
                extra = "  (%d/%d rows so far)" % (rows, N_QUESTIONS) if rows else ""
                print("  %-32s%s" % (name, extra))

    if groups["DONE"]:
        print("\nDONE:")
        print("  " + ", ".join(n for n, _ in groups["DONE"]))

    if groups["TO SUBMIT"]:
        print("\nRun 'bash submit_all.sh' to submit the TO SUBMIT jobs (it stops at the 40-job queue limit).")


if __name__ == "__main__":
    main()
