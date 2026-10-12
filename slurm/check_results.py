"""
NarrativeShield - quick quality check of result files (no packages needed).

    python check_results.py              # all conditions
    python check_results.py B2 NS        # only conditions starting with these

For every result file it prints, per persona: accuracy and parse rate
(share of responses with a usable answer letter). For NarrativeShield it also
shows how often Agent 1 produced valid JSON. It flags likely problems:
  LOW ACC   accuracy near/below chance (0.25)  -> prompt/parsing problem
  LOW PARSE parse rate < 90%                   -> answers not being extracted
  GAP       alpha-vs-marked gap > 0.10         -> (expected for B1; interesting elsewhere)
Rows that are not yet 1000 are marked PARTIAL.
"""
import glob
import json
import os
import re
import sys

SCRATCH = os.environ.get("SCRATCH_BASE", "/path/to/scratch")
N_QUESTIONS = int(os.environ.get("N_QUESTIONS", "1000"))
PERSONAS = ["alpha", "beta", "gamma"]


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def answer_of(ev):
    # B1-B4 store extracted_answer; NarrativeShield stores final_answer
    return ev.get("extracted_answer", ev.get("final_answer"))


def main():
    wanted = [a.upper() for a in sys.argv[1:]]
    files = sorted(glob.glob("%s/*_results/results_*.jsonl" % SCRATCH))
    if not files:
        print("No result files found under %s" % SCRATCH)
        return

    n_flags = 0
    print("%-34s %6s  %-26s %-26s %s" % ("file", "rows", "accuracy a/b/g", "parsed a/b/g", "flags"))
    print("-" * 120)
    for path in files:
        name = os.path.basename(path)[len("results_"):-len(".jsonl")]
        if wanted and not any(name.upper().startswith(w) for w in wanted):
            continue
        rows = load(path)
        if not rows:
            continue

        acc, parsed, a1ok = [], [], []
        for p in PERSONAS:
            evs = [r["personas_eval"][p] for r in rows if p in r.get("personas_eval", {})]
            n = max(len(evs), 1)
            acc.append(sum(bool(e.get("is_correct")) for e in evs) / n)
            parsed.append(sum(answer_of(e) is not None for e in evs) / n)
            if "agent1_parse_success" in (evs[0] if evs else {}):
                a1ok.append(sum(bool(e.get("agent1_parse_success")) for e in evs) / n)

        flags = []
        if len(rows) < N_QUESTIONS:
            flags.append("PARTIAL(%d)" % len(rows))
        if min(acc) < 0.30:
            flags.append("LOW ACC")
        if min(parsed) < 0.90:
            flags.append("LOW PARSE")
        if acc[0] - min(acc[1], acc[2]) > 0.10:
            flags.append("GAP %.2f" % (acc[0] - min(acc[1], acc[2])))
        if a1ok:
            flags.append("agent1 JSON ok %.0f%%" % (100 * sum(a1ok) / len(a1ok)))
            if sum(a1ok) / len(a1ok) < 0.80:
                flags.append("LOW AGENT1")
        # Chain-of-thought files (B2/B4) should end with an explicit "Answer: X".
        # Responses without it were scored by a fallback that can pick a stray
        # letter (e.g. the article "A"), so they are close to random guesses.
        raws = [e.get("raw_response") for r in rows for e in r.get("personas_eval", {}).values()
                if isinstance(e.get("raw_response"), str)]
        if raws and name.startswith(("B2", "B4")):
            expl = sum(bool(re.search(r"(?i)\banswer\s*:\s*\W*([A-D])\b", t)) for t in raws) / len(raws)
            think = sum("<think>" in t for t in raws) / len(raws)
            flags.append("explicit-Answer %.0f%%" % (100 * expl))
            if expl < 0.90:
                flags.append("LOW EXPLICIT")
            if think > 0.01:
                flags.append("think-leak %.0f%%" % (100 * think))
        n_flags += any(f.startswith(("LOW", "PARTIAL")) for f in flags)

        print("%-34s %6d  %-26s %-26s %s" % (
            name, len(rows),
            "/".join("%.3f" % a for a in acc),
            "/".join("%.3f" % p for p in parsed),
            "  ".join(flags)))

    print("-" * 120)
    print("%d file(s) with a LOW/PARTIAL flag. Chance level is 0.25." % n_flags)


if __name__ == "__main__":
    main()
