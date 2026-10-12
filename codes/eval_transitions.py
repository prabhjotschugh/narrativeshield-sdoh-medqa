"""
NarrativeShield — Paired Transition-Table Analysis
=================================================================================
A central methodological critique of the mitigation claim:

  For Llama-3.2, B1 accuracy drops from 0.665 (control) to 0.509, while marked
  personas move from 0.529/0.552 to 0.518/0.512 under NarrativeShield. The
  GAP closes, but neither marked persona actually IMPROVES — the honest
  reading is "everyone got worse, roughly evenly," not "the disadvantaged
  personas caught up." An aggregate scalar (NAG, OMR delta) cannot distinguish
  these two very different mechanisms.

This script answers the question directly with per-question paired data,
which already exists in results/b1_results/ and results/ns_results/ — no new
inference is needed, this is pure post-hoc analysis.

For every model x persona-pair (alpha vs each marked persona), builds the
2x2 transition table:

                              |  NS correct   |  NS incorrect
    B1-control correct,      |   RECOVERED    |  STILL FAILING
    B1-marked incorrect      |  (anchoring    |
    ("anchoring victim")     |   victim fixed)|
    --------------------------+----------------+----------------
    B1-control incorrect,    |  (reverse      |       —
    B1-marked correct        |   anchoring:   |
    ("reverse anchoring")    |   n/a — no B1  |
                              |   deficit to   |
                              |   fix here)    |
    --------------------------+----------------+----------------
    Both correct under B1    |   MAINTAINED   |  NEWLY BROKEN
    --------------------------+----------------+----------------
    Both incorrect under B1  |  NEWLY FIXED   |  STILL FAILING

The key number for defending the mitigation claim is the "anchoring victim
recovered" rate: of the questions where the model specifically failed BECAUSE
of the marked persona (correct under control, wrong under marked), what
fraction does NarrativeShield actually fix? If this is low while the
aggregate NAG still looks good, that confirms the "everyone got worse
evenly" reading rather than genuine mitigation.

Outputs
-------
  transition_tables.csv        — one row per (model, persona_pair) with all
                                  2x2 cell counts + rates
  transition_tables_detail.csv — per-question transition category, for
                                  drill-down / qualitative review

Usage
-----
  python eval_transitions.py
"""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics_common as mc

B1_RESULT_DIR = os.environ.get("B1_RESULT_DIR", "/path/to/scratch/b1_results")
NS_RESULT_DIR = os.environ.get("NS_RESULT_DIR", "/path/to/scratch/ns_results")
OUTPUT_DIR    = os.environ.get("OUTPUT_DIR", NS_RESULT_DIR)
NS_ABLATION   = os.environ.get("NS_ABLATION", "full")   # which NS condition to compare against B1

MODEL_KEYS = mc.MODEL_KEYS
PERSONAS = mc.PERSONAS


def load_b1(model_key: str) -> list:
    return mc.load_jsonl_results(B1_RESULT_DIR, "results_B1_{model}.jsonl", model_key)


def load_ns(model_key: str, ablation: str) -> list:
    return mc.load_jsonl_results(NS_RESULT_DIR, f"results_NS_{ablation}_" + "{model}.jsonl", model_key)


def build_correctness_by_qid(data: list, persona: str) -> dict:
    """Returns {question_id: bool_correct} for one persona, from a list of result rows."""
    out = {}
    for row in data:
        qid = row.get("question_id")
        if qid is None:
            continue
        out[qid] = bool(mc.safe_get(row, "personas_eval", persona, "is_correct", default=False))
    return out


def classify_transition(b1_control_correct: bool, b1_marked_correct: bool, ns_correct: bool) -> str:
    """
    Classifies one (question, persona-pair) triple into one of 6 categories.
    "anchoring_victim" cases are the ones the mitigation claim lives or dies on.
    """
    if b1_control_correct and not b1_marked_correct:
        return "anchoring_victim_recovered" if ns_correct else "anchoring_victim_still_failing"
    if not b1_control_correct and b1_marked_correct:
        return "reverse_anchoring_recovered" if ns_correct else "reverse_anchoring_still_failing"
    if b1_control_correct and b1_marked_correct:
        return "maintained" if ns_correct else "newly_broken"
    # neither correct under B1
    return "newly_fixed" if ns_correct else "still_failing_both"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print(f"  NarrativeShield — Paired Transition-Table Analysis (NS ablation={NS_ABLATION})")
    print("=" * 70)

    summary_rows = []
    detail_rows = []

    for model_key in MODEL_KEYS:
        b1_data = load_b1(model_key)
        ns_data = load_ns(model_key, NS_ABLATION)
        if not b1_data or not ns_data:
            print(f"  [skip] {model_key}: missing B1 ({len(b1_data)} rows) or NS ({len(ns_data)} rows)")
            continue

        b1_control = build_correctness_by_qid(b1_data, "alpha")
        ns_correct_by_persona = {
            p: build_correctness_by_qid(ns_data, p) for p in PERSONAS
        }

        for marked_persona in PERSONAS:
            if marked_persona == "alpha":
                continue
            b1_marked = build_correctness_by_qid(b1_data, marked_persona)
            ns_marked = ns_correct_by_persona.get(marked_persona, {})

            common_qids = set(b1_control) & set(b1_marked) & set(ns_marked)
            if not common_qids:
                continue

            counts = {
                "anchoring_victim_recovered": 0, "anchoring_victim_still_failing": 0,
                "reverse_anchoring_recovered": 0, "reverse_anchoring_still_failing": 0,
                "maintained": 0, "newly_broken": 0,
                "newly_fixed": 0, "still_failing_both": 0,
            }
            for qid in common_qids:
                cat = classify_transition(b1_control[qid], b1_marked[qid], ns_marked[qid])
                counts[cat] += 1
                detail_rows.append({
                    "model": model_key,
                    "persona_pair": f"alpha_vs_{marked_persona}",
                    "question_id": qid,
                    "b1_control_correct": b1_control[qid],
                    "b1_marked_correct": b1_marked[qid],
                    "ns_correct": ns_marked[qid],
                    "transition": cat,
                })

            n_victims = counts["anchoring_victim_recovered"] + counts["anchoring_victim_still_failing"]
            n_both_correct = counts["maintained"] + counts["newly_broken"]
            n_both_wrong = counts["newly_fixed"] + counts["still_failing_both"]
            recovery_rate = (
                counts["anchoring_victim_recovered"] / n_victims if n_victims > 0 else float("nan")
            )
            collateral_damage_rate = (
                counts["newly_broken"] / n_both_correct if n_both_correct > 0 else float("nan")
            )

            summary_rows.append({
                "model": model_key,
                "persona_pair": f"alpha_vs_{marked_persona}",
                "n": len(common_qids),
                "n_anchoring_victims": n_victims,
                "anchoring_victim_recovery_rate": recovery_rate,
                "anchoring_victims_recovered": counts["anchoring_victim_recovered"],
                "anchoring_victims_still_failing": counts["anchoring_victim_still_failing"],
                "n_reverse_anchoring": (
                    counts["reverse_anchoring_recovered"] + counts["reverse_anchoring_still_failing"]
                ),
                "reverse_anchoring_recovered": counts["reverse_anchoring_recovered"],
                "reverse_anchoring_still_failing": counts["reverse_anchoring_still_failing"],
                "n_both_correct_under_b1": n_both_correct,
                "maintained": counts["maintained"],
                "newly_broken_collateral_damage": counts["newly_broken"],
                "collateral_damage_rate": collateral_damage_rate,
                "n_both_wrong_under_b1": n_both_wrong,
                "newly_fixed": counts["newly_fixed"],
                "still_failing_both": counts["still_failing_both"],
            })

            print(f"\n  {model_key} | alpha vs {marked_persona}  (n={len(common_qids)})")
            print(f"    Anchoring victims (B1 control correct, B1 {marked_persona} wrong): {n_victims}")
            if n_victims > 0:
                print(f"      -> recovered by NS: {counts['anchoring_victim_recovered']} "
                      f"({recovery_rate*100:.1f}%)  |  still failing: {counts['anchoring_victim_still_failing']}")
            print(f"    Both correct under B1: {n_both_correct}  "
                  f"-> maintained: {counts['maintained']}  |  newly broken (collateral damage): "
                  f"{counts['newly_broken']}"
                  + (f" ({collateral_damage_rate*100:.1f}%)" if n_both_correct > 0 else ""))
            print(f"    Both wrong under B1: {n_both_wrong}  "
                  f"-> newly fixed: {counts['newly_fixed']}  |  still failing: {counts['still_failing_both']}")

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        out_path = f"{OUTPUT_DIR}/transition_tables.csv"
        df_summary.to_csv(out_path, index=False)
        print(f"\n[OK] Transition table summary -> {out_path}")
    else:
        print("\n[WARN] No transition data computed — check that both B1 and NS result files exist.")

    if detail_rows:
        df_detail = pd.DataFrame(detail_rows)
        detail_path = f"{OUTPUT_DIR}/transition_tables_detail.csv"
        df_detail.to_csv(detail_path, index=False)
        print(f"[OK] Per-question detail       -> {detail_path}  ({len(df_detail)} rows)")

    print("\n" + "=" * 70)
    print("  INTERPRETATION NOTE")
    print("=" * 70)
    print("""
The "anchoring_victim_recovery_rate" is the load-bearing number for the
mitigation claim. A near-zero NAG under NarrativeShield is consistent with
TWO very different underlying mechanisms:

  (a) Genuine mitigation: anchoring victims are recovered at a high rate
      (this is what the paper wants to claim).
  (b) Uniform degradation: NarrativeShield lowers accuracy roughly evenly
      across control and marked personas, closing the GAP without actually
      helping the disadvantaged persona (the Llama-3.2 reading).

Report BOTH the recovery rate AND the collateral-damage rate (newly-broken
questions that both personas previously got right) alongside NAG for every
model — a model can only support the mitigation narrative if recovery rate
is meaningfully higher than what random accuracy churn would produce, and
collateral damage is not absorbing the apparent gain.
""")


if __name__ == "__main__":
    main()
