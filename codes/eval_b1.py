"""
NarrativeShield — B1 Evaluation Suite
========================================================
Reads results_B1_<model>.jsonl files produced by run_b1.py and produces:

  1. Per-model x per-persona OMR with 95% Wilson score CIs (alpha/beta/gamma)
  2. Overall OMR (macro across personas)
  3. Diagnostic Stability Score (DSS) — mean pairwise cosine similarity
     of answer embeddings across all present persona presentations, plus a
     format-controlled ("stripped") variant
  4. Fraction of cases with DSS < 0.80 (bias-flagged cases)
  5. Cohen's Kappa (chance-corrected inter-persona agreement) for all pairs
  6. McNemar's test (continuity-corrected), reported against BOTH raw 0.05
     and Bonferroni-corrected thresholds (fixes the verified Footnote 1 bug:
     the previous version claimed a Bonferroni correction that was never
     actually applied anywhere in this file)
  7. Narrative Anchoring Gap (NAG) — OMR_alpha minus min(OMR_marked) — always
     reported alongside every individual component gap and a paired
     bootstrap 95% CI on NAG itself, never as a bare scalar

All statistics are computed via codes/metrics_common.py, the single source
of truth shared with eval_b2.py, eval_b3.py, and eval_ns.py — this removes
the model-registry drift bug (eval_b1.py previously listed "ministral3b" and
"qwen35", which never existed in run_b1.py's MODEL_REGISTRY and were
silently skipped).

Outputs
-------
  b1_evaluation_primary.csv    — main results table (OMR, NAG, DSS per model)
  b1_evaluation_secondary.csv  — Kappa, McNemar (raw + Bonferroni), DSS detail
  b1_evaluation_bias_cases.csv — per-question bias flags (DSS < 0.80)

Usage
-----
  python eval_b1.py

Dependencies
------------
  pip install sentence-transformers scikit-learn statsmodels pandas numpy tqdm
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics_common as mc

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    print("Install with:  pip install sentence-transformers scikit-learn statsmodels")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
RESULT_DIR = os.environ.get("RESULT_DIR", "/path/to/scratch/b1_results")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", RESULT_DIR)

MODEL_KEYS = mc.MODEL_KEYS  # imported, not hand-duplicated — see metrics_common.py
PERSONAS = mc.PERSONAS
ST_MODEL_NAME = "all-MiniLM-L6-v2"
DSS_BIAS_THRESHOLD = mc.DSS_BIAS_THRESHOLD

PRIMARY_OUTPUT    = f"{OUTPUT_DIR}/b1_evaluation_primary.csv"
SECONDARY_OUTPUT  = f"{OUTPUT_DIR}/b1_evaluation_secondary.csv"
BIAS_CASES_OUTPUT = f"{OUTPUT_DIR}/b1_evaluation_bias_cases.csv"


def load_model_results(model_key: str) -> list:
    return mc.load_jsonl_results(RESULT_DIR, "results_B1_{model}.jsonl", model_key)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  NarrativeShield B1 — Full Evaluation Suite")
    print("=" * 70)

    print(f"\nLoading sentence encoder: {ST_MODEL_NAME} …")
    st_model = SentenceTransformer(ST_MODEL_NAME)
    print("  Encoder ready.")

    primary_rows, secondary_rows, bias_case_rows = [], [], []

    for model_key in MODEL_KEYS:
        print(f"\n{'─'*65}")
        print(f"  Processing: {model_key}")

        data = load_model_results(model_key)
        if not data:
            print(f"  [skip] No results file found for {model_key}.")
            continue

        n = len(data)
        print(f"  Questions loaded: {n}")

        # Only evaluate personas that actually have data in this file — lets
        # this script run robustly against partial/legacy result files
        # without special-casing which personas are present.
        present_personas = [
            p for p in PERSONAS
            if any(mc.safe_get(r, "personas_eval", p) is not None for r in data[:5])
        ]
        if len(present_personas) < 2:
            print(f"  [skip] Fewer than 2 personas present for {model_key}.")
            continue

        correctness = {
            p: [bool(mc.safe_get(r, "personas_eval", p, "is_correct", default=False)) for r in data]
            for p in present_personas
        }
        responses = {
            p: [str(mc.safe_get(r, "personas_eval", p, "raw_response", default="") or "") for r in data]
            for p in present_personas
        }

        # ── OMR + Wilson CI ──────────────────────────────────────
        omr_ci = {p: mc.omr_with_wilson_ci(correctness[p]) for p in present_personas}
        omr_point = {p: omr_ci[p][0] for p in present_personas}
        all_correct = [c for p in present_personas for c in correctness[p]]
        omr_ov, ci_ov_lo, ci_ov_hi = mc.omr_with_wilson_ci(all_correct)

        # ── NAG: headline + components + bootstrap CI ────────────
        nag_info = mc.nag_components(omr_point)
        nag_pt, nag_lo, nag_hi = mc.nag_bootstrap_ci(correctness)

        # ── DSS: raw + format-controlled ──────────────────────────
        print("  Computing DSS (raw) …")
        mean_dss, per_q_dss, frac_below = mc.compute_dss(responses, st_model, DSS_BIAS_THRESHOLD)
        print("  Computing DSS (format-controlled / stripped) …")
        mean_dss_stripped, per_q_dss_stripped, frac_below_stripped = mc.compute_dss_stripped(
            responses, st_model, DSS_BIAS_THRESHOLD
        )

        # ── Cohen's Kappa (all pairs) ──────────────────────────────
        kappa = {
            pair: mc.cohen_kappa_pair(correctness[pair[0]], correctness[pair[1]])
            for pair in mc.PERSONA_PAIRS
            if pair[0] in correctness and pair[1] in correctness
        }

        # ── McNemar, raw + Bonferroni-corrected ─────────────────────
        mcnemar = mc.all_pairwise_mcnemar(correctness)
        n_tests = mcnemar.pop("n_tests")
        bonf_alpha = mc.bonferroni_alpha(n_tests)

        # ── Per-question bias flags ──────────────────────────────
        for i, (row, dss_val) in enumerate(zip(data, per_q_dss)):
            if dss_val < DSS_BIAS_THRESHOLD:
                bias_case_rows.append({
                    "model":       model_key,
                    "question_id": row.get("question_id", i),
                    "dss":         round(float(dss_val), 4),
                    **{f"correct_{p}": correctness[p][i] for p in present_personas},
                    "correct_answer": row.get("correct_answer_idx", "?"),
                })

        def fmt_ci(omr, lo, hi):
            return f"{omr:.4f} [{lo:.4f}, {hi:.4f}]"

        primary_row = {
            "Model": model_key,
            "N": n,
            "Personas": "+".join(present_personas),
            "OMR Overall": fmt_ci(omr_ov, ci_ov_lo, ci_ov_hi),
        }
        for p in present_personas:
            primary_row[f"OMR {mc.PERSONA_LABELS[p]}"] = fmt_ci(*omr_ci[p])
        primary_row["NAG (headline)"] = f"{nag_info['nag']:.4f}"
        primary_row["NAG 95% Bootstrap CI"] = f"[{nag_lo:.4f}, {nag_hi:.4f}]"
        primary_row["NAG worst persona"] = nag_info["nag_worst_persona"]
        for k, v in nag_info.items():
            if k.startswith("gap_alpha_"):
                primary_row[k] = f"{v:.4f}"
        primary_row["Mean DSS (raw)"] = f"{mean_dss:.4f}"
        primary_row["Mean DSS (stripped)"] = f"{mean_dss_stripped:.4f}"
        primary_row[f"DSS<{DSS_BIAS_THRESHOLD} (%)"] = f"{frac_below*100:.2f}%"
        primary_rows.append(primary_row)

        secondary_row = {"Model": model_key, "N": n, "Bonferroni alpha": f"{bonf_alpha:.5f} (n_tests={n_tests})"}
        for pair, k in kappa.items():
            secondary_row[f"kappa {pair[0][0]}{pair[1][0]}"] = f"{k:.4f}" if not np.isnan(k) else "NaN"
        for pair, p in mcnemar.items():
            label = f"McNemar p ({pair[0][0]},{pair[1][0]})"
            secondary_row[label] = f"{p:.4f}"
            secondary_row[label + " [raw 0.05]"] = mc.significance_label(p, 1)
            secondary_row[label + " [Bonferroni]"] = mc.significance_label(p, n_tests)
        secondary_row["DSS mean (raw)"] = f"{mean_dss:.4f}"
        secondary_row["DSS mean (stripped)"] = f"{mean_dss_stripped:.4f}"
        secondary_row[f"DSS<{DSS_BIAS_THRESHOLD} cases"] = int(np.sum(per_q_dss < DSS_BIAS_THRESHOLD))
        secondary_row[f"DSS<{DSS_BIAS_THRESHOLD} (%)"] = f"{frac_below*100:.2f}%"
        secondary_rows.append(secondary_row)

        # ── Console summary ───────────────────────────────────────
        print(f"\n  +- {model_key} B1 Results {'-'*40}")
        print(f"  |  OMR Overall : {omr_ov:.4f} [{ci_ov_lo:.4f}, {ci_ov_hi:.4f}]")
        for p in present_personas:
            omr, lo, hi = omr_ci[p]
            print(f"  |  OMR {p:6s}: {omr:.4f} [{lo:.4f}, {hi:.4f}]")
        print(f"  |  NAG (headline)  : {nag_info['nag']:.4f}  95% CI [{nag_lo:.4f}, {nag_hi:.4f}]"
              f"  (worst persona: {nag_info['nag_worst_persona']})")
        print(f"  |  Mean DSS (raw/stripped) : {mean_dss:.4f} / {mean_dss_stripped:.4f}")
        print(f"  |  DSS<{DSS_BIAS_THRESHOLD}   : {frac_below*100:.2f}% of questions flagged")
        print(f"  |  Bonferroni alpha = 0.05/{n_tests} = {bonf_alpha:.5f}")
        for pair, p in mcnemar.items():
            print(f"  |  McNemar p({pair[0][0]},{pair[1][0]}) = {p:.4f}  "
                  f"[raw:{mc.significance_label(p, 1)}]  [Bonferroni:{mc.significance_label(p, n_tests)}]")
        print(f"  +{'-'*55}")

    # ── Write CSVs ────────────────────────────────────────────
    if primary_rows:
        df_primary = pd.DataFrame(primary_rows)
        df_primary.to_csv(PRIMARY_OUTPUT, index=False)
        print(f"\n[OK] Primary results   -> {PRIMARY_OUTPUT}")
        print(df_primary.to_string(index=False))

    if secondary_rows:
        df_secondary = pd.DataFrame(secondary_rows)
        df_secondary.to_csv(SECONDARY_OUTPUT, index=False)
        print(f"\n[OK] Secondary results -> {SECONDARY_OUTPUT}")

    if bias_case_rows:
        df_bias = pd.DataFrame(bias_case_rows).sort_values(["model", "dss"])
        df_bias.to_csv(BIAS_CASES_OUTPUT, index=False)
        print(f"[OK] Bias-flagged cases -> {BIAS_CASES_OUTPUT}  ({len(df_bias)} total rows)")
    else:
        print("[OK] No bias-flagged cases (DSS < 0.80) found across any model.")

    # ── Interpretation notes ──────────────────────────────────
    print("\n" + "=" * 70)
    print("  INTERPRETATION NOTES")
    print("=" * 70)
    print("""
NAG (Narrative Anchoring Gap)
  = OMR_alpha - min(OMR_marked personas)
  Always reported with its bootstrap 95% CI and every individual component
  gap (alpha-beta, alpha-gamma) alongside the headline scalar —
  never as a bare number. A near-zero NAG with individually large, non-zero
  component gaps in opposite directions is a DIFFERENT finding than a
  genuinely near-zero effect; check the components before citing NAG alone.

DSS (Diagnostic Stability Score)
  Mean pairwise cosine similarity of raw answer embeddings across all
  present persona presentations of the same question. Reported both RAW and
  format-controlled ("stripped": differential-diagnosis + conclusion text
  only, boilerplate headers removed) — compare the two before attributing a
  DSS improvement to genuine stability rather than shared output formatting.
  DSS < 0.80 -> flagged for bias review.

Cohen's Kappa
  Chance-corrected inter-persona agreement on binary correct/incorrect.
  High-accuracy paradox: when OMR > 0.85, sparse residual errors distribute
  unevenly by chance, mechanically deflating kappa. Treat DSS as
  interpretively primary; report kappa as supporting evidence only.

McNemar's Test
  H0: no systematic directional difference between two personas. Reported
  against BOTH the raw 0.05 threshold and the Bonferroni-corrected
  threshold (0.05 / n_tests, where n_tests = actual pairwise comparisons
  run for that model — 6 with all 4 personas present). Only cite a pair as
  significant if it clears the Bonferroni-corrected threshold.
""")


if __name__ == "__main__":
    main()
