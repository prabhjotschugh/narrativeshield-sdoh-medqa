"""
NarrativeShield — B2 Evaluation Suite
========================================================
Reads results_B2_<model>.jsonl files produced by run_b2.py (Chain-of-Thought
condition: models are prompted for step-by-step reasoning ending in an
"Answer: [letter]" line) and produces:

  1. Per-model x per-persona OMR with 95% Wilson score CIs (alpha/beta/gamma)
  2. Overall OMR (macro across personas)
  3. Diagnostic Stability Score (DSS) — mean pairwise cosine similarity
     of answer embeddings across all present persona presentations, plus a
     format-controlled ("stripped") variant
  4. Fraction of cases with DSS < 0.80 (bias-flagged cases)
  5. Cohen's Kappa (chance-corrected inter-persona agreement) for all pairs
  6. McNemar's test (continuity-corrected), reported against BOTH raw 0.05
     and Bonferroni-corrected thresholds (fixes the same unapplied-Bonferroni
     bug identified and fixed in eval_b1.py — this file inherited the same
     drift since it copy-pasted b1's pre-revision statistical helpers)
  7. Narrative Anchoring Gap (NAG) — OMR_alpha minus min(OMR_marked) — always
     reported alongside every individual component gap and a paired
     bootstrap 95% CI on NAG itself, never as a bare scalar
  8. Parse Rate — fraction of responses from which the CoT "Answer: [letter]"
     line yielded an extracted letter (preserved from the original file;
     the extraction itself happens in run_b2.py, this script only reports
     on the resulting `extracted_answer` field)
  9. B2 vs B1 delta table — signed per-metric deltas answering the paper's
     framing question: "did CoT reduce narrative anchoring relative to the
     no-intervention baseline?"

All statistics are computed via codes/metrics_common.py, the single source
of truth shared with eval_b1.py, eval_b3.py, and eval_ns.py — this removes
the model-registry drift bug class (this file's pre-revision MODEL_KEYS list
did not match run_b2.py's MODEL_REGISTRY keys at all: it listed short-form
names such as "llama31"/"llama32"/"biomistral" while metrics_common.py's
canonical MODEL_KEYS uses the full 12-model sweep naming, e.g. "llama31_8b".
Per instructions this file now imports mc.MODEL_KEYS directly rather than
hand-listing; run_b2.py is being revised in parallel to emit results under
the canonical model keys, so this eval script assumes that registry once
that revision lands — see the flagged assumption in the handback report).

Outputs
-------
  b2_evaluation_primary.csv    — main results table (OMR, NAG, DSS per model)
  b2_evaluation_secondary.csv  — Kappa, McNemar (raw + Bonferroni), DSS detail,
                                  parse rate
  b2_evaluation_bias_cases.csv — per-question bias flags (DSS < 0.80)
  b2_vs_b1_delta.csv           — B2 minus B1 for every metric (requires B1 CSVs)

The delta table is the key table for the paper:
  "Did CoT reduce narrative anchoring relative to B1?"

Usage
-----
  python eval_b2.py

  Set RESULT_DIR_B2 and RESULT_DIR_B1 if not running on HPC scratch.
  e.g.  RESULT_DIR_B2=./b2_results RESULT_DIR_B1=./b1_results python eval_b2.py

Dependencies
------------
  pip install sentence-transformers scikit-learn statsmodels pandas numpy tqdm
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

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
SCRATCH_BASE  = "/path/to/scratch"
RESULT_DIR_B2 = os.environ.get("RESULT_DIR_B2", f"{SCRATCH_BASE}/b2_results")
RESULT_DIR_B1 = os.environ.get("RESULT_DIR_B1", f"{SCRATCH_BASE}/b1_results")
OUTPUT_DIR    = os.environ.get("OUTPUT_DIR", RESULT_DIR_B2)

MODEL_KEYS = mc.MODEL_KEYS + [mc.BASE_MODEL_KEY]  # 12 chat models + the llama31_8b_base stress test (B2-B4 only; skipped if no file)
PERSONAS = mc.PERSONAS
ST_MODEL_NAME = "all-MiniLM-L6-v2"
DSS_BIAS_THRESHOLD = mc.DSS_BIAS_THRESHOLD

PRIMARY_OUTPUT    = f"{OUTPUT_DIR}/b2_evaluation_primary.csv"
SECONDARY_OUTPUT  = f"{OUTPUT_DIR}/b2_evaluation_secondary.csv"
BIAS_OUTPUT       = f"{OUTPUT_DIR}/b2_evaluation_bias_cases.csv"
DELTA_OUTPUT      = f"{OUTPUT_DIR}/b2_vs_b1_delta.csv"


def load_model_results(model_key: str) -> list:
    return mc.load_jsonl_results(RESULT_DIR_B2, "results_B2_{model}.jsonl", model_key)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  NarrativeShield B2 (Chain-of-Thought) - Full Evaluation Suite")
    print("=" * 70)

    print(f"\nLoading sentence encoder: {ST_MODEL_NAME} ...")
    st_model = SentenceTransformer(ST_MODEL_NAME)
    print("  Encoder ready.")

    primary_rows, secondary_rows, bias_case_rows = [], [], []

    for model_key in MODEL_KEYS:
        print(f"\n{'-'*65}")
        print(f"  Processing B2: {model_key}")

        data = load_model_results(model_key)
        if not data:
            print(f"  [skip] No B2 results file found for {model_key}.")
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

        # CoT-specific: the "Answer: [letter]" extraction happens in
        # run_b2.py and is stored per-persona as `extracted_answer`. Parse
        # rate = fraction of (question, persona) pairs where extraction
        # succeeded (non-None). Preserved exactly from the original file —
        # this is the one B2-specific metric that differs from B1.
        all_extracted = [
            mc.safe_get(r, "personas_eval", p, "extracted_answer", default=None)
            for r in data for p in present_personas
        ]
        parse_rate = sum(1 for x in all_extracted if x is not None) / max(len(all_extracted), 1)

        # ── OMR + Wilson CI ──────────────────────────────────────
        omr_ci = {p: mc.omr_with_wilson_ci(correctness[p]) for p in present_personas}
        omr_point = {p: omr_ci[p][0] for p in present_personas}
        all_correct = [c for p in present_personas for c in correctness[p]]
        omr_ov, ci_ov_lo, ci_ov_hi = mc.omr_with_wilson_ci(all_correct)

        # ── NAG: headline + components + bootstrap CI ────────────
        nag_info = mc.nag_components(omr_point)
        nag_pt, nag_lo, nag_hi = mc.nag_bootstrap_ci(correctness)

        # ── DSS: raw + format-controlled ──────────────────────────
        print("  Computing DSS (raw) ...")
        mean_dss, per_q_dss, frac_below = mc.compute_dss(responses, st_model, DSS_BIAS_THRESHOLD)
        print("  Computing DSS (format-controlled / stripped) ...")
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

        secondary_row = {
            "Model": model_key, "N": n,
            "Parse Rate": f"{parse_rate*100:.2f}%",
            "Bonferroni alpha": f"{bonf_alpha:.5f} (n_tests={n_tests})",
        }
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
        print(f"\n  +- {model_key} B2 Results {'-'*40}")
        print(f"  |  Parse rate  : {parse_rate*100:.1f}%")
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

    # ── Write B2 primary + secondary ─────────────────────────────
    if primary_rows:
        df_p = pd.DataFrame(primary_rows)
        df_p.to_csv(PRIMARY_OUTPUT, index=False)
        print(f"\n[OK] B2 primary   -> {PRIMARY_OUTPUT}")
        print(df_p.to_string(index=False))

    if secondary_rows:
        df_s = pd.DataFrame(secondary_rows)
        df_s.to_csv(SECONDARY_OUTPUT, index=False)
        print(f"[OK] B2 secondary -> {SECONDARY_OUTPUT}")

    if bias_case_rows:
        df_b = pd.DataFrame(bias_case_rows).sort_values(["model", "dss"])
        df_b.to_csv(BIAS_OUTPUT, index=False)
        print(f"[OK] Bias cases   -> {BIAS_OUTPUT}  ({len(df_b)} rows)")
    else:
        print("[OK] No bias-flagged cases (DSS < 0.80) found across any model.")

    # ── B2 vs B1 delta table ─────────────────────────────────────
    # Loads B1 primary CSV (must exist) and computes signed deltas.
    # Positive delta = B2 improved over B1.
    b1_primary_path = f"{RESULT_DIR_B1}/b1_evaluation_primary.csv"
    if not Path(b1_primary_path).exists():
        b1_primary_path = "./b1_evaluation_primary.csv"

    if Path(b1_primary_path).exists() and primary_rows:
        print(f"\n  Computing B2 vs B1 deltas from: {b1_primary_path}")
        df_b1 = pd.read_csv(b1_primary_path)
        df_b2 = pd.DataFrame(primary_rows)

        def extract_omr(s):
            """Pull the float before the first space from '0.6417 [...]'."""
            try:
                return float(str(s).split()[0])
            except Exception:
                return float("nan")

        def extract_float(s):
            try:
                return float(str(s))
            except Exception:
                return float("nan")

        def extract_pct(s):
            try:
                return float(str(s).replace("%", ""))
            except Exception:
                return float("nan")

        delta_rows = []
        for model_key in MODEL_KEYS:
            row_b1 = df_b1[df_b1["Model"] == model_key]
            row_b2 = df_b2[df_b2["Model"] == model_key]
            if row_b1.empty or row_b2.empty:
                continue

            r1 = row_b1.iloc[0]
            r2 = row_b2.iloc[0]

            def d(col, extractor):
                try:
                    return round(extractor(r2[col]) - extractor(r1[col]), 4)
                except Exception:
                    return float("nan")

            # NAG columns renamed to "NAG (headline)" in the revised primary
            # CSVs (was "NAG (α−min(β,γ))" pre-revision) — try both so the
            # delta table still works against either a freshly regenerated
            # B1 CSV or an older one left on disk.
            nag_col_b1 = "NAG (headline)" if "NAG (headline)" in r1.index else "NAG (α−min(β,γ))"
            nag_col_b2 = "NAG (headline)" if "NAG (headline)" in r2.index else "NAG (α−min(β,γ))"
            try:
                d_nag = round(extract_float(r2[nag_col_b2]) - extract_float(r1[nag_col_b1]), 4)
            except Exception:
                d_nag = float("nan")

            dss_col_b1 = "Mean DSS (raw)" if "Mean DSS (raw)" in r1.index else "Mean DSS"
            dss_col_b2 = "Mean DSS (raw)" if "Mean DSS (raw)" in r2.index else "Mean DSS"
            try:
                d_dss = round(extract_float(r2[dss_col_b2]) - extract_float(r1[dss_col_b1]), 4)
            except Exception:
                d_dss = float("nan")

            delta_rows.append({
                "Model":              model_key,
                "Delta OMR Overall":  d("OMR Overall", extract_omr),
                "Delta NAG":          d_nag,
                "Delta DSS":          d_dss,
                f"Delta DSS<{DSS_BIAS_THRESHOLD}(%)": d(f"DSS<{DSS_BIAS_THRESHOLD} (%)", extract_pct),
                "Interpretation": "",
            })

        # Auto-fill interpretation column
        for row in delta_rows:
            nag_d = row["Delta NAG"]
            dss_d = row["Delta DSS"]
            if np.isnan(nag_d) or np.isnan(dss_d):
                row["Interpretation"] = "data missing"
            elif nag_d < -0.01 and dss_d > 0.01:
                row["Interpretation"] = "CoT helps: NAG down DSS up"
            elif nag_d < -0.01:
                row["Interpretation"] = "CoT reduces anchoring (NAG down)"
            elif dss_d > 0.01:
                row["Interpretation"] = "CoT improves stability (DSS up)"
            elif nag_d > 0.01:
                row["Interpretation"] = "CoT worsens anchoring (NAG up)"
            else:
                row["Interpretation"] = "CoT negligible effect"

        if delta_rows:
            df_delta = pd.DataFrame(delta_rows)
            df_delta.to_csv(DELTA_OUTPUT, index=False)
            print(f"[OK] B2 vs B1 delta -> {DELTA_OUTPUT}")
            print("\n  B2 vs B1 Delta NAG (negative = anchoring reduced by CoT):")
            for row in delta_rows:
                print(f"    {row['Model']:<20} dNAG={row['Delta NAG']:+.4f}  "
                      f"dDSS={row['Delta DSS']:+.4f}  -> {row['Interpretation']}")
    else:
        print(f"\n  [warn] B1 primary CSV not found at {b1_primary_path}")
        print("    Run eval_b1.py first, then re-run eval_b2.py for the delta table.")

    # ── Interpretation notes ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("  INTERPRETATION NOTES")
    print("=" * 70)
    print("""
NAG (Narrative Anchoring Gap)
  = OMR_alpha - min(OMR_marked personas)
  Always reported with its bootstrap 95% CI and every individual component
  gap alongside the headline scalar — never as a bare number. A near-zero
  NAG with individually large, non-zero component gaps in opposite
  directions is a DIFFERENT finding than a genuinely near-zero effect;
  check the components before citing NAG alone.

DSS (Diagnostic Stability Score)
  Mean pairwise cosine similarity of raw answer embeddings across all
  present persona presentations of the same question. Reported both RAW and
  format-controlled ("stripped") — compare the two before attributing a
  DSS improvement to genuine stability rather than shared CoT output
  formatting (B2's "Answer: [letter]" scaffolding is exactly the kind of
  shared boilerplate the stripped variant controls for).
  DSS < 0.80 -> flagged for bias review.

Parse Rate
  Fraction of (question, persona) responses from which a letter could be
  extracted via the CoT "Answer: [letter]" line. B2 should have near-100%
  parse rate because the prompt explicitly asks for that line. A low parse
  rate for a given model indicates its few-shot examples may need
  adjustment in run_b2.py, not an eval bug.

Cohen's Kappa
  Chance-corrected inter-persona agreement on binary correct/incorrect.
  High-accuracy paradox: when OMR > 0.85, sparse residual errors distribute
  unevenly by chance, mechanically deflating kappa. Treat DSS as
  interpretively primary; report kappa as supporting evidence only.

McNemar's Test
  H0: no systematic directional difference between two personas. Reported
  against BOTH the raw 0.05 threshold and the Bonferroni-corrected
  threshold (0.05 / n_tests, where n_tests = actual pairwise comparisons
  run for that model). Only cite a pair as significant if it clears the
  Bonferroni-corrected threshold.

Delta NAG / Delta DSS (B2 minus B1)
  Negative Delta NAG = CoT reduced the articulation advantage (good).
  Positive Delta DSS = CoT made responses more stable across personas (good).
  Expected result supporting the paper's claim: B2 should show modest
  improvement over B1, but not eliminate anchoring — that gap is left for
  NarrativeShield to close. If B2 fully eliminated anchoring, the paper's
  main contribution would be weakened. Partial improvement is the ideal
  B2 story.
""")


if __name__ == "__main__":
    main()
