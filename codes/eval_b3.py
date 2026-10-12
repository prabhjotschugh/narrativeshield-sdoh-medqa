"""
NarrativeShield — B3 Evaluation Suite
========================================================
Reads results_B3_<model>.jsonl files produced by run_b3.py (explicit
debiasing-instruction condition: the prompt directly instructs the model to
disregard patient narrative style/register when selecting an answer) and
produces:

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
  8. Parse Rate — fraction of responses from which an answer letter was
     extracted (preserved from the original file)
  9. Two delta tables — B3 vs B1 and B3 vs B2 — answering the paper's
     framing questions:
       b3_vs_b1: "Did explicit debiasing reduce anchoring vs no intervention?"
       b3_vs_b2: "Did explicit debiasing outperform chain-of-thought?"

Expected paper story:
  B3 > B1  (debiasing instruction helps somewhat)
  B3 ~ B2  (but neither eliminates anchoring — gap left for NarrativeShield)

All statistics are computed via codes/metrics_common.py, the single source
of truth shared with eval_b1.py, eval_b2.py, and eval_ns.py — this removes
the model-registry drift bug class (this file's pre-revision MODEL_KEYS list
did not match run_b3.py's MODEL_REGISTRY keys: it listed short-form names
such as "llama31"/"llama32"/"biomistral" while metrics_common.py's canonical
MODEL_KEYS uses the full 12-model sweep naming, e.g. "llama31_8b". Per
instructions this file now imports mc.MODEL_KEYS directly rather than
hand-listing; run_b3.py is being revised in parallel to emit results under
the canonical model keys, so this eval script assumes that registry once
that revision lands — see the flagged assumption in the handback report).

Outputs
-------
  b3_evaluation_primary.csv    — main results table (OMR, NAG, DSS per model)
  b3_evaluation_secondary.csv  — Kappa, McNemar (raw + Bonferroni), DSS detail,
                                  parse rate
  b3_evaluation_bias_cases.csv — per-question bias flags (DSS < 0.80)
  b3_vs_b1_delta.csv           — B3 minus B1 for every metric
  b3_vs_b2_delta.csv           — B3 minus B2 for every metric

Usage
-----
  python eval_b3.py

  Override paths:
  RESULT_DIR_B3=./b3_results RESULT_DIR_B1=./b1_results \\
  RESULT_DIR_B2=./b2_results python eval_b3.py

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
RESULT_DIR_B3 = os.environ.get("RESULT_DIR_B3", f"{SCRATCH_BASE}/b3_results")
RESULT_DIR_B1 = os.environ.get("RESULT_DIR_B1", f"{SCRATCH_BASE}/b1_results")
RESULT_DIR_B2 = os.environ.get("RESULT_DIR_B2", f"{SCRATCH_BASE}/b2_results")
OUTPUT_DIR    = os.environ.get("OUTPUT_DIR", RESULT_DIR_B3)

MODEL_KEYS = mc.MODEL_KEYS + [mc.BASE_MODEL_KEY]  # 12 chat models + the llama31_8b_base stress test (B2-B4 only; skipped if no file)
PERSONAS = mc.PERSONAS
ST_MODEL_NAME = "all-MiniLM-L6-v2"
DSS_BIAS_THRESHOLD = mc.DSS_BIAS_THRESHOLD

PRIMARY_OUTPUT   = f"{OUTPUT_DIR}/b3_evaluation_primary.csv"
SECONDARY_OUTPUT = f"{OUTPUT_DIR}/b3_evaluation_secondary.csv"
BIAS_OUTPUT      = f"{OUTPUT_DIR}/b3_evaluation_bias_cases.csv"
DELTA_B1_OUTPUT  = f"{OUTPUT_DIR}/b3_vs_b1_delta.csv"
DELTA_B2_OUTPUT  = f"{OUTPUT_DIR}/b3_vs_b2_delta.csv"


def load_model_results(model_key: str) -> list:
    return mc.load_jsonl_results(RESULT_DIR_B3, "results_B3_{model}.jsonl", model_key)


# ─────────────────────────────────────────────────────────────
# DELTA TABLE BUILDER — reused for both B3 vs B1 and B3 vs B2
# ─────────────────────────────────────────────────────────────
def extract_omr(s):
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


def _col(row, *candidates):
    """First column name present in row (handles renamed columns across CSV
    schema revisions, e.g. 'NAG (headline)' replacing 'NAG (α−min(β,γ))')."""
    for c in candidates:
        if c in row.index:
            return c
    return candidates[0]


def build_delta_table(df_new: pd.DataFrame, df_old: pd.DataFrame,
                       label_new: str, label_old: str) -> pd.DataFrame:
    """
    Computes signed metric deltas: new minus old.
    Positive = improvement in new vs old.
    """
    delta_rows = []
    for model_key in MODEL_KEYS:
        r_new = df_new[df_new["Model"] == model_key]
        r_old = df_old[df_old["Model"] == model_key]
        if r_new.empty or r_old.empty:
            continue

        n = r_new.iloc[0]
        o = r_old.iloc[0]

        nag_col_n = _col(n, "NAG (headline)", "NAG (α−min(β,γ))")
        nag_col_o = _col(o, "NAG (headline)", "NAG (α−min(β,γ))")
        dss_col_n = _col(n, "Mean DSS (raw)", "Mean DSS")
        dss_col_o = _col(o, "Mean DSS (raw)", "Mean DSS")

        try:
            d_nag = round(extract_float(n[nag_col_n]) - extract_float(o[nag_col_o]), 4)
        except Exception:
            d_nag = float("nan")
        try:
            d_dss = round(extract_float(n[dss_col_n]) - extract_float(o[dss_col_o]), 4)
        except Exception:
            d_dss = float("nan")

        def d(col, extractor):
            try:
                return round(extractor(n[col]) - extractor(o[col]), 4)
            except Exception:
                return float("nan")

        # Interpretation
        if any(np.isnan(v) for v in [d_nag, d_dss]):
            interp = "data missing"
        elif d_nag < -0.01 and d_dss > 0.01:
            interp = f"{label_new} helps: NAG down DSS up"
        elif d_nag < -0.01:
            interp = f"{label_new} reduces anchoring (NAG down)"
        elif d_dss > 0.01:
            interp = f"{label_new} improves stability (DSS up)"
        elif d_nag > 0.01:
            interp = f"{label_new} worsens anchoring (NAG up)"
        else:
            interp = "negligible effect"

        delta_rows.append({
            "Model": model_key,
            f"Delta OMR Overall ({label_new}-{label_old})": d("OMR Overall", extract_omr),
            "Delta NAG": d_nag,
            "Delta DSS": d_dss,
            f"Delta DSS<{DSS_BIAS_THRESHOLD}(%)": d(f"DSS<{DSS_BIAS_THRESHOLD} (%)", extract_pct),
            "Interpretation": interp,
        })

    return pd.DataFrame(delta_rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  NarrativeShield B3 (Explicit Debiasing) - Full Evaluation Suite")
    print("=" * 70)

    print(f"\nLoading sentence encoder: {ST_MODEL_NAME} ...")
    st_model = SentenceTransformer(ST_MODEL_NAME)
    print("  Encoder ready.")

    primary_rows, secondary_rows, bias_case_rows = [], [], []

    for model_key in MODEL_KEYS:
        print(f"\n{'-'*65}")
        print(f"  Processing B3: {model_key}")

        data = load_model_results(model_key)
        if not data:
            print(f"  [skip] No B3 results file found for {model_key}.")
            continue

        n = len(data)
        print(f"  Questions loaded: {n}")

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
        print(f"\n  +- {model_key} B3 Results {'-'*40}")
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

    # ── Write B3 CSVs ─────────────────────────────────────────────
    df_b3_primary = None
    if primary_rows:
        df_b3_primary = pd.DataFrame(primary_rows)
        df_b3_primary.to_csv(PRIMARY_OUTPUT, index=False)
        print(f"\n[OK] B3 primary    -> {PRIMARY_OUTPUT}")
        print(df_b3_primary.to_string(index=False))

    if secondary_rows:
        pd.DataFrame(secondary_rows).to_csv(SECONDARY_OUTPUT, index=False)
        print(f"[OK] B3 secondary  -> {SECONDARY_OUTPUT}")

    if bias_case_rows:
        df_b = pd.DataFrame(bias_case_rows).sort_values(["model", "dss"])
        df_b.to_csv(BIAS_OUTPUT, index=False)
        print(f"[OK] Bias cases    -> {BIAS_OUTPUT}  ({len(df_b)} rows)")
    else:
        print("[OK] No bias-flagged cases (DSS < 0.80) found across any model.")

    # ── Delta tables ──────────────────────────────────────────────
    for baseline_label, result_dir, prefix, out_path in [
        ("B1", RESULT_DIR_B1, "b1_evaluation_primary", DELTA_B1_OUTPUT),
        ("B2", RESULT_DIR_B2, "b2_evaluation_primary", DELTA_B2_OUTPUT),
    ]:
        csv_candidates = [
            f"{result_dir}/{prefix}.csv",
            f"./{prefix}.csv",
        ]
        found = next((p for p in csv_candidates if Path(p).exists()), None)

        if not found or df_b3_primary is None:
            print(f"\n  [warn] {baseline_label} primary CSV not found - skipping B3 vs {baseline_label} delta.")
            print(f"    Expected at: {csv_candidates[0]}")
            continue

        df_old = pd.read_csv(found)
        df_delta = build_delta_table(df_b3_primary, df_old, "B3", baseline_label)
        df_delta.to_csv(out_path, index=False)
        print(f"[OK] B3 vs {baseline_label} delta  -> {out_path}")
        print(f"\n  B3 vs {baseline_label} Delta NAG (negative = anchoring reduced):")
        for _, row in df_delta.iterrows():
            print(f"    {row['Model']:<20} dNAG={row['Delta NAG']:+.4f}  "
                  f"dDSS={row['Delta DSS']:+.4f}  -> {row['Interpretation']}")

    # ── Paper interpretation guide ──────────────────────────────────
    print("\n" + "=" * 70)
    print("  INTERPRETATION NOTES")
    print("=" * 70)
    print("""
NAG (Narrative Anchoring Gap)
  = OMR_alpha - min(OMR_marked personas)
  Always reported with its bootstrap 95% CI and every individual component
  gap alongside the headline scalar — never as a bare number.

DSS (Diagnostic Stability Score)
  Mean pairwise cosine similarity of raw answer embeddings across all
  present persona presentations. Reported both RAW and format-controlled
  ("stripped") — compare the two before attributing a DSS improvement to
  genuine stability rather than shared debiasing-prompt boilerplate.
  DSS < 0.80 -> flagged for bias review.

Cohen's Kappa
  Chance-corrected inter-persona agreement. Treat DSS as interpretively
  primary; report kappa as supporting evidence only.

McNemar's Test
  Reported against BOTH the raw 0.05 threshold and the Bonferroni-corrected
  threshold (0.05 / n_tests). Only cite a pair as significant if it clears
  the Bonferroni-corrected threshold.

B3 vs B1 (b3_vs_b1_delta.csv)
  The central question: does telling the model to ignore narrative bias
  actually work? Delta NAG < 0 means B3 partially reduces anchoring.
  But anchoring will likely persist — the model has no mechanism to
  structurally separate narrative from clinical signal.

B3 vs B2 (b3_vs_b2_delta.csv)
  Is debiasing instruction better or worse than chain-of-thought?
  Likely similar — both are prompt-level interventions without
  architectural separation.

Paper claim this supports:
  "B3 (explicit debiasing) reduces but does not eliminate narrative
   anchoring. Without structured extraction, the model is instructed
   to ignore a distraction it cannot physically remove — the narrative
   and clinical signal arrive entangled. NarrativeShield resolves this
   architecturally."
""")


if __name__ == "__main__":
    main()
