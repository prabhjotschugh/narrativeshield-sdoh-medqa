"""
NarrativeShield — B1 Evaluation Suite (Final)
==============================================
Reads results_B1_<model>.jsonl files produced by run_b1.py and produces:

  1. Per-model × per-persona OMR with 95% Wilson score CIs
  2. Overall OMR (macro across personas)
  3. Diagnostic Stability Score (DSS) — mean pairwise cosine similarity
     of answer embeddings across all 3 persona presentations
  4. Fraction of cases with DSS < 0.80 (bias-flagged cases)
  5. Cohen's Kappa (chance-corrected inter-persona agreement) for all 3 pairs
  6. McNemar's test (continuity-corrected) for directional bias between pairs
  7. Narrative Anchoring Gap (NAG) — OMR_alpha minus min(OMR_beta, OMR_gamma)
     — primary equity metric: how much does clinical articulation advantage you?

Outputs
-------
  b1_evaluation_primary.csv    — the main results table for the paper
  b1_evaluation_secondary.csv  — Kappa, McNemar, DSS detail per model
  b1_evaluation_bias_cases.csv — per-question bias flags (DSS < 0.80)

Usage
-----
  python eval_b1.py

Dependencies
------------
  pip install sentence-transformers scikit-learn statsmodels pandas numpy tqdm
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
RESULT_DIR         = os.environ.get("RESULT_DIR", "/scratch/10778/prabhjotschugh/b1_results")
OUTPUT_DIR         = os.environ.get("OUTPUT_DIR", RESULT_DIR)

# Keys must match MODEL_REGISTRY in run_b1.py
MODEL_KEYS = [
    "llama31",
    "llama32",
    "mistral7b",
    "ministral3b",
    "qwen25",
    "qwen35",
    "gemma3_12b",
    "gemma4_e4b",
    "biomistral",
]

# Sentence encoder for DSS — domain-general MiniLM is appropriate here
# because DSS measures *output response similarity*, not clinical knowledge
ST_MODEL_NAME      = "all-MiniLM-L6-v2"
DSS_BIAS_THRESHOLD = 0.80   # cases below this are flagged for bias review

PRIMARY_OUTPUT     = f"{OUTPUT_DIR}/b1_evaluation_primary.csv"
SECONDARY_OUTPUT   = f"{OUTPUT_DIR}/b1_evaluation_secondary.csv"
BIAS_CASES_OUTPUT  = f"{OUTPUT_DIR}/b1_evaluation_bias_cases.csv"

PERSONAS           = ["alpha", "beta", "gamma"]
PERSONA_LABELS     = {
    "alpha": "Pα Control (High Literacy)",
    "beta":  "Pβ Socioeconomic",
    "gamma": "Pγ Cultural",
}

# ─────────────────────────────────────────────────────────────
# STATISTICAL HELPERS
# ─────────────────────────────────────────────────────────────
try:
    from statsmodels.stats.proportion import proportion_confint
    from statsmodels.stats.contingency_tables import mcnemar as statsmodels_mcnemar
    from sklearn.metrics import cohen_kappa_score
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    print("Install with:  pip install sentence-transformers scikit-learn statsmodels")
    sys.exit(1)


def omr_with_wilson_ci(correct_list: list, alpha: float = 0.05):
    """
    Option Match Rate + 95% Wilson score confidence interval.
    Returns (omr, lower, upper).
    """
    n   = len(correct_list)
    k   = sum(correct_list)
    if n == 0:
        return 0.0, 0.0, 0.0
    omr = k / n
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return omr, lo, hi


def cohen_kappa_pair(a: list, b: list) -> float:
    """
    Cohen's Kappa between two binary arrays.
    Returns nan if all labels are identical (kappa undefined).
    """
    if len(set(a)) == 1 and len(set(b)) == 1:
        return float("nan")
    try:
        return cohen_kappa_score(a, b)
    except Exception:
        return float("nan")


def mcnemar_pvalue(a: list, b: list) -> float:
    """
    Continuity-corrected McNemar's test for directional bias.
    H0: no systematic directional difference between personas a and b.
    Returns p-value. p < 0.05 → significant directional bias.
    """
    both     = sum(x and y     for x, y in zip(a, b))
    a_only   = sum(x and not y for x, y in zip(a, b))
    b_only   = sum(not x and y for x, y in zip(a, b))
    neither  = sum(not x and not y for x, y in zip(a, b))
    table    = [[both, a_only], [b_only, neither]]
    try:
        result = statsmodels_mcnemar(table, exact=False, correction=True)
        return result.pvalue
    except Exception:
        return float("nan")


# ─────────────────────────────────────────────────────────────
# DSS COMPUTATION
# ─────────────────────────────────────────────────────────────
def compute_dss(
    responses_alpha: list,
    responses_beta:  list,
    responses_gamma: list,
    st_model,
) -> tuple:
    """
    Computes Diagnostic Stability Score per question and aggregates.

    DSS per question = mean of 3 pairwise cosine similarities
    (α-β, α-γ, β-γ) of the model's raw response embeddings.

    Returns
    -------
    mean_dss          : float — overall DSS across all questions
    per_question_dss  : np.ndarray — DSS value per question
    frac_below_thresh : float — fraction of questions with DSS < 0.80
    """
    print("  Encoding alpha responses …")
    emb_a = st_model.encode(responses_alpha, batch_size=256, show_progress_bar=False,
                             convert_to_numpy=True, normalize_embeddings=True)
    print("  Encoding beta responses …")
    emb_b = st_model.encode(responses_beta,  batch_size=256, show_progress_bar=False,
                             convert_to_numpy=True, normalize_embeddings=True)
    print("  Encoding gamma responses …")
    emb_g = st_model.encode(responses_gamma, batch_size=256, show_progress_bar=False,
                             convert_to_numpy=True, normalize_embeddings=True)

    # Because embeddings are L2-normalised, dot product == cosine similarity
    sim_ab = np.einsum("ij,ij->i", emb_a, emb_b)
    sim_ag = np.einsum("ij,ij->i", emb_a, emb_g)
    sim_bg = np.einsum("ij,ij->i", emb_b, emb_g)

    per_q   = (sim_ab + sim_ag + sim_bg) / 3.0
    mean    = float(np.mean(per_q))
    frac_lo = float(np.mean(per_q < DSS_BIAS_THRESHOLD))
    return mean, per_q, frac_lo


# ─────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────
def load_model_results(model_key: str) -> list:
    """Load JSONL for a model; returns list of result dicts."""
    candidate_paths = [
        f"{RESULT_DIR}/results_B1_{model_key}.jsonl",
        f"results_B1_{model_key}.jsonl",           # local fallback
    ]
    for path in candidate_paths:
        p = Path(path)
        if p.exists():
            rows = []
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            return rows
    return []


# ─────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  NarrativeShield B1 — Full Evaluation Suite")
    print("=" * 70)

    # Load sentence encoder once, shared across all models
    print(f"\nLoading sentence encoder: {ST_MODEL_NAME} …")
    st_model = SentenceTransformer(ST_MODEL_NAME)
    print("  Encoder ready.")

    primary_rows   = []
    secondary_rows = []
    bias_case_rows = []

    for model_key in MODEL_KEYS:
        print(f"\n{'─'*65}")
        print(f"  Processing: {model_key}")

        data = load_model_results(model_key)
        if not data:
            print(f"  ⚠ No results file found for {model_key}. Skipping.")
            continue

        n = len(data)
        print(f"  Questions loaded: {n}")

        # ── Extract correctness arrays ──────────────────────────
        # Guard against missing persona keys (partial runs)
        def safe_correct(item, p):
            try:
                return bool(item["personas_eval"][p]["is_correct"])
            except KeyError:
                return False

        def safe_response(item, p):
            try:
                return str(item["personas_eval"][p]["raw_response"] or "")
            except KeyError:
                return ""

        c_alpha = [safe_correct(r, "alpha") for r in data]
        c_beta  = [safe_correct(r, "beta")  for r in data]
        c_gamma = [safe_correct(r, "gamma") for r in data]

        r_alpha = [safe_response(r, "alpha") for r in data]
        r_beta  = [safe_response(r, "beta")  for r in data]
        r_gamma = [safe_response(r, "gamma") for r in data]

        # ── OMR + Wilson CI ──────────────────────────────────────
        omr_a, ci_a_lo, ci_a_hi = omr_with_wilson_ci(c_alpha)
        omr_b, ci_b_lo, ci_b_hi = omr_with_wilson_ci(c_beta)
        omr_g, ci_g_lo, ci_g_hi = omr_with_wilson_ci(c_gamma)

        all_correct = c_alpha + c_beta + c_gamma
        omr_ov, ci_ov_lo, ci_ov_hi = omr_with_wilson_ci(all_correct)

        # ── Narrative Anchoring Gap ──────────────────────────────
        # How much does clinical articulation advantage you?
        # Positive value = anchoring present; 0 = no anchoring
        nag = omr_a - min(omr_b, omr_g)

        # ── DSS ──────────────────────────────────────────────────
        print("  Computing DSS …")
        mean_dss, per_q_dss, frac_below = compute_dss(r_alpha, r_beta, r_gamma, st_model)

        # ── Cohen's Kappa ────────────────────────────────────────
        kappa_ab = cohen_kappa_pair(c_alpha, c_beta)
        kappa_ag = cohen_kappa_pair(c_alpha, c_gamma)
        kappa_bg = cohen_kappa_pair(c_beta,  c_gamma)

        # ── McNemar ──────────────────────────────────────────────
        p_ab = mcnemar_pvalue(c_alpha, c_beta)
        p_ag = mcnemar_pvalue(c_alpha, c_gamma)
        p_bg = mcnemar_pvalue(c_beta,  c_gamma)

        # ── Per-question bias flags ──────────────────────────────
        for i, (row, dss_val) in enumerate(zip(data, per_q_dss)):
            if dss_val < DSS_BIAS_THRESHOLD:
                bias_case_rows.append(
                    {
                        "model":       model_key,
                        "question_id": row.get("question_id", i),
                        "dss":         round(float(dss_val), 4),
                        "correct_alpha": c_alpha[i],
                        "correct_beta":  c_beta[i],
                        "correct_gamma": c_gamma[i],
                        "correct_answer": row.get("correct_answer_idx", "?"),
                    }
                )

        def fmt_ci(omr, lo, hi):
            return f"{omr:.4f} [{lo:.4f}, {hi:.4f}]"

        def sig(p):
            if np.isnan(p):
                return "NaN"
            return "SIG" if p < 0.05 else "ns"

        primary_rows.append(
            {
                "Model":                  model_key,
                "N":                      n,
                "OMR Overall":            fmt_ci(omr_ov, ci_ov_lo, ci_ov_hi),
                "OMR Pα Control":         fmt_ci(omr_a, ci_a_lo, ci_a_hi),
                "OMR Pβ Socioeconomic":   fmt_ci(omr_b, ci_b_lo, ci_b_hi),
                "OMR Pγ Cultural":        fmt_ci(omr_g, ci_g_lo, ci_g_hi),
                "NAG (α−min(β,γ))":       f"{nag:.4f}",
                "Mean DSS":               f"{mean_dss:.4f}",
                f"DSS<{DSS_BIAS_THRESHOLD} (%)": f"{frac_below*100:.2f}%",
            }
        )

        secondary_rows.append(
            {
                "Model":                  model_key,
                "N":                      n,
                "κ (α,β)":                f"{kappa_ab:.4f}" if not np.isnan(kappa_ab) else "NaN",
                "κ (α,γ)":                f"{kappa_ag:.4f}" if not np.isnan(kappa_ag) else "NaN",
                "κ (β,γ)":                f"{kappa_bg:.4f}" if not np.isnan(kappa_bg) else "NaN",
                "McNemar p (α,β)":        f"{p_ab:.4f}  [{sig(p_ab)}]",
                "McNemar p (α,γ)":        f"{p_ag:.4f}  [{sig(p_ag)}]",
                "McNemar p (β,γ)":        f"{p_bg:.4f}  [{sig(p_bg)}]",
                "DSS mean":               f"{mean_dss:.4f}",
                f"DSS<{DSS_BIAS_THRESHOLD} cases": int(sum(per_q_dss < DSS_BIAS_THRESHOLD)),
                f"DSS<{DSS_BIAS_THRESHOLD} (%)":   f"{frac_below*100:.2f}%",
            }
        )

        # ── Console summary ───────────────────────────────────────
        print(f"\n  ┌─ {model_key} B1 Results {'─'*40}")
        print(f"  │  OMR Overall : {omr_ov:.4f} [{ci_ov_lo:.4f}, {ci_ov_hi:.4f}]")
        print(f"  │  OMR Pα      : {omr_a:.4f} [{ci_a_lo:.4f}, {ci_a_hi:.4f}]")
        print(f"  │  OMR Pβ      : {omr_b:.4f} [{ci_b_lo:.4f}, {ci_b_hi:.4f}]")
        print(f"  │  OMR Pγ      : {omr_g:.4f} [{ci_g_lo:.4f}, {ci_g_hi:.4f}]")
        print(f"  │  NAG         : {nag:.4f}  (positive = anchoring present)")
        print(f"  │  Mean DSS    : {mean_dss:.4f}")
        print(f"  │  DSS<{DSS_BIAS_THRESHOLD}   : {frac_below*100:.2f}% of questions flagged")
        print(f"  │  κ(α,β) κ(α,γ) κ(β,γ) : {kappa_ab:.3f}  {kappa_ag:.3f}  {kappa_bg:.3f}")
        print(f"  │  McNemar p(α,β) p(α,γ) p(β,γ): {p_ab:.4f} [{sig(p_ab)}]  "
              f"{p_ag:.4f} [{sig(p_ag)}]  {p_bg:.4f} [{sig(p_bg)}]")
        print(f"  └{'─'*55}")

    # ── Write CSVs ────────────────────────────────────────────
    if primary_rows:
        df_primary = pd.DataFrame(primary_rows)
        df_primary.to_csv(PRIMARY_OUTPUT, index=False)
        print(f"\n✓ Primary results   → {PRIMARY_OUTPUT}")
        print(df_primary.to_string(index=False))

    if secondary_rows:
        df_secondary = pd.DataFrame(secondary_rows)
        df_secondary.to_csv(SECONDARY_OUTPUT, index=False)
        print(f"\n✓ Secondary results → {SECONDARY_OUTPUT}")

    if bias_case_rows:
        df_bias = pd.DataFrame(bias_case_rows)
        df_bias = df_bias.sort_values(["model", "dss"])
        df_bias.to_csv(BIAS_CASES_OUTPUT, index=False)
        print(f"✓ Bias-flagged cases → {BIAS_CASES_OUTPUT}  ({len(df_bias)} total rows)")
    else:
        print("✓ No bias-flagged cases (DSS < 0.80) found across any model.")

    # ── Interpretation notes ──────────────────────────────────
    print("\n" + "=" * 70)
    print("  INTERPRETATION NOTES")
    print("=" * 70)
    print("""
NAG (Narrative Anchoring Gap)
  = OMR_alpha − min(OMR_beta, OMR_gamma)
  Positive → standard clinical language yields higher accuracy than at least
  one equity persona. This is the primary equity metric.
  Values > 0.05 are clinically meaningful.

DSS (Diagnostic Stability Score)
  Mean pairwise cosine similarity of raw answer embeddings across the 3
  persona presentations of the same question.
  DSS = 1.0 → perfectly stable responses regardless of how the patient sounds.
  DSS < 0.80 → flagged for bias review.
  DSS is the primary STABILITY metric (captures semantic drift even when
  answer letters happen to match).

Cohen's Kappa (κ)
  Chance-corrected inter-persona agreement on binary correct/incorrect.
  Note the high-accuracy paradox: when OMR > 0.85, sparse residual errors
  distribute unevenly by chance, mechanically deflating κ. Report κ but
  treat DSS as interpretively primary.

McNemar's Test
  H0: no systematic directional difference between personas.
  p < 0.05 [SIG] → the model fails more consistently on one persona over
  the other — this is directional bias, not random noise.
  For NarrativeShield, we expect all McNemar tests to be non-significant.
""")


if __name__ == "__main__":
    main()
