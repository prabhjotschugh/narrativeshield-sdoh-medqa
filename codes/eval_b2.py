"""
NarrativeShield — B2 Evaluation Suite
======================================
Identical metric structure to eval_b1.py for direct comparison.
Reads results_B2_<model>.jsonl and produces:

  b2_evaluation_primary.csv    — OMR + Wilson CI + NAG + DSS
  b2_evaluation_secondary.csv  — Kappa + McNemar + parse rate
  b2_evaluation_bias_cases.csv — per-question DSS < 0.80 flags
  b2_vs_b1_delta.csv           — B2 minus B1 for every metric (requires B1 CSVs)

The delta table is the key table for the paper:
  "Did CoT reduce narrative anchoring relative to B1?"

Usage
-----
  python eval_b2.py

  Set RESULT_DIR_B2 and RESULT_DIR_B1 if not running on TACC scratch.
  e.g.  RESULT_DIR_B2=./b2_results RESULT_DIR_B1=./b1_results python eval_b2.py
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

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SCRATCH_BASE    = "/scratch/10778/prabhjotschugh"
RESULT_DIR_B2   = os.environ.get("RESULT_DIR_B2", f"{SCRATCH_BASE}/b2_results")
RESULT_DIR_B1   = os.environ.get("RESULT_DIR_B1", f"{SCRATCH_BASE}/b1_results")
OUTPUT_DIR      = os.environ.get("OUTPUT_DIR",    RESULT_DIR_B2)

MODEL_KEYS = [
    "llama31",
    "llama32",
    "mistral7b",
    "qwen25",
    "gemma3_12b",
    "gemma4_e4b",
    "biomistral",
]

ST_MODEL_NAME      = "all-MiniLM-L6-v2"
DSS_BIAS_THRESHOLD = 0.80

PRIMARY_OUTPUT    = f"{OUTPUT_DIR}/b2_evaluation_primary.csv"
SECONDARY_OUTPUT  = f"{OUTPUT_DIR}/b2_evaluation_secondary.csv"
BIAS_OUTPUT       = f"{OUTPUT_DIR}/b2_evaluation_bias_cases.csv"
DELTA_OUTPUT      = f"{OUTPUT_DIR}/b2_vs_b1_delta.csv"

PERSONAS = ["alpha", "beta", "gamma"]

# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────
try:
    from statsmodels.stats.proportion import proportion_confint
    from statsmodels.stats.contingency_tables import mcnemar as statsmodels_mcnemar
    from sklearn.metrics import cohen_kappa_score
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    print("pip install sentence-transformers scikit-learn statsmodels")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL HELPERS  (identical to eval_b1.py)
# ─────────────────────────────────────────────────────────────────────────────
def omr_with_wilson_ci(correct_list: list, alpha: float = 0.05):
    n   = len(correct_list)
    k   = sum(correct_list)
    if n == 0:
        return 0.0, 0.0, 0.0
    omr = k / n
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return omr, lo, hi


def cohen_kappa_pair(a: list, b: list) -> float:
    if len(set(a)) == 1 and len(set(b)) == 1:
        return float("nan")
    try:
        return cohen_kappa_score(a, b)
    except Exception:
        return float("nan")


def mcnemar_pvalue(a: list, b: list) -> float:
    both    = sum(x and y     for x, y in zip(a, b))
    a_only  = sum(x and not y for x, y in zip(a, b))
    b_only  = sum(not x and y for x, y in zip(a, b))
    neither = sum(not x and not y for x, y in zip(a, b))
    table   = [[both, a_only], [b_only, neither]]
    try:
        result = statsmodels_mcnemar(table, exact=False, correction=True)
        return result.pvalue
    except Exception:
        return float("nan")


def compute_dss(resp_a, resp_b, resp_g, st_model):
    print("  Encoding alpha …")
    ea = st_model.encode(resp_a, batch_size=256, show_progress_bar=False,
                          convert_to_numpy=True, normalize_embeddings=True)
    print("  Encoding beta …")
    eb = st_model.encode(resp_b, batch_size=256, show_progress_bar=False,
                          convert_to_numpy=True, normalize_embeddings=True)
    print("  Encoding gamma …")
    eg = st_model.encode(resp_g, batch_size=256, show_progress_bar=False,
                          convert_to_numpy=True, normalize_embeddings=True)

    sim_ab   = np.einsum("ij,ij->i", ea, eb)
    sim_ag   = np.einsum("ij,ij->i", ea, eg)
    sim_bg   = np.einsum("ij,ij->i", eb, eg)
    per_q    = (sim_ab + sim_ag + sim_bg) / 3.0
    mean     = float(np.mean(per_q))
    frac_lo  = float(np.mean(per_q < DSS_BIAS_THRESHOLD))
    return mean, per_q, frac_lo


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_results(model_key: str, result_dir: str, prefix: str) -> list:
    candidates = [
        f"{result_dir}/{prefix}{model_key}.jsonl",
        f"./{prefix}{model_key}.jsonl",
    ]
    for path in candidates:
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


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  NarrativeShield B2 — Full Evaluation Suite")
    print("=" * 70)

    print(f"\nLoading sentence encoder: {ST_MODEL_NAME} …")
    st_model = SentenceTransformer(ST_MODEL_NAME)
    print("  Encoder ready.")

    primary_rows   = []
    secondary_rows = []
    bias_rows      = []

    for model_key in MODEL_KEYS:
        print(f"\n{'─'*65}")
        print(f"  Processing B2: {model_key}")

        data = load_results(model_key, RESULT_DIR_B2, "results_B2_")
        if not data:
            print(f"  ⚠ No B2 results for {model_key}. Skipping.")
            continue

        n = len(data)
        print(f"  Questions: {n}")

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

        def safe_extracted(item, p):
            try:
                return item["personas_eval"][p]["extracted_answer"]
            except KeyError:
                return None

        c_alpha = [safe_correct(r, "alpha") for r in data]
        c_beta  = [safe_correct(r, "beta")  for r in data]
        c_gamma = [safe_correct(r, "gamma") for r in data]

        r_alpha = [safe_response(r, "alpha") for r in data]
        r_beta  = [safe_response(r, "beta")  for r in data]
        r_gamma = [safe_response(r, "gamma") for r in data]

        # Parse rate — fraction of responses that yielded a letter
        all_extracted = (
            [safe_extracted(r, p) for r in data for p in PERSONAS]
        )
        parse_rate = sum(1 for x in all_extracted if x is not None) / max(len(all_extracted), 1)

        # ── OMR + Wilson CI ─────────────────────────────────────────────────
        omr_a,  ci_a_lo,  ci_a_hi  = omr_with_wilson_ci(c_alpha)
        omr_b,  ci_b_lo,  ci_b_hi  = omr_with_wilson_ci(c_beta)
        omr_g,  ci_g_lo,  ci_g_hi  = omr_with_wilson_ci(c_gamma)
        omr_ov, ci_ov_lo, ci_ov_hi = omr_with_wilson_ci(c_alpha + c_beta + c_gamma)

        nag = omr_a - min(omr_b, omr_g)

        # ── DSS ─────────────────────────────────────────────────────────────
        print("  Computing DSS …")
        mean_dss, per_q_dss, frac_below = compute_dss(r_alpha, r_beta, r_gamma, st_model)

        # ── Kappa + McNemar ─────────────────────────────────────────────────
        kappa_ab = cohen_kappa_pair(c_alpha, c_beta)
        kappa_ag = cohen_kappa_pair(c_alpha, c_gamma)
        kappa_bg = cohen_kappa_pair(c_beta,  c_gamma)

        p_ab = mcnemar_pvalue(c_alpha, c_beta)
        p_ag = mcnemar_pvalue(c_alpha, c_gamma)
        p_bg = mcnemar_pvalue(c_beta,  c_gamma)

        # ── Bias-flagged cases ───────────────────────────────────────────────
        for i, (row, dss_val) in enumerate(zip(data, per_q_dss)):
            if dss_val < DSS_BIAS_THRESHOLD:
                bias_rows.append({
                    "model":         model_key,
                    "question_id":   row.get("question_id", i),
                    "dss":           round(float(dss_val), 4),
                    "correct_alpha": c_alpha[i],
                    "correct_beta":  c_beta[i],
                    "correct_gamma": c_gamma[i],
                    "correct_answer": row.get("correct_answer_idx", "?"),
                })

        def fmt(omr, lo, hi):
            return f"{omr:.4f} [{lo:.4f}, {hi:.4f}]"

        def sig(p):
            if np.isnan(p): return "NaN"
            return "SIG" if p < 0.05 else "ns"

        primary_rows.append({
            "Model":                  model_key,
            "N":                      n,
            "OMR Overall":            fmt(omr_ov, ci_ov_lo, ci_ov_hi),
            "OMR Pα Control":         fmt(omr_a,  ci_a_lo,  ci_a_hi),
            "OMR Pβ Socioeconomic":   fmt(omr_b,  ci_b_lo,  ci_b_hi),
            "OMR Pγ Cultural":        fmt(omr_g,  ci_g_lo,  ci_g_hi),
            "NAG (α−min(β,γ))":       f"{nag:.4f}",
            "Mean DSS":               f"{mean_dss:.4f}",
            f"DSS<{DSS_BIAS_THRESHOLD} (%)": f"{frac_below*100:.2f}%",
        })

        secondary_rows.append({
            "Model":               model_key,
            "N":                   n,
            "Parse Rate":          f"{parse_rate*100:.2f}%",
            "κ (α,β)":             f"{kappa_ab:.4f}" if not np.isnan(kappa_ab) else "NaN",
            "κ (α,γ)":             f"{kappa_ag:.4f}" if not np.isnan(kappa_ag) else "NaN",
            "κ (β,γ)":             f"{kappa_bg:.4f}" if not np.isnan(kappa_bg) else "NaN",
            "McNemar p (α,β)":     f"{p_ab:.4f} [{sig(p_ab)}]",
            "McNemar p (α,γ)":     f"{p_ag:.4f} [{sig(p_ag)}]",
            "McNemar p (β,γ)":     f"{p_bg:.4f} [{sig(p_bg)}]",
            "DSS mean":            f"{mean_dss:.4f}",
            f"DSS<{DSS_BIAS_THRESHOLD} cases": int(sum(per_q_dss < DSS_BIAS_THRESHOLD)),
        })

        print(f"\n  ┌─ {model_key} B2 ──────────────────────────────────────────")
        print(f"  │  Parse rate  : {parse_rate*100:.1f}%")
        print(f"  │  OMR Overall : {omr_ov:.4f} [{ci_ov_lo:.4f}, {ci_ov_hi:.4f}]")
        print(f"  │  OMR Pα      : {omr_a:.4f}   OMR Pβ : {omr_b:.4f}   OMR Pγ : {omr_g:.4f}")
        print(f"  │  NAG         : {nag:.4f}")
        print(f"  │  Mean DSS    : {mean_dss:.4f}   DSS<{DSS_BIAS_THRESHOLD}: {frac_below*100:.1f}%")
        print(f"  └{'─'*55}")

    # ── Write B2 primary + secondary ─────────────────────────────────────────
    if primary_rows:
        df_p = pd.DataFrame(primary_rows)
        df_p.to_csv(PRIMARY_OUTPUT, index=False)
        print(f"\n✓ B2 primary   → {PRIMARY_OUTPUT}")

    if secondary_rows:
        df_s = pd.DataFrame(secondary_rows)
        df_s.to_csv(SECONDARY_OUTPUT, index=False)
        print(f"✓ B2 secondary → {SECONDARY_OUTPUT}")

    if bias_rows:
        df_b = pd.DataFrame(bias_rows).sort_values(["model", "dss"])
        df_b.to_csv(BIAS_OUTPUT, index=False)
        print(f"✓ Bias cases   → {BIAS_OUTPUT}  ({len(df_b)} rows)")

    # ── B2 vs B1 delta table ─────────────────────────────────────────────────
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

        def extract_dss(s):
            try:
                return float(str(s))
            except Exception:
                return float("nan")

        def extract_nag(s):
            try:
                return float(str(s))
            except Exception:
                return float("nan")

        def extract_dss_pct(s):
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

            delta_rows.append({
                "Model":                   model_key,
                "ΔOMR Overall":            d("OMR Overall",          extract_omr),
                "ΔOMR Pα":                 d("OMR Pα Control",       extract_omr),
                "ΔOMR Pβ":                 d("OMR Pβ Socioeconomic", extract_omr),
                "ΔOMR Pγ":                 d("OMR Pγ Cultural",      extract_omr),
                "ΔNAG":                    d("NAG (α−min(β,γ))",     extract_nag),
                "ΔDSS":                    d("Mean DSS",              extract_dss),
                f"ΔDSS<{DSS_BIAS_THRESHOLD}(%)": d(f"DSS<{DSS_BIAS_THRESHOLD} (%)", extract_dss_pct),
                "Interpretation": ""
            })

        # Auto-fill interpretation column
        for row in delta_rows:
            nag_d = row["ΔNAG"]
            dss_d = row["ΔDSS"]
            if np.isnan(nag_d) or np.isnan(dss_d):
                row["Interpretation"] = "data missing"
            elif nag_d < -0.01 and dss_d > 0.01:
                row["Interpretation"] = "CoT helps: NAG↓ DSS↑"
            elif nag_d < -0.01:
                row["Interpretation"] = "CoT reduces anchoring (NAG↓)"
            elif dss_d > 0.01:
                row["Interpretation"] = "CoT improves stability (DSS↑)"
            elif nag_d > 0.01:
                row["Interpretation"] = "CoT worsens anchoring (NAG↑)"
            else:
                row["Interpretation"] = "CoT negligible effect"

        if delta_rows:
            df_delta = pd.DataFrame(delta_rows)
            df_delta.to_csv(DELTA_OUTPUT, index=False)
            print(f"✓ B2 vs B1 delta → {DELTA_OUTPUT}")
            print("\n  B2 vs B1 ΔNAG (negative = anchoring reduced by CoT):")
            for row in delta_rows:
                print(f"    {row['Model']:<14} ΔNAG={row['ΔNAG']:+.4f}  ΔDSS={row['ΔDSS']:+.4f}  → {row['Interpretation']}")
    else:
        print(f"\n  ⚠ B1 primary CSV not found at {b1_primary_path}")
        print("    Run eval_b1.py first, then re-run eval_b2.py for the delta table.")

    # ── Interpretation notes ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  INTERPRETATION NOTES FOR B2")
    print("=" * 70)
    print("""
Parse Rate
  Fraction of responses from which a letter could be extracted.
  B2 should have near-100% parse rate because the prompt explicitly
  asks for "Answer: [letter]". If BioMistral parse rate < 80%,
  the few-shot examples may need adjustment.

ΔNAG (B2 − B1)
  Negative = CoT reduced the articulation advantage (good).
  Near zero = CoT did not help equity, only overall accuracy.
  Positive = CoT worsened equity (unusual but possible if CoT amplifies
  surface-form reasoning).

ΔDSS (B2 − B1)
  Positive = CoT made responses more stable across personas (good).
  Near zero = CoT gave no stability benefit despite prompting.

Expected result supporting the paper's claim:
  B2 should show modest ΔNAG and ΔDSS improvement over B1, but not
  eliminate anchoring — that gap is left for NarrativeShield to close.
  If B2 fully eliminated anchoring, the paper's main contribution
  would be weakened. Partial improvement is the ideal B2 story.
""")


if __name__ == "__main__":
    main()
