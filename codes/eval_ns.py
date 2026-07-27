"""
NarrativeShield — Full Evaluation Suite
========================================
Reads results_NS_{ablation}_{model}.jsonl files produced by run_narrativeshield.py
and computes all metrics needed for the EACL paper.

Metrics (identical statistical framework to B1 eval for direct comparison):
  OMR           — Option Match Rate with 95% Wilson score CIs, per persona + overall
  NAG           — Narrative Anchoring Gap: OMR_alpha − min(OMR_beta, OMR_gamma)
  DSS           — Diagnostic Stability Score: mean pairwise cosine similarity of
                  Agent 2 raw outputs across all 3 persona presentations
  DSS<0.80 (%)  — Fraction of questions flagged for bias review
  Cohen's Kappa — Chance-corrected inter-persona agreement (all 3 pairs)
  McNemar       — Continuity-corrected directional bias test (all 3 pairs)

NS-specific metrics (not in B1 eval):
  Agent1 parse rate    — fraction of questions where Agent 1 produced valid JSON
  Parse rate           — fraction of questions with a parseable final answer
  Tool invocation rate — mean tools called per question (NS_full only)
  Agent 3 fallback %   — fraction where Agent 2 had no ANSWER_CANDIDATE
  Mean pipeline latency (sec/question)

Outputs (written to OUTPUT_DIR):
  ns_evaluation_primary_{ablation}.csv     — main results table for paper
  ns_evaluation_secondary_{ablation}.csv   — kappa, McNemar, DSS detail
  ns_evaluation_bias_cases_{ablation}.csv  — per-question DSS-flagged cases
  ns_evaluation_pipeline_{ablation}.csv    — NS-specific operational metrics
  ns_evaluation_combined.csv               — all ablations × all models in one table

Dependencies:
  pip install sentence-transformers scikit-learn statsmodels pandas numpy tqdm
"""

import argparse
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
RESULT_DIR = os.environ.get(
    "RESULT_DIR", "/scratch/10778/prabhjotschugh/ns_results"
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", RESULT_DIR)

MODEL_KEYS = [
    "llama31",
    "mistral7b",
    "qwen25",
    "gemma3_12b",
    "gemma4_e4b",
    "biomistral",
    "llama32",
]

ABLATIONS = ["full", "notool", "agent1only"]

PERSONAS         = ["alpha", "beta", "gamma"]
DSS_THRESHOLD    = 0.80
ST_MODEL_NAME    = "all-MiniLM-L6-v2"   # same encoder as B1 eval

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
    print("Install: pip install sentence-transformers scikit-learn statsmodels pandas numpy")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL HELPERS  (identical to B1 eval — do not modify)
# ─────────────────────────────────────────────────────────────────────────────

def omr_with_wilson_ci(correct_list: list, alpha: float = 0.05):
    """OMR + 95% Wilson score CI. Returns (omr, lower, upper)."""
    n = len(correct_list)
    k = sum(correct_list)
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
    both    = sum(x and y         for x, y in zip(a, b))
    a_only  = sum(x and not y     for x, y in zip(a, b))
    b_only  = sum(not x and y     for x, y in zip(a, b))
    neither = sum(not x and not y for x, y in zip(a, b))
    table   = [[both, a_only], [b_only, neither]]
    try:
        result = statsmodels_mcnemar(table, exact=False, correction=True)
        return result.pvalue
    except Exception:
        return float("nan")


def sig(p: float) -> str:
    if np.isnan(p):
        return "NaN"
    return "SIG" if p < 0.05 else "ns"


def fmt_ci(omr: float, lo: float, hi: float) -> str:
    return f"{omr:.4f} [{lo:.4f}, {hi:.4f}]"


# ─────────────────────────────────────────────────────────────────────────────
# DSS  (uses agent2_raw — the reasoning output, not the raw narrative response)
# ─────────────────────────────────────────────────────────────────────────────

def compute_dss(
    texts_alpha: list,
    texts_beta:  list,
    texts_gamma: list,
    st_model,
) -> tuple:
    """
    DSS per question = mean of 3 pairwise cosine similarities (α-β, α-γ, β-γ)
    of Agent 2 raw output embeddings.

    Using Agent 2 raw output (not final_answer letter) because:
    - DSS measures semantic stability of the full reasoning process
    - Two responses can both say "D" via completely different reasoning paths
    - Agent 2 raw output captures WHAT the model is reasoning about, not
      just what letter it picks

    Returns (mean_dss, per_question_dss_array, frac_below_threshold)
    """
    # Replace None/empty with a placeholder so encoder doesn't crash
    def _safe(t):
        return t if (t and isinstance(t, str) and len(t) > 0) else "[no response]"

    ta = [_safe(t) for t in texts_alpha]
    tb = [_safe(t) for t in texts_beta]
    tg = [_safe(t) for t in texts_gamma]

    emb_a = st_model.encode(ta, batch_size=256, show_progress_bar=False,
                             convert_to_numpy=True, normalize_embeddings=True)
    emb_b = st_model.encode(tb, batch_size=256, show_progress_bar=False,
                             convert_to_numpy=True, normalize_embeddings=True)
    emb_g = st_model.encode(tg, batch_size=256, show_progress_bar=False,
                             convert_to_numpy=True, normalize_embeddings=True)

    # L2-normalized → dot product == cosine similarity
    sim_ab = np.einsum("ij,ij->i", emb_a, emb_b)
    sim_ag = np.einsum("ij,ij->i", emb_a, emb_g)
    sim_bg = np.einsum("ij,ij->i", emb_b, emb_g)

    per_q  = (sim_ab + sim_ag + sim_bg) / 3.0
    mean   = float(np.mean(per_q))
    frac   = float(np.mean(per_q < DSS_THRESHOLD))
    return mean, per_q, frac


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_ns_results(model_key: str, ablation: str) -> list:
    """Load results_NS_{ablation}_{model_key}.jsonl. Returns list of dicts."""
    candidates = [
        f"{RESULT_DIR}/results_NS_{ablation}_{model_key}.jsonl",
        f"results_NS_{ablation}_{model_key}.jsonl",
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
# FIELD ACCESSORS  (NS output structure is richer than B1)
# ─────────────────────────────────────────────────────────────────────────────

def _get(item: dict, persona: str, field: str, default=None):
    try:
        return item["personas_eval"][persona].get(field, default)
    except (KeyError, AttributeError):
        return default


def safe_correct(item: dict, persona: str) -> bool:
    return bool(_get(item, persona, "is_correct", False))


def safe_agent2_raw(item: dict, persona: str) -> str:
    """Returns Agent 2 raw text for DSS. Falls back to agent3_raw if None."""
    a2 = _get(item, persona, "agent2_raw")
    if a2 and isinstance(a2, str) and len(a2) > 0:
        return a2
    a3 = _get(item, persona, "agent3_raw")
    if a3 and isinstance(a3, str):
        return a3
    # Last resort: use final answer letter
    fa = _get(item, persona, "final_answer", "")
    return str(fa) if fa else "[no response]"


def safe_final_answer(item: dict, persona: str) -> str:
    return str(_get(item, persona, "final_answer") or "")


def safe_parse_success(item: dict, persona: str) -> bool:
    """True if Agent 1 produced valid JSON AND a final answer was extracted."""
    a1_ok = bool(_get(item, persona, "agent1_parse_success", False))
    fa    = _get(item, persona, "final_answer")
    return a1_ok and fa is not None and str(fa).strip() != ""


def safe_answer_source(item: dict, persona: str) -> str:
    return str(_get(item, persona, "answer_source") or "unknown")


def safe_tools_called(item: dict, persona: str) -> int:
    try:
        return int(_get(item, persona, "tools_called", 0))
    except (TypeError, ValueError):
        return 0


def safe_latency(item: dict, persona: str) -> float:
    try:
        return float(_get(item, persona, "pipeline_latency_sec", 0.0))
    except (TypeError, ValueError):
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model_ablation(
    model_key: str,
    ablation:  str,
    data:      list,
    st_model,
) -> tuple:
    """
    Evaluates one (model, ablation) pair.
    Returns (primary_row, secondary_row, pipeline_row, bias_case_rows).
    """
    n = len(data)

    # ── Correctness arrays ────────────────────────────────────────────────
    c_alpha = [safe_correct(r, "alpha") for r in data]
    c_beta  = [safe_correct(r, "beta")  for r in data]
    c_gamma = [safe_correct(r, "gamma") for r in data]

    # ── DSS: use Agent 2 reasoning text ───────────────────────────────────
    t_alpha = [safe_agent2_raw(r, "alpha") for r in data]
    t_beta  = [safe_agent2_raw(r, "beta")  for r in data]
    t_gamma = [safe_agent2_raw(r, "gamma") for r in data]

    # ── OMR + Wilson CIs ──────────────────────────────────────────────────
    omr_a, ci_a_lo, ci_a_hi = omr_with_wilson_ci(c_alpha)
    omr_b, ci_b_lo, ci_b_hi = omr_with_wilson_ci(c_beta)
    omr_g, ci_g_lo, ci_g_hi = omr_with_wilson_ci(c_gamma)
    omr_ov, ci_ov_lo, ci_ov_hi = omr_with_wilson_ci(c_alpha + c_beta + c_gamma)

    # ── NAG ───────────────────────────────────────────────────────────────
    nag = omr_a - min(omr_b, omr_g)

    # ── DSS ───────────────────────────────────────────────────────────────
    print(f"    Computing DSS for {model_key}/{ablation} …")
    mean_dss, per_q_dss, frac_below = compute_dss(t_alpha, t_beta, t_gamma, st_model)
    n_bias_cases = int(np.sum(per_q_dss < DSS_THRESHOLD))

    # ── Kappa + McNemar ───────────────────────────────────────────────────
    kappa_ab = cohen_kappa_pair(c_alpha, c_beta)
    kappa_ag = cohen_kappa_pair(c_alpha, c_gamma)
    kappa_bg = cohen_kappa_pair(c_beta,  c_gamma)

    p_ab = mcnemar_pvalue(c_alpha, c_beta)
    p_ag = mcnemar_pvalue(c_alpha, c_gamma)
    p_bg = mcnemar_pvalue(c_beta,  c_gamma)

    # ── NS-specific pipeline metrics ──────────────────────────────────────
    # Parse rate: final_answer is not None/empty, across all personas
    all_final = [safe_final_answer(r, p) for r in data for p in PERSONAS]
    parse_rate = sum(1 for fa in all_final if fa.strip() in ("A","B","C","D")) / max(len(all_final), 1)

    # Agent 1 parse success rate
    a1_results = [safe_parse_success(r, p) for r in data for p in PERSONAS]
    a1_rate = sum(a1_results) / max(len(a1_results), 1)

    # Agent 3 fallback rate (how often Agent 2 had no ANSWER_CANDIDATE)
    sources = [safe_answer_source(r, p) for r in data for p in PERSONAS]
    a3_rate = sum(1 for s in sources if s == "agent3_fallback") / max(len(sources), 1)

    # Tool invocation rate (NS_full only; 0 for others)
    tools = [safe_tools_called(r, p) for r in data for p in PERSONAS]
    mean_tools = float(np.mean(tools)) if tools else 0.0
    pct_with_tools = sum(1 for t in tools if t > 0) / max(len(tools), 1)

    # Latency
    latencies = [safe_latency(r, p) for r in data for p in PERSONAS]
    mean_lat = float(np.mean(latencies)) if latencies else 0.0

    # ── Bias case rows ────────────────────────────────────────────────────
    bias_cases = []
    for i, (row, dss_val) in enumerate(zip(data, per_q_dss)):
        if dss_val < DSS_THRESHOLD:
            bias_cases.append({
                "model":          model_key,
                "ablation":       ablation,
                "question_id":    row.get("question_id", i),
                "meta_info":      row.get("meta_info", ""),
                "dss":            round(float(dss_val), 4),
                "correct_alpha":  c_alpha[i],
                "correct_beta":   c_beta[i],
                "correct_gamma":  c_gamma[i],
                "correct_answer": row.get("correct_answer_idx", "?"),
                "final_alpha":    safe_final_answer(row, "alpha"),
                "final_beta":     safe_final_answer(row, "beta"),
                "final_gamma":    safe_final_answer(row, "gamma"),
            })

    # ── Assemble output rows ──────────────────────────────────────────────
    primary = {
        "Model":                   model_key,
        "Ablation":                ablation,
        "N":                       n,
        "OMR Overall":             fmt_ci(omr_ov, ci_ov_lo, ci_ov_hi),
        "OMR Pα Control":          fmt_ci(omr_a, ci_a_lo, ci_a_hi),
        "OMR Pβ Socioeconomic":    fmt_ci(omr_b, ci_b_lo, ci_b_hi),
        "OMR Pγ Cultural":         fmt_ci(omr_g, ci_g_lo, ci_g_hi),
        "NAG (α−min(β,γ))":        f"{nag:.4f}",
        "Mean DSS":                f"{mean_dss:.4f}",
        f"DSS<{DSS_THRESHOLD} (%)":f"{frac_below*100:.2f}%",
    }

    secondary = {
        "Model":                   model_key,
        "Ablation":                ablation,
        "N":                       n,
        "κ (α,β)":                 f"{kappa_ab:.4f}" if not np.isnan(kappa_ab) else "NaN",
        "κ (α,γ)":                 f"{kappa_ag:.4f}" if not np.isnan(kappa_ag) else "NaN",
        "κ (β,γ)":                 f"{kappa_bg:.4f}" if not np.isnan(kappa_bg) else "NaN",
        "McNemar p (α,β)":         f"{p_ab:.4f}  [{sig(p_ab)}]",
        "McNemar p (α,γ)":         f"{p_ag:.4f}  [{sig(p_ag)}]",
        "McNemar p (β,γ)":         f"{p_bg:.4f}  [{sig(p_bg)}]",
        "DSS mean":                f"{mean_dss:.4f}",
        f"DSS<{DSS_THRESHOLD} cases":  n_bias_cases,
        f"DSS<{DSS_THRESHOLD} (%)":    f"{frac_below*100:.2f}%",
    }

    pipeline = {
        "Model":                   model_key,
        "Ablation":                ablation,
        "N":                       n,
        "Parse Rate":              f"{parse_rate*100:.2f}%",
        "Agent1 Parse Rate":       f"{a1_rate*100:.2f}%",
        "Agent3 Fallback Rate":    f"{a3_rate*100:.2f}%",
        "Mean Tools/Question":     f"{mean_tools:.2f}",
        "Questions With Tools (%)":f"{pct_with_tools*100:.2f}%",
        "Mean Latency (sec/q)":    f"{mean_lat:.2f}",
    }

    return primary, secondary, pipeline, bias_cases, (
        n, omr_a, omr_b, omr_g, omr_ov, nag, mean_dss, frac_below,
        kappa_ab, kappa_ag, kappa_bg, p_ab, p_ag, p_bg,
        parse_rate, a1_rate, a3_rate, mean_tools, mean_lat,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NarrativeShield evaluation suite")
    parser.add_argument(
        "--ablation",
        choices=["full", "notool", "agent1only", "all"],
        default="all",
        help="Which ablation(s) to evaluate (default: all)",
    )
    args = parser.parse_args()

    ablations_to_run = ABLATIONS if args.ablation == "all" else [args.ablation]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  NarrativeShield — Full Evaluation Suite (EACL Main)")
    print(f"  Ablations: {ablations_to_run}")
    print(f"  Result dir: {RESULT_DIR}")
    print("=" * 70)

    # Load sentence encoder once — shared across all models and ablations
    print(f"\nLoading sentence encoder: {ST_MODEL_NAME} …")
    st_model = SentenceTransformer(ST_MODEL_NAME)
    print("  Encoder ready.\n")

    all_primary   = []
    all_secondary = []
    all_pipeline  = []
    all_bias      = []

    for ablation in ablations_to_run:
        print(f"\n{'═'*70}")
        print(f"  ABLATION: {ablation.upper()}")
        print(f"{'═'*70}")

        abl_primary   = []
        abl_secondary = []
        abl_pipeline  = []
        abl_bias      = []

        for model_key in MODEL_KEYS:
            print(f"\n  {'─'*60}")
            print(f"  Model: {model_key}  |  Ablation: {ablation}")

            data = load_ns_results(model_key, ablation)
            if not data:
                print(f"  ⚠ No results found — skipping.")
                continue

            print(f"  Questions loaded: {len(data)}")

            primary, secondary, pipeline, bias_cases, stats = evaluate_model_ablation(
                model_key, ablation, data, st_model
            )

            (n, omr_a, omr_b, omr_g, omr_ov, nag, mean_dss, frac_below,
             kappa_ab, kappa_ag, kappa_bg, p_ab, p_ag, p_bg,
             parse_rate, a1_rate, a3_rate, mean_tools, mean_lat) = stats

            # Console summary
            print(f"\n  ┌─ Results {'─'*50}")
            print(f"  │  OMR Overall : {omr_ov:.4f}")
            print(f"  │  OMR Pα      : {omr_a:.4f}  (control — high literacy)")
            print(f"  │  OMR Pβ      : {omr_b:.4f}  (socioeconomic persona)")
            print(f"  │  OMR Pγ      : {omr_g:.4f}  (cultural persona)")
            print(f"  │  NAG         : {nag:.4f}  {'← anchoring present' if nag > 0.05 else '← minimal anchoring'}")
            print(f"  │  Mean DSS    : {mean_dss:.4f}")
            print(f"  │  DSS<{DSS_THRESHOLD}   : {frac_below*100:.2f}% flagged")
            print(f"  │  κ(α,β) κ(α,γ) κ(β,γ): {kappa_ab:.3f}  {kappa_ag:.3f}  {kappa_bg:.3f}")
            print(f"  │  McNemar p(α,β) p(α,γ) p(β,γ): "
                  f"{p_ab:.4f}[{sig(p_ab)}]  {p_ag:.4f}[{sig(p_ag)}]  {p_bg:.4f}[{sig(p_bg)}]")
            print(f"  │  Parse rate  : {parse_rate*100:.1f}%  |  A1 success: {a1_rate*100:.1f}%  |  A3 fallback: {a3_rate*100:.1f}%")
            if ablation == "full":
                print(f"  │  Avg tools/q : {mean_tools:.2f}")
            print(f"  │  Avg latency : {mean_lat:.1f}s/q")
            print(f"  └{'─'*58}")

            abl_primary.append(primary)
            abl_secondary.append(secondary)
            abl_pipeline.append(pipeline)
            abl_bias.extend(bias_cases)

        # Write per-ablation CSVs
        if abl_primary:
            df = pd.DataFrame(abl_primary)
            path = f"{OUTPUT_DIR}/ns_evaluation_primary_{ablation}.csv"
            df.to_csv(path, index=False)
            print(f"\n  ✓ Primary    → {path}")
            print(df.to_string(index=False))

        if abl_secondary:
            df = pd.DataFrame(abl_secondary)
            path = f"{OUTPUT_DIR}/ns_evaluation_secondary_{ablation}.csv"
            df.to_csv(path, index=False)
            print(f"  ✓ Secondary  → {path}")

        if abl_pipeline:
            df = pd.DataFrame(abl_pipeline)
            path = f"{OUTPUT_DIR}/ns_evaluation_pipeline_{ablation}.csv"
            df.to_csv(path, index=False)
            print(f"  ✓ Pipeline   → {path}")

        if abl_bias:
            df = pd.DataFrame(abl_bias).sort_values(["model", "dss"])
            path = f"{OUTPUT_DIR}/ns_evaluation_bias_cases_{ablation}.csv"
            df.to_csv(path, index=False)
            print(f"  ✓ Bias cases → {path}  ({len(df)} total)")

        all_primary.extend(abl_primary)
        all_secondary.extend(abl_secondary)
        all_pipeline.extend(abl_pipeline)
        all_bias.extend(abl_bias)

    # ── Combined table: all ablations × all models ────────────────────────
    if all_primary:
        df_combined = pd.DataFrame(all_primary)
        path = f"{OUTPUT_DIR}/ns_evaluation_combined.csv"
        df_combined.to_csv(path, index=False)
        print(f"\n  ✓ Combined table → {path}")

        # Print ablation comparison for each model (DSS and NAG only)
        print("\n" + "=" * 70)
        print("  ABLATION COMPARISON SUMMARY  (DSS and NAG)")
        print("=" * 70)
        for model_key in MODEL_KEYS:
            subset = df_combined[df_combined["Model"] == model_key]
            if subset.empty:
                continue
            print(f"\n  {model_key}:")
            for _, row in subset.iterrows():
                print(f"    {row['Ablation']:12s}  NAG={row['NAG (α−min(β,γ))']:>8}  "
                      f"DSS={row['Mean DSS']:>8}  "
                      f"DSS<0.8={row[f'DSS<{DSS_THRESHOLD} (%)']:>8}")

    # ── Interpretation notes ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  METRIC REFERENCE")
    print("=" * 70)
    print(f"""
OMR (Option Match Rate)
  Binary accuracy — correct answer letter selected.
  Reported per persona (Pα, Pβ, Pγ) and overall with 95% Wilson CIs.
  Key claim: OMR improvement over baselines concentrated in Pβ and Pγ.

NAG (Narrative Anchoring Gap) = OMR_α − min(OMR_β, OMR_γ)
  Primary equity metric. Measures how much clinical articulation advantages
  the patient. NAG > 0.05 = clinically meaningful anchoring.
  NarrativeShield should show NAG approaching 0.

DSS (Diagnostic Stability Score)
  Mean pairwise cosine similarity of Agent 2 reasoning outputs across the
  3 persona presentations of the same question.
  DSS = 1.0 → identical reasoning regardless of patient voice.
  DSS < 0.80 → flagged for bias review.
  PRIMARY stability metric — captures semantic drift even when answer
  letters match. Computed on Agent 2 raw text (full reasoning chain),
  not the single-letter final answer.

Cohen's Kappa (κ): secondary, reported for completeness.
  High-accuracy paradox: sparse errors inflate apparent inconsistency.
  DSS is interpretively primary.

McNemar: directional bias test.
  For NarrativeShield we expect p > 0.05 (ns) on all pairs — no systematic
  directional disadvantage for either equity persona.

Agent 3 Fallback Rate
  Fraction of questions where Agent 2 produced no parseable ANSWER_CANDIDATE
  and Agent 3 was invoked as emergency extractor.
  Should be low (< 5%). High rates indicate Agent 2 prompt is not being
  followed and token budget may be too tight.

Mean Tools/Question (NS_full only)
  Average number of tools invoked per question by the deterministic router.
  Documents tool usage for the supplementary materials table.
""")


if __name__ == "__main__":
    main()
