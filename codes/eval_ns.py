"""
NarrativeShield — Full Evaluation Suite
==========================================================
Reads results_NS_{ablation}_{model}.jsonl files produced by run_ns.py and
computes all metrics needed for the paper, across all 3 ablations:

  ABLATIONS = ["full", "notool", "agent1only"]

Metrics (identical statistical framework to eval_b1.py for direct comparison):
  OMR              — Option Match Rate with 95% Wilson score CIs, per persona
                      (alpha/beta/gamma) + overall
  NAG              — Narrative Anchoring Gap: OMR_alpha - min(OMR_marked),
                      reported with every component gap + paired bootstrap
                      95% CI, never as a bare scalar (same fix as eval_b1.py)
  DSS              — Diagnostic Stability Score: mean pairwise cosine
                      similarity of Agent 2 raw reasoning outputs across all
                      present persona presentations, raw + format-controlled
                      ("stripped") variants
  Structured DSS   — added analysis:
                      field-level exact/fuzzy match rate across Agent 1's
                      structured JSON extraction (parsed from
                      personas_eval[p]["agent1_json_str"]) across personas.
                      Isolates whether EXTRACTION itself is register-invariant,
                      separate from whether Agent 2's downstream reasoning
                      converges (which the text-based DSS above measures).
                      A model can have high text DSS (converges on similar
                      reasoning/answers) while having low structured DSS
                      (Agent 1 extracted different clinical facts per
                      persona and Agent 2 recovered anyway) — or vice versa —
                      and that dissociation is itself a paper-relevant finding.
  DSS<0.80 (%)     — Fraction of questions flagged for bias review
  Cohen's Kappa    — Chance-corrected inter-persona agreement (all present pairs)
  McNemar          — Continuity-corrected directional bias test, reported
                      against BOTH raw 0.05 and Bonferroni-corrected
                      thresholds via mc.significance_label (same Footnote 1
                      fix as eval_b1.py, applied per-ablation since each
                      ablation's pairwise comparisons are a separate family)

NS-specific metrics (not in B1/B2/B3 eval) — semantics preserved EXACTLY from
the original file, including the known, now-documented, non-bug contradiction
where Parse Rate can be ~100% while Agent1 Parse Rate is ~1%:
  Parse Rate           — fraction of (question, persona) pairs where
                          final_answer is a clean "A"/"B"/"C"/"D" string,
                          REGARDLESS of which agent/fallback produced it.
  Agent1 Parse Rate    — fraction of (question, persona) pairs satisfying the
                          COMPOUND condition: agent1_parse_success is True
                          AND final_answer is non-empty. This is a much
                          stricter, different-purpose metric than Parse Rate
                          above (it demands Agent 1's own JSON parse
                          succeeded, not just that *some* answer letter was
                          eventually produced by the pipeline) — do not
                          merge these two metrics or "fix" the apparent
                          contradiction between them; it reflects how often
                          the final answer was salvaged by Agent 2/3 despite
                          Agent 1 extraction failing.
  Agent3 Fallback Rate — fraction of (question, persona) pairs where
                          answer_source == "agent3_fallback" EXACTLY (Agent 2
                          produced no parseable ANSWER_CANDIDATE and Agent 3
                          was invoked as emergency single-letter extractor).
  Tool invocation rate — mean tools called per question (NS_full only; 0 for
                          notool/agent1only ablations, which never route to
                          tools)
  Mean pipeline latency (sec/question)

Outputs (written to OUTPUT_DIR):
  ns_evaluation_primary_{ablation}.csv     — main results table for paper
  ns_evaluation_secondary_{ablation}.csv   — kappa, McNemar (raw + Bonferroni),
                                              DSS detail (text + structured)
  ns_evaluation_bias_cases_{ablation}.csv  — per-question DSS-flagged cases
  ns_evaluation_pipeline_{ablation}.csv    — NS-specific operational metrics
  ns_evaluation_combined.csv               — all ablations x all models in one table

All core statistics are computed via codes/metrics_common.py, the single
source of truth shared with eval_b1.py, eval_b2.py, and eval_b3.py — this
removes the model-registry drift bug class (this file's pre-revision
MODEL_KEYS list did not match run_ns.py's MODEL_REGISTRY keys: it listed
short-form names such as "llama31"/"biomistral" while metrics_common.py's
canonical MODEL_KEYS uses the full 12-model sweep naming, e.g. "llama31_8b".
Per instructions this file now imports mc.MODEL_KEYS and mc.PERSONAS (the
alpha/beta/gamma set — the doctor-verified persona dataset is frozen for
this revision and is not being extended) directly rather than hand-listing;
run_ns.py was revised in parallel to emit results under the canonical
model/persona set. The present-persona detection below still degrades
gracefully against any partial/legacy result files, same as eval_b1.py.

FIELD-NAME ASSUMPTION (flagged per instructions): Agent 1's structured JSON
is stored per-persona as `personas_eval[p]["agent1_json_str"]` — a JSON
STRING (or None), not a pre-parsed dict, and NOT under the key
"agent1_json" (that key exists only on the flat per-persona pipeline record
inside run_ns.py's run_pipeline(), before it's copied into q_record; the
copy step only carries over agent1_json_str, not agent1_json — verified by
reading run_ns.py's q_record assembly around the "personas_eval[persona] ="
block). This script parses that string with json.loads() defensively (via
_load_agent1_json below) and treats parse failures as a missing structured
record (contributes 0.0 pairwise field-similarity for that question, per
metrics_common.compute_structured_dss's own handling of non-dict inputs).

Usage
-----
  python eval_ns.py                    # all 3 ablations
  python eval_ns.py --ablation full    # single ablation

Dependencies
------------
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

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics_common as mc

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    print("Install: pip install sentence-transformers scikit-learn statsmodels pandas numpy")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
RESULT_DIR = os.environ.get("RESULT_DIR", "/path/to/scratch/ns_results")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", RESULT_DIR)

MODEL_KEYS = mc.MODEL_KEYS  # imported, not hand-duplicated — see metrics_common.py
PERSONAS = mc.PERSONAS
ABLATIONS = ["full", "notool", "agent1only"]

DSS_THRESHOLD = mc.DSS_BIAS_THRESHOLD
ST_MODEL_NAME = "all-MiniLM-L6-v2"   # same encoder as B1/B2/B3 eval


# ─────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────
def load_ns_results(model_key: str, ablation: str) -> list:
    """Load results_NS_{ablation}_{model_key}.jsonl. Returns list of dicts."""
    return mc.load_jsonl_results(
        RESULT_DIR, f"results_NS_{ablation}_{{model}}.jsonl", model_key
    )


# ─────────────────────────────────────────────────────────────
# FIELD ACCESSORS  (NS output structure is richer than B1/B2/B3)
# ─────────────────────────────────────────────────────────────
def _get(item: dict, persona: str, field: str, default=None):
    return mc.safe_get(item, "personas_eval", persona, field, default=default)


def safe_correct(item: dict, persona: str) -> bool:
    return bool(_get(item, persona, "is_correct", False))


def safe_agent2_raw(item: dict, persona: str) -> str:
    """Returns Agent 2 raw text for DSS. Falls back to agent3_raw, then the
    final answer letter, if Agent 2 produced nothing usable."""
    a2 = _get(item, persona, "agent2_raw")
    if a2 and isinstance(a2, str) and len(a2) > 0:
        return a2
    a3 = _get(item, persona, "agent3_raw")
    if a3 and isinstance(a3, str):
        return a3
    fa = _get(item, persona, "final_answer", "")
    return str(fa) if fa else "[no response]"


def safe_final_answer(item: dict, persona: str) -> str:
    return str(_get(item, persona, "final_answer") or "")


def safe_parse_success(item: dict, persona: str) -> bool:
    """True if Agent 1 produced valid JSON AND a final answer was extracted.
    Compound condition — preserved exactly from the original file. This is
    intentionally stricter than (and different from) the plain Parse Rate
    metric below; the two can legitimately diverge dramatically."""
    a1_ok = bool(_get(item, persona, "agent1_parse_success", False))
    fa = _get(item, persona, "final_answer")
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


def _load_agent1_json(item: dict, persona: str):
    """
    Parses personas_eval[p]["agent1_json_str"] (a JSON string, or None) into
    a dict for structured DSS. Returns None on missing/unparseable input —
    metrics_common.compute_structured_dss treats non-dict entries as a 0.0
    pairwise field-similarity for that question rather than crashing.
    """
    raw = _get(item, persona, "agent1_json_str")
    if not raw or not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────
# SINGLE-MODEL EVALUATION
# ─────────────────────────────────────────────────────────────
def evaluate_model_ablation(model_key: str, ablation: str, data: list, st_model) -> dict:
    """
    Evaluates one (model, ablation) pair. Returns a dict of everything the
    caller needs: primary/secondary/pipeline row dicts, bias case rows, and
    the raw correctness arrays (for the per-ablation McNemar/NAG-CI pass
    that needs correctness pooled across models... actually those stats are
    per-model like eval_b1.py, so correctness is returned per this call).
    """
    n = len(data)

    present_personas = [
        p for p in PERSONAS
        if any(mc.safe_get(r, "personas_eval", p) is not None for r in data[:5])
    ]
    if len(present_personas) < 2:
        return None

    # ── Correctness arrays ────────────────────────────────────
    correctness = {p: [safe_correct(r, p) for r in data] for p in present_personas}

    # ── DSS: Agent 2 reasoning text (raw + format-controlled) ──
    texts = {p: [safe_agent2_raw(r, p) for r in data] for p in present_personas}

    # ── Structured DSS: Agent 1 JSON extraction (NEW) ───────────
    agent1_jsons = {p: [_load_agent1_json(r, p) for r in data] for p in present_personas}

    # ── OMR + Wilson CIs ─────────────────────────────────────
    omr_ci = {p: mc.omr_with_wilson_ci(correctness[p]) for p in present_personas}
    omr_point = {p: omr_ci[p][0] for p in present_personas}
    all_correct = [c for p in present_personas for c in correctness[p]]
    omr_ov, ci_ov_lo, ci_ov_hi = mc.omr_with_wilson_ci(all_correct)

    # ── NAG: headline + components + bootstrap CI ────────────
    nag_info = mc.nag_components(omr_point)
    nag_pt, nag_lo, nag_hi = mc.nag_bootstrap_ci(correctness)

    # ── DSS: raw + format-controlled (text-level, Agent 2) ────
    print(f"    Computing DSS (raw) for {model_key}/{ablation} ...")
    mean_dss, per_q_dss, frac_below = mc.compute_dss(texts, st_model, DSS_THRESHOLD)
    print(f"    Computing DSS (stripped) for {model_key}/{ablation} ...")
    mean_dss_stripped, per_q_dss_stripped, frac_below_stripped = mc.compute_dss_stripped(
        texts, st_model, DSS_THRESHOLD
    )
    n_bias_cases = int(np.sum(per_q_dss < DSS_THRESHOLD))

    # ── Structured DSS (field-level, Agent 1) ──────────────────
    # Falls back to NaN if fewer than 2 personas have any parseable JSON at
    # all (compute_structured_dss requires >=2 present persona keys, which
    # is already guaranteed by present_personas, but every row could still
    # fail to parse — that's handled internally as 0.0 per-question, not a
    # crash, so this only guards the truly-degenerate len(present)<2 case).
    try:
        mean_structured_dss, per_q_structured_dss = mc.compute_structured_dss(agent1_jsons)
    except ValueError:
        mean_structured_dss, per_q_structured_dss = float("nan"), np.array([])

    # ── Cohen's Kappa (all present pairs) ───────────────────────
    kappa = {
        pair: mc.cohen_kappa_pair(correctness[pair[0]], correctness[pair[1]])
        for pair in mc.PERSONA_PAIRS
        if pair[0] in correctness and pair[1] in correctness
    }

    # ── McNemar, raw + Bonferroni-corrected ─────────────────────
    mcnemar = mc.all_pairwise_mcnemar(correctness)
    n_tests = mcnemar.pop("n_tests")
    bonf_alpha = mc.bonferroni_alpha(n_tests)

    # ── NS-specific pipeline metrics (semantics preserved exactly) ──
    # Parse Rate: fraction of final_answer fields that are clean A-D
    # letters, regardless of source (Agent1/Agent2/Agent3 fallback/etc).
    all_final = [safe_final_answer(r, p) for r in data for p in present_personas]
    parse_rate = sum(1 for fa in all_final if fa.strip() in ("A", "B", "C", "D")) / max(len(all_final), 1)

    # Agent 1 Parse Rate: compound condition — agent1_parse_success AND a
    # non-empty final_answer. Deliberately distinct from Parse Rate above;
    # a 100% Parse Rate can coexist with a ~1% Agent1 Parse Rate when Agent
    # 2/3 salvage the answer despite Agent 1 JSON extraction failing. This
    # is a real, documented pipeline phenomenon, not a bug to reconcile.
    a1_results = [safe_parse_success(r, p) for r in data for p in present_personas]
    a1_rate = sum(a1_results) / max(len(a1_results), 1)

    # Agent 3 Fallback Rate: answer_source == "agent3_fallback" exactly.
    sources = [safe_answer_source(r, p) for r in data for p in present_personas]
    a3_rate = sum(1 for s in sources if s == "agent3_fallback") / max(len(sources), 1)

    # Tool invocation rate (NS_full only; 0 for notool/agent1only).
    tools = [safe_tools_called(r, p) for r in data for p in present_personas]
    mean_tools = float(np.mean(tools)) if tools else 0.0
    pct_with_tools = sum(1 for t in tools if t > 0) / max(len(tools), 1)

    # Latency.
    latencies = [safe_latency(r, p) for r in data for p in present_personas]
    mean_lat = float(np.mean(latencies)) if latencies else 0.0

    # ── Bias case rows ────────────────────────────────────────
    bias_cases = []
    for i, (row, dss_val) in enumerate(zip(data, per_q_dss)):
        if dss_val < DSS_THRESHOLD:
            bias_cases.append({
                "model":       model_key,
                "ablation":    ablation,
                "question_id": row.get("question_id", i),
                "meta_info":   row.get("meta_info", ""),
                "dss":         round(float(dss_val), 4),
                "structured_dss": (
                    round(float(per_q_structured_dss[i]), 4)
                    if len(per_q_structured_dss) > i else None
                ),
                **{f"correct_{p}": correctness[p][i] for p in present_personas},
                "correct_answer": row.get("correct_answer_idx", "?"),
                **{f"final_{p}": safe_final_answer(row, p) for p in present_personas},
            })

    def fmt_ci(omr, lo, hi):
        return f"{omr:.4f} [{lo:.4f}, {hi:.4f}]"

    primary = {
        "Model": model_key,
        "Ablation": ablation,
        "N": n,
        "Personas": "+".join(present_personas),
        "OMR Overall": fmt_ci(omr_ov, ci_ov_lo, ci_ov_hi),
    }
    for p in present_personas:
        primary[f"OMR {mc.PERSONA_LABELS[p]}"] = fmt_ci(*omr_ci[p])
    primary["NAG (headline)"] = f"{nag_info['nag']:.4f}"
    primary["NAG 95% Bootstrap CI"] = f"[{nag_lo:.4f}, {nag_hi:.4f}]"
    primary["NAG worst persona"] = nag_info["nag_worst_persona"]
    for k, v in nag_info.items():
        if k.startswith("gap_alpha_"):
            primary[k] = f"{v:.4f}"
    primary["Mean DSS (raw)"] = f"{mean_dss:.4f}"
    primary["Mean DSS (stripped)"] = f"{mean_dss_stripped:.4f}"
    primary["Mean Structured DSS"] = (
        f"{mean_structured_dss:.4f}" if not np.isnan(mean_structured_dss) else "NaN"
    )
    primary[f"DSS<{DSS_THRESHOLD} (%)"] = f"{frac_below*100:.2f}%"

    secondary = {"Model": model_key, "Ablation": ablation, "N": n,
                 "Bonferroni alpha": f"{bonf_alpha:.5f} (n_tests={n_tests})"}
    for pair, k in kappa.items():
        secondary[f"kappa {pair[0][0]}{pair[1][0]}"] = f"{k:.4f}" if not np.isnan(k) else "NaN"
    for pair, p in mcnemar.items():
        label = f"McNemar p ({pair[0][0]},{pair[1][0]})"
        secondary[label] = f"{p:.4f}"
        secondary[label + " [raw 0.05]"] = mc.significance_label(p, 1)
        secondary[label + " [Bonferroni]"] = mc.significance_label(p, n_tests)
    secondary["DSS mean (raw)"] = f"{mean_dss:.4f}"
    secondary["DSS mean (stripped)"] = f"{mean_dss_stripped:.4f}"
    secondary["Structured DSS mean"] = (
        f"{mean_structured_dss:.4f}" if not np.isnan(mean_structured_dss) else "NaN"
    )
    secondary[f"DSS<{DSS_THRESHOLD} cases"] = n_bias_cases
    secondary[f"DSS<{DSS_THRESHOLD} (%)"] = f"{frac_below*100:.2f}%"

    pipeline = {
        "Model": model_key,
        "Ablation": ablation,
        "N": n,
        "Parse Rate": f"{parse_rate*100:.2f}%",
        "Agent1 Parse Rate": f"{a1_rate*100:.2f}%",
        "Agent3 Fallback Rate": f"{a3_rate*100:.2f}%",
        "Mean Tools/Question": f"{mean_tools:.2f}",
        "Questions With Tools (%)": f"{pct_with_tools*100:.2f}%",
        "Mean Latency (sec/q)": f"{mean_lat:.2f}",
    }

    return {
        "primary": primary,
        "secondary": secondary,
        "pipeline": pipeline,
        "bias_cases": bias_cases,
        "present_personas": present_personas,
        "console": dict(
            n=n, omr_ci=omr_ci, omr_ov=omr_ov, ci_ov_lo=ci_ov_lo, ci_ov_hi=ci_ov_hi,
            nag_info=nag_info, nag_lo=nag_lo, nag_hi=nag_hi,
            mean_dss=mean_dss, mean_dss_stripped=mean_dss_stripped,
            mean_structured_dss=mean_structured_dss, frac_below=frac_below,
            kappa=kappa, mcnemar=mcnemar, n_tests=n_tests, bonf_alpha=bonf_alpha,
            parse_rate=parse_rate, a1_rate=a1_rate, a3_rate=a3_rate,
            mean_tools=mean_tools, mean_lat=mean_lat,
        ),
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
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
    print("  NarrativeShield -- Full Evaluation Suite")
    print(f"  Ablations: {ablations_to_run}")
    print(f"  Result dir: {RESULT_DIR}")
    print("=" * 70)

    print(f"\nLoading sentence encoder: {ST_MODEL_NAME} ...")
    st_model = SentenceTransformer(ST_MODEL_NAME)
    print("  Encoder ready.\n")

    all_primary, all_secondary, all_pipeline, all_bias = [], [], [], []

    for ablation in ablations_to_run:
        print(f"\n{'='*70}")
        print(f"  ABLATION: {ablation.upper()}")
        print(f"{'='*70}")

        abl_primary, abl_secondary, abl_pipeline, abl_bias = [], [], [], []

        for model_key in MODEL_KEYS:
            print(f"\n  {'-'*60}")
            print(f"  Model: {model_key}  |  Ablation: {ablation}")

            data = load_ns_results(model_key, ablation)
            if not data:
                print(f"  [skip] No results found.")
                continue

            print(f"  Questions loaded: {len(data)}")

            result = evaluate_model_ablation(model_key, ablation, data, st_model)
            if result is None:
                print(f"  [skip] Fewer than 2 personas present.")
                continue

            c = result["console"]

            print(f"\n  +- Results {'-'*50}")
            print(f"  |  OMR Overall : {c['omr_ov']:.4f} [{c['ci_ov_lo']:.4f}, {c['ci_ov_hi']:.4f}]")
            for p in result["present_personas"]:
                omr, lo, hi = c["omr_ci"][p]
                print(f"  |  OMR {p:6s}: {omr:.4f} [{lo:.4f}, {hi:.4f}]")
            print(f"  |  NAG (headline)  : {c['nag_info']['nag']:.4f}  "
                  f"95% CI [{c['nag_lo']:.4f}, {c['nag_hi']:.4f}]  "
                  f"(worst persona: {c['nag_info']['nag_worst_persona']})")
            print(f"  |  Mean DSS (raw/stripped) : {c['mean_dss']:.4f} / {c['mean_dss_stripped']:.4f}")
            sdss = c["mean_structured_dss"]
            sdss_str = f"{sdss:.4f}" if not np.isnan(sdss) else "NaN"
            print(f"  |  Mean Structured DSS (Agent 1 JSON) : {sdss_str}")
            print(f"  |  DSS<{DSS_THRESHOLD}   : {c['frac_below']*100:.2f}% flagged")
            print(f"  |  Bonferroni alpha = 0.05/{c['n_tests']} = {c['bonf_alpha']:.5f}")
            for pair, p in c["mcnemar"].items():
                print(f"  |  McNemar p({pair[0][0]},{pair[1][0]}) = {p:.4f}  "
                      f"[raw:{mc.significance_label(p, 1)}]  "
                      f"[Bonferroni:{mc.significance_label(p, c['n_tests'])}]")
            print(f"  |  Parse Rate : {c['parse_rate']*100:.1f}%  |  "
                  f"Agent1 Parse Rate: {c['a1_rate']*100:.1f}%  |  "
                  f"Agent3 Fallback: {c['a3_rate']*100:.1f}%")
            if ablation == "full":
                print(f"  |  Avg tools/q : {c['mean_tools']:.2f}")
            print(f"  |  Avg latency : {c['mean_lat']:.1f}s/q")
            print(f"  +{'-'*58}")

            abl_primary.append(result["primary"])
            abl_secondary.append(result["secondary"])
            abl_pipeline.append(result["pipeline"])
            abl_bias.extend(result["bias_cases"])

        # Write per-ablation CSVs
        if abl_primary:
            df = pd.DataFrame(abl_primary)
            path = f"{OUTPUT_DIR}/ns_evaluation_primary_{ablation}.csv"
            df.to_csv(path, index=False)
            print(f"\n  [OK] Primary    -> {path}")
            print(df.to_string(index=False))

        if abl_secondary:
            df = pd.DataFrame(abl_secondary)
            path = f"{OUTPUT_DIR}/ns_evaluation_secondary_{ablation}.csv"
            df.to_csv(path, index=False)
            print(f"  [OK] Secondary  -> {path}")

        if abl_pipeline:
            df = pd.DataFrame(abl_pipeline)
            path = f"{OUTPUT_DIR}/ns_evaluation_pipeline_{ablation}.csv"
            df.to_csv(path, index=False)
            print(f"  [OK] Pipeline   -> {path}")

        if abl_bias:
            df = pd.DataFrame(abl_bias).sort_values(["model", "dss"])
            path = f"{OUTPUT_DIR}/ns_evaluation_bias_cases_{ablation}.csv"
            df.to_csv(path, index=False)
            print(f"  [OK] Bias cases -> {path}  ({len(df)} total)")

        all_primary.extend(abl_primary)
        all_secondary.extend(abl_secondary)
        all_pipeline.extend(abl_pipeline)
        all_bias.extend(abl_bias)

    # ── Combined table: all ablations x all models ────────────
    if all_primary:
        df_combined = pd.DataFrame(all_primary)
        path = f"{OUTPUT_DIR}/ns_evaluation_combined.csv"
        df_combined.to_csv(path, index=False)
        print(f"\n  [OK] Combined table -> {path}")

        print("\n" + "=" * 70)
        print("  ABLATION COMPARISON SUMMARY  (DSS, Structured DSS, NAG)")
        print("=" * 70)
        for model_key in MODEL_KEYS:
            subset = df_combined[df_combined["Model"] == model_key]
            if subset.empty:
                continue
            print(f"\n  {model_key}:")
            for _, row in subset.iterrows():
                print(f"    {row['Ablation']:12s}  NAG={row['NAG (headline)']:>8}  "
                      f"DSS={row['Mean DSS (raw)']:>8}  "
                      f"StructDSS={row['Mean Structured DSS']:>8}  "
                      f"DSS<0.8={row[f'DSS<{DSS_THRESHOLD} (%)']:>8}")

    # ── Interpretation notes ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  METRIC REFERENCE")
    print("=" * 70)
    print(f"""
OMR (Option Match Rate)
  Binary accuracy — correct answer letter selected.
  Reported per persona (alpha/beta/gamma) and overall with 95% Wilson CIs.

NAG (Narrative Anchoring Gap) = OMR_alpha - min(OMR_marked)
  Primary equity metric, always reported with its bootstrap 95% CI and every
  individual component gap alongside the headline scalar — never as a bare
  number. NarrativeShield should show NAG approaching 0.

DSS (Diagnostic Stability Score) — text-level, Agent 2 reasoning
  Mean pairwise cosine similarity of Agent 2 raw reasoning outputs across
  the present persona presentations of the same question. Reported both RAW
  and format-controlled ("stripped": NS_reasoning-protocol boilerplate
  headers removed). DSS = 1.0 -> identical reasoning regardless of patient
  voice. DSS < 0.80 -> flagged for bias review. Captures semantic drift even
  when final answer letters match.

Structured DSS — field-level, Agent 1 JSON extraction (NEW this revision)
  Field-level exact/fuzzy match rate across Agent 1's structured clinical
  JSON extraction (parsed from agent1_json_str) for the same question
  across personas. Unlike text DSS, this is NOT an embedding similarity —
  it directly measures whether EXTRACTION is register-invariant, separate
  from whether Agent 2's downstream reasoning converges. A dissociation
  between DSS and Structured DSS (e.g. high text DSS despite low structured
  DSS) indicates Agent 2 is recovering from inconsistent extraction rather
  than extraction itself being stable — an important distinction for the
  architectural claim this paper makes about NarrativeShield.

Cohen's Kappa (kappa): secondary, reported for completeness.
  High-accuracy paradox: sparse errors inflate apparent inconsistency.
  DSS is interpretively primary.

McNemar: directional bias test, reported against BOTH the raw 0.05
  threshold and the Bonferroni-corrected threshold (0.05 / n_tests, where
  n_tests = actual pairwise comparisons for that model/ablation). For
  NarrativeShield we expect p > Bonferroni-corrected-alpha (ns) on all
  pairs — no systematic directional disadvantage for either equity persona.
  Only cite a pair as significant if it clears the Bonferroni threshold.

Parse Rate
  Fraction of (question, persona) pairs where final_answer is a clean
  A/B/C/D letter, REGARDLESS of which stage of the pipeline produced it
  (Agent 1 direct, Agent 2, or Agent 3 fallback all count).

Agent1 Parse Rate
  Fraction of (question, persona) pairs satisfying the COMPOUND condition:
  Agent 1's own JSON parse succeeded (agent1_parse_success) AND a non-empty
  final_answer resulted. This is deliberately stricter and can be far lower
  than Parse Rate above — e.g. 100% Parse Rate coexisting with ~1% Agent1
  Parse Rate is a known, documented pipeline phenomenon (Agent 2/3 salvaging
  the answer despite Agent 1 extraction failing), not a bug to reconcile.

Agent3 Fallback Rate
  Fraction of (question, persona) pairs where answer_source ==
  "agent3_fallback" exactly (Agent 2 produced no parseable ANSWER_CANDIDATE
  and Agent 3 was invoked as emergency extractor). Should be low (< 5%).
  High rates indicate the Agent 2 prompt is not being followed and token
  budget may be too tight.

Mean Tools/Question (NS_full only)
  Average number of tools invoked per question by the deterministic router.
  Documents tool usage for the supplementary materials table.
""")


if __name__ == "__main__":
    main()
