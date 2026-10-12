"""
NarrativeShield — Agent 1 Output Fact-Recall Audit
====================================================================================
Concern: the fact-preservation guarantee cited from dataset
section 2.3 concerns the DATASET's persona rewrites (original question vs.
Pα/Pβ/Pγ narrative), not Agent 1's extracted JSON at inference time. The
paper currently asserts "Agent 1 is the load-bearing debiasing step" without
directly auditing whether Agent 1's OUTPUT, per model, actually preserves
the same clinical facts as the narrative it was extracting from. This is a natural concern for any
extraction-based pipeline.

This script reuses the exact same independent-auditor-model logic already
built and validated for dataset construction (build_audit_prompt in
dataset/2. add_persona_code.py) — imported directly, not reimplemented —
but points it at (persona_narrative, agent1_json_as_text) pairs instead of
(original_question, persona_narrative) pairs. Same auditor, same
verification standard, applied one level deeper into the pipeline.

Method
------
For a stratified sample (default: 100 questions x 3 personas x 12 models =
3,600 audits), for each model's NarrativeShield run:
  1. Read Agent 1's raw JSON output (already logged in results/ns_results/,
     no new inference needed) for that (question, persona).
  2. Compare it against the persona narrative it was extracted FROM (not the
     original MedQA question — Agent 1 never sees the original, only the
     persona-rewritten narrative for whichever Pα/Pβ/Pγ variant is being
     evaluated).
  3. Call the SAME auditor prompt/model used for dataset validation, framed
     as "does this structured JSON preserve the facts in this narrative?"
  4. Record per-model, per-persona fact-recall rate.

This produces the evidence that Agent 1 genuinely preserves
clinical signal (or, if it doesn't for some model, that finding is itself
important and should be reported honestly rather than assumed away).

Cost note: this uses the Gemini auditor API (same as dataset construction),
NOT GPU compute — it can run on any machine with GEMINI_API_KEY set, in
parallel with or after the HPC GPU runs, since it only reads already-logged
Agent 1 outputs from results/ns_results/*.jsonl.

Outputs
-------
  agent1_fact_recall_{model}.csv — per-question, per-persona audit result
  agent1_fact_recall_summary.csv — per-model, per-persona aggregate recall rate

Usage
-----
  python audit_agent1_outputs.py [--sample-size 100] [--ablation full]
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics_common as mc

import json
import re

import google.generativeai as genai

# Self-contained copy of the auditor prompt/parsing used for dataset
# construction (dataset/2. add_persona_code.py, one-directional fact-
# preservation audit). Copied verbatim in behavior rather than imported, so
# this script has no dependency on the dataset/ folder (which is frozen and
# not present in the flat HPC folder).
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
AUDIT_MODEL_NAME = os.environ.get("AUDIT_MODEL", "gemini-3.1-flash-lite")
TEMP_AUDIT = 0.0
genai.configure(api_key=GEMINI_API_KEY)
_audit_model = None


def get_model(temperature: float = TEMP_AUDIT, is_audit: bool = True):
    global _audit_model
    if _audit_model is None:
        _audit_model = genai.GenerativeModel(
            model_name=AUDIT_MODEL_NAME,
            generation_config=genai.GenerationConfig(
                temperature=temperature, max_output_tokens=4096, candidate_count=1,
            ),
        )
    return _audit_model


def build_audit_prompt(original_question: str, persona_text: str, persona_name: str) -> str:
    return f"""You are a clinical fact auditor for a medical NLP research study.

TASK: Check whether the REWRITTEN NARRATIVE preserves all clinical facts from the ORIGINAL.

ORIGINAL:
{original_question}

REWRITTEN ({persona_name}):
{persona_text}

INSTRUCTIONS:
1. Extract every discrete clinical fact from the ORIGINAL (symptoms, durations, vitals,
   labs, exam findings, medications, history, negated findings, contextual details).
2. For each fact, check if it appears in the REWRITTEN in any form (a structured JSON
   field value counts as present). Absent = MISSING.
3. Return ONLY the JSON below. No prose. No markdown. No explanation before or after.
   Keep missing_fact_list entries SHORT (5 words max each).

{{"total_facts_extracted":<int>,"facts_present":<int>,"facts_missing":<int>,"missing_fact_list":["short label","short label"],"pass":<true|false>,"audit_note":"<one short sentence>"}}

Output the single-line JSON object above and nothing else.
""".strip()


def parse_audit_response(raw: str) -> dict:
    raw = re.sub(r"```(?:json)?\s*", "", (raw or "").strip()).strip().rstrip("`").strip()
    block = re.search(r"\{.*\}", raw, re.DOTALL)
    for candidate in (raw, block.group(0) if block else None):
        if not candidate:
            continue
        try:
            r = json.loads(candidate)
            r.setdefault("pass", False)
            r.setdefault("total_facts_extracted", 0)
            r.setdefault("facts_present", 0)
            r.setdefault("facts_missing", 0)
            r.setdefault("audit_note", "")
            return r
        except json.JSONDecodeError:
            continue
    return {"pass": False, "audit_note": f"Unparseable audit response: {raw[:150]}",
            "total_facts_extracted": 0, "facts_present": 0, "facts_missing": -1}

NS_RESULT_DIR = os.environ.get("NS_RESULT_DIR", "/path/to/scratch/ns_results")
# Default source: the same published HF dataset every run_*.py script reads
# from (dataset/medqa_1000.csv) — this repo does NOT
# contain a local medqa_1000_with_personas_v2.csv (that was the dataset-
# generation pipeline's intermediate output; the finalized, doctor-verified
# dataset lives on the Hub, not as a local file). Set DATASET_CSV to a local
# path instead if you have one and prefer not to hit the network.
DATASET_NAME = os.environ.get("DATASET_NAME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset", "medqa_1000.csv"))
DATASET_CSV = os.environ.get("DATASET_CSV")  # optional local override
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", NS_RESULT_DIR)

MODEL_KEYS = [m for m in mc.MODEL_KEYS if not os.environ.get('AUDIT_MODELS') or m in os.environ['AUDIT_MODELS'].split(',')]
PERSONAS = mc.PERSONAS
DEFAULT_SAMPLE_SIZE = 100
RETRY_DELAY_SEC = 6
MAX_RETRIES = 3


def load_persona_narratives(csv_path: str | None) -> dict:
    """
    Returns {question_id: {persona: narrative_text}}. Loads from a local CSV
    if DATASET_CSV was set, otherwise from the local dataset CSV
    (DATASET_NAME) — same source every run_*.py script uses.
    """
    if csv_path:
        p = Path(csv_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Dataset CSV not found at {csv_path} (set via DATASET_CSV env var)."
            )
        df = pd.read_csv(p)
    else:
        from datasets import load_dataset
        print(f"  Loading dataset from local CSV: {DATASET_NAME} ...")
        df = load_dataset("csv", data_files=DATASET_NAME, split="train").to_pandas()

    out = {}
    for _, row in df.iterrows():
        qid = row["question_id"]
        out[qid] = {p: row.get(mc.PERSONA_KEYS[p]) for p in PERSONAS if mc.PERSONA_KEYS[p] in df.columns}
    return out


def audit_one(narrative: str, agent1_json_str: str, persona: str) -> dict:
    """
    Runs the shared auditor prompt comparing Agent 1's structured JSON
    output against the persona narrative it was extracted from. Retries
    on transient API failure.
    """
    if not narrative or not agent1_json_str:
        return {"pass": False, "audit_note": "Missing narrative or Agent1 JSON — skipped.",
                "total_facts_extracted": 0, "facts_present": 0, "facts_missing": -1}

    prompt = build_audit_prompt(
        original_question=narrative,
        persona_text=agent1_json_str,
        persona_name=f"Agent1 JSON extraction of Persona {persona.upper()}",
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            audit_model = get_model(TEMP_AUDIT, is_audit=True)
            response = audit_model.generate_content(prompt)
            return parse_audit_response(response.text)
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC * attempt)
            else:
                return {"pass": False, "audit_note": f"Audit API error after {MAX_RETRIES} attempts: {e}",
                         "total_facts_extracted": 0, "facts_present": 0, "facts_missing": -1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                         help="Number of questions to sample per model (stratified random, seeded).")
    parser.add_argument("--ablation", default="full", choices=["full", "notool", "agent1only"],
                         help="Which NS ablation's Agent1 output to audit (agent1's extraction "
                              "step is identical across ablations, but result files are split by "
                              "ablation, so this selects which file to read from).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print(f"  Agent 1 Output Fact-Recall Audit  (sample={args.sample_size}/model, ablation={args.ablation})")
    print("=" * 70)

    narratives = load_persona_narratives(DATASET_CSV)
    print(f"Loaded persona narratives for {len(narratives)} questions.")

    summary_rows = []
    rng = random.Random(args.seed)

    for model_key in MODEL_KEYS:
        data = mc.load_jsonl_results(NS_RESULT_DIR, f"results_NS_{args.ablation}_" + "{model}.jsonl", model_key)
        if not data:
            print(f"\n  [skip] {model_key}: no NS results found for ablation={args.ablation}.")
            continue

        sampled = data if len(data) <= args.sample_size else rng.sample(data, args.sample_size)
        print(f"\n  {model_key}: auditing {len(sampled)} questions x {len(PERSONAS)} personas "
              f"= {len(sampled) * len(PERSONAS)} audits")

        from concurrent.futures import ThreadPoolExecutor
        jobs = []
        for row in sampled:
            qid = row.get("question_id")
            for persona in PERSONAS:
                jobs.append((qid, persona,
                             mc.safe_get(row, "personas_eval", persona, "agent1_json_str"),
                             mc.safe_get(row, "personas_eval", persona, "agent1_parse_success", default=False),
                             narratives.get(qid, {}).get(persona)))

        def run_job(j):
            qid, persona, js, ok, narrative = j
            base = {"model": model_key, "question_id": qid, "persona": persona}
            if not ok or not js:
                # Agent 1 produced no parseable JSON: recorded, not skipped.
                return {**base, "agent1_parse_success": False, "audit_pass": False, "facts_present": 0,
                        "facts_missing": -1, "total_facts_extracted": 0,
                        "audit_note": "Agent1 JSON extraction failed — no audit possible."}
            a = audit_one(narrative, js, persona)
            return {**base, "agent1_parse_success": True, "audit_pass": a.get("pass", False),
                    "facts_present": a.get("facts_present", 0), "facts_missing": a.get("facts_missing", -1),
                    "total_facts_extracted": a.get("total_facts_extracted", 0), "audit_note": a.get("audit_note", "")}

        with ThreadPoolExecutor(int(os.environ.get("AUDIT_WORKERS", "16"))) as ex:
            detail_rows = list(tqdm(ex.map(run_job, jobs), total=len(jobs), desc=f"  {model_key}"))

        df_detail = pd.DataFrame(detail_rows)
        detail_path = f"{OUTPUT_DIR}/agent1_fact_recall_{model_key}.csv"
        df_detail.to_csv(detail_path, index=False)
        print(f"  [OK] -> {detail_path}")

        for persona in PERSONAS:
            sub = df_detail[df_detail["persona"] == persona]
            if sub.empty:
                continue
            auditable = sub[sub["agent1_parse_success"]]
            recall_rate = (
                (auditable["facts_present"] / auditable["total_facts_extracted"].replace(0, pd.NA))
                .dropna().mean()
                if not auditable.empty else float("nan")
            )
            summary_rows.append({
                "model": model_key,
                "persona": persona,
                "n_sampled": len(sub),
                "n_agent1_parse_success": len(auditable),
                "agent1_parse_success_rate": len(auditable) / len(sub) if len(sub) else float("nan"),
                "mean_fact_recall_rate_when_parsed": recall_rate,
                "audit_pass_rate_when_parsed": auditable["audit_pass"].mean() if not auditable.empty else float("nan"),
            })

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        summary_path = f"{OUTPUT_DIR}/agent1_fact_recall_summary.csv"
        df_summary.to_csv(summary_path, index=False)
        print(f"\n[OK] Summary -> {summary_path}")
        print(df_summary.to_string(index=False))
    else:
        print("\n[WARN] No audits completed — check NS_RESULT_DIR and DATASET_CSV paths.")

    print("\n" + "=" * 70)
    print("  INTERPRETATION NOTE")
    print("=" * 70)
    print("""
"agent1_parse_success_rate" alone is the same misleading headline metric
flagged for BioMistral in the original submission (100% overall Parse Rate
next to ~1% genuine Agent1 JSON success) — always read it alongside
"mean_fact_recall_rate_when_parsed", which is conditioned on Agent 1 actually
having produced valid JSON. A model with low parse success AND low recall
when it does parse has a genuinely broken extraction step; a model with low
parse success but high recall when it does parse has a brittle-but-accurate
extractor. These are different findings and the paper should report which
one applies to each model, not a single blended number.
""")


if __name__ == "__main__":
    main()
