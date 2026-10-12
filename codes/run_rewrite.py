"""
Two new mitigation conditions (added after the run_ns.py vignette leak was fixed).
==============================================================================================
--method rewrite    Rewrite-then-Reason. Call 1: the evaluated model rewrites the patient's
                    narrative as a third-person clinical vignette (all facts kept, voice dropped).
                    Call 2: it answers from that rewrite + the question sentence + options.
                    The original narrative and the original MedQA vignette never reach call 2.
--method storyform  Story + Form. One call: the patient's narrative + the Agent-1 JSON already
                    saved in ns_results (agent1only file) + question sentence + options.

The question sentence comes from question_stem() in rerun_ns_stem.py (no vignette text).
Output: {NS_NEW_DIR}/results_NS_{method}_{model}.jsonl, same schema as the other NS files,
so check_v2.py and the analysis read it unchanged. Resume-safe (per question).

Usage: python run_rewrite.py --model qwen25_7b --method rewrite
"""
import argparse
import ast
import json
import os
import time

from rerun_ns_stem import question_stem, load_done, OLD_DIR, NEW_DIR, PERSONAS

TOKENS_REWRITE = 512
TOKENS_ANSWER = 16

REWRITE_SYSTEM = """You are a clinician writing up a patient's own account as a standard clinical case vignette.

Rewrite the patient narrative as a concise, third-person clinical vignette in standard medical terminology, the way a USMLE case is written.

KEEP, exactly and completely:
- the patient's age and sex (if the age is given only approximately, state it as given)
- every symptom, with its onset, duration, timing and course, including symptoms the patient denies
- every number: vitals, lab values, doses, durations, measurements
- all past medical, surgical, family, medication, allergy, exposure and social history that bears on health (e.g. smoking, alcohol, travel, occupation exposures, substances taken, remedies or medicines the patient took)
- all physical examination, laboratory and imaging findings

DROP: financial worries, insurance, work or family pressures, who brought the patient in, emotions, cultural or metaphorical phrasing, and any other wording that is about how the patient speaks rather than what happened clinically.

Do not add, infer or interpret anything that is not in the narrative. Do not suggest a diagnosis. Output only the vignette."""

ANSWER_SYSTEM = "Select the best answer letter (A, B, C, or D). Output exactly one letter."


def msgs(model_key, system, content, content_block_models):
    if model_key in content_block_models:
        return [{"role": "user", "content": [{"type": "text", "text": system + "\n\n" + content}]}]
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", "--ablation", dest="method", choices=["rewrite", "storyform"], required=True)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from tqdm import tqdm
    import run_ns as R

    os.makedirs(NEW_DIR, exist_ok=True)
    out_path = f"{NEW_DIR}/results_NS_{args.method}_{args.model}.jsonl"
    rows = {r["question_id"]: r for r in load_dataset("csv", data_files=R.DATASET_NAME, split="train")}

    saved_json = {}
    if args.method == "storyform":
        with open(f"{OLD_DIR}/results_NS_agent1only_{args.model}.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    saved_json[r["question_id"]] = {p: r["personas_eval"][p] for p in PERSONAS}

    done = load_done(out_path)
    todo = [q for q in rows if q not in done]
    print(f"{args.model}/{args.method}: {len(done)} done, {len(todo)} to run -> {out_path}")
    if not todo:
        return

    repo_id, strategy = R.MODEL_REGISTRY[args.model]
    loader_type, obj = R.load_model(args.model, repo_id, strategy)
    CB = R.CONTENT_BLOCK_MODELS
    n_corr = {p: 0 for p in PERSONAS}

    with open(out_path, "a", encoding="utf-8") as out:
        for qid in tqdm(todo, desc=f"{args.model}/{args.method}"):
            row = rows[qid]
            try:
                options = ast.literal_eval(row["options"])
            except Exception:
                options = row["options"] if isinstance(row["options"], dict) else {}
            opts = "\n".join(f"  {k}: {v}" for k, v in options.items())
            stem = question_stem(row["question"])
            rec = {"question_id": qid, "correct_answer_idx": row["answer_idx"], "meta_info": row.get("meta_info", ""),
                   "ablation": args.method, "model": args.model, "question_stem": stem, "personas_eval": {}}
            for p in PERSONAS:
                narrative = row[R.PERSONA_KEYS[p]].strip()
                t0 = time.time()
                rewrite, json_str, a1_ok, raw = None, None, None, None
                try:
                    if args.method == "rewrite":
                        rewrite = R.infer(loader_type, obj, msgs(args.model, REWRITE_SYSTEM,
                                          f"PATIENT NARRATIVE:\n{narrative}\n\nWrite the clinical vignette.", CB),
                                          TOKENS_REWRITE).strip()
                        content = f"CLINICAL VIGNETTE:\n{rewrite}\n\nQUESTION:\n{stem}\n\nOPTIONS:\n{opts}\n\nAnswer letter:"
                    else:
                        pe = saved_json.get(qid, {}).get(p, {})
                        json_str, a1_ok = pe.get("agent1_json_str") or "{}", pe.get("agent1_parse_success", False)
                        content = (f"PATIENT NARRATIVE:\n{narrative}\n\nEXTRACTED CLINICAL FACTS (JSON):\n{json_str}\n\n"
                                   f"QUESTION:\n{stem}\n\nOPTIONS:\n{opts}\n\nAnswer letter:")
                    raw = R.infer(loader_type, obj, msgs(args.model, ANSWER_SYSTEM, content, CB), TOKENS_ANSWER)
                    final = R.extract_answer_letter(raw) or R.letter_from_option_text(raw, options)
                    source = args.method
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    torch.cuda.empty_cache()
                    final, source = None, "oom"
                ok = final == row["answer_idx"]
                n_corr[p] += ok
                rec["personas_eval"][p] = {
                    "narrative_length": len(narrative), "rewrite": rewrite,
                    "agent1_json_str": json_str, "agent1_parse_success": a1_ok,
                    "agent3_raw": raw, "final_answer": final, "answer_source": source, "is_correct": ok,
                    "pipeline_latency_sec": round(time.time() - t0, 2),
                }
            out.write(json.dumps(rec) + "\n")
            out.flush()

    n = len(todo)
    print("accuracy on new rows: " + ", ".join(f"{p} {n_corr[p] / n:.3f}" for p in PERSONAS))
    R.unload_model(loader_type, obj)


if __name__ == "__main__":
    main()
