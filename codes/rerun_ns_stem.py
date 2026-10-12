"""
NarrativeShield v2 rerun: reasoning step without the original vignette.
========================================================================
BUG IN run_ns.py (all versions, including the one used for the first
submission): the reasoning call (Agent 2, and the answer call in the
agent1only ablation) received `question = row["question"]`, which is the FULL
ORIGINAL MedQA VIGNETTE in clinical wording, identical for all three personas.
The reasoning model therefore saw the clean clinical case regardless of the
persona, and the NarrativeShield results are not valid.

Fix used here: the reasoning call receives only the item's QUESTION SENTENCE
(e.g. "Which of the following is the most likely diagnosis?"), taken from the
original item, together with Agent 1's JSON and the options. The narrative and
the vignette never reach the reasoning call.

Agent 1 is NOT rerun: it only ever saw the persona narrative, so its saved
JSON (agent1_json_str in every results_NS_*.jsonl) is valid and reused as is.
Tool routing (full ablation) is recomputed from that JSON and the stem.

Input : {OLD_DIR}/results_NS_{ablation}_{model}.jsonl   (old results, read only)
Output: {NEW_DIR}/results_NS_{ablation}_{model}.jsonl   (same schema + "question_stem")
Resume-safe: questions already in the output file are skipped.

Usage (one model x one ablation per job):
  python rerun_ns_stem.py --model qwen25_7b --ablation agent1only
"""
import argparse
import ast
import json
import os
import re
import time

SCRATCH_BASE = os.environ.get("SCRATCH_BASE", "/path/to/scratch")
OLD_DIR = os.environ.get("NS_OLD_DIR", f"{SCRATCH_BASE}/ns_results")
NEW_DIR = os.environ.get("NS_NEW_DIR", f"{SCRATCH_BASE}/ns_results_v2")
PERSONAS = ["alpha", "beta", "gamma"]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_LAB_TAIL = re.compile(r"\d[\d.,]*\s*(?:mg/dL|mEq/L|mmol/L|mOsm/kg|IU/L|U/L|g/dL|/hpf|/mm3|%|μm3|pg/cell)\)?|\b(?:Positive|Negative|Many)\b\)?")
_QWORD = re.compile(r"\b(Which|What|Where|Why|How|Who|When|In which|At which|Based on|Given|Assuming|If)\b")


def question_stem(question: str) -> str:
    """Final question sentence of a MedQA item (the part after the vignette).

    Lab tables often have no sentence breaks, so the stem is anchored on the
    last question word before the final '?' rather than on sentence splitting.
    """
    text = re.sub(r"\s+", " ", (question or "").strip()).rstrip("\"' ")
    q = text.rfind("?")
    if q != -1:
        head = text[:q]
        starts = [m.start() for m in _QWORD.finditer(head)]
        sent_start = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
        if starts and starts[-1] > sent_start:
            stem = text[starts[-1]:q + 1]
        else:
            stem = text[sent_start + 2 if sent_start != -1 else 0:q + 1].strip()
        # Drop a lab table that runs into a mid-sentence question ("... 20 mmol/L A deficiency of which ...?")
        lab_end = list(_LAB_TAIL.finditer(stem))
        if len(stem.split()) > 25 and lab_end:
            stem = stem[lab_end[-1].end():].strip()
        return stem
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return sents[-1] if sents else text


def load_done(path: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["question_id"])
                except Exception:
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ablation", choices=["agent1only", "notool", "full"], required=True)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from tqdm import tqdm
    import run_ns as R  # reuse loaders, prompts, parsers and tools unchanged

    os.makedirs(NEW_DIR, exist_ok=True)
    old_path = f"{OLD_DIR}/results_NS_{args.ablation}_{args.model}.jsonl"
    new_path = f"{NEW_DIR}/results_NS_{args.ablation}_{args.model}.jsonl"
    old = {}
    with open(old_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                old[r["question_id"]] = r
    rows = {r["question_id"]: r for r in load_dataset("csv", data_files=R.DATASET_NAME, split="train")}
    done = load_done(new_path)
    todo = [q for q in old if q not in done]
    print(f"{args.model}/{args.ablation}: {len(old)} old rows, {len(done)} done, {len(todo)} to run -> {new_path}")
    if not todo:
        return

    repo_id, strategy = R.MODEL_REGISTRY[args.model]
    loader_type, obj = R.load_model(args.model, repo_id, strategy)
    n_corr = {p: 0 for p in PERSONAS}

    with open(new_path, "a", encoding="utf-8") as out:
        for qid in tqdm(todo, desc=f"{args.model}/{args.ablation}"):
            row, prev = rows[qid], old[qid]
            try:
                options = ast.literal_eval(row["options"])
            except Exception:
                options = row["options"] if isinstance(row["options"], dict) else {}
            stem = question_stem(row["question"])
            rec = {"question_id": qid, "correct_answer_idx": row["answer_idx"], "meta_info": row.get("meta_info", ""),
                   "ablation": args.ablation, "model": args.model, "question_stem": stem, "personas_eval": {}}
            for p in PERSONAS:
                pe = prev["personas_eval"][p]
                json_str = pe.get("agent1_json_str") or "{}"
                t0 = time.time()
                tool_logs, tool_context, a2_raw, a3_raw = [], None, None, None
                try:
                    if args.ablation == "agent1only":
                        a3_raw = R.infer(loader_type, obj, R.build_agent1only_messages(args.model, json_str, stem, options), R.TOKENS_AGENT3)
                        final = R.extract_answer_letter(a3_raw) or R.letter_from_option_text(a3_raw, options)
                        source = "agent1only"
                    else:
                        if args.ablation == "full":
                            _, clinical = R.extract_clinical_json(json_str)
                            outs, tool_logs = R.run_tool_router(clinical, stem, options)
                            tool_context = "\n".join(f"• {o}" for o in outs) if outs else None
                        a2_raw = R.infer(loader_type, obj, R.build_agent2_messages(args.model, json_str, stem, options, tool_context), R.TOKENS_AGENT2)
                        final = R.extract_answer_letter(a2_raw)
                        source = "agent2_direct"
                        if not final:
                            a3_raw = R.infer(loader_type, obj, R.build_agent3_messages(args.model, a2_raw, options), R.TOKENS_AGENT3)
                            final = R.extract_answer_letter(a3_raw) or R.extract_answer_letter(a2_raw) or R.letter_from_option_text(a3_raw, options)
                            source = "agent3_fallback"
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    torch.cuda.empty_cache()
                    final, source = None, "oom"
                ok = final == row["answer_idx"]
                n_corr[p] += ok
                rec["personas_eval"][p] = {
                    "narrative_length": pe.get("narrative_length"),
                    "agent1_parse_success": pe.get("agent1_parse_success", False),
                    "agent1_json_str": pe.get("agent1_json_str"),
                    "tool_context": tool_context, "agent2_raw": a2_raw, "agent3_raw": a3_raw,
                    "final_answer": final, "answer_source": source, "is_correct": ok,
                    "tools_called": len(tool_logs),
                    "pipeline_latency_sec": round(time.time() - t0, 2),
                }
            out.write(json.dumps(rec) + "\n")
            out.flush()

    n = len(todo)
    print("accuracy on rerun rows: " + ", ".join(f"{p} {n_corr[p] / n:.3f}" for p in PERSONAS))
    R.unload_model(loader_type, obj)


if __name__ == "__main__":
    main()
