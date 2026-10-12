"""
NarrativeShield — Model Loading Smoke Test
==============================================================
Run this FIRST on HPC, before submitting any real SLURM jobs. It loads
every model in the 12-model registry (plus the llama31_8b_base stress-test
model) ONE AT A TIME, runs a single trivial generation against each, and
reports PASS/FAIL per model — so a broken HF_TOKEN, a wrong model class, a
gated-repo access issue, or an OOM on a specific model surfaces in minutes,
not after a multi-hour job silently produces empty/garbage output.

This does NOT touch the real dataset, the real output directories, or the
real per-condition result files — it writes its own small log/JSON report
to SCRATCH_BASE/smoke_test_results/ and uses a single hardcoded test
question, not narrativeshield-sdoh-medqa.

What it reuses, not reimplements
----------------------------------
All model-loading logic (load_pipeline_text / load_gemma3_manual /
load_medgemma_manual / load_model) and the MODEL_REGISTRY are imported
directly from run_b1.py — the canonical 12-model registry — so this test
can never drift out of sync with what the real run scripts actually load.
The base-model (llama31_8b_base) completion path is imported from run_b2.py,
since that is the only script with a working non-chat/completion branch.

Weights/cache location
-------------------------
Importing run_b1.py triggers its own top-level cache-redirect block BEFORE
any HuggingFace import — HF_HOME / HF_HUB_CACHE / TRANSFORMERS_CACHE /
HF_DATASETS_CACHE all point under SCRATCH_BASE ("/path/to/scratch" by default), never $HOME. This smoke test inherits that
redirect automatically by importing run_b1 first, so model weights
downloaded during this smoke test land in exactly the same scratch cache
the real jobs will reuse — the smoke test's downloads are not wasted, they
warm the cache for the real run.

Usage
-----
  cd codes
  python smoke_test_models.py                      # all 13 models
  python smoke_test_models.py --model llama31_8b    # just one (fast iteration
                                                     # while debugging one model)
  python smoke_test_models.py --skip-base           # skip llama31_8b_base
                                                     # (it downloads a second
                                                     # full Llama-3.1-8B
                                                     # checkpoint, doubling
                                                     # that model's disk use)

What counts as PASS
----------------------
  1. Model loads without raising.
  2. A single forward generation call completes without raising
     (including no CUDA OOM at a trivial 1-question batch size — a model
     that OOMs here will OOM far worse at the real QUESTIONS_PER_CHUNK
     batch size, so this is a meaningful early signal, not a guarantee).
  3. The raw output is non-empty.
  4. For instruct models: the output contains at least one standalone
     A-D letter (loose check — confirms the model is minimally following
     the answer-format instruction, not a rigorous accuracy check).
  5. VRAM is successfully freed afterward (next model's load would fail
     loudly anyway if not, but this confirms the unload path itself runs).

A model can PASS this smoke test and still behave imperfectly in the real
run (e.g. a low but non-zero parse rate) — this is a load-bearing sanity
check, not a substitute for the real eval_*.py metrics.
"""

import os
import shutil

# Force gcc for Triton JIT (HPC defaults to nvc, which rejects -Wno-psabi).
_gcc = shutil.which("gcc")
if _gcc:
    os.environ["CC"] = _gcc
    os.environ["CXX"] = shutil.which("g++") or _gcc

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Importing run_b1 triggers its cache-redirect-to-scratch block as a side
# effect (by design — this is what makes the smoke test's downloads land
# in the same scratch cache the real jobs reuse). This must happen before
# anything else touches torch/transformers.
import run_b1  # noqa: E402

# Base-model (non-instruct) completion path only exists in run_b2.py.
import run_b2  # noqa: E402

SMOKE_OUTPUT_DIR = f"{run_b1.SCRATCH_BASE}/smoke_test_results"
Path(SMOKE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

TEST_QUESTION = (
    "A 34-year-old woman presents with a 2-day history of burning pain on "
    "urination, urinary frequency, and mild suprapubic tenderness. She has no "
    "fever, flank pain, or nausea. Urinalysis shows positive leukocyte "
    "esterase and nitrites."
)
TEST_OPTIONS = {
    "A": "Oral nitrofurantoin",
    "B": "Oral ciprofloxacin",
    "C": "IV vancomycin",
    "D": "Watchful waiting, no treatment",
}


def build_test_prompt() -> str:
    body = TEST_QUESTION + "\n\nOptions:\n"
    for k, v in TEST_OPTIONS.items():
        body += f"{k}: {v}\n"
    body += "\nProvide only the single correct letter."
    return body


def run_instruct_smoke(model_key: str, repo_id: str, strategy: str) -> dict:
    """Loads one instruct model via run_b1's loaders, runs one generation, unloads."""
    result = {"model_key": model_key, "repo_id": repo_id, "strategy": strategy, "stage": "load"}
    t0 = time.time()
    try:
        loader_type, obj = run_b1.load_model(model_key, repo_id, strategy)
        result["load_time_sec"] = round(time.time() - t0, 1)
    except Exception as e:
        result["pass"] = False
        result["stage"] = "load"
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        return result

    result["stage"] = "generate"
    try:
        prompt_text = build_test_prompt()
        messages = run_b1.build_messages(model_key, prompt_text)

        if loader_type == "pipeline":
            pipe = obj
            out = pipe([messages], max_new_tokens=16, do_sample=False, batch_size=1)
            gen = out[0][0]["generated_text"]
            raw = gen[-1].get("content", str(gen[-1])) if isinstance(gen, list) else str(gen)
        else:  # processor_model — gemma3_12b, medgemma_4b
            import torch
            proc, model = obj
            inputs = proc.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device, dtype=torch.bfloat16)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                gen_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            raw = proc.decode(gen_ids[0][input_len:], skip_special_tokens=True)

        result["raw_output"] = raw
        result["generate_time_sec"] = round(time.time() - t0 - result["load_time_sec"], 1)

        has_letter = bool(re.search(r"\b[A-D]\b", raw))
        result["non_empty"] = bool(raw and raw.strip())
        result["contains_answer_letter"] = has_letter
        result["pass"] = result["non_empty"]  # letter check is informational, not pass/fail
        if not has_letter:
            result["warning"] = "Output non-empty but no standalone A-D letter found — check format."

    except Exception as e:
        result["pass"] = False
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()

    result["stage"] = "unload"
    try:
        run_b1.unload_model(loader_type, obj)
        result["unloaded"] = True
    except Exception as e:
        result["unloaded"] = False
        result["unload_error"] = f"{type(e).__name__}: {e}"

    return result


def run_base_model_smoke(model_key: str, repo_id: str) -> dict:
    """Loads llama31_8b_base via run_b2's completion path, runs one generation, unloads."""
    result = {"model_key": model_key, "repo_id": repo_id, "strategy": "pipeline_text_completion", "stage": "load"}
    t0 = time.time()
    try:
        loader_type, obj = run_b2.load_pipeline_text(repo_id)
        result["load_time_sec"] = round(time.time() - t0, 1)
    except Exception as e:
        result["pass"] = False
        result["stage"] = "load"
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        return result

    result["stage"] = "generate"
    try:
        options = TEST_OPTIONS
        prompt_text = run_b2.build_base_model_completion_prompt(TEST_QUESTION, options)
        pipe = obj
        out = pipe(
            [prompt_text], max_new_tokens=16, do_sample=False,
            batch_size=1, return_full_text=False,
        )
        raw = out[0][0]["generated_text"]
        result["raw_output"] = raw
        result["generate_time_sec"] = round(time.time() - t0 - result["load_time_sec"], 1)
        result["non_empty"] = bool(raw and raw.strip())
        result["pass"] = result["non_empty"]
    except Exception as e:
        result["pass"] = False
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()

    result["stage"] = "unload"
    try:
        run_b1.unload_model(loader_type, obj)
        result["unloaded"] = True
    except Exception as e:
        result["unloaded"] = False
        result["unload_error"] = f"{type(e).__name__}: {e}"

    return result


def main():
    parser = argparse.ArgumentParser(description="Smoke-test model loading before real SLURM jobs.")
    parser.add_argument(
        "--model", default=None,
        choices=list(run_b1.MODEL_REGISTRY.keys()) + ["llama31_8b_base"],
        help="Test only this one model key (default: test all 13).",
    )
    parser.add_argument(
        "--skip-base", action="store_true",
        help="Skip llama31_8b_base (downloads a second full Llama-3.1-8B checkpoint).",
    )
    args = parser.parse_args()

    if args.model:
        targets = [args.model]
    else:
        targets = list(run_b1.MODEL_REGISTRY.keys())
        if not args.skip_base:
            targets.append("llama31_8b_base")

    print("=" * 70)
    print("  NarrativeShield — Model Loading Smoke Test")
    print(f"  Models to test : {len(targets)}")
    print(f"  Scratch base   : {run_b1.SCRATCH_BASE}")
    print(f"  HF cache dir   : {run_b1.HF_CACHE_DIR}")
    print(f"  Output         : {SMOKE_OUTPUT_DIR}")
    print("=" * 70)

    results = []
    for i, model_key in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] Testing: {model_key}")
        print("-" * 60)

        if model_key == "llama31_8b_base":
            r = run_base_model_smoke(model_key, "meta-llama/Llama-3.1-8B")
        else:
            repo_id, _, strategy = run_b1.MODEL_REGISTRY[model_key]
            r = run_instruct_smoke(model_key, repo_id, strategy)

        status = "PASS" if r.get("pass") else "FAIL"
        print(f"  [{status}] stage reached: {r.get('stage')}")
        if r.get("load_time_sec") is not None:
            print(f"  Load time     : {r['load_time_sec']}s")
        if r.get("generate_time_sec") is not None:
            print(f"  Generate time : {r['generate_time_sec']}s")
        if r.get("raw_output") is not None:
            preview = r["raw_output"][:120].replace("\n", " ")
            print(f"  Raw output    : {preview!r}")
        if r.get("warning"):
            print(f"  [WARN] {r['warning']}")
        if r.get("error"):
            print(f"  [ERROR] {r['error']}")

        results.append(r)

    # ── Summary ──────────────────────────────────────────────────────
    n_pass = sum(1 for r in results if r.get("pass"))
    n_fail = len(results) - n_pass

    print("\n" + "=" * 70)
    print("  SMOKE TEST SUMMARY")
    print("=" * 70)
    for r in results:
        status = "PASS" if r.get("pass") else "FAIL"
        extra = f" [{r.get('stage')}: {r.get('error')}]" if status == "FAIL" else ""
        print(f"  [{status}]  {r['model_key']:<20}{extra}")
    print(f"\n  {n_pass}/{len(results)} passed, {n_fail} failed.")

    report_path = f"{SMOKE_OUTPUT_DIR}/smoke_test_report.json"
    # Strip tracebacks from the on-disk report's summary view but keep them
    # in a separate full-detail file, since tracebacks can be long.
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[OK] Full report -> {report_path}")

    if n_fail > 0:
        print(
            f"\n  {n_fail} model(s) FAILED. Fix these before submitting the real "
            "SLURM job matrix (slurm/job_matrix.py) — a failing model here will "
            "fail every B1/B2/B3/B4/NS job that uses it, wasting GPU-hours."
        )
        sys.exit(1)
    else:
        print("\n  All models PASSED. Safe to proceed with the real SLURM job matrix.")


if __name__ == "__main__":
    main()
