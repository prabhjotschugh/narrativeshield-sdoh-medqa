"""
NarrativeShield — B1 Direct Prompting Baseline (Final)
=======================================================
Runs 12 generative models sequentially. Each model is fully evaluated
then unloaded before the next one loads — you never hold two models
in VRAM at the same time.

Run order:
  1.  llama32_3b          meta-llama/Llama-3.2-3B-Instruct
  2.  medgemma_4b         google/medgemma-4b-it
  3.  phi4_mini           microsoft/Phi-4-mini-instruct
  4.  qwen25_7b           Qwen/Qwen2.5-7B-Instruct
  5.  mistral7b           mistralai/Mistral-7B-Instruct-v0.3
  6.  qwen3_8b            Qwen/Qwen3-8B
  7.  ministral8b         mistralai/Ministral-8B-Instruct-2410
  8.  llama31_8b          meta-llama/Llama-3.1-8B-Instruct
  9.  openbiollm_8b       aaditya/Llama3-OpenBioLLM-8B
  10. gemma3_12b          google/gemma-3-12b-it
  11. qwen25_14b          Qwen/Qwen2.5-14B-Instruct
  12. mistral_small_24b   mistralai/Mistral-Small-24B-Instruct-2501

Resume-safe: each model writes to its own .jsonl file.
Re-running after a preemption skips already-completed question_ids.

Output files (written to OUTPUT_DIR):
  results_B1_{model_key}.jsonl   (one per MODEL_REGISTRY key)

Usage
-----
  python run_b1.py                    # all 12 models, one process
  python run_b1.py --model llama31_8b # one model (SLURM array-job friendly,
                                       # matches run_b2.py/run_b3.py/run_ns.py)
"""

# ─────────────────────────────────────────────────────────────────────────────
# CACHE REDIRECT — must happen before any HuggingFace import
# ─────────────────────────────────────────────────────────────────────────────
import os
import shutil

# HPC (aarch64 GH nodes) defaults CC to NVIDIA's `nvc`, which Triton then uses
# to JIT-compile helpers and chokes on gcc-only flags (-Wno-psabi), crashing
# generate(). Force gcc before torch/transformers are imported.
_gcc = shutil.which("gcc")
if _gcc:
    os.environ["CC"] = _gcc
    os.environ["CXX"] = shutil.which("g++") or _gcc

SCRATCH_BASE = "/path/to/scratch"
HF_CACHE_DIR = f"{SCRATCH_BASE}/hf_cache"
OUTPUT_DIR   = f"{SCRATCH_BASE}/b1_results"
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,   exist_ok=True)

for _var, _path in [
    ("HF_HOME",               HF_CACHE_DIR),
    ("HF_DATASETS_CACHE",     f"{HF_CACHE_DIR}/datasets"),
    ("TRANSFORMERS_CACHE",    f"{HF_CACHE_DIR}/transformers"),
    ("HUGGINGFACE_HUB_CACHE", f"{HF_CACHE_DIR}/hub"),
    ("HF_HUB_CACHE",          f"{HF_CACHE_DIR}/hub"),
]:
    os.environ[_var] = _path

# ─────────────────────────────────────────────────────────────────────────────
# STANDARD IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import ast
import gc
import json
import re
import time
import warnings
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import login, snapshot_download
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    pipeline,
)

warnings.filterwarnings("ignore", category=UserWarning)

# TF32 gives a large free speedup on A100 / H100
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
HF_TOKEN            = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN_HERE")
DATASET_NAME        = os.environ.get("DATASET_NAME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset", "medqa_1000.csv"))
MAX_NEW_TOKENS      = 128
QUESTIONS_PER_CHUNK = 16   # lower to 8 if you hit CUDA OOM
NUM_WORKERS         = 4    # CPU workers for parallel prompt prep
PERSONAS            = ["alpha", "beta", "gamma"]

# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# Ordered dict — execution order is exactly this sequence.
# key → (hf_repo_id, output_jsonl_path, load_strategy)
#
# load_strategy options:
#   "pipeline_text"    — standard pipeline("text-generation", ...)
#   "gemma3_manual"    — Gemma3ForConditionalGeneration + AutoProcessor
#                        (manual download required for HPC Lustre filesystem)
#   "medgemma_manual"  — same manual-load pattern as gemma3_manual
#                        (MedGemma-4B-it is Gemma-3-architecture)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "llama32_3b":        (
        "meta-llama/Llama-3.2-3B-Instruct",
        f"{OUTPUT_DIR}/results_B1_llama32_3b.jsonl",
        "pipeline_text",
    ),
    "medgemma_4b":       (
        "google/medgemma-4b-it",
        f"{OUTPUT_DIR}/results_B1_medgemma_4b.jsonl",
        "medgemma_manual",
    ),
    "phi4_mini":         (
        "microsoft/Phi-4-mini-instruct",
        f"{OUTPUT_DIR}/results_B1_phi4_mini.jsonl",
        "pipeline_text",
    ),
    "qwen25_7b":         (
        "Qwen/Qwen2.5-7B-Instruct",
        f"{OUTPUT_DIR}/results_B1_qwen25_7b.jsonl",
        "pipeline_text",
    ),
    "mistral7b":         (
        "mistralai/Mistral-7B-Instruct-v0.3",
        f"{OUTPUT_DIR}/results_B1_mistral7b.jsonl",
        "pipeline_text",
    ),
    "qwen3_8b":          (
        "Qwen/Qwen3-8B",
        f"{OUTPUT_DIR}/results_B1_qwen3_8b.jsonl",
        "pipeline_text",
    ),
    "ministral8b":       (
        "mistralai/Ministral-8B-Instruct-2410",
        f"{OUTPUT_DIR}/results_B1_ministral8b.jsonl",
        "pipeline_text",
    ),
    "llama31_8b":        (
        "meta-llama/Llama-3.1-8B-Instruct",
        f"{OUTPUT_DIR}/results_B1_llama31_8b.jsonl",
        "pipeline_text",
    ),
    "openbiollm_8b":     (
        "aaditya/Llama3-OpenBioLLM-8B",
        f"{OUTPUT_DIR}/results_B1_openbiollm_8b.jsonl",
        "pipeline_text",
    ),
    "gemma3_12b":        (
        "google/gemma-3-12b-it",
        f"{OUTPUT_DIR}/results_B1_gemma3_12b.jsonl",
        "gemma3_manual",
    ),
    "qwen25_14b":        (
        "Qwen/Qwen2.5-14B-Instruct",
        f"{OUTPUT_DIR}/results_B1_qwen25_14b.jsonl",
        "pipeline_text",
    ),
    "mistral_small_24b": (
        "mistralai/Mistral-Small-24B-Instruct-2501",
        f"{OUTPUT_DIR}/results_B1_mistral_small_24b.jsonl",
        "pipeline_text",
    ),
}

# Models that require content-block message format ([{"type": "text", ...}])
# instead of plain string content — both are manually-loaded Gemma-family
# checkpoints (gemma3_manual / medgemma_manual loader strategies).
CONTENT_BLOCK_MODELS = {"gemma3_12b", "medgemma_4b"}

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER — B1 is completely unmitigated zero-shot
# ─────────────────────────────────────────────────────────────────────────────
def build_vanilla_prompt(narrative: str, options: dict) -> str:
    body = narrative + "\n\nOptions:\n"
    for k, v in options.items():
        body += f"{k}: {v}\n"
    body += "\nWhich of the following is the correct option? Provide only the single correct letter."
    return body


def build_messages(model_key: str, prompt_text: str) -> list:
    """Returns the correct chat-message structure for each model family."""
    if model_key == "mistral7b":
        # Mistral v0.3 expects an explicit (even empty) system turn
        return [
            {"role": "system", "content": ""},
            {"role": "user",   "content": prompt_text},
        ]
    elif model_key in CONTENT_BLOCK_MODELS:
        # Manually-loaded Gemma-family models (gemma3_12b, medgemma_4b):
        # processor expects content as a list of typed blocks
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            }
        ]
    else:
        # all other models — standard single-turn
        return [{"role": "user", "content": prompt_text}]


# ─────────────────────────────────────────────────────────────────────────────
# ANSWER PARSER — deterministic regex cascade, no LLM calls
# ─────────────────────────────────────────────────────────────────────────────
def extract_answer_letter(text: str):
    if not text:
        return None
    text = text.strip()

    # 1. Natural language prefix: "The answer is B", "Correct option: C", etc.
    m = re.search(
        r'(?i)(?:the\s+)?(?:correct\s+)?(?:option|answer)\s+is\s*:?\s*([A-D])\b',
        text,
    )
    if m:
        return m.group(1).upper()

    # 2. Exact single-letter line: "B" or "B." or "B)"
    m = re.match(r'^([A-D])[\.\):]?\s*$', text)
    if m:
        return m.group(1).upper()

    # 3. Letter wrapped in markdown bold or quotes: **B**, "B", `B`
    m = re.search(r'[\*\'\"`]([A-D])[\*\'\"`]', text)
    if m:
        return m.group(1).upper()

    # 4. Fallback — first standalone A/B/C/D word boundary found in text
    m = re.search(r'\b([A-D])\b', text)
    if m:
        return m.group(1).upper()

    return None


# ─────────────────────────────────────────────────────────────────────────────
# TORCH DATASET — CPU workers pre-build prompts while GPU infers
# ─────────────────────────────────────────────────────────────────────────────
class PersonaDataset(Dataset):
    """Flat dataset: one record per (question × persona) pair."""

    def __init__(self, rows: list, model_key: str):
        self.records = []
        for row in rows:
            options = ast.literal_eval(row["options"])
            for persona in PERSONAS:
                narrative = row[f"persona_{persona}"]
                prompt    = build_vanilla_prompt(narrative, options)
                messages  = build_messages(model_key, prompt)
                self.records.append({
                    "question_id": row["question_id"],
                    "answer_idx":  row["answer_idx"],
                    "persona":     persona,
                    "messages":    messages,
                })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def collate_fn(batch):
    """Keep records as a plain list — no tensor stacking needed."""
    return batch


# ─────────────────────────────────────────────────────────────────────────────
# RESUME HELPER
# ─────────────────────────────────────────────────────────────────────────────
def load_processed_ids(output_file: str) -> set:
    """Return set of question_ids already written to the output file."""
    done = set()
    p = Path(output_file)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["question_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


# ─────────────────────────────────────────────────────────────────────────────
# SCRATCH DOWNLOAD HELPER
# snapshot_download with local_dir_use_symlinks=False is required on HPC
# because the Lustre scratch filesystem does not support symlinks.
# ─────────────────────────────────────────────────────────────────────────────
def _download_to_scratch(repo_id: str, subdir: str) -> str:
    local = f"{HF_CACHE_DIR}/{subdir}"
    # Always call snapshot_download: it resumes partial downloads and is a
    # near-instant no-op when complete. (Checking only isdir() was a bug - a
    # failed/gated download leaves an empty folder that looked "cached".)
    print(f"  Syncing {repo_id} -> {local}")
    try:
        snapshot_download(
            repo_id=repo_id,
            token=HF_TOKEN,
            local_dir=local,
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        if os.path.isfile(os.path.join(local, "config.json")):
            print(f"  [warn] sync failed ({type(e).__name__}); using existing files in {local}")
        else:
            raise
    return local


LLAMA3_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'"
    " + message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)


def _build_text_pipeline(repo_id: str):
    """
    Text-generation pipeline with three model-specific fixes found in the
    HPC smoke test (transformers v5):
      1. Mistral tekken-tokenizer repos (Ministral-8B, Mistral-Small-24B):
         pipeline() forwards device_map/_from_pipeline kwargs into
         MistralCommonBackend.from_pretrained, which rejects them. Fall back
         to loading tokenizer and model separately.
      2. OpenBioLLM ships no chat template -> use the Llama-3 template.
      3. Qwen3 defaults to "thinking" mode, which burns the whole token
         budget on <think> text before answering -> disable thinking.
    """
    try:
        pipe = pipeline("text-generation", model=repo_id, device_map="auto",
                        torch_dtype=torch.bfloat16, token=HF_TOKEN)
    except ValueError as e:
        if "MistralCommonBackend" not in str(e):
            raise
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(repo_id, token=HF_TOKEN)
        mdl = AutoModelForCausalLM.from_pretrained(
            repo_id, device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN)
        pipe = pipeline("text-generation", model=mdl, tokenizer=tok)

    if "OpenBioLLM" in repo_id and not getattr(pipe.tokenizer, "chat_template", None):
        pipe.tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
    if "Qwen3" in repo_id:
        import functools
        pipe.tokenizer.apply_chat_template = functools.partial(
            pipe.tokenizer.apply_chat_template, enable_thinking=False)
    return pipe


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADERS
# ─────────────────────────────────────────────────────────────────────────────
def load_pipeline_text(repo_id: str):
    """Standard HF text-generation pipeline. Works for most models."""
    pipe = _build_text_pipeline(repo_id)
    # Prevent padding warnings during batch inference
    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    return "pipeline", pipe


def load_gemma3_manual(repo_id: str):
    """
    Gemma 3 12B — manual download + Gemma3ForConditionalGeneration.
    Required because the HF pipeline task is image-text-to-text and
    HPC Lustre needs symlink-free downloads.
    """
    local = _download_to_scratch(repo_id, "gemma-3-12b-it")
    proc  = AutoProcessor.from_pretrained(local, token=HF_TOKEN)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        local,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    ).eval()
    return "processor_model", (proc, model)


def load_medgemma_manual(repo_id: str):
    """
    MedGemma-4B-it — manual download + Gemma3ForConditionalGeneration.
    Same HPC Lustre pattern as Gemma 3 (MedGemma is Gemma-3-architecture).
    Confirmed: google/medgemma-4b-it's config.json lists
    architectures=["Gemma3ForConditionalGeneration"], model_type="gemma3".
    """
    local = _download_to_scratch(repo_id, "medgemma-4b-it")
    proc  = AutoProcessor.from_pretrained(local, token=HF_TOKEN)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        local,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    ).eval()
    return "processor_model", (proc, model)


def load_model(model_key: str, repo_id: str, strategy: str):
    print(f"  Loading {model_key}  ({repo_id}) …")
    t0 = time.time()
    if   strategy == "pipeline_text":    loader_type, obj = load_pipeline_text(repo_id)
    elif strategy == "gemma3_manual":    loader_type, obj = load_gemma3_manual(repo_id)
    elif strategy == "medgemma_manual":  loader_type, obj = load_medgemma_manual(repo_id)
    else:
        raise ValueError(f"Unknown load strategy: {strategy!r}")
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return loader_type, obj


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────
def run_inference_batch(loader_type: str, obj, batch: list) -> list:
    """
    Runs a batch through the loaded model.
    Returns a list of raw response strings, same length as batch.
    """
    messages_list = [item["messages"] for item in batch]

    if loader_type == "pipeline":
        pipe    = obj
        outputs = pipe(
            messages_list,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            batch_size=len(batch),
        )
        results = []
        for out in outputs:
            gen = out[0]["generated_text"]
            # HF pipeline returns the full conversation list;
            # last element is the new assistant turn
            if isinstance(gen, list):
                raw = gen[-1].get("content", str(gen[-1]))
            else:
                raw = str(gen)
            results.append(raw)
        return results

    elif loader_type == "processor_model":
        # Gemma 3 and Gemma 4: apply_chat_template → generate → decode
        proc, model = obj
        results     = []
        for messages in messages_list:
            inputs = proc.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )

            raw = proc.decode(gen_ids[0][input_len:], skip_special_tokens=True)
            results.append(raw)
        return results

    raise ValueError(f"Unknown loader_type: {loader_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# VRAM CLEANUP — called between every model
# ─────────────────────────────────────────────────────────────────────────────
def unload_model(loader_type: str, obj):
    if loader_type == "pipeline":
        del obj
    else:
        proc, model = obj
        del proc, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("  VRAM freed.\n")


# ─────────────────────────────────────────────────────────────────────────────
# PER-MODEL EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model_key: str, repo_id: str, strategy: str,
                   output_file: str, all_rows: list):

    print(f"\n{'='*65}")
    print(f"  MODEL  : {model_key}")
    print(f"  REPO   : {repo_id}")
    print(f"  OUTPUT : {output_file}")
    print(f"{'='*65}")

    # ── Resume: skip already-completed questions ───────────────────────────
    processed_ids = load_processed_ids(output_file)
    remaining     = [r for r in all_rows if r["question_id"] not in processed_ids]

    if not remaining:
        print("  All questions already processed — skipping this model.")
        return

    print(f"  Already done : {len(processed_ids)}")
    print(f"  To process   : {len(remaining)}")

    # ── Build flat task dataset; DataLoader runs prompt prep on CPU ────────
    ds     = PersonaDataset(remaining, model_key)
    loader = DataLoader(
        ds,
        batch_size=QUESTIONS_PER_CHUNK * len(PERSONAS),   # questions × 3 personas
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=(NUM_WORKERS > 0),
    )

    # ── Load model ─────────────────────────────────────────────────────────
    loader_type, obj = load_model(model_key, repo_id, strategy)

    # ── Inference loop ─────────────────────────────────────────────────────
    n_correct   = {p: 0 for p in PERSONAS}
    n_processed = 0

    with open(output_file, "a", encoding="utf-8") as out_f:
        for batch in tqdm(loader, desc=f"  [{model_key}]", unit="batch"):
            try:
                raw_responses = run_inference_batch(loader_type, obj, batch)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(
                        f"\n  ⚠ CUDA OOM on batch — skipping chunk.\n"
                        f"  Reduce QUESTIONS_PER_CHUNK (currently {QUESTIONS_PER_CHUNK})."
                    )
                    torch.cuda.empty_cache()
                    continue
                raise

            # ── Group results back by question_id ──────────────────────────
            by_qid: dict = {}
            for item, raw in zip(batch, raw_responses):
                qid = item["question_id"]
                if qid not in by_qid:
                    by_qid[qid] = {
                        "question_id":        qid,
                        "correct_answer_idx": item["answer_idx"],
                        "personas_eval":      {},
                    }
                extracted = extract_answer_letter(raw)
                correct   = extracted == item["answer_idx"]

                by_qid[qid]["personas_eval"][item["persona"]] = {
                    "raw_response":     raw,
                    "extracted_answer": extracted,
                    "is_correct":       correct,
                }
                if correct:
                    n_correct[item["persona"]] += 1

            # ── Write only fully-complete rows (all 3 personas present) ────
            # This means a crash mid-batch never writes a partial row,
            # so the resume mechanism is always safe.
            for qid, row in by_qid.items():
                if len(row["personas_eval"]) == len(PERSONAS):
                    out_f.write(json.dumps(row) + "\n")

            # Flush after every chunk — safe to kill at any point
            out_f.flush()
            n_processed += len(by_qid)

    # ── Per-persona accuracy summary ───────────────────────────────────────
    print(f"\n  ── Quick accuracy summary for {model_key} ──────────────")
    for p in PERSONAS:
        pct = 100.0 * n_correct[p] / max(n_processed, 1)
        print(f"     Persona {p}: {n_correct[p]} / {n_processed}  ({pct:.1f}%)")

    # ── Unload before next model ───────────────────────────────────────────
    unload_model(loader_type, obj)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=list(MODEL_REGISTRY.keys()) + ["all"], default="all",
        help="Run a single model (for one-model-per-SLURM-job submission, matching "
             "run_b2.py/run_b3.py/run_ns.py's per-job design) or 'all' to run every "
             "model sequentially in one process (the original single-script design). "
             "Default 'all' preserves prior behavior for local/non-SLURM use.",
    )
    args = parser.parse_args()

    registry = (
        MODEL_REGISTRY if args.model == "all"
        else {args.model: MODEL_REGISTRY[args.model]}
    )

    print("=" * 65)
    print("  NarrativeShield — B1 Direct Prompting Baseline")
    print(f"  {len(registry)} model(s) this run | greedy decoding | resume-safe")
    print(f"  Output directory : {OUTPUT_DIR}")
    print("=" * 65)
    print()
    print("  Run order:")
    for i, (key, (repo, _, _)) in enumerate(registry.items(), 1):
        print(f"    {i}. {key:<20} {repo}")
    print()

    # No login() call: HF_TOKEN is read from the environment and passed as token=...
    # everywhere. login() writes a shared stored_tokens file in the HF cache,
    # which concurrent jobs corrupt (race) and crash on.
    print("  HuggingFace auth OK\n")

    print(f"  Loading dataset: {DATASET_NAME} …")
    raw_ds   = load_dataset("csv", data_files=DATASET_NAME, split="train")
    all_rows = list(raw_ds)
    print(f"  Total questions : {len(all_rows)}\n")

    t_start = time.time()

    for model_key, (repo_id, output_file, strategy) in registry.items():
        evaluate_model(
            model_key=model_key,
            repo_id=repo_id,
            strategy=strategy,
            output_file=output_file,
            all_rows=all_rows,
        )

    elapsed = time.time() - t_start
    hrs, rem = divmod(int(elapsed), 3600)
    mins     = rem // 60
    print(f"\n{'='*65}")
    print(f"  {'ALL MODELS' if args.model == 'all' else args.model} COMPLETE.")
    print(f"  Total wall time : {hrs}h {mins}m")
    print(f"  Results in      : {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
