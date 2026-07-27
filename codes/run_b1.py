"""
NarrativeShield — B1 Direct Prompting Baseline (Final)
=======================================================
Runs 7 generative models sequentially. Each model is fully evaluated
then unloaded before the next one loads — you never hold two models
in VRAM at the same time.

Run order (llama32 intentionally last — access approval pending):
  1. llama31      meta-llama/Llama-3.1-8B-Instruct
  2. mistral7b    mistralai/Mistral-7B-Instruct-v0.3
  3. qwen25       Qwen/Qwen2.5-7B-Instruct
  4. gemma3_12b   google/gemma-3-12b-it
  5. gemma4_e4b   google/gemma-4-E4B-it
  6. biomistral   BioMistral/BioMistral-7B
  7. llama32      meta-llama/Llama-3.2-3B-Instruct   ← runs last

Resume-safe: each model writes to its own .jsonl file.
Re-running after a preemption skips already-completed question_ids.

Output files (written to OUTPUT_DIR):
  results_B1_llama31.jsonl
  results_B1_mistral7b.jsonl
  results_B1_qwen25.jsonl
  results_B1_gemma3_12b.jsonl
  results_B1_gemma4_e4b.jsonl
  results_B1_biomistral.jsonl
  results_B1_llama32.jsonl

Usage
-----
  python run_b1.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# CACHE REDIRECT — must happen before any HuggingFace import
# ─────────────────────────────────────────────────────────────────────────────
import os

SCRATCH_BASE = "/scratch/10778/prabhjotschugh"
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
    AutoModelForImageTextToText,
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
DATASET_NAME        = "Prabhjotschugh/narrativeshield-sdoh-medqa"
MAX_NEW_TOKENS      = 128
QUESTIONS_PER_CHUNK = 16   # lower to 8 if you hit CUDA OOM
NUM_WORKERS         = 4    # CPU workers for parallel prompt prep
PERSONAS            = ["alpha", "beta", "gamma"]

# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# Ordered dict — execution order is exactly this sequence.
# llama32 is intentionally last so HF access approval has time to arrive.
# key → (hf_repo_id, output_jsonl_path, load_strategy)
#
# load_strategy options:
#   "pipeline_text"  — standard pipeline("text-generation", ...)
#   "gemma3_manual"  — Gemma3ForConditionalGeneration + AutoProcessor
#                      (manual download required for TACC Lustre filesystem)
#   "gemma4_manual"  — AutoModelForImageTextToText + AutoProcessor
#                      (same manual download pattern)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "llama31":    (
        "meta-llama/Llama-3.1-8B-Instruct",
        f"{OUTPUT_DIR}/results_B1_llama31.jsonl",
        "pipeline_text",
    ),
    "mistral7b":  (
        "mistralai/Mistral-7B-Instruct-v0.3",
        f"{OUTPUT_DIR}/results_B1_mistral7b.jsonl",
        "pipeline_text",
    ),
    "qwen25":     (
        "Qwen/Qwen2.5-7B-Instruct",
        f"{OUTPUT_DIR}/results_B1_qwen25.jsonl",
        "pipeline_text",
    ),
    "gemma3_12b": (
        "google/gemma-3-12b-it",
        f"{OUTPUT_DIR}/results_B1_gemma3_12b.jsonl",
        "gemma3_manual",
    ),
    "gemma4_e4b": (
        "google/gemma-4-E4B-it",
        f"{OUTPUT_DIR}/results_B1_gemma4_e4b.jsonl",
        "gemma4_manual",
    ),
    "biomistral": (
        "BioMistral/BioMistral-7B",
        f"{OUTPUT_DIR}/results_B1_biomistral.jsonl",
        "pipeline_text",
    ),
    "llama32":    (
        "meta-llama/Llama-3.2-3B-Instruct",
        f"{OUTPUT_DIR}/results_B1_llama32.jsonl",
        "pipeline_text",
    ),
}

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
    elif model_key in ("gemma3_12b", "gemma4_e4b"):
        # Gemma processor expects content as a list of typed blocks
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            }
        ]
    else:
        # llama31, llama32, qwen25, biomistral — standard single-turn
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
# snapshot_download with local_dir_use_symlinks=False is required on TACC
# because the Lustre scratch filesystem does not support symlinks.
# ─────────────────────────────────────────────────────────────────────────────
def _download_to_scratch(repo_id: str, subdir: str) -> str:
    local = f"{HF_CACHE_DIR}/{subdir}"
    if not os.path.isdir(local):
        print(f"  Downloading {repo_id} → {local}  (runs once, cached after)")
        snapshot_download(
            repo_id=repo_id,
            token=HF_TOKEN,
            local_dir=local,
            local_dir_use_symlinks=False,
        )
    else:
        print(f"  Using cached download: {local}")
    return local


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADERS
# ─────────────────────────────────────────────────────────────────────────────
def load_pipeline_text(repo_id: str):
    """Standard HF text-generation pipeline. Works for most models."""
    pipe = pipeline(
        "text-generation",
        model=repo_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    # Prevent padding warnings during batch inference
    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    return "pipeline", pipe


def load_gemma3_manual(repo_id: str):
    """
    Gemma 3 12B — manual download + Gemma3ForConditionalGeneration.
    Required because the HF pipeline task is image-text-to-text and
    TACC Lustre needs symlink-free downloads.
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


def load_gemma4_manual(repo_id: str):
    """
    Gemma 4 E4B — manual download + AutoModelForImageTextToText.
    Same TACC Lustre pattern as Gemma 3.
    """
    local = _download_to_scratch(repo_id, "gemma-4-E4B-it")
    proc  = AutoProcessor.from_pretrained(local, token=HF_TOKEN)
    model = AutoModelForImageTextToText.from_pretrained(
        local,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    ).eval()
    return "processor_model", (proc, model)


def load_model(model_key: str, repo_id: str, strategy: str):
    print(f"  Loading {model_key}  ({repo_id}) …")
    t0 = time.time()
    if   strategy == "pipeline_text": loader_type, obj = load_pipeline_text(repo_id)
    elif strategy == "gemma3_manual": loader_type, obj = load_gemma3_manual(repo_id)
    elif strategy == "gemma4_manual": loader_type, obj = load_gemma4_manual(repo_id)
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
    print("=" * 65)
    print("  NarrativeShield — B1 Direct Prompting Baseline")
    print("  7 models | greedy decoding | resume-safe")
    print(f"  Output directory : {OUTPUT_DIR}")
    print("=" * 65)
    print()
    print("  Run order:")
    for i, (key, (repo, _, _)) in enumerate(MODEL_REGISTRY.items(), 1):
        tag = "  ← runs last (gated access)" if key == "llama32" else ""
        print(f"    {i}. {key:<14} {repo}{tag}")
    print()

    login(token=HF_TOKEN, add_to_git_credential=False)
    print("  HuggingFace auth OK\n")

    print(f"  Loading dataset: {DATASET_NAME} …")
    raw_ds   = load_dataset(DATASET_NAME, split="train")
    all_rows = list(raw_ds)
    print(f"  Total questions : {len(all_rows)}\n")

    t_start = time.time()

    for model_key, (repo_id, output_file, strategy) in MODEL_REGISTRY.items():
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
    print(f"  ALL MODELS COMPLETE.")
    print(f"  Total wall time : {hrs}h {mins}m")
    print(f"  Results in      : {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
