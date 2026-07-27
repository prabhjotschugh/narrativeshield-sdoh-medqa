"""
NarrativeShield — B3 Explicit Debiasing Baseline
=================================================
One script, one model per SLURM job.

Usage
-----
  python run_b3.py --model llama31
  python run_b3.py --model mistral7b
  python run_b3.py --model qwen25
  python run_b3.py --model gemma3_12b
  python run_b3.py --model gemma4_e4b
  python run_b3.py --model biomistral
  python run_b3.py --model llama32

Architecture
------------
Single model, no agents, no structured extraction, no chain-of-thought.
This is the critical baseline that tests whether a plain debiasing instruction
can achieve what NarrativeShield achieves architecturally.

The debiasing instruction is delivered as a SYSTEM PROMPT (not a suffix)
so the model processes it as a persistent behavioral constraint rather than
part of the question — the strongest possible implementation of this approach.

Prompt strategy
---------------
Instruct models:
  System prompt:
    "Ignore how the patient describes their symptoms. Focus exclusively on
     the objective clinical facts, test results, and medical history. Do not
     let the patient's language style, health literacy, or cultural framing
     influence your reasoning."
  User turn: identical to B1 (narrative + options + "Provide only the single
     correct letter.")

BioMistral (base model — no chat template):
  The debiasing instruction is prepended as a strong directive line before
  the question in a raw completion format, using the same few-shot pattern
  as B2 so the output format is consistent.

Parser
------
  MAX_NEW_TOKENS = 128  (same as B1 — no CoT, direct answer expected)
  Same B1 cascade: natural language prefix → exact single letter →
  markdown bold/quotes → first standalone A-D fallback.
  B3 should have near-identical parse rate to B1.

Output
------
  results_B3_<model>.jsonl  — same schema as B1 and B2
"""

# ─────────────────────────────────────────────────────────────────────────────
# CACHE REDIRECT — before any HuggingFace import
# ─────────────────────────────────────────────────────────────────────────────
import os

SCRATCH_BASE = "/scratch/10778/prabhjotschugh"
HF_CACHE_DIR = f"{SCRATCH_BASE}/hf_cache"
OUTPUT_DIR   = f"{SCRATCH_BASE}/b3_results"
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
# IMPORTS
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
    AutoModelForImageTextToText,
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    pipeline,
)

warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
HF_TOKEN            = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN_HERE")
DATASET_NAME        = "Prabhjotschugh/narrativeshield-sdoh-medqa"
MAX_NEW_TOKENS      = 128    # same as B1 — no CoT, direct answer expected
QUESTIONS_PER_CHUNK = 16
NUM_WORKERS         = 4
PERSONAS            = ["alpha", "beta", "gamma"]

# ─────────────────────────────────────────────────────────────────────────────
# THE DEBIASING INSTRUCTION
# Delivered as system prompt for instruct models.
# Names all three bias axes explicitly (language style, health literacy,
# cultural framing) — maps directly to Pα/Pβ/Pγ persona design.
# This is the strongest single-instruction debiasing implementation possible.
# ─────────────────────────────────────────────────────────────────────────────
DEBIASING_SYSTEM_PROMPT = (
    "Ignore how the patient describes their symptoms. Focus exclusively on "
    "the objective clinical facts, test results, and medical history. Do not "
    "let the patient's language style, health literacy, or cultural framing "
    "influence your reasoning."
)

# For BioMistral (base model): prepended as a directive line in raw text
BIOMISTRAL_DEBIASING_PREFIX = (
    "INSTRUCTION: Ignore how the patient describes their symptoms. "
    "Focus exclusively on the objective clinical facts, test results, and "
    "medical history. Do not let the patient's language style, health "
    "literacy, or cultural framing influence your reasoning.\n\n"
)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# key → (repo_id, output_file, load_strategy, model_type)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "llama31":    (
        "meta-llama/Llama-3.1-8B-Instruct",
        f"{OUTPUT_DIR}/results_B3_llama31.jsonl",
        "pipeline_text",
        "instruct",
    ),
    "llama32":    (
        "meta-llama/Llama-3.2-3B-Instruct",
        f"{OUTPUT_DIR}/results_B3_llama32.jsonl",
        "pipeline_text",
        "instruct",
    ),
    "mistral7b":  (
        "mistralai/Mistral-7B-Instruct-v0.3",
        f"{OUTPUT_DIR}/results_B3_mistral7b.jsonl",
        "pipeline_text",
        "instruct",
    ),
    "qwen25":     (
        "Qwen/Qwen2.5-7B-Instruct",
        f"{OUTPUT_DIR}/results_B3_qwen25.jsonl",
        "pipeline_text",
        "instruct",
    ),
    "gemma3_12b": (
        "google/gemma-3-12b-it",
        f"{OUTPUT_DIR}/results_B3_gemma3_12b.jsonl",
        "gemma3_manual",
        "instruct",
    ),
    "gemma4_e4b": (
        "google/gemma-4-E4B-it",
        f"{OUTPUT_DIR}/results_B3_gemma4_e4b.jsonl",
        "gemma4_manual",
        "instruct",
    ),
    "biomistral": (
        "BioMistral/BioMistral-7B",
        f"{OUTPUT_DIR}/results_B3_biomistral.jsonl",
        "pipeline_text",
        "base",
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def build_base_question(narrative: str, options: dict) -> str:
    """The question body — identical to B1. Only the system context changes."""
    body = narrative + "\n\nOptions:\n"
    for k, v in options.items():
        body += f"{k}: {v}\n"
    body += "\nWhich of the following is the correct option? Provide only the single correct letter."
    return body


def build_messages(model_key: str, model_type: str,
                   narrative: str, options: dict) -> list:
    """
    Instruct models: debiasing instruction in the system role.
      - Mistral: explicit system turn (required by its template)
      - Gemma:   content-list format
      - Others:  standard system + user turns

    BioMistral: raw completion string with INSTRUCTION prefix prepended.
    """
    question = build_base_question(narrative, options)

    if model_type == "base":
        # Raw text completion — debiasing as a capitalised INSTRUCTION line
        return BIOMISTRAL_DEBIASING_PREFIX + question

    if model_key == "mistral7b":
        return [
            {"role": "system", "content": DEBIASING_SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]
    elif model_key in ("gemma3_12b", "gemma4_e4b"):
        # Gemma system turn uses the same content-list format
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": DEBIASING_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": question}],
            },
        ]
    else:
        # llama31, llama32, qwen25
        return [
            {"role": "system", "content": DEBIASING_SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]


# ─────────────────────────────────────────────────────────────────────────────
# ANSWER PARSER — identical to B1 cascade
# B3 asks for a single letter with no CoT, so the B1 parser is appropriate.
# ─────────────────────────────────────────────────────────────────────────────
def extract_answer_letter(text: str):
    if not text:
        return None
    text = text.strip()

    # 1. Natural language prefix
    m = re.search(
        r'(?i)(?:the\s+)?(?:correct\s+)?(?:option|answer)\s+is\s*:?\s*([A-D])\b',
        text,
    )
    if m:
        return m.group(1).upper()

    # 2. Exact single-letter line
    m = re.match(r'^([A-D])[\.\):]?\s*$', text)
    if m:
        return m.group(1).upper()

    # 3. Letter in markdown bold or quotes
    m = re.search(r'[\*\'\"`]([A-D])[\*\'\"`]', text)
    if m:
        return m.group(1).upper()

    # 4. First standalone A-D in text
    m = re.search(r'\b([A-D])\b', text)
    if m:
        return m.group(1).upper()

    return None


# ─────────────────────────────────────────────────────────────────────────────
# TORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────
class PersonaDataset(Dataset):
    def __init__(self, rows: list, model_key: str, model_type: str):
        self.records = []
        for row in rows:
            options = ast.literal_eval(row["options"])
            for persona in PERSONAS:
                narrative = row[f"persona_{persona}"]
                messages  = build_messages(model_key, model_type,
                                           narrative, options)
                self.records.append({
                    "question_id": row["question_id"],
                    "answer_idx":  row["answer_idx"],
                    "persona":     persona,
                    "messages":    messages,
                    "model_type":  model_type,
                })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def collate_fn(batch):
    return batch


# ─────────────────────────────────────────────────────────────────────────────
# RESUME HELPER
# ─────────────────────────────────────────────────────────────────────────────
def load_processed_ids(output_file: str) -> set:
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
# SCRATCH DOWNLOAD (TACC Lustre — no symlinks)
# ─────────────────────────────────────────────────────────────────────────────
def _download_to_scratch(repo_id: str, subdir: str) -> str:
    local = f"{HF_CACHE_DIR}/{subdir}"
    if not os.path.isdir(local):
        print(f"  Downloading {repo_id} → {local}  (runs once)")
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
    pipe = pipeline(
        "text-generation",
        model=repo_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    return "pipeline", pipe


def load_gemma3_manual(repo_id: str):
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
        raise ValueError(f"Unknown strategy: {strategy!r}")
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return loader_type, obj


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────
def run_inference_batch(loader_type: str, obj, batch: list) -> list:
    model_type    = batch[0]["model_type"]
    messages_list = [item["messages"] for item in batch]

    if loader_type == "pipeline":
        pipe = obj

        if model_type == "base":
            # BioMistral: raw completion, return only new tokens
            outputs = pipe(
                messages_list,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                batch_size=len(batch),
                return_full_text=False,
            )
            return [out[0]["generated_text"] for out in outputs]

        else:
            outputs = pipe(
                messages_list,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                batch_size=len(batch),
            )
            results = []
            for out in outputs:
                gen = out[0]["generated_text"]
                if isinstance(gen, list):
                    raw = gen[-1].get("content", str(gen[-1]))
                else:
                    raw = str(gen)
                results.append(raw)
            return results

    elif loader_type == "processor_model":
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
# VRAM CLEANUP
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
    print("  VRAM freed.")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model_key: str, repo_id: str, strategy: str,
                   model_type: str, output_file: str, all_rows: list):

    print(f"\n{'='*65}")
    print(f"  BASELINE : B3 Explicit Debiasing")
    print(f"  MODEL    : {model_key}  ({model_type})")
    print(f"  REPO     : {repo_id}")
    print(f"  OUTPUT   : {output_file}")
    print(f"  DELIVERY : {'INSTRUCTION prefix (base)' if model_type == 'base' else 'system prompt (instruct)'}")
    print(f"{'='*65}")

    # ── Resume ─────────────────────────────────────────────────────────────
    processed_ids = load_processed_ids(output_file)
    remaining     = [r for r in all_rows if r["question_id"] not in processed_ids]

    if not remaining:
        print("  All questions already processed — skipping.")
        return

    print(f"  Already done : {len(processed_ids)}")
    print(f"  To process   : {len(remaining)}")

    # ── Dataset + DataLoader ───────────────────────────────────────────────
    ds     = PersonaDataset(remaining, model_key, model_type)
    loader = DataLoader(
        ds,
        batch_size=QUESTIONS_PER_CHUNK * len(PERSONAS),
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=(NUM_WORKERS > 0),
    )

    # ── Load model ─────────────────────────────────────────────────────────
    loader_type, obj = load_model(model_key, repo_id, strategy)

    # ── Inference ──────────────────────────────────────────────────────────
    n_correct   = {p: 0 for p in PERSONAS}
    n_parsed    = 0
    n_processed = 0

    with open(output_file, "a", encoding="utf-8") as out_f:
        for batch in tqdm(loader, desc=f"  [B3/{model_key}]", unit="batch"):
            try:
                raw_responses = run_inference_batch(loader_type, obj, batch)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(
                        f"\n  ⚠ CUDA OOM — skipping batch.\n"
                        f"  Reduce QUESTIONS_PER_CHUNK (currently {QUESTIONS_PER_CHUNK})."
                    )
                    torch.cuda.empty_cache()
                    continue
                raise

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
                if extracted:
                    n_parsed += 1

                by_qid[qid]["personas_eval"][item["persona"]] = {
                    "raw_response":     raw,
                    "extracted_answer": extracted,
                    "is_correct":       correct,
                }
                if correct:
                    n_correct[item["persona"]] += 1

            # Write only complete rows (all 3 personas) — crash-safe
            for qid, row in by_qid.items():
                if len(row["personas_eval"]) == len(PERSONAS):
                    out_f.write(json.dumps(row) + "\n")

            out_f.flush()
            n_processed += len(by_qid)

    # ── Summary ────────────────────────────────────────────────────────────
    total_responses = n_processed * len(PERSONAS)
    parse_rate      = 100.0 * n_parsed / max(total_responses, 1)
    print(f"\n  ── B3 summary for {model_key} ──────────────────────────")
    print(f"     Parse rate  : {n_parsed}/{total_responses}  ({parse_rate:.1f}%)")
    for p in PERSONAS:
        pct = 100.0 * n_correct[p] / max(n_processed, 1)
        print(f"     Persona {p}  : {n_correct[p]}/{n_processed}  ({pct:.1f}%)")

    unload_model(loader_type, obj)


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING + MAIN
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="NarrativeShield B3 — run one model per SLURM job"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help=f"Model key. Choices: {list(MODEL_REGISTRY.keys())}",
    )
    return parser.parse_args()


def main():
    args      = parse_args()
    model_key = args.model

    repo_id, output_file, strategy, model_type = MODEL_REGISTRY[model_key]

    print("=" * 65)
    print("  NarrativeShield — B3 Explicit Debiasing Baseline")
    print(f"  Running : {model_key}")
    print(f"  Output  : {OUTPUT_DIR}")
    print("=" * 65)
    print(f"\n  Debiasing instruction:\n  \"{DEBIASING_SYSTEM_PROMPT}\"\n")

    login(token=HF_TOKEN, add_to_git_credential=False)
    print("  HuggingFace auth OK")

    print(f"\n  Loading dataset: {DATASET_NAME} …")
    raw_ds   = load_dataset(DATASET_NAME, split="train")
    all_rows = list(raw_ds)
    print(f"  Total questions: {len(all_rows)}")

    t_start = time.time()

    evaluate_model(
        model_key=model_key,
        repo_id=repo_id,
        strategy=strategy,
        model_type=model_type,
        output_file=output_file,
        all_rows=all_rows,
    )

    elapsed  = time.time() - t_start
    hrs, rem = divmod(int(elapsed), 3600)
    mins     = rem // 60
    print(f"\n{'='*65}")
    print(f"  DONE.  {model_key}  |  Wall time: {hrs}h {mins}m")
    print(f"  Output: {output_file}")
    print("=" * 65)


if __name__ == "__main__":
    main()
