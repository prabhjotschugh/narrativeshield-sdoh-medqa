"""
NarrativeShield — B2 Chain-of-Thought Baseline
===============================================
One script, one model per SLURM job.

Usage
-----
  python run_b2.py --model llama31
  python run_b2.py --model mistral7b
  python run_b2.py --model qwen25
  python run_b2.py --model gemma3_12b
  python run_b2.py --model gemma4_e4b
  python run_b2.py --model biomistral
  python run_b2.py --model llama32

Prompt strategy
---------------
Instruct models (all except biomistral):
  Standard CoT suffix appended to the B1 prompt —
  "Think step by step. Consider each option carefully based only on the
   clinical facts presented. Then state your final answer as: Answer: [letter]"

BioMistral (base/pretrained, no chat template):
  2-shot completion prompting. Two fully solved examples are prepended
  showing the CoT reasoning pattern AND the "Answer: X" terminal token.
  The model sees the format from context instead of from instruction-following.

Parser
------
  MAX_NEW_TOKENS = 512 (CoT needs space to reason)
  Priority cascade:
    1. Explicit terminal token  "Answer: X"  or  "answer: X"
    2. Last non-empty line of the response (letter or letter + text)
    3. Full B1 regex cascade on entire response
  This is documented in the paper as methodologically clean — B2 asks
  for a specific terminal token; the parser honours it first.

Output
------
  results_B2_<model>.jsonl  — same schema as B1, directly comparable
"""

# ─────────────────────────────────────────────────────────────────────────────
# CACHE REDIRECT — before any HuggingFace import
# ─────────────────────────────────────────────────────────────────────────────
import os

SCRATCH_BASE = "/scratch/10778/prabhjotschugh"
HF_CACHE_DIR = f"{SCRATCH_BASE}/hf_cache"
OUTPUT_DIR   = f"{SCRATCH_BASE}/b2_results"
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
    AutoTokenizer,
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
MAX_NEW_TOKENS      = 512    # CoT needs more space than B1's 128
QUESTIONS_PER_CHUNK = 8      # halved vs B1 because responses are ~4x longer
NUM_WORKERS         = 4
PERSONAS            = ["alpha", "beta", "gamma"]

# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "llama31":    (
        "meta-llama/Llama-3.1-8B-Instruct",
        f"{OUTPUT_DIR}/results_B2_llama31.jsonl",
        "pipeline_text",
        "instruct",
    ),
    "llama32":    (
        "meta-llama/Llama-3.2-3B-Instruct",
        f"{OUTPUT_DIR}/results_B2_llama32.jsonl",
        "pipeline_text",
        "instruct",
    ),
    "mistral7b":  (
        "mistralai/Mistral-7B-Instruct-v0.3",
        f"{OUTPUT_DIR}/results_B2_mistral7b.jsonl",
        "pipeline_text",
        "instruct",
    ),
    "qwen25":     (
        "Qwen/Qwen2.5-7B-Instruct",
        f"{OUTPUT_DIR}/results_B2_qwen25.jsonl",
        "pipeline_text",
        "instruct",
    ),
    "gemma3_12b": (
        "google/gemma-3-12b-it",
        f"{OUTPUT_DIR}/results_B2_gemma3_12b.jsonl",
        "gemma3_manual",
        "instruct",
    ),
    "gemma4_e4b": (
        "google/gemma-4-E4B-it",
        f"{OUTPUT_DIR}/results_B2_gemma4_e4b.jsonl",
        "gemma4_manual",
        "instruct",
    ),
    "biomistral": (
        "BioMistral/BioMistral-7B",
        f"{OUTPUT_DIR}/results_B2_biomistral.jsonl",
        "pipeline_text",
        "base",          # ← triggers few-shot completion path
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# FEW-SHOT EXAMPLES FOR BIOMISTRAL
# Two fully worked examples that demonstrate:
#   (a) step-by-step clinical reasoning
#   (b) the "Answer: X" terminal token the parser looks for
# These are generic enough not to overlap with MedQA topics.
# ─────────────────────────────────────────────────────────────────────────────
BIOMISTRAL_FEW_SHOT_EXAMPLES = """The following are medical multiple-choice questions. For each question, reason step by step through the clinical options, then state the final answer.

Question: A 45-year-old man presents with severe chest pain radiating to the left arm, diaphoresis, and nausea for the past 30 minutes. ECG shows ST-segment elevation in leads II, III, and aVF. What is the most appropriate immediate management?

Options:
A: Administer oral aspirin and arrange outpatient cardiology follow-up
B: Perform immediate percutaneous coronary intervention (PCI)
C: Start intravenous heparin and monitor
D: Prescribe nitrates and discharge

Step-by-step reasoning:
The patient presents with classic signs of an acute ST-elevation myocardial infarction (STEMI): chest pain radiating to the left arm, diaphoresis, nausea, and inferior ST elevations (II, III, aVF). STEMI requires immediate reperfusion therapy. PCI is the gold-standard treatment when available within 90 minutes. Aspirin alone (A) is insufficient. Heparin monitoring alone (C) does not provide reperfusion. Nitrates and discharge (D) are dangerous in STEMI.

Answer: B

Question: A 7-year-old boy is brought in by his mother for a 3-day history of sore throat, fever of 38.9°C, and difficulty swallowing. Physical examination reveals tonsillar exudates and tender anterior cervical lymphadenopathy. Rapid strep test is positive. What is the most appropriate treatment?

Options:
A: Supportive care with antipyretics only
B: Amoxicillin for 10 days
C: Azithromycin for 3 days
D: Watchful waiting with repeat culture in 48 hours

Step-by-step reasoning:
The clinical picture — exudative tonsillitis, fever, tender cervical nodes, and positive rapid strep test — confirms Group A Streptococcal pharyngitis. First-line treatment is penicillin or amoxicillin for 10 days (B). Amoxicillin is preferred in children due to palatability. Azithromycin (C) is reserved for penicillin-allergic patients. Supportive care only (A) risks rheumatic fever. Watchful waiting (D) is inappropriate with a confirmed positive test.

Answer: B

"""

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
COT_SUFFIX = (
    "\n\nThink step by step. Consider each option carefully based only on the "
    "clinical facts presented. Then state your final answer as: Answer: [letter]"
)


def build_instruct_cot_prompt(narrative: str, options: dict) -> str:
    """
    B2 prompt for instruct-tuned models.
    Identical to B1 base prompt + CoT suffix requesting terminal Answer: token.
    """
    body = narrative + "\n\nOptions:\n"
    for k, v in options.items():
        body += f"{k}: {v}\n"
    body += COT_SUFFIX
    return body


def build_biomistral_completion_prompt(narrative: str, options: dict) -> str:
    """
    B2 prompt for BioMistral (base model, no chat template).
    Prepends two worked few-shot examples that establish the CoT format
    and the Answer: X terminal token via in-context learning.
    The test case is appended as a raw completion target.
    """
    body  = BIOMISTRAL_FEW_SHOT_EXAMPLES
    body += "Question: " + narrative + "\n\nOptions:\n"
    for k, v in options.items():
        body += f"{k}: {v}\n"
    body += "\nStep-by-step reasoning:\n"
    return body


def build_messages(model_key: str, model_type: str, prompt_text: str) -> list:
    """
    Returns correct message structure.
    BioMistral uses raw string completion (no messages wrapper).
    """
    if model_type == "base":
        # Raw string — passed directly to pipeline as text, not messages
        return prompt_text

    if model_key == "mistral7b":
        return [
            {"role": "system", "content": ""},
            {"role": "user",   "content": prompt_text},
        ]
    elif model_key in ("gemma3_12b", "gemma4_e4b"):
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            }
        ]
    else:
        return [{"role": "user", "content": prompt_text}]


# ─────────────────────────────────────────────────────────────────────────────
# ANSWER PARSER
# Priority: (1) explicit "Answer: X" terminal token
#           (2) last non-empty line containing a standalone letter
#           (3) full B1 regex cascade on entire text
# ─────────────────────────────────────────────────────────────────────────────
def extract_answer_letter(text: str) -> str | None:
    if not text:
        return None
    text = text.strip()

    # ── Priority 1: explicit terminal token ──────────────────────────────────
    # Matches "Answer: B", "answer: B", "Answer: B." etc.
    m = re.search(r'(?i)\banswer\s*:\s*([A-D])\b', text)
    if m:
        return m.group(1).upper()

    # ── Priority 2: last non-empty line ──────────────────────────────────────
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        last = lines[-1]
        # Exact single letter on the last line
        m = re.match(r'^([A-D])[\.\):]?\s*$', last)
        if m:
            return m.group(1).upper()
        # "The answer is X" on last line
        m = re.search(
            r'(?i)(?:the\s+)?(?:correct\s+)?(?:option|answer)\s+is\s*:?\s*([A-D])\b',
            last,
        )
        if m:
            return m.group(1).upper()
        # First standalone letter on last line
        m = re.search(r'\b([A-D])\b', last)
        if m:
            return m.group(1).upper()

    # ── Priority 3: B1 cascade on full text (fallback) ───────────────────────
    m = re.search(
        r'(?i)(?:the\s+)?(?:correct\s+)?(?:option|answer)\s+is\s*:?\s*([A-D])\b',
        text,
    )
    if m:
        return m.group(1).upper()
    m = re.match(r'^([A-D])[\.\):]?\s*$', text)
    if m:
        return m.group(1).upper()
    m = re.search(r'[\*\'\"`]([A-D])[\*\'\"`]', text)
    if m:
        return m.group(1).upper()
    m = re.search(r'\b([A-D])\b', text)
    if m:
        return m.group(1).upper()

    return None


# ─────────────────────────────────────────────────────────────────────────────
# TORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────
class PersonaDataset(Dataset):
    def __init__(self, rows: list, model_key: str, model_type: str):
        self.records   = []
        self.model_type = model_type

        for row in rows:
            options = ast.literal_eval(row["options"])
            for persona in PERSONAS:
                narrative = row[f"persona_{persona}"]

                if model_type == "base":
                    prompt_text = build_biomistral_completion_prompt(narrative, options)
                else:
                    prompt_text = build_instruct_cot_prompt(narrative, options)

                messages = build_messages(model_key, model_type, prompt_text)

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
    """
    Two paths:
      - instruct models: messages list → pipeline / processor
      - BioMistral (base): raw completion string → pipeline with text_inputs
    """
    model_type    = batch[0]["model_type"]
    messages_list = [item["messages"] for item in batch]

    if loader_type == "pipeline":
        pipe = obj

        if model_type == "base":
            # BioMistral: raw text completion, not chat
            # pipeline() accepts plain strings when passed as positional arg
            outputs = pipe(
                messages_list,          # list of raw strings
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                batch_size=len(batch),
                return_full_text=False, # only return newly generated tokens
            )
            return [out[0]["generated_text"] for out in outputs]

        else:
            # Instruct models: standard chat pipeline
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
        # Gemma 3 and Gemma 4 — both instruct, same path
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
    print(f"  BASELINE : B2 Chain-of-Thought")
    print(f"  MODEL    : {model_key}  ({model_type})")
    print(f"  REPO     : {repo_id}")
    print(f"  OUTPUT   : {output_file}")
    print(f"  PROMPT   : {'few-shot completion' if model_type == 'base' else 'instruct CoT + Answer: token'}")
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
    n_parsed    = 0        # how many responses yielded a parseable letter
    n_processed = 0

    with open(output_file, "a", encoding="utf-8") as out_f:
        for batch in tqdm(loader, desc=f"  [B2/{model_key}]", unit="batch"):
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

            # Write only complete rows
            for qid, row in by_qid.items():
                if len(row["personas_eval"]) == len(PERSONAS):
                    out_f.write(json.dumps(row) + "\n")

            out_f.flush()
            n_processed += len(by_qid)

    # ── Summary ────────────────────────────────────────────────────────────
    total_responses = n_processed * len(PERSONAS)
    parse_rate = 100.0 * n_parsed / max(total_responses, 1)
    print(f"\n  ── B2 summary for {model_key} ──────────────────────────")
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
        description="NarrativeShield B2 — run one model per SLURM job"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help=f"Model key to run. Choices: {list(MODEL_REGISTRY.keys())}",
    )
    return parser.parse_args()


def main():
    args      = parse_args()
    model_key = args.model

    repo_id, output_file, strategy, model_type = MODEL_REGISTRY[model_key]

    print("=" * 65)
    print("  NarrativeShield — B2 Chain-of-Thought Baseline")
    print(f"  Running: {model_key}")
    print(f"  Output : {OUTPUT_DIR}")
    print("=" * 65)

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

    elapsed = time.time() - t_start
    hrs, rem = divmod(int(elapsed), 3600)
    mins     = rem // 60
    print(f"\n{'='*65}")
    print(f"  DONE.  {model_key}  |  Wall time: {hrs}h {mins}m")
    print(f"  Output: {output_file}")
    print("=" * 65)


if __name__ == "__main__":
    main()
