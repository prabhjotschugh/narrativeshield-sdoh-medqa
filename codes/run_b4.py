"""
NarrativeShield — B4 Chain-of-Thought + Explicit Register-Neutrality Baseline
=================================================================================
One script, one model per SLURM job.

Usage
-----
  python run_b4.py --model llama32_3b
  python run_b4.py --model medgemma_4b
  python run_b4.py --model phi4_mini
  python run_b4.py --model qwen25_7b
  python run_b4.py --model mistral7b
  python run_b4.py --model qwen3_8b
  python run_b4.py --model ministral8b
  python run_b4.py --model llama31_8b
  python run_b4.py --model openbiollm_8b
  python run_b4.py --model gemma3_12b
  python run_b4.py --model qwen25_14b
  python run_b4.py --model mistral_small_24b
  python run_b4.py --model llama31_8b_base

Why this baseline exists
--------------------------
B2 (chain-of-thought) and B3 (one-shot explicit debiasing instruction) each
test ONE mechanism in isolation. Neither tests whether the two interact:
does forcing VISIBLE step-by-step reasoning make an explicit
register-neutrality instruction MORE effective than either mechanism alone?
This is a genuinely open question B1-B3 do not answer, and — unlike a bare
repeat of B3's instruction — it varies a real, distinct axis (reasoning
depth x instruction specificity), not a restatement of an existing
condition.

Scope note: this is NOT a substitute for isolating the register-vs-new-
information confound (that would require a
generation-time neutral control persona, i.e. new dataset rows — out of
scope for this revision, since the existing Pα/Pβ/Pγ dataset is finalized
and doctor-verified, and is not being regenerated or extended). B4 tests a
different question: whether models CAN be steered to suppress their own
register-sensitivity given both an explicit instruction AND room to reason
about it, using the SAME frozen dataset as every other condition.

Architecture
------------
Single model, no agents, no structured extraction. Combines:
  - B3's debiasing system prompt (explicitly names register as the thing
    to ignore, not a generic fairness instruction)
  - B2's chain-of-thought suffix + "Answer: [letter]" terminal token
    (forces visible step-by-step reasoning instead of a bare final letter)

Prompt strategy
---------------
Instruct models (all except llama31_8b_base):
  System prompt (identical to B3's DEBIASING_SYSTEM_PROMPT):
    "Ignore how the patient describes their symptoms. Focus exclusively on
     the objective clinical facts, test results, and medical history. Do not
     let the patient's language style, health literacy, or cultural framing
     influence your reasoning."
  User turn: identical to B2's CoT prompt — narrative + options + "Think
     step by step ... Then state your final answer as: Answer: [letter]"

llama31_8b_base (meta-llama/Llama-3.1-8B — base model, no chat template):
  Same 2-shot completion pattern as B2/B3's base-model path, with the
  debiasing directive prepended before the few-shot examples so it is the
  first thing the model conditions on, then the CoT-demonstrating examples,
  then the test question.

Parser
------
  MAX_NEW_TOKENS = 512 (same as B2 — CoT needs space to reason)
  Identical 3-tier cascade to B2: explicit "Answer: X" terminal token →
  last non-empty line → full B1 regex cascade fallback.

Output
------
  results_B4_<model>.jsonl  — same schema as B1/B2/B3, directly comparable
"""

# ─────────────────────────────────────────────────────────────────────────────
# CACHE REDIRECT — before any HuggingFace import
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
OUTPUT_DIR   = f"{SCRATCH_BASE}/b4_results"
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
DATASET_NAME        = os.environ.get("DATASET_NAME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset", "medqa_1000.csv"))
MAX_NEW_TOKENS      = 1280   # raised from 512: Qwen3/Gemma CoT answers were cut off before "Answer: X"
QUESTIONS_PER_CHUNK = 8      # same as B2 — responses are ~4x longer than B1/B3
NUM_WORKERS         = 4
PERSONAS            = ["alpha", "beta", "gamma"]

# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "llama32_3b":         (
        "meta-llama/Llama-3.2-3B-Instruct",
        f"{OUTPUT_DIR}/results_B4_llama32_3b.jsonl",
        "pipeline_text",
        "instruct",
    ),
    "medgemma_4b":        (
        "google/medgemma-4b-it",
        f"{OUTPUT_DIR}/results_B4_medgemma_4b.jsonl",
        "medgemma_manual",
        "instruct",
    ),
    "phi4_mini":          (
        "microsoft/Phi-4-mini-instruct",
        f"{OUTPUT_DIR}/results_B4_phi4_mini.jsonl",
        "pipeline_text",
        "instruct",
    ),
    "qwen25_7b":          (
        "Qwen/Qwen2.5-7B-Instruct",
        f"{OUTPUT_DIR}/results_B4_qwen25_7b.jsonl",
        "pipeline_text",
        "instruct",
    ),
    "mistral7b":          (
        "mistralai/Mistral-7B-Instruct-v0.3",
        f"{OUTPUT_DIR}/results_B4_mistral7b.jsonl",
        "pipeline_text",
        "instruct",
    ),
    "qwen3_8b":           (
        "Qwen/Qwen3-8B",
        f"{OUTPUT_DIR}/results_B4_qwen3_8b.jsonl",
        "pipeline_text",
        "instruct",
    ),
    "ministral8b":        (
        "mistralai/Ministral-8B-Instruct-2410",
        f"{OUTPUT_DIR}/results_B4_ministral8b.jsonl",
        "pipeline_text",
        "instruct",
    ),
    "llama31_8b":         (
        "meta-llama/Llama-3.1-8B-Instruct",
        f"{OUTPUT_DIR}/results_B4_llama31_8b.jsonl",
        "pipeline_text",
        "instruct",
    ),
    "openbiollm_8b":      (
        "aaditya/Llama3-OpenBioLLM-8B",
        f"{OUTPUT_DIR}/results_B4_openbiollm_8b.jsonl",
        "pipeline_text",
        "instruct",
    ),
    "gemma3_12b":         (
        "google/gemma-3-12b-it",
        f"{OUTPUT_DIR}/results_B4_gemma3_12b.jsonl",
        "gemma3_manual",
        "instruct",
    ),
    "qwen25_14b":         (
        "Qwen/Qwen2.5-14B-Instruct",
        f"{OUTPUT_DIR}/results_B4_qwen25_14b.jsonl",
        "pipeline_text",
        "instruct",
    ),
    "mistral_small_24b":  (
        "mistralai/Mistral-Small-24B-Instruct-2501",
        f"{OUTPUT_DIR}/results_B4_mistral_small_24b.jsonl",
        "pipeline_text",
        "instruct",
    ),
    "llama31_8b_base":    (
        "meta-llama/Llama-3.1-8B",
        f"{OUTPUT_DIR}/results_B4_llama31_8b_base.jsonl",
        "pipeline_text",   # same loader as instruct models — the "base"
                           # model_type flag below is what routes it to the
                           # completion (not chat) inference path at runtime
        "base",            # ← triggers few-shot completion path
    ),
}

# Models that require content-block message format ([{"type": "text", ...}])
# instead of plain string content — both are manually-loaded Gemma-family
# checkpoints (gemma3_manual / medgemma_manual loader strategies).
CONTENT_BLOCK_MODELS = {"gemma3_12b", "medgemma_4b"}

# ─────────────────────────────────────────────────────────────────────────────
# DEBIASING INSTRUCTION — identical text to B3's DEBIASING_SYSTEM_PROMPT,
# kept as a separate copy (not imported from run_b3.py) so each run_*.py
# script stays fully self-contained for independent SLURM submission.
# ─────────────────────────────────────────────────────────────────────────────
DEBIASING_SYSTEM_PROMPT = (
    "Ignore how the patient describes their symptoms. Focus exclusively on "
    "the objective clinical facts, test results, and medical history. Do not "
    "let the patient's language style, health literacy, or cultural framing "
    "influence your reasoning."
)

# For llama31_8b_base (base model): prepended as a directive line in raw
# text, before the CoT-demonstrating few-shot examples.
BASE_MODEL_DEBIASING_PREFIX = (
    "INSTRUCTION: Ignore how the patient describes their symptoms. "
    "Focus exclusively on the objective clinical facts, test results, and "
    "medical history. Do not let the patient's language style, health "
    "literacy, or cultural framing influence your reasoning.\n\n"
)

# ─────────────────────────────────────────────────────────────────────────────
# FEW-SHOT EXAMPLES FOR THE BASE (NON-INSTRUCT) MODEL
# Identical worked examples to run_b2.py/run_b3.py's base-model path —
# demonstrates step-by-step reasoning + the "Answer: X" terminal token.
# ─────────────────────────────────────────────────────────────────────────────
BASE_MODEL_FEW_SHOT_EXAMPLES = """The following are medical multiple-choice questions. For each question, reason step by step through the clinical options, then state the final answer.

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
    B4 user-turn prompt for instruct-tuned models — identical structure to
    B2's CoT prompt. The debiasing instruction itself lives in the SYSTEM
    prompt (build_messages below), not here, matching B3's placement.
    """
    body = narrative + "\n\nOptions:\n"
    for k, v in options.items():
        body += f"{k}: {v}\n"
    body += COT_SUFFIX
    return body


def build_base_model_completion_prompt(narrative: str, options: dict) -> str:
    """
    B4 prompt for llama31_8b_base (base model, no chat template).
    Debiasing directive first, then the two worked CoT examples, then the
    test case — the model conditions on the instruction before it ever
    sees a worked example, then applies the same reasoning pattern to the
    real question.
    """
    body  = BASE_MODEL_DEBIASING_PREFIX
    body += BASE_MODEL_FEW_SHOT_EXAMPLES
    body += "Question: " + narrative + "\n\nOptions:\n"
    for k, v in options.items():
        body += f"{k}: {v}\n"
    body += "\nStep-by-step reasoning:\n"
    return body


def build_messages(model_key: str, model_type: str, prompt_text: str) -> list:
    """
    Returns correct message structure. Debiasing instruction is delivered
    as a SYSTEM prompt (persistent behavioral constraint), CoT prompt as
    the user turn — mirrors B3's system/user split, not B2's user-only split.
    llama31_8b_base uses raw string completion (no messages wrapper); the
    debiasing directive is already baked into prompt_text by
    build_base_model_completion_prompt above.
    """
    if model_type == "base":
        # Raw string — passed directly to pipeline as text, not messages
        return prompt_text

    if model_key == "mistral7b":
        return [
            {"role": "system", "content": DEBIASING_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt_text},
        ]
    elif model_key in CONTENT_BLOCK_MODELS:
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": DEBIASING_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            },
        ]
    else:
        return [
            {"role": "system", "content": DEBIASING_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt_text},
        ]


# ─────────────────────────────────────────────────────────────────────────────
# ANSWER PARSER — identical to B2's cascade (CoT + terminal token expected)
# Priority: (1) explicit "Answer: X" terminal token
#           (2) last non-empty line containing a standalone letter
#           (3) full B1 regex cascade on entire text
# ─────────────────────────────────────────────────────────────────────────────
def extract_answer_letter(text: str) -> str | None:
    if not text:
        return None
    text = text.strip()

    # ── Priority 1: explicit terminal token ──────────────────────────────────
    m = re.search(r'(?i)\banswer\s*:\s*([A-D])\b', text)
    if m:
        return m.group(1).upper()

    # ── Priority 2: last non-empty line ──────────────────────────────────────
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        last = lines[-1]
        m = re.match(r'^([A-D])[\.\):]?\s*$', last)
        if m:
            return m.group(1).upper()
        m = re.search(
            r'(?i)(?:the\s+)?(?:correct\s+)?(?:option|answer)\s+is\s*:?\s*([A-D])\b',
            last,
        )
        if m:
            return m.group(1).upper()
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
                    prompt_text = build_base_model_completion_prompt(narrative, options)
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
# SCRATCH DOWNLOAD (HPC Lustre — no symlinks)
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
    pipe = _build_text_pipeline(repo_id)
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
        raise ValueError(f"Unknown strategy: {strategy!r}")
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return loader_type, obj


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────
def run_inference_batch(loader_type: str, obj, batch: list) -> list:
    """
    Two paths:
      - instruct models: messages list (system + user) → pipeline / processor
      - llama31_8b_base (base): raw completion string → pipeline with text_inputs
    """
    model_type    = batch[0]["model_type"]
    messages_list = [item["messages"] for item in batch]

    if loader_type == "pipeline":
        pipe = obj

        if model_type == "base":
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
        # gemma3_12b and medgemma_4b — both instruct, same path
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
    print(f"  BASELINE : B4 Chain-of-Thought + Explicit Register-Neutrality")
    print(f"  MODEL    : {model_key}  ({model_type})")
    print(f"  REPO     : {repo_id}")
    print(f"  OUTPUT   : {output_file}")
    print(f"  PROMPT   : {'debiasing prefix + few-shot completion' if model_type == 'base' else 'debiasing system prompt + instruct CoT + Answer: token'}")
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
        for batch in tqdm(loader, desc=f"  [B4/{model_key}]", unit="batch"):
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
    print(f"\n  ── B4 summary for {model_key} ──────────────────────────")
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
        description="NarrativeShield B4 — run one model per SLURM job"
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
    print("  NarrativeShield — B4 CoT + Explicit Register-Neutrality Baseline")
    print(f"  Running: {model_key}")
    print(f"  Output : {OUTPUT_DIR}")
    print("=" * 65)

    # No login() call: HF_TOKEN is read from the environment and passed as token=...
    # everywhere. login() writes a shared stored_tokens file in the HF cache,
    # which concurrent jobs corrupt (race) and crash on.
    print("  HuggingFace auth OK")

    print(f"\n  Loading dataset: {DATASET_NAME} …")
    raw_ds   = load_dataset("csv", data_files=DATASET_NAME, split="train")
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
