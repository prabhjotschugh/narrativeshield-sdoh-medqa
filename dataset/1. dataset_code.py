"""
MedQA-USMLE Dataset Filter
============================
Extracts 1000 narrative-suitable questions for SDoH persona generation.

A "narrative-suitable" question must:
1. Have a patient vignette stem (real patient with symptoms)
2. Be long enough for persona variation to be meaningful
3. Contain actual clinical presentation (not pure knowledge recall)
4. Be about diagnosis or treatment (not pathophysiology / embryology / mechanism)

Usage:
    pip install datasets pandas
    python filter_medqa.py
"""

import re
import json
import random
import pandas as pd
from datasets import load_dataset

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)

# ── Load dataset ───────────────────────────────────────────────────────────────
print("Loading MedQA-USMLE-4-options ...")
ds = load_dataset("GBaker/MedQA-USMLE-4-options")

# Combine train + test so we have the full pool to sample from
train_df = ds["train"].to_pandas()
test_df  = ds["test"].to_pandas()
full_df  = pd.concat([train_df, test_df], ignore_index=True)
print(f"Total questions in dataset: {len(full_df)}")


# ══════════════════════════════════════════════════════════════════════════════
# FILTER 1 — Must have a patient vignette stem
# These are questions that start with a real patient presentation.
# Pattern: "A/An X-year-old [man/woman/boy/girl/patient/...]"
# ══════════════════════════════════════════════════════════════════════════════
VIGNETTE_PATTERN = re.compile(
    r"^(A|An)\s+\d+[\-\s]?(year|month|week|day)[\-\s]?old",
    re.IGNORECASE
)

def has_patient_vignette(text: str) -> bool:
    return bool(VIGNETTE_PATTERN.match(text.strip()))

# ══════════════════════════════════════════════════════════════════════════════
# FILTER 2 — Minimum length: question must be at least 300 characters
# Short questions are usually pure recall ("Which enzyme..."), not narratives.
# 300 chars gives at least 2-3 sentences of clinical context.
# ══════════════════════════════════════════════════════════════════════════════
MIN_CHARS = 300

# ══════════════════════════════════════════════════════════════════════════════
# FILTER 3 — Must contain at least ONE clinical signal word
# Ensures there's actual clinical content to vary across personas.
# ══════════════════════════════════════════════════════════════════════════════
CLINICAL_SIGNALS = [
    # Symptoms
    "pain", "fever", "cough", "shortness of breath", "dyspnea", "fatigue",
    "nausea", "vomiting", "diarrhea", "bleeding", "swelling", "rash",
    "headache", "dizziness", "weakness", "numbness", "tingling", "chest",
    "abdominal", "urination", "burning", "discharge", "weight loss",
    "weight gain", "palpitation", "syncope", "seizure", "confusion",
    # Presentation context
    "presents to", "comes to", "brought to", "emergency department",
    "physician", "clinic", "office", "hospital", "admitted",
    # Vitals / labs (indicates clinical workup is present)
    "blood pressure", "heart rate", "temperature", "pulse", "hemoglobin",
    "white blood cell", "platelet", "creatinine", "sodium", "glucose",
    "bilirubin", "mmHg", "bpm",
    # Physical exam findings
    "physical exam", "examination", "palpation", "auscultation",
    "tenderness", "murmur", "edema", "jaundice",
]

def has_clinical_signal(text: str) -> bool:
    text_lower = text.lower()
    return any(signal in text_lower for signal in CLINICAL_SIGNALS)

# ══════════════════════════════════════════════════════════════════════════════
# FILTER 4 — Exclude pure mechanism / pathophysiology questions
# These questions don't benefit from narrative variation because there's
# no patient-reported symptom framing to vary.
# ══════════════════════════════════════════════════════════════════════════════
EXCLUDE_QUESTION_TYPES = [
    # Mechanism / pathophysiology
    "most likely mechanism", "pathogenesis", "embryologic",
    "which enzyme", "which receptor", "which pathway",
    "most likely explains the finding",
    # Anatomy
    "which structure", "which nerve", "which muscle", "which artery",
    "which vein", "which bone",
    # Pure genetics / biochemistry
    "which gene", "which chromosome", "which protein",
    "which amino acid",
    # Histology
    "histological", "histologic", "microscopy", "biopsy shows",
    "which cell type",
]

def is_excluded_type(text: str) -> bool:
    text_lower = text.lower()
    return any(excl in text_lower for excl in EXCLUDE_QUESTION_TYPES)

# ══════════════════════════════════════════════════════════════════════════════
# FILTER 5 — Question must be about diagnosis OR treatment
# These are the question types where narrative framing will actually
# affect LLM output — the core of your paper's claim.
# ══════════════════════════════════════════════════════════════════════════════
TARGET_QUESTION_TYPES = [
    # Diagnosis
    "most likely diagnosis", "most likely cause", "most likely etiology",
    "most likely condition", "most likely pathology", "most likely finding",
    "most likely responsible", "what is the diagnosis",
    # Treatment / management
    "best treatment", "best management", "best next step",
    "most appropriate treatment", "most appropriate management",
    "most appropriate next step", "most appropriate therapy",
    "next best step", "next step in management",
    "which of the following should", "which of the following would",
    "which medication", "which drug", "which antibiotic",
    # Investigation
    "best initial test", "most appropriate test", "which test",
    "which of the following tests", "confirm the diagnosis",
    "most likely to confirm",
]

def is_target_question_type(text: str) -> bool:
    text_lower = text.lower()
    return any(target in text_lower for target in TARGET_QUESTION_TYPES)

# ══════════════════════════════════════════════════════════════════════════════
# FILTER 6 — Must have some narrative richness
# The question should have the patient describing symptoms themselves,
# OR have a caregiver bringing in a patient with described symptoms.
# This is what gives you material for persona variation.
# Proxy: look for first-person reporting words or symptom history words.
# ══════════════════════════════════════════════════════════════════════════════
NARRATIVE_RICHNESS = [
    "she states", "he states", "she reports", "he reports",
    "she says", "he says", "she complains", "he complains",
    "she denies", "he denies", "she notes", "he notes",
    "she has been", "he has been", "she has had", "he has had",
    "the patient states", "the patient reports", "the patient says",
    "the patient complains", "the patient notes", "the patient denies",
    "history of", "for the past", "for the last", "started",
    "worsening", "improving", "associated with", "accompanied by",
    "she also", "he also", "she additionally", "he additionally",
    "mother reports", "father reports", "parent reports",
    "caregiver reports",
]

def has_narrative_richness(text: str) -> bool:
    text_lower = text.lower()
    return any(signal in text_lower for signal in NARRATIVE_RICHNESS)


# ══════════════════════════════════════════════════════════════════════════════
# APPLY ALL FILTERS
# ══════════════════════════════════════════════════════════════════════════════
print("\nApplying filters...")

mask = (
    full_df["question"].apply(has_patient_vignette)       # F1: patient vignette
    & (full_df["question"].str.len() >= MIN_CHARS)        # F2: min length
    & full_df["question"].apply(has_clinical_signal)      # F3: clinical signal
    & ~full_df["question"].apply(is_excluded_type)        # F4: exclude mechanism
    & full_df["question"].apply(is_target_question_type)  # F5: dx or tx question
    & full_df["question"].apply(has_narrative_richness)   # F6: narrative richness
)

filtered_df = full_df[mask].copy()
print(f"Questions passing all filters: {len(filtered_df)}")


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE 1000 questions
# Stratify by answer_idx (A/B/C/D) to avoid answer-distribution skew.
# Also stratify by meta_info (step1 vs step2&3) to keep topic diversity.
# ══════════════════════════════════════════════════════════════════════════════
TARGET_N = 1000

if len(filtered_df) < TARGET_N:
    print(f"WARNING: Only {len(filtered_df)} questions passed filters. "
          f"Using all of them. Consider relaxing Filter 4 or 6.")
    sampled_df = filtered_df.copy()
else:
    # Stratified sample: proportional by (meta_info × answer_idx)
    sampled_df = (
        filtered_df
        .groupby(["meta_info", "answer_idx"], group_keys=False)
        .apply(lambda g: g.sample(
            n=min(len(g), max(1, int(TARGET_N * len(g) / len(filtered_df)))),
            random_state=SEED
        ))
    )
    # Top up to exactly 1000 if stratification leaves us short
    if len(sampled_df) < TARGET_N:
        remaining = filtered_df.drop(sampled_df.index)
        topup = remaining.sample(
            n=TARGET_N - len(sampled_df),
            random_state=SEED
        )
        sampled_df = pd.concat([sampled_df, topup], ignore_index=True)
    
    sampled_df = sampled_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    sampled_df = sampled_df.head(TARGET_N)

print(f"\nFinal sample size: {len(sampled_df)}")


# ══════════════════════════════════════════════════════════════════════════════
# QUALITY CHECK — print distributions
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Answer distribution (should be roughly balanced) ──")
print(sampled_df["answer_idx"].value_counts().sort_index())

print("\n── Meta info distribution ──")
print(sampled_df["meta_info"].value_counts())

print("\n── Question length stats (characters) ──")
sampled_df["q_length"] = sampled_df["question"].str.len()
print(sampled_df["q_length"].describe().round(0))


# ══════════════════════════════════════════════════════════════════════════════
# CLEAN & FORMAT for downstream use
# Each row will have:
#   - question_id     : unique ID for your experiments
#   - question        : original question text (the vignette)
#   - options         : dict {A, B, C, D}
#   - answer          : correct answer text
#   - answer_idx      : correct answer letter
#   - meta_info       : step1 / step2&3
#   - question_length : char count (useful for analysis)
# ══════════════════════════════════════════════════════════════════════════════
sampled_df = sampled_df.reset_index(drop=True)
sampled_df["question_id"] = [f"MQ_{i+1:04d}" for i in range(len(sampled_df))]

output_cols = [
    "question_id", "question", "options", "answer", "answer_idx",
    "meta_info", "q_length"
]
final_df = sampled_df[output_cols].copy()


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════

# 1. CSV — easy to inspect
final_df.to_csv("medqa_1000_filtered.csv", index=False)

# 2. JSONL — best format for LLM inference pipelines
with open("medqa_1000_filtered.jsonl", "w", encoding="utf-8") as f:
    for _, row in final_df.iterrows():
        record = {
            "question_id": row["question_id"],
            "question":    row["question"],
            "options":     row["options"],   # already a dict
            "answer":      row["answer"],
            "answer_idx":  row["answer_idx"],
            "meta_info":   row["meta_info"],
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
print("\n✓ Saved: medqa_1000_filtered.csv")
print("✓ Saved: medqa_1000_filtered.jsonl")


# ══════════════════════════════════════════════════════════════════════════════
# SHOW 3 SAMPLE QUESTIONS so you can verify quality visually
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("SAMPLE QUESTIONS (verify these look like good vignettes)")
print("═"*70)

for i, row in final_df.sample(3, random_state=SEED).iterrows():
    print(f"\n[{row['question_id']}] ({row['meta_info']}) — Answer: {row['answer_idx']}")
    print(f"Q: {row['question'][:400]}...")
    print(f"Options: {row['options']}")
    print(f"Correct: {row['answer']}")
    print("-"*70)

print("\nDone. Use medqa_1000_filtered.jsonl for your persona generation pipeline.")