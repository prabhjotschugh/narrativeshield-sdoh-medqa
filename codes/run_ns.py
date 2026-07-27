"""
NarrativeShield — Full Agentic Pipeline (v2 — EACL Main)
=========================================================

Architecture (3-agent, deterministic tool injection):

  Agent 1 — Adversarial Intake Extractor
    Input : raw patient narrative (Pα / Pβ / Pγ)
    Output: structured clinical JSON — all sociolinguistic signal stripped
    Role  : THE load-bearing debiasing step. Persona voice is gone by the
            time any downstream component sees the data.

  [Tool Router — deterministic Python code, not model decision]
    Input : Agent 1 JSON + question + options
    Action: auto-selects and executes tools based on clinical content rules
    Output: tool_context block injected into Agent 2's prompt

  Agent 2 — Clinical Reasoning Engine
    Input : Agent 1 JSON + injected tool context + question + options
    Output: structured reasoning with explicit ANSWER_CANDIDATE: X line
    Role  : Reasons from clean facts + grounded tool evidence.
            Never sees the original narrative.

  Agent 3 — Emergency Fallback Extractor (called ONLY when Agent 2 has
            no parseable ANSWER_CANDIDATE line)
    Input : Agent 2 raw output + options
    Output: single letter A/B/C/D

Tools (all deterministic — no live API calls):
  T1  Static Drug KB   — 60+ USMLE drugs: indications, CI, pregnancy safety,
                         withdrawal profiles, mechanism, key facts
  T2  Lab Interpreter  — static reference ranges, clinical interpretation
  T3  Clinical Scorers — Wells DVT/PE, CURB-65, Glasgow, Apgar
  [T4  openFDA REST     — called when drug not in static KB and TACC network allows]
  [T5  RxNorm REST      — called as fallback for drug class lookup]

Key design decisions vs v1:
  - Tools are called by DETERMINISTIC Python routing, not model discretion.
    7B models do not reliably emit structured TOOL_CALL JSON. Routing by code
    guarantees 100% tool invocation rate where relevant.
  - Tool results are injected BEFORE Agent 2 reasons (not after).
  - Agent 3 does NOT re-reason. It only extracts the ANSWER_CANDIDATE letter
    from Agent 2's text when regex parsing fails.
  - Static Drug KB surfaces opioid withdrawal profiles, pregnancy
    contraindications, etc. — the exact cases where 7B models fail silently.
  - Agent 1 prompt explicitly handles cultural medication naming.

Output files (written to OUTPUT_DIR):
  results_NS_{ablation}_{model}.jsonl
  tool_call_log_{model}.jsonl

Usage:
  python run_narrativeshield.py       
  python run_narrativeshield.py --model llama31
"""

# ─────────────────────────────────────────────────────────────────────────────
# CACHE REDIRECT — before any HuggingFace import
# ─────────────────────────────────────────────────────────────────────────────
import os

SCRATCH_BASE = "/scratch/10778/prabhjotschugh"
HF_CACHE_DIR = f"{SCRATCH_BASE}/hf_cache"
OUTPUT_DIR   = f"{SCRATCH_BASE}/ns_results"
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
from typing import Optional

import torch
from datasets import load_dataset
from huggingface_hub import login, snapshot_download
from tqdm import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    pipeline,
)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
HF_TOKEN     = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN_HERE")
DATASET_NAME = "Prabhjotschugh/narrativeshield-sdoh-medqa"
PERSONAS     = ["alpha", "beta", "gamma"]
PERSONA_KEYS = {"alpha": "persona_alpha", "beta": "persona_beta", "gamma": "persona_gamma"}

# Token budgets — keep Agent 2 generous, Agent 3 tight (it should output 1 char)
TOKENS_AGENT1 = 600
TOKENS_AGENT2 = 700
TOKENS_AGENT3 = 16   # letter only

MODEL_REGISTRY = {
    "llama31":    ("meta-llama/Llama-3.1-8B-Instruct",    "pipeline_text"),
    "mistral7b":  ("mistralai/Mistral-7B-Instruct-v0.3",  "pipeline_text"),
    "qwen25":     ("Qwen/Qwen2.5-7B-Instruct",            "pipeline_text"),
    "gemma3_12b": ("google/gemma-3-12b-it",               "gemma3_manual"),
    "gemma4_e4b": ("google/gemma-4-E4B-it",               "gemma4_manual"),
    "biomistral": ("BioMistral/BioMistral-7B",            "pipeline_text"),
    "llama32":    ("meta-llama/Llama-3.2-3B-Instruct",    "pipeline_text"),
}


# ═════════════════════════════════════════════════════════════════════════════
#  TOOL LAYER
#  All tools are deterministic.
#  Tools are invoked by Python routing code — never by model discretion.
#  The original narrative NEVER reaches any tool.
# ═════════════════════════════════════════════════════════════════════════════

# ── T1: Static Drug Knowledge Base ───────────────────────────────────────────
# Covers the USMLE drug classes most represented in MedQA.
# Constructed from First Aid Step 1/2, AMBOSS, and UpToDate drug summaries.
# Built BLIND to the specific 1,000 test questions — sourced from USMLE
# content outline category frequencies, not individual question inspection.

DRUG_KB = {
    # ── Antibiotics ─────────────────────────────────────────────────────────
    "nitrofurantoin": {
        "class": "Nitrofuran antibiotic",
        "indications": ["uncomplicated UTI", "UTI prophylaxis"],
        "contraindications": ["GFR < 30 mL/min", "term pregnancy (38-42 weeks)", "G6PD deficiency", "neonates < 1 month"],
        "pregnancy_safety": "SAFE in 1st and 2nd trimester. AVOID at term (38-42 weeks): risk of neonatal hemolytic anemia.",
        "mechanism": "Bactericidal — reactive intermediates damage bacterial DNA, ribosomes, and cell wall",
        "key_fact": "First-line for uncomplicated UTI in pregnancy (1st/2nd trimester). Avoid at term.",
    },
    "ciprofloxacin": {
        "class": "Fluoroquinolone antibiotic",
        "indications": ["complicated UTI", "pyelonephritis", "anthrax post-exposure", "gonorrhea", "Pseudomonas"],
        "contraindications": ["pregnancy", "age < 18 (cartilage toxicity)", "myasthenia gravis", "QT prolongation"],
        "pregnancy_safety": "CONTRAINDICATED. Cartilage toxicity in animal studies; avoid in pregnancy.",
        "mechanism": "Inhibits bacterial DNA gyrase (gram-neg) and topoisomerase IV (gram-pos)",
        "key_fact": "CONTRAINDICATED in pregnancy. Use nitrofurantoin or cephalexin instead for UTI.",
    },
    "trimethoprim-sulfamethoxazole": {
        "class": "Sulfonamide + dihydrofolate reductase inhibitor",
        "indications": ["UTI", "PCP pneumonia treatment and prophylaxis", "Nocardia", "Toxoplasma prophylaxis"],
        "contraindications": ["1st trimester pregnancy (neural tube defects)", "3rd trimester (kernicterus)", "G6PD deficiency", "sulfa allergy"],
        "pregnancy_safety": "CONTRAINDICATED throughout pregnancy: folate antagonist (1st trimester NTD), kernicterus risk at term.",
        "key_fact": "Avoid in pregnancy. Covers PCP; add leucovorin when treating Toxoplasma.",
    },
    "amoxicillin": {
        "class": "Aminopenicillin (beta-lactam)",
        "indications": ["strep pharyngitis", "otitis media", "sinusitis", "H. pylori (triple therapy)", "Lyme disease (early)"],
        "contraindications": ["penicillin allergy (cross-reactivity ~1-2%)"],
        "pregnancy_safety": "SAFE — category B. Widely used in pregnancy.",
        "mechanism": "Inhibits transpeptidation (PBP binding) → cell wall synthesis failure",
        "key_fact": "Safe in pregnancy. First-line for strep, otitis media, early Lyme.",
    },
    "amoxicillin-clavulanate": {
        "class": "Aminopenicillin + beta-lactamase inhibitor",
        "indications": ["complicated UTI", "sinusitis (when resistant)", "animal bites", "diabetic foot"],
        "contraindications": ["penicillin allergy", "prior cholestatic jaundice with amoxicillin-clavulanate"],
        "pregnancy_safety": "SAFE — category B. Second-line UTI option in pregnancy.",
        "key_fact": "Beta-lactamase inhibitor expands coverage to resistant organisms including MRSA-negative staph.",
    },
    "cephalexin": {
        "class": "1st generation cephalosporin",
        "indications": ["uncomplicated UTI", "skin/soft tissue infections", "strep pharyngitis (penicillin alternative)"],
        "contraindications": ["cephalosporin allergy (< 1% cross-react with penicillin)"],
        "pregnancy_safety": "SAFE — category B. Preferred alternative to nitrofurantoin at term.",
        "key_fact": "Safe in pregnancy including at term. Good UTI option when nitrofurantoin contraindicated.",
    },
    "vancomycin": {
        "class": "Glycopeptide antibiotic",
        "indications": ["MRSA infections", "C. diff (oral, severe)", "beta-lactam allergic gram-pos coverage"],
        "contraindications": ["vancomycin allergy", "use with caution in renal failure (nephrotoxic)"],
        "adverse_effects": ["nephrotoxicity", "ototoxicity", "Red Man Syndrome (rate-related, not allergy)"],
        "key_fact": "Drug of choice for MRSA. Red Man Syndrome prevented by slow infusion. Monitor troughs.",
    },
    "metronidazole": {
        "class": "Nitroimidazole antibiotic/antiprotozoal",
        "indications": ["C. diff (mild-moderate)", "bacterial vaginosis", "Trichomoniasis", "Giardia", "anaerobic infections", "H. pylori"],
        "contraindications": ["avoid alcohol (disulfiram reaction)", "1st trimester pregnancy (controversial)"],
        "pregnancy_safety": "AVOID 1st trimester — teratogenicity concern. Use in 2nd/3rd trimester acceptable.",
        "key_fact": "Disulfiram-like reaction with alcohol. Turns urine dark. First-line for C. diff mild.",
    },
    "azithromycin": {
        "class": "Macrolide antibiotic",
        "indications": ["community-acquired pneumonia (atypicals)", "STIs (chlamydia, M. genitalium)", "MAC prophylaxis", "pertussis"],
        "contraindications": ["QT prolongation", "liver disease (use erythromycin instead)", "macrolide allergy"],
        "mechanism": "Inhibits 50S ribosomal subunit (23S rRNA) → protein synthesis inhibition",
        "key_fact": "Z-pack. Covers atypicals (Mycoplasma, Legionella, Chlamydophila). QT prolongation risk.",
    },
    "doxycycline": {
        "class": "Tetracycline antibiotic",
        "indications": ["Lyme disease (all stages)", "RMSF", "chlamydia", "atypical pneumonia", "malaria prophylaxis", "acne"],
        "contraindications": ["pregnancy (teeth discoloration, bone toxicity)", "age < 8 (same reason)", "esophageal stricture (take upright)"],
        "pregnancy_safety": "CONTRAINDICATED — deposits in fetal teeth and bones; discoloration and growth inhibition.",
        "key_fact": "Drug of choice for RMSF (even in children despite caveat). Avoid in pregnancy.",
    },
    "clindamycin": {
        "class": "Lincosamide antibiotic",
        "indications": ["anaerobic infections above diaphragm", "MRSA skin infections", "bacterial vaginosis", "toxoplasmosis (with pyrimethamine)", "group B strep in penicillin-allergic pregnant patients"],
        "adverse_effects": ["pseudomembranous colitis (C. diff — highest risk of all antibiotics)"],
        "key_fact": "Highest risk of C. diff of any antibiotic. Safe in pregnancy for penicillin-allergic patients.",
    },

    # ── Opioids and withdrawal ───────────────────────────────────────────────
    "oxycodone": {
        "class": "Opioid analgesic (Schedule II, full mu-agonist)",
        "indications": ["moderate to severe pain"],
        "contraindications": ["respiratory depression", "paralytic ileus", "concurrent MAOIs", "hypersensitivity"],
        "adverse_effects": ["constipation (tolerance does NOT develop)", "nausea", "respiratory depression", "miosis", "physical dependence", "sedation"],
        "withdrawal_symptoms": ["nausea", "vomiting", "diarrhea", "myalgias", "arthralgias", "rhinorrhea", "lacrimation", "diaphoresis", "piloerection", "anxiety", "insomnia", "yawning", "tachycardia", "hypertension"],
        "pregnancy_safety": "AVOID — neonatal abstinence syndrome (NAS) in neonate.",
        "key_fact": "WITHDRAWAL mimics flu: nausea/vomiting/diarrhea/myalgias/rhinorrhea. Key differentiator: patient on opioids with dose change or access interruption.",
    },
    "morphine": {
        "class": "Opioid analgesic (Schedule II, full mu-agonist)",
        "withdrawal_symptoms": ["nausea", "vomiting", "diarrhea", "myalgias", "rhinorrhea", "diaphoresis", "anxiety", "tachycardia"],
        "adverse_effects": ["constipation", "respiratory depression", "histamine release (avoid in asthma/anaphylaxis)"],
        "key_fact": "Releases histamine — avoid in asthma and hemodynamically unstable patients. Use fentanyl instead.",
    },
    "fentanyl": {
        "class": "Opioid analgesic (Schedule II, highly lipophilic)",
        "indications": ["procedural sedation", "ICU analgesia", "chronic pain patches"],
        "adverse_effects": ["chest wall rigidity (rapid IV push — 'wooden chest syndrome')", "respiratory depression"],
        "key_fact": "Does NOT release histamine. Preferred in hemodynamically unstable or asthmatic patients.",
    },
    "naloxone": {
        "class": "Opioid antagonist",
        "indications": ["opioid overdose reversal", "opioid-induced respiratory depression"],
        "key_fact": "Short half-life (30-90 min) — may need redosing or infusion for long-acting opioid overdose.",
    },
    "methadone": {
        "class": "Opioid agonist + NMDA antagonist (long-acting)",
        "indications": ["opioid use disorder maintenance", "chronic pain"],
        "key_fact": "QT prolongation risk. Long half-life causes accumulation. Used in pregnancy for OUD.",
    },
    "buprenorphine": {
        "class": "Partial opioid agonist (mu) + kappa antagonist",
        "indications": ["opioid use disorder (Suboxone = buprenorphine + naloxone)", "moderate pain"],
        "key_fact": "Partial agonist — ceiling effect on respiratory depression. Safer in OUD than full agonists.",
    },

    # ── Smoking cessation ────────────────────────────────────────────────────
    "varenicline": {
        "class": "Nicotinic acetylcholine receptor partial agonist",
        "indications": ["smoking cessation"],
        "adverse_effects": ["nausea (most common)", "insomnia", "vivid dreams", "neuropsychiatric symptoms (black box)", "suicidal ideation"],
        "key_fact": "Most effective pharmacotherapy for smoking cessation. Black box: neuropsychiatric symptoms. Nausea dose-limiting.",
    },
    "bupropion": {
        "class": "Atypical antidepressant / dopamine-norepinephrine reuptake inhibitor",
        "indications": ["depression", "smoking cessation", "ADHD (off-label)", "seasonal affective disorder"],
        "contraindications": ["seizure disorder", "anorexia/bulimia nervosa", "MAOI use within 14 days", "abrupt alcohol/benzo withdrawal"],
        "key_fact": "Lowers seizure threshold. CONTRAINDICATED in eating disorders and seizure history.",
    },

    # ── Cardiovascular ───────────────────────────────────────────────────────
    "aspirin": {
        "class": "NSAID / irreversible COX-1 and COX-2 inhibitor / antiplatelet",
        "indications": ["ACS (first-line antiplatelet)", "stroke prevention (TIA/stroke)", "pre-eclampsia prophylaxis (low dose)", "Kawasaki disease"],
        "contraindications": ["children with viral illness (Reye syndrome)", "active peptic ulcer", "aspirin-exacerbated respiratory disease (AERD/Samter triad)"],
        "pregnancy_safety": "Low dose (81mg) SAFE for pre-eclampsia prevention after 12 weeks. High dose: AVOID.",
        "key_fact": "Reye syndrome in children with viral illness. AERD = aspirin + asthma + nasal polyps.",
    },
    "warfarin": {
        "class": "Vitamin K antagonist anticoagulant",
        "indications": ["atrial fibrillation", "DVT/PE treatment", "mechanical heart valve", "hypercoagulable states"],
        "contraindications": ["pregnancy", "active major bleeding", "uncontrolled hypertension"],
        "pregnancy_safety": "CONTRAINDICATED — crosses placenta. Warfarin embryopathy: nasal hypoplasia, stippled epiphyses (1st trimester), CNS defects (any trimester), fetal/neonatal bleeding.",
        "reversal": "Vitamin K (hours-days) + FFP (immediate) + 4-factor PCC (fastest, preferred in life-threatening bleed)",
        "monitoring": "INR (target 2-3 for most indications; 2.5-3.5 for mechanical mitral valve)",
        "key_fact": "Use LMWH (enoxaparin) or unfractionated heparin in pregnancy instead.",
    },
    "heparin": {
        "class": "Indirect thrombin inhibitor (activates antithrombin III)",
        "indications": ["DVT/PE (acute treatment)", "ACS (NSTEMI/UA)", "pregnancy anticoagulation", "bridging therapy", "hemodialysis"],
        "contraindications": ["HIT (heparin-induced thrombocytopenia + thrombosis)", "active bleeding"],
        "pregnancy_safety": "SAFE — does NOT cross placenta. Drug of choice for anticoagulation in pregnancy.",
        "monitoring": "aPTT (target 60-100 sec, or 2-2.5x normal)",
        "reversal": "Protamine sulfate (1 mg per 100 units heparin given in last 2-3 hours)",
        "key_fact": "Check CBC on day 4-10 for HIT. If platelets fall >50%, stop heparin immediately.",
    },
    "enoxaparin": {
        "class": "Low molecular weight heparin (LMWH)",
        "indications": ["DVT/PE treatment and prophylaxis", "ACS", "pregnancy anticoagulation (preferred)"],
        "contraindications": ["severe renal failure (GFR < 30) — use UFH instead", "HIT"],
        "pregnancy_safety": "SAFE — does not cross placenta. Preferred over UFH in pregnancy for convenience.",
        "monitoring": "Anti-Xa levels (unlike UFH, aPTT not reliable for monitoring)",
        "key_fact": "No reliable aPTT monitoring. Use anti-Xa. Preferred in pregnancy.",
    },
    "metoprolol": {
        "class": "Selective beta-1 adrenergic antagonist",
        "indications": ["hypertension", "angina", "heart failure (HFrEF — start low, titrate up)", "rate control in AF", "post-MI"],
        "contraindications": ["decompensated HF (acutely)", "severe bradycardia", "2nd/3rd degree AV block", "cardiogenic shock"],
        "key_fact": "Beta-1 selective (cardioselective) — safer in asthma/COPD than non-selective. Do NOT stop abruptly.",
    },
    "lisinopril": {
        "class": "ACE inhibitor",
        "indications": ["hypertension", "heart failure (HFrEF)", "diabetic nephropathy (proteinuria)", "post-MI (reduces remodeling)"],
        "contraindications": ["pregnancy (2nd/3rd trimester fetotoxicity)", "bilateral renal artery stenosis", "hyperkalemia", "history of ACE inhibitor-induced angioedema"],
        "pregnancy_safety": "CONTRAINDICATED in 2nd/3rd trimester — causes fetal renal agenesis, oligohydramnios, Potter sequence, limb contractures.",
        "adverse_effects": ["dry cough (bradykinin accumulation)", "hyperkalemia", "angioedema (rare but life-threatening)"],
        "key_fact": "Dry cough → switch to ARB (losartan). First-line for diabetic nephropathy and post-MI.",
    },
    "losartan": {
        "class": "Angiotensin II receptor blocker (ARB)",
        "indications": ["hypertension", "heart failure (HFrEF, ACE-intolerant)", "diabetic nephropathy", "Marfan syndrome"],
        "contraindications": ["pregnancy (same as ACE inhibitors)", "bilateral renal artery stenosis", "hyperkalemia"],
        "pregnancy_safety": "CONTRAINDICATED in 2nd/3rd trimester — same fetotoxicity as ACE inhibitors.",
        "key_fact": "Use when patient cannot tolerate ACE inhibitor cough. NO cough (no bradykinin effect).",
    },
    "amlodipine": {
        "class": "Dihydropyridine calcium channel blocker",
        "indications": ["hypertension", "stable angina", "Raynaud phenomenon", "hypertension in pregnancy (safe)"],
        "adverse_effects": ["peripheral edema (most common)", "reflex tachycardia", "headache", "flushing"],
        "pregnancy_safety": "ACCEPTABLE — used for hypertension in pregnancy (2nd line after methyldopa/labetalol)",
        "key_fact": "Peripheral edema is dose-dependent. Safer in pregnancy than ACE inhibitors/ARBs.",
    },
    "hydralazine": {
        "class": "Direct arterial vasodilator",
        "indications": ["hypertensive emergency in pregnancy", "heart failure (with nitrates)", "resistant hypertension"],
        "adverse_effects": ["reflex tachycardia", "lupus-like syndrome (drug-induced lupus)", "fluid retention"],
        "pregnancy_safety": "SAFE — first-line for hypertensive emergency in pregnancy.",
        "key_fact": "Drug-induced lupus (+ anti-histone antibodies). Used IV for acute severe HTN in pregnancy.",
    },
    "labetalol": {
        "class": "Alpha and non-selective beta-adrenergic antagonist",
        "indications": ["hypertensive emergency in pregnancy", "hypertension", "pheochromocytoma (combined alpha/beta block)"],
        "pregnancy_safety": "SAFE — first-line oral antihypertensive in pregnancy.",
        "key_fact": "Combined alpha + beta block. Drug of choice for chronic HTN in pregnancy.",
    },
    "methyldopa": {
        "class": "Central alpha-2 agonist (sympatholytic)",
        "indications": ["hypertension in pregnancy (historical first-line)", "hypertension"],
        "pregnancy_safety": "SAFE — longstanding safety record in pregnancy.",
        "key_fact": "Causes hemolytic anemia (positive Coombs). First-line in many pregnancy guidelines.",
    },

    # ── Diabetes ─────────────────────────────────────────────────────────────
    "metformin": {
        "class": "Biguanide antidiabetic",
        "indications": ["type 2 diabetes (first-line)", "PCOS", "prediabetes prevention"],
        "contraindications": ["GFR < 30 mL/min", "iodinated contrast (hold 48h before and after)", "alcoholism", "hepatic failure", "metabolic acidosis"],
        "mechanism": "Inhibits mitochondrial complex I → reduces hepatic gluconeogenesis → increases insulin sensitivity",
        "adverse_effects": ["GI (nausea, diarrhea — take with food)", "lactic acidosis (rare but life-threatening)", "vitamin B12 deficiency (long-term)"],
        "key_fact": "First-line T2DM. Does NOT cause hypoglycemia alone. Hold before contrast. Check B12 yearly.",
    },
    "insulin": {
        "class": "Pancreatic hormone / antidiabetic",
        "indications": ["type 1 diabetes", "T2DM (if oral agents insufficient)", "DKA", "hyperkalemia (drives K into cells)", "gestational diabetes"],
        "adverse_effects": ["hypoglycemia (most important)", "weight gain", "lipodystrophy at injection sites"],
        "key_fact": "Only safe antidiabetic in pregnancy (besides metformin in some guidelines). Used for hyperkalemia with glucose.",
    },

    # ── Psychiatry ───────────────────────────────────────────────────────────
    "lithium": {
        "class": "Mood stabilizer",
        "indications": ["bipolar disorder (gold standard — acute mania + maintenance)", "SIADH (causes nephrogenic DI as side effect)", "cluster headache prophylaxis"],
        "contraindications": ["severe renal failure", "pregnancy (1st trimester Ebstein anomaly risk)"],
        "pregnancy_safety": "AVOID 1st trimester — Ebstein anomaly (tricuspid valve dysplasia, right ventricular hypoplasia). Risk lower than historical estimates but still caution advised.",
        "toxicity": "Narrow therapeutic index (0.6-1.2 mEq/L). Toxicity signs: tremor (coarse) → ataxia → confusion → seizures → coma → cardiac arrhythmias. Treat with dialysis.",
        "drug_interactions": "NSAIDs, thiazides, ACE inhibitors all INCREASE lithium levels (reduce excretion).",
        "key_fact": "Monitor levels. Nephrogenic DI side effect (patient drinks more water, urinates more).",
    },
    "haloperidol": {
        "class": "Typical (1st gen) antipsychotic — high potency",
        "indications": ["schizophrenia", "acute agitation (IM)", "Tourette syndrome", "delirium"],
        "adverse_effects": ["EPS: akathisia, dystonia, parkinsonian sx, tardive dyskinesia", "NMS (neuroleptic malignant syndrome)", "QT prolongation"],
        "key_fact": "High EPS risk. NMS = fever + rigidity + autonomic instability + elevated CK. Treat with bromocriptine + dantrolene.",
    },
    "clozapine": {
        "class": "Atypical (2nd gen) antipsychotic",
        "indications": ["treatment-resistant schizophrenia", "suicidality in schizophrenia"],
        "contraindications": ["ANC < 1500 (absolute)", "history of clozapine-induced agranulocytosis"],
        "adverse_effects": ["agranulocytosis (monitor ANC weekly x6 months)", "seizures (dose-dependent)", "myocarditis", "metabolic syndrome", "hypersalivation", "sedation"],
        "key_fact": "Most effective antipsychotic but requires mandatory ANC monitoring. Only antipsychotic proven to reduce suicide.",
    },
    "fluoxetine": {
        "class": "SSRI (selective serotonin reuptake inhibitor)",
        "indications": ["depression", "OCD", "panic disorder", "bulimia nervosa", "PMDD"],
        "contraindications": ["MAOI use within 14 days (serotonin syndrome)", "linezolid (serotonin syndrome)"],
        "pregnancy_safety": "Generally considered safest SSRI (category C). Neonatal withdrawal syndrome with late-pregnancy use.",
        "key_fact": "Longest half-life of SSRIs (4-6 days). Less discontinuation syndrome. Inhibits CYP2D6.",
    },
    "sertraline": {
        "class": "SSRI",
        "indications": ["depression", "OCD", "PTSD", "panic disorder", "social anxiety", "PMDD"],
        "pregnancy_safety": "Preferred SSRI in pregnancy — best safety data.",
        "key_fact": "Most prescribed SSRI in pregnancy. Neonatal abstinence syndrome risk with 3rd trimester use.",
    },

    # ── Corticosteroids ──────────────────────────────────────────────────────
    "prednisone": {
        "class": "Systemic corticosteroid",
        "indications": ["autoimmune diseases", "asthma exacerbation", "COPD exacerbation", "adrenal insufficiency", "inflammatory conditions"],
        "adverse_effects": ["hyperglycemia", "osteoporosis (chronic)", "Cushingoid features", "immunosuppression", "peptic ulcer", "avascular necrosis (femoral head)", "adrenal suppression"],
        "key_fact": "Give PPI prophylaxis with NSAIDs but NOT steroids alone. Taper slowly to prevent adrenal crisis.",
    },
    "betamethasone": {
        "class": "Corticosteroid",
        "indications": ["fetal lung maturity (24-34 weeks preterm labor)", "skin conditions (topical)"],
        "pregnancy_safety": "SAFE and indicated — given IM to mother to accelerate fetal surfactant production.",
        "key_fact": "Standard of care for preterm labor 24-34 weeks. Promotes fetal lung maturity (surfactant).",
    },

    # ── Obstetrics / Gynecology ──────────────────────────────────────────────
    "magnesium-sulfate": {
        "class": "Tocolytic / anticonvulsant",
        "indications": ["eclampsia seizure prevention and treatment", "neuroprotection in preterm < 32 weeks", "tocolysis (adjunct)"],
        "toxicity": "Loss of patellar reflex (first sign) → respiratory depression → cardiac arrest",
        "reversal": "Calcium gluconate IV",
        "key_fact": "Monitor reflexes and respiratory rate. Antidote is calcium gluconate.",
    },
    "oxytocin": {
        "class": "Uterotonic hormone",
        "indications": ["labor induction/augmentation", "postpartum hemorrhage prevention"],
        "adverse_effects": ["water retention/hyponatremia (antidiuretic effect)", "uterine hyperstimulation", "fetal distress"],
        "key_fact": "Antidiuretic effect at high doses — can cause hyponatremia. Avoid in placenta previa.",
    },
    "misoprostol": {
        "class": "Prostaglandin E1 analog",
        "indications": ["cervical ripening/labor induction", "postpartum hemorrhage", "medical abortion (with mifepristone)", "peptic ulcer prevention with NSAIDs"],
        "contraindications": ["prior uterine surgery/cesarean (uterine rupture risk)", "active labor"],
        "key_fact": "Risk of uterine rupture with prior cesarean. Used for PPH when other uterotonics fail.",
    },

    # ── Rheumatology/Immunology ──────────────────────────────────────────────
    "methotrexate": {
        "class": "Antifolate (DMARD / antineoplastic)",
        "indications": ["rheumatoid arthritis", "psoriasis", "ectopic pregnancy", "chemotherapy", "gestational trophoblastic disease"],
        "contraindications": ["pregnancy (teratogen — fetal death, neural tube defects)", "hepatic disease", "renal failure", "immunodeficiency"],
        "pregnancy_safety": "ABSOLUTELY CONTRAINDICATED — major teratogen. Ensure pregnancy test negative before starting. Discontinue 3 months before conception attempt.",
        "key_fact": "Leucovorin (folinic acid) rescue reduces toxicity. Monitor CBC and LFTs. Highly teratogenic.",
    },
    "hydroxychloroquine": {
        "class": "Antimalarial / DMARD",
        "indications": ["SLE", "rheumatoid arthritis", "malaria prophylaxis and treatment"],
        "adverse_effects": ["retinal toxicity (bull's eye maculopathy — baseline eye exam required)", "QT prolongation"],
        "pregnancy_safety": "SAFE in pregnancy — recommended to continue in SLE patients (reduces flares).",
        "key_fact": "Annual ophthalmology exam for retinal toxicity. Safest DMARD in pregnancy.",
    },

    # ── Pulmonology ──────────────────────────────────────────────────────────
    "albuterol": {
        "class": "Short-acting beta-2 agonist (SABA)",
        "indications": ["acute asthma exacerbation (first-line rescue)", "COPD bronchospasm", "hyperkalemia (drives K into cells)", "exercise-induced bronchospasm"],
        "adverse_effects": ["tachycardia", "hypokalemia (high doses)", "tremor"],
        "key_fact": "Rescue inhaler. Use before LABA in stepwise asthma therapy. Hypokalemia with frequent use.",
    },
    "ipratropium": {
        "class": "Short-acting muscarinic antagonist (SAMA)",
        "indications": ["COPD (first-line)", "asthma exacerbation (add-on to albuterol)", "rhinorrhea"],
        "key_fact": "Preferred bronchodilator in COPD over beta-agonists. Combine with albuterol in acute exacerbation.",
    },
    "n-acetylcysteine": {
        "class": "Mucolytic / antidote",
        "indications": ["acetaminophen toxicity (antidote)", "contrast-induced nephropathy prophylaxis", "mucolytic in CF/COPD"],
        "key_fact": "Antidote for acetaminophen overdose — replenishes glutathione. Give within 8-10 hours for best effect.",
    },
}

# Normalize keys for lookup (lowercase, remove hyphens/spaces)
_DRUG_KB_NORMALIZED = {
    re.sub(r'[-\s]', '', k.lower()): v
    for k, v in DRUG_KB.items()
}


def lookup_drug(drug_name: str) -> Optional[dict]:
    """
    Looks up a drug in the static KB.
    Handles partial matches and common name variations.
    """
    key = re.sub(r'[-\s]', '', drug_name.lower().strip())
    # Exact match
    if key in _DRUG_KB_NORMALIZED:
        return _DRUG_KB_NORMALIZED[key]
    # Partial match (drug_name is substring of KB key or vice versa)
    for kb_key, entry in _DRUG_KB_NORMALIZED.items():
        if key in kb_key or kb_key in key:
            return entry
    return None


# ── T2: Lab Reference Ranges (static) ────────────────────────────────────────
LAB_RANGES = {
    "WBC":      {"low": 4.5,  "high": 11.0, "unit": "10^3/uL",
                 "high_interp": "Leukocytosis — infection/inflammation/leukemia/stress",
                 "low_interp":  "Leukopenia — immunosuppression/viral illness/bone marrow failure"},
    "HGB_M":    {"low": 13.5, "high": 17.5, "unit": "g/dL", "low_interp": "Anemia in male"},
    "HGB_F":    {"low": 12.0, "high": 15.5, "unit": "g/dL", "low_interp": "Anemia in female"},
    "HCT_M":    {"low": 41.0, "high": 53.0, "unit": "%"},
    "HCT_F":    {"low": 36.0, "high": 46.0, "unit": "%"},
    "PLT":      {"low": 150,  "high": 400,  "unit": "10^3/uL",
                 "high_interp": "Thrombocytosis — reactive (infection/iron deficiency) or primary",
                 "low_interp":  "Thrombocytopenia — ITP, TTP, DIC, HIT, drug-induced, marrow failure"},
    "Na":       {"low": 136,  "high": 145,  "unit": "mEq/L",
                 "high_interp": "Hypernatremia — free water deficit (dehydration, DI)",
                 "low_interp":  "Hyponatremia — SIADH/HF/cirrhosis/hypothyroidism"},
    "K":        {"low": 3.5,  "high": 5.0,  "unit": "mEq/L",
                 "high_interp": "Hyperkalemia — renal failure/ACEi/K-sparing diuretics/acidosis — EKG changes: peaked T → PR prolonged → wide QRS → sine wave",
                 "low_interp":  "Hypokalemia — diuretics/vomiting/Conn syndrome — U waves on EKG"},
    "Cr":       {"low": 0.6,  "high": 1.2,  "unit": "mg/dL",
                 "high_interp": "Elevated creatinine — AKI or CKD; check BUN/Cr ratio (>20:1 prerenal)"},
    "BUN":      {"low": 7.0,  "high": 20.0, "unit": "mg/dL",
                 "high_interp": "Elevated BUN — dehydration/GI bleed/renal failure; BUN/Cr ratio >20:1 → prerenal"},
    "Glucose":  {"low": 70.0, "high": 100.0,"unit": "mg/dL",
                 "high_interp": "Hyperglycemia (fasting) — diabetes/stress/steroids/pancreatitis",
                 "low_interp":  "Hypoglycemia — insulin excess/Addison/liver failure/insulinoma"},
    "ALT":      {"low": 7.0,  "high": 56.0, "unit": "U/L",
                 "high_interp": "Hepatocellular injury — hepatitis/NAFLD/drug toxicity/ischemia"},
    "AST":      {"low": 10.0, "high": 40.0, "unit": "U/L",
                 "high_interp": "Hepatocellular or muscle injury; AST:ALT > 2:1 suggests alcoholic hepatitis"},
    "ALP":      {"low": 44.0, "high": 147.0,"unit": "U/L",
                 "high_interp": "Cholestasis/bone disease/infiltrative liver disease/pregnancy (placental)"},
    "TBili":    {"low": 0.1,  "high": 1.2,  "unit": "mg/dL",
                 "high_interp": "Jaundice >2.5; direct (conjugated) → cholestasis; indirect (unconjugated) → hemolysis/Gilbert/neonatal"},
    "Ca":       {"low": 8.5,  "high": 10.5, "unit": "mg/dL",
                 "high_interp": "Hypercalcemia — Bones Stones Groans Moans: hyperPTH/malignancy/sarcoidosis/vitamin D toxicity",
                 "low_interp":  "Hypocalcemia — hypoPTH/vitamin D deficiency/pancreatitis/hypoMg — Chvostek/Trousseau signs"},
    "Mg":       {"low": 1.7,  "high": 2.2,  "unit": "mEq/L",
                 "low_interp":  "Hypomagnesemia — diuretics/alcoholism/malabsorption — causes refractory hypoK and hypoCa"},
    "Phos":     {"low": 2.5,  "high": 4.5,  "unit": "mg/dL",
                 "low_interp":  "Hypophosphatemia — refeeding syndrome/hyperPTH/antacids/poor nutrition"},
    "TSH":      {"low": 0.4,  "high": 4.0,  "unit": "mIU/L",
                 "high_interp": "Hypothyroidism (or recovery from hyperthyroid Rx)",
                 "low_interp":  "Hyperthyroidism or exogenous thyroid hormone"},
    "PT":       {"low": 11.0, "high": 13.5, "unit": "sec",
                 "high_interp": "Prolonged PT/INR — liver disease/vitamin K deficiency/warfarin/DIC (factor VII deficiency first)"},
    "INR":      {"low": 0.8,  "high": 1.1,  "unit": "ratio",
                 "high_interp": "Coagulopathy — same as elevated PT; target 2-3 on warfarin"},
    "PTT":      {"low": 25.0, "high": 35.0, "unit": "sec",
                 "high_interp": "Prolonged PTT — heparin/hemophilia A or B/vWD/lupus anticoagulant/DIC"},
    "CRP":      {"low": 0.0,  "high": 1.0,  "unit": "mg/dL",
                 "high_interp": "Acute phase reactant — infection/autoimmune/malignancy/MI"},
    "ESR_M":    {"low": 0.0,  "high": 15.0, "unit": "mm/hr"},
    "ESR_F":    {"low": 0.0,  "high": 20.0, "unit": "mm/hr",
                 "high_interp": "Nonspecific — infection/autoimmune/malignancy/anemia/pregnancy"},
    "HbA1c":    {"low": 4.0,  "high": 5.6,  "unit": "%",
                 "high_interp": "5.7-6.4 = prediabetes; ≥6.5 = diabetes (two readings); target <7.0 if diabetic"},
    "LDL":      {"low": 0.0,  "high": 100.0,"unit": "mg/dL",
                 "high_interp": "Elevated LDL — cardiovascular risk; target <70 in very high risk (prior MI/DM+CVD)"},
    "HDL_M":    {"low": 40.0, "high": 60.0, "unit": "mg/dL",
                 "low_interp":  "Low HDL — independent CVD risk factor"},
    "HDL_F":    {"low": 50.0, "high": 60.0, "unit": "mg/dL",
                 "low_interp":  "Low HDL — independent CVD risk factor"},
    "TG":       {"low": 0.0,  "high": 150.0,"unit": "mg/dL",
                 "high_interp": ">500: pancreatitis risk; 150-499: borderline/high CVD risk; >1000: eruptive xanthomas"},
    "Bicarb":   {"low": 22.0, "high": 29.0, "unit": "mEq/L",
                 "high_interp": "Metabolic alkalosis (or compensating respiratory acidosis)",
                 "low_interp":  "Metabolic acidosis (or compensating respiratory alkalosis)"},
    "pO2":      {"low": 75.0, "high": 100.0,"unit": "mmHg",
                 "low_interp":  "Hypoxemia — check A-a gradient; <60 = respiratory failure threshold"},
    "pH":       {"low": 7.35, "high": 7.45, "unit": "units",
                 "high_interp": "Alkalemia — respiratory (↓pCO2) or metabolic (↑HCO3)",
                 "low_interp":  "Acidemia — respiratory (↑pCO2) or metabolic (↓HCO3)"},
    "pCO2":     {"low": 35.0, "high": 45.0, "unit": "mmHg",
                 "high_interp": "Hypercapnia — respiratory acidosis or compensatory for metabolic alkalosis",
                 "low_interp":  "Hypocapnia — respiratory alkalosis or compensatory for metabolic acidosis"},
    "Troponin": {"low": 0.0,  "high": 0.04, "unit": "ng/mL",
                 "high_interp": "Myocardial injury — ACS (NSTEMI/STEMI), myocarditis, PE, severe sepsis, demand ischemia"},
    "BNP":      {"low": 0.0,  "high": 100.0,"unit": "pg/mL",
                 "high_interp": "Heart failure (>400 = high likelihood); also elevated in PE, renal failure"},
    "Lipase":   {"low": 0.0,  "high": 160.0,"unit": "U/L",
                 "high_interp": "Pancreatitis (>3x ULN = diagnostic); also gallbladder disease, renal failure"},
    "Amylase":  {"low": 25.0, "high": 125.0,"unit": "U/L",
                 "high_interp": "Pancreatitis or salivary gland disease; less specific than lipase"},
    "PSA":      {"low": 0.0,  "high": 4.0,  "unit": "ng/mL",
                 "high_interp": "Elevated PSA — prostate cancer (consider biopsy if >4); also BPH, prostatitis, DRE"},
}

_LAB_ALIASES = {
    "wbc": "WBC", "white blood cell": "WBC", "white count": "WBC",
    "hgb": "HGB_F", "hb": "HGB_F", "hemoglobin": "HGB_F",
    "hct": "HCT_F", "hematocrit": "HCT_F",
    "plt": "PLT", "platelets": "PLT", "platelet count": "PLT",
    "sodium": "Na", "na": "Na",
    "potassium": "K", "k": "K",
    "creatinine": "Cr", "cr": "Cr",
    "bun": "BUN", "blood urea nitrogen": "BUN",
    "glucose": "Glucose", "blood glucose": "Glucose", "fasting glucose": "Glucose",
    "alt": "ALT", "sgpt": "ALT",
    "ast": "AST", "sgot": "AST",
    "alp": "ALP", "alkaline phosphatase": "ALP",
    "tbili": "TBili", "total bilirubin": "TBili", "bilirubin": "TBili",
    "calcium": "Ca", "ca": "Ca",
    "magnesium": "Mg", "mg": "Mg",
    "phosphate": "Phos", "phosphorus": "Phos",
    "tsh": "TSH",
    "pt": "PT", "prothrombin time": "PT",
    "inr": "INR",
    "ptt": "PTT", "aptt": "PTT", "activated ptt": "PTT",
    "crp": "CRP", "c-reactive protein": "CRP",
    "esr": "ESR_F", "sed rate": "ESR_F",
    "hba1c": "HbA1c", "a1c": "HbA1c", "glycated hemoglobin": "HbA1c",
    "ldl": "LDL",
    "hdl": "HDL_F",
    "triglycerides": "TG", "tg": "TG",
    "bicarbonate": "Bicarb", "bicarb": "Bicarb", "hco3": "Bicarb",
    "po2": "pO2", "o2": "pO2",
    "ph": "pH",
    "pco2": "pCO2", "co2": "pCO2",
    "troponin": "Troponin", "troponin i": "Troponin", "troponin t": "Troponin",
    "bnp": "BNP", "pro-bnp": "BNP",
    "lipase": "Lipase",
    "amylase": "Amylase",
    "psa": "PSA",
}


def interpret_lab(lab_name: str, value: float, sex: str = "unknown") -> dict:
    """T2: Deterministic lab interpretation."""
    key = _LAB_ALIASES.get(lab_name.lower().strip())
    if not key:
        key = lab_name.upper().strip()

    # Apply sex-specific range
    if key in ("HGB", "HGB_F", "HGB_M"):
        key = "HGB_M" if sex.lower() in ("m", "male") else "HGB_F"
    elif key in ("HCT", "HCT_F", "HCT_M"):
        key = "HCT_M" if sex.lower() in ("m", "male") else "HCT_F"
    elif key in ("HDL", "HDL_F", "HDL_M"):
        key = "HDL_M" if sex.lower() in ("m", "male") else "HDL_F"
    elif key in ("ESR", "ESR_F", "ESR_M"):
        key = "ESR_M" if sex.lower() in ("m", "male") else "ESR_F"

    if key not in LAB_RANGES:
        return {"lab": lab_name, "value": value, "status": "unknown_lab",
                "interpretation": f"No reference range for '{lab_name}'"}

    ref = LAB_RANGES[key]
    low, high = ref.get("low"), ref.get("high")

    if low is not None and value < low:
        status = "LOW"
        interp = ref.get("low_interp", f"{lab_name} below reference range ({low} {ref['unit']})")
    elif high is not None and value > high:
        status = "HIGH"
        interp = ref.get("high_interp", f"{lab_name} above reference range ({high} {ref['unit']})")
    else:
        status = "NORMAL"
        interp = f"{lab_name} {value} {ref['unit']} — within reference range ({low}-{high})"

    return {
        "lab": lab_name, "value": value, "unit": ref.get("unit", ""),
        "reference": f"{low}–{high}", "status": status,
        "interpretation": interp,
    }


# ── T3: Clinical Scoring Calculators ─────────────────────────────────────────

def score_wells_dvt(p: dict) -> str:
    s  = int(bool(p.get("active_cancer")))
    s += int(bool(p.get("paralysis_paresis_immobilization")))
    s += int(bool(p.get("bedridden_3d_or_surgery_12wk")))
    s += int(bool(p.get("localized_tenderness")))
    s += int(bool(p.get("entire_leg_swollen")))
    s += int(bool(p.get("calf_swelling_3cm")))
    s += int(bool(p.get("pitting_edema")))
    s += int(bool(p.get("collateral_veins")))
    s -= 2 * int(bool(p.get("alt_dx_as_likely")))
    risk = "HIGH (75%)" if s >= 3 else "MODERATE (17%)" if s in (1, 2) else "LOW (3%)"
    return f"Wells DVT score {s} → {risk} DVT probability"


def score_wells_pe(p: dict) -> str:
    s  = 3.0 * int(bool(p.get("dvt_signs")))
    s += 3.0 * int(bool(p.get("pe_most_likely")))
    s += 1.5 * int(bool(p.get("hr_over_100")))
    s += 1.5 * int(bool(p.get("immobilization_or_surgery_4wk")))
    s += 1.5 * int(bool(p.get("prior_dvt_pe")))
    s += 1.0 * int(bool(p.get("hemoptysis")))
    s += 1.0 * int(bool(p.get("malignancy")))
    risk = "HIGH (~66%)" if s > 6 else "MODERATE (~29%)" if s > 2 else "LOW (~2%)"
    return f"Wells PE score {s} → {risk} PE probability"


def score_curb65(p: dict) -> str:
    s = sum([bool(p.get("confusion")), bool(p.get("bun_over_19")),
             bool(p.get("rr_30_or_more")), bool(p.get("sbp_under_90_or_dbp_under_60")),
             bool(p.get("age_65_or_more"))])
    mgmt = ["Outpatient OK", "Outpatient OK", "Consider admission", "Admit (ICU if 4-5)"][min(s, 3)]
    return f"CURB-65 score {s}/5 → 30-day mortality ~{[0.7,2.1,9.2,14.5,40][min(s,4)]}%. {mgmt}"


def score_gcs(p: dict) -> str:
    total = int(p.get("eye", 4)) + int(p.get("verbal", 5)) + int(p.get("motor", 6))
    sev = "Mild TBI/normal" if total >= 13 else "Moderate TBI" if total >= 9 else "Severe TBI — intubation"
    return f"GCS {total}/15 → {sev}"


def score_apgar(p: dict) -> str:
    total = sum(int(p.get(k, 0)) for k in ["appearance", "pulse", "grimace", "activity", "respiration"])
    status = "Normal (≥7)" if total >= 7 else "Moderate concern (4-6)" if total >= 4 else "Critical (<4) — resuscitate"
    return f"Apgar {total}/10 → {status}"


# ── Deterministic Tool Router ─────────────────────────────────────────────────

def is_likely_drug(text: str) -> bool:
    """Heuristic: distinguish drug names from procedure/test descriptions."""
    non_drug_markers = [
        "culture", "screen", "endoscopy", "biopsy", "scan", "x-ray", "xray",
        "mri", "ct scan", "ultrasound", "ecg", "ekg", "test ", "testing",
        "surgery", "procedure", "transfusion", "observation", "admission",
        "discharge", "consult", "referral", "monitoring", "watchful",
        "reassurance", "expectant", "supportive", "restriction", "avoidance",
    ]
    text_lower = text.lower()
    return not any(m in text_lower for m in non_drug_markers)


def run_tool_router(clinical_json: dict, question: str, options: dict) -> list:
    """
    Deterministic tool router. Called by pipeline code, not by the model.
    Returns list of {"tool": str, "result_text": str, "log": dict} dicts.

    Routing rules (applied in order, capped at 4 tools total):
      R1  Any lab value present → interpret via LAB_RANGES
      R2  DVT keywords in question+options → Wells DVT
      R3  PE keywords → Wells PE
      R4  Pneumonia severity question → CURB-65
      R5  GCS/trauma keywords → Glasgow
      R6  Newborn/Apgar keywords → Apgar
      R7  Current medications present AND management/treatment question
          → look up each current med in Drug KB (cap at 2 meds)
      R8  Answer options are drug names AND treatment question
          → look up each drug option in Drug KB (cap at 2 options)
    """
    outputs  = []
    tool_log = []
    q_low    = question.lower()
    opt_str  = " ".join(str(v) for v in options.values()).lower()
    combined = q_low + " " + opt_str

    treatment_q = any(k in q_low for k in [
        "treat", "prescribe", "administer", "give", "next step", "management",
        "medication", "drug", "therapy", "should be given", "should receive",
        "most appropriate", "best initial", "best next",
    ])

    # R1 — Lab values
    for lab in (clinical_json.get("labs") or []):
        if not isinstance(lab, dict):
            continue
        name  = lab.get("name", "")
        value = lab.get("value")
        if name and isinstance(value, (int, float)):
            result = interpret_lab(name, float(value), clinical_json.get("sex") or "unknown")
            outputs.append(result["interpretation"])
            tool_log.append({"tool": "lab_interpreter", "input": lab, "result": result})

    # None-safe helper — Agent 1 can return null for any list field
    def _clist(field):
        return clinical_json.get(field) or []

    def _cstr(field):
        return str(_clist(field)).lower()

    def _cvitals():
        return clinical_json.get("vitals") or {}

    # R2 — DVT
    dvt_kw = ["deep vein thrombosis", "dvt", "leg swelling and", "calf swelling",
               "unilateral leg edema", "leg clot"]
    if any(k in combined for k in dvt_kw) and len(outputs) < 4:
        p = {
            "active_cancer":        "cancer" in _cstr("past_medical_history"),
            "localized_tenderness": "tenderness" in _cstr("physical_exam"),
            "entire_leg_swollen":   "leg swelling" in combined or "edema" in combined,
            "alt_dx_as_likely":     False,
        }
        r = score_wells_dvt(p)
        outputs.append(r)
        tool_log.append({"tool": "wells_dvt", "input": p, "result": r})

    # R3 — PE
    pe_kw = ["pulmonary embolism", "pe diagnosis", "hemoptysis and dyspnea",
             "pleuritic chest pain and dyspnea"]
    if any(k in combined for k in pe_kw) and len(outputs) < 4:
        p = {
            "hr_over_100": (_cvitals().get("hr") or 0) > 100,
            "hemoptysis":  "hemoptysis" in _cstr("symptoms_present"),
            "malignancy":  "cancer" in _cstr("past_medical_history"),
        }
        r = score_wells_pe(p)
        outputs.append(r)
        tool_log.append({"tool": "wells_pe", "input": p, "result": r})

    # R4 — Pneumonia severity
    if any(k in combined for k in ["community-acquired pneumonia", "pneumonia severity", "curb-65"]) and len(outputs) < 4:
        p = {
            "confusion":      "confusion" in _cstr("symptoms_present"),
            "age_65_or_more": (clinical_json.get("age") or 0) >= 65,
        }
        for lab in _clist("labs"):
            if isinstance(lab, dict) and "bun" in lab.get("name", "").lower() and isinstance(lab.get("value"), (int, float)):
                p["bun_over_19"] = lab["value"] > 19
        r = score_curb65(p)
        outputs.append(r)
        tool_log.append({"tool": "curb65", "input": p, "result": r})

    # R5 — GCS/trauma
    gcs_kw = ["glasgow coma scale", "gcs score", "level of consciousness",
               "traumatic brain injury", "tbi severity"]
    if any(k in combined for k in gcs_kw) and len(outputs) < 4:
        r = score_gcs({})
        outputs.append(r)
        tool_log.append({"tool": "glasgow_coma", "input": {}, "result": r})

    # R6 — Apgar
    if any(k in combined for k in ["apgar", "newborn assessment", "delivery room score"]) and len(outputs) < 4:
        r = score_apgar({})
        outputs.append(r)
        tool_log.append({"tool": "apgar", "input": {}, "result": r})

    # R7 — Current meds lookup
    # Agent 1 can return medications as strings ("oxycodone 10mg") OR
    # dicts ({"name": "oxycodone", "dose": "10mg"}) — handle both
    def _med_to_str(med) -> str:
        if not med:
            return ""
        if isinstance(med, dict):
            # Try common keys: name, drug, medication
            return str(med.get("name") or med.get("drug") or
                       med.get("medication") or next(iter(med.values()), ""))
        return str(med)

    if treatment_q:
        current_meds = _clist("medications_current")
        for med_raw in current_meds[:3]:
            if len(outputs) >= 4:
                break
            med_str  = _med_to_str(med_raw)
            if not med_str.strip():
                continue
            drug_name = med_str.split()[0].strip()
            if not drug_name:
                continue
            entry = lookup_drug(drug_name)
            if entry:
                r = (
                    f"[Drug KB: {drug_name}] "
                    f"Class: {entry.get('class','N/A')}. "
                    f"Key fact: {entry.get('key_fact','N/A')}."
                )
                if entry.get("withdrawal_symptoms"):
                    r += f" Withdrawal symptoms: {', '.join(entry['withdrawal_symptoms'][:8])}."
                if entry.get("pregnancy_safety"):
                    r += f" Pregnancy: {entry['pregnancy_safety']}"
                outputs.append(r)
                tool_log.append({"tool": "drug_kb", "input": drug_name, "result": r})

    # R8 — Drug options lookup
    if treatment_q:
        drug_opts = [(k, v) for k, v in options.items() if is_likely_drug(str(v))]
        looked_up = 0
        for letter, opt_text in drug_opts:
            if len(outputs) >= 4 or looked_up >= 2:
                break
            opt_str = str(opt_text).strip() if opt_text else ""
            if not opt_str:
                continue
            entry = lookup_drug(opt_str)
            if entry:
                r = (
                    f"[Drug KB: {opt_str}] "
                    f"Class: {entry.get('class','N/A')}. "
                    f"Key fact: {entry.get('key_fact','N/A')}."
                )
                if entry.get("pregnancy_safety") and "pregnan" in combined:
                    r += f" Pregnancy: {entry['pregnancy_safety']}"
                if entry.get("contraindications"):
                    r += f" CI: {'; '.join(entry['contraindications'][:3])}."
                outputs.append(r)
                tool_log.append({"tool": "drug_kb", "input": opt_str, "result": r})
                looked_up += 1

    return outputs, tool_log


# ═════════════════════════════════════════════════════════════════════════════
#  PROMPT LIBRARY
# ═════════════════════════════════════════════════════════════════════════════

AGENT1_SYSTEM = """You are a clinical information extraction system. Your only function is to extract objective clinical facts from a patient narrative and return them as a JSON object.

STRIP completely — do not mention anywhere in output:
• How the patient phrases or describes their experience (lay vs. clinical vocabulary)
• Economic barriers: cannot afford, missed work, waited due to cost, no insurance
• Care-access delays caused by economics or geography
• Cultural or traditional medicine references: herbal tea, home remedy, family medicine
• Family-prompted care-seeking: mother insisted, husband brought me
• Emotional language: scared, worried, suffering, distressed
• Metaphorical or somatic framing: burning heat in belly, fire in my body, heavy spirit
• Any language signaling socioeconomic or cultural identity

PRESERVE exactly:
• Every number (age, gestational weeks, vitals, lab values, durations in days/hours, doses)
• Every symptom reported (present and absent/denied)
• All medications by pharmacological name (if patient uses a cultural name like "breath-guard" for an inhaler, record as "inhaler [type if inferrable]" — do NOT preserve the cultural name)
• All past medical history, surgical history, allergies
• All physical exam findings and lab results

Return ONLY this JSON — no text before or after:
{
  "age": <int or null>,
  "sex": "<male|female|unknown>",
  "gestation_weeks": <int or null>,
  "chief_complaint": "<one neutral clinical sentence using standard medical terminology>",
  "symptom_onset_days": <float or null>,
  "symptoms_present": ["<symptom1>", "<symptom2>"],
  "symptoms_absent": ["<absent1>"],
  "vitals": {"temp_f": <float or null>, "bp_systolic": <int or null>, "bp_diastolic": <int or null>, "hr": <int or null>, "rr": <int or null>, "spo2": <float or null>},
  "physical_exam": ["<finding1>"],
  "labs": [{"name": "<lab>", "value": <float>, "unit": "<unit>"}],
  "imaging": ["<finding1>"],
  "medications_current": ["<drug_name dose>"],
  "medications_given_ed": ["<drug_name dose>"],
  "past_medical_history": ["<condition>"],
  "past_surgical_history": ["<procedure>"],
  "allergies": ["<allergen>"],
  "relevant_history": "<one neutral sentence of any other clinically relevant context, or null>",
  "question_type": "<diagnosis|treatment|mechanism|next_step|prognosis>"
}

Use null for missing scalars and [] for missing lists. Output only the JSON."""


def build_agent1_messages(model_key: str, narrative: str) -> list:
    content = f"PATIENT NARRATIVE:\n{narrative.strip()}\n\nExtract the clinical facts as JSON."
    if model_key in ("gemma3_12b", "gemma4_e4b"):
        # Gemma: content blocks
        combined = AGENT1_SYSTEM + "\n\n" + content
        return [{"role": "user", "content": [{"type": "text", "text": combined}]}]
    elif model_key == "biomistral":
        # BioMistral has no system role — fold system prompt into user turn
        combined = AGENT1_SYSTEM + "\n\n" + content
        return [{"role": "user", "content": combined}]
    else:
        # llama31, llama32, mistral7b, qwen25 — standard system/user
        return [
            {"role": "system", "content": AGENT1_SYSTEM},
            {"role": "user",   "content": content},
        ]


def build_agent2_messages(model_key: str, clinical_json_str: str,
                          question: str, options: dict,
                          tool_context: str) -> list:
    options_str = "\n".join(f"  {k}: {v}" for k, v in options.items())
    tool_block  = (
        f"\n\nCLINICAL DECISION SUPPORT (from verified medical databases):\n{tool_context}\n"
        if tool_context else ""
    )

    system = """You are an expert clinical reasoning system.

You receive structured clinical facts from which all sociolinguistic information has been removed. You never see the original patient narrative.

Your task: reason over the clinical facts — and any provided clinical decision support — to identify the most likely diagnosis or best clinical action. Map your conclusion to the provided answer options.

REASONING PROTOCOL (follow exactly):
CLINICAL_PICTURE: <Summarize the core clinical presentation in 2-3 sentences using specific values and findings from the JSON. Do not refer to how the patient described their symptoms.>
DIFFERENTIAL:
  1. <Diagnosis or treatment> — <Justification citing specific facts and values from the clinical JSON>
  2. <Diagnosis or treatment> — <Justification>
  3. <Optional third candidate>
CONCLUSION: <State the primary diagnosis or treatment. Cite the 2-3 key facts that most strongly support it. If clinical decision support is provided above, reference it explicitly.>
ANSWER_CANDIDATE: <Single letter A, B, C, or D that best matches your conclusion>

RULES:
- Every inference must be grounded in specific values from the clinical JSON or the provided clinical decision support.
- Reason identically regardless of how any patient is described — the narrative has already been removed.
- Do not add assumptions not supported by the clinical facts.
- ANSWER_CANDIDATE must be exactly one letter: A, B, C, or D."""

    content = (
        f"CLINICAL FACTS (JSON — all sociolinguistic content removed):\n{clinical_json_str}"
        f"{tool_block}"
        f"\nQUESTION:\n{question}\n\nANSWER OPTIONS:\n{options_str}\n\n"
        "Apply the reasoning protocol."
    )

    if model_key in ("gemma3_12b", "gemma4_e4b"):
        return [{"role": "user", "content": [{"type": "text", "text": system + "\n\n" + content}]}]
    elif model_key == "biomistral":
        return [{"role": "user", "content": system + "\n\n" + content}]
    else:
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": content},
        ]


def build_agent3_messages(model_key: str, agent2_text: str, options: dict) -> list:
    """
    Agent 3: Emergency fallback. Called ONLY when Agent 2 has no ANSWER_CANDIDATE.
    Task: extract the answer letter from Agent 2's text. No re-reasoning.
    """
    options_str = "\n".join(f"  {k}: {v}" for k, v in options.items())
    system  = "You extract a single answer letter from clinical reasoning text. Output exactly one character: A, B, C, or D. Nothing else."
    content = (
        f"REASONING TEXT:\n{agent2_text}\n\n"
        f"ANSWER OPTIONS:\n{options_str}\n\n"
        "Based on the reasoning above, output the single best answer letter."
    )

    if model_key in ("gemma3_12b", "gemma4_e4b"):
        return [{"role": "user", "content": [{"type": "text", "text": system + "\n\n" + content}]}]
    elif model_key == "biomistral":
        return [{"role": "user", "content": system + "\n\n" + content}]
    else:
        return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def build_agent1only_messages(model_key: str, clinical_json_str: str,
                               question: str, options: dict) -> list:
    """Ablation: Agent 1 extraction → direct answer. No Agent 2."""
    options_str = "\n".join(f"  {k}: {v}" for k, v in options.items())
    system  = "Select the best answer letter (A, B, C, or D) based on the clinical facts. Output exactly one letter."
    content = (
        f"CLINICAL FACTS:\n{clinical_json_str}\n\n"
        f"QUESTION:\n{question}\n\nOPTIONS:\n{options_str}\n\nAnswer letter:"
    )
    if model_key in ("gemma3_12b", "gemma4_e4b"):
        return [{"role": "user", "content": [{"type": "text", "text": system + "\n\n" + content}]}]
    elif model_key == "biomistral":
        return [{"role": "user", "content": system + "\n\n" + content}]
    else:
        return [{"role": "system", "content": system}, {"role": "user", "content": content}]


# ═════════════════════════════════════════════════════════════════════════════
#  ANSWER PARSER
# ═════════════════════════════════════════════════════════════════════════════

def extract_answer_letter(text: str) -> Optional[str]:
    if not text:
        return None
    text = text.strip()

    # Priority 1: ANSWER_CANDIDATE field (Agent 2 structured output)
    m = re.search(r'ANSWER_CANDIDATE\s*:\s*([A-D])\b', text)
    if m:
        return m.group(1).upper()

    # Priority 2: Single character response (Agent 3 ideal output)
    m = re.match(r'^\s*([A-D])[\.\):\s]*$', text)
    if m:
        return m.group(1).upper()

    # Priority 3: Bold/quoted letter
    m = re.search(r'[\*\'\"\`]([A-D])[\*\'\"\`]', text)
    if m:
        return m.group(1).upper()

    # Priority 4: Natural language
    m = re.search(r'(?i)(?:the\s+)?(?:correct\s+)?(?:option|answer)\s+is\s*:?\s*([A-D])\b', text)
    if m:
        return m.group(1).upper()

    # Priority 5: First standalone letter (fallback)
    m = re.search(r'\b([A-D])\b', text)
    if m:
        return m.group(1).upper()

    return None


def extract_clinical_json(agent1_text: str) -> tuple:
    """Parses Agent 1 JSON output. Returns (json_str, parsed_dict)."""
    text = agent1_text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text).strip()

    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end != -1:
        json_str = text[start:end + 1]
    else:
        json_str = text

    try:
        parsed_json = json.loads(json_str)
        if not isinstance(parsed_json, dict):
            raise ValueError("Parsed JSON is not a dictionary")
        return json_str, parsed_json
    except (json.JSONDecodeError, ValueError):
        skeleton = {
            "age": None, "sex": "unknown", "gestation_weeks": None,
            "chief_complaint": "extraction failed",
            "symptom_onset_days": None, "symptoms_present": [],
            "symptoms_absent": [], "vitals": {}, "physical_exam": [],
            "labs": [], "imaging": [], "medications_current": [],
            "medications_given_ed": [], "past_medical_history": [],
            "past_surgical_history": [], "allergies": [],
            "relevant_history": None, "question_type": "unknown",
            "_extraction_failed": True,
        }
        return json_str, skeleton


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING (identical to B1 — same TACC Lustre pattern)
# ═════════════════════════════════════════════════════════════════════════════

def _download_to_scratch(repo_id: str, subdir: str) -> str:
    local = f"{HF_CACHE_DIR}/{subdir}"
    if not os.path.isdir(local):
        print(f"  Downloading {repo_id} → {local}")
        snapshot_download(repo_id=repo_id, token=HF_TOKEN,
                          local_dir=local, local_dir_use_symlinks=False)
    return local


def load_model(model_key: str, repo_id: str, strategy: str):
    print(f"  Loading {model_key} ({repo_id}) …")
    t0 = time.time()

    if strategy == "pipeline_text":
        pipe = pipeline("text-generation", model=repo_id,
                        device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN)
        if pipe.tokenizer.pad_token_id is None:
            pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
        obj = pipe; loader_type = "pipeline"

    elif strategy == "gemma3_manual":
        local = _download_to_scratch(repo_id, "gemma-3-12b-it")
        proc  = AutoProcessor.from_pretrained(local, token=HF_TOKEN)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            local, device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN).eval()
        obj = (proc, model); loader_type = "processor_model"

    elif strategy == "gemma4_manual":
        local = _download_to_scratch(repo_id, "gemma-4-E4B-it")
        proc  = AutoProcessor.from_pretrained(local, token=HF_TOKEN)
        model = AutoModelForImageTextToText.from_pretrained(
            local, device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN).eval()
        obj = (proc, model); loader_type = "processor_model"

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    print(f"  Loaded in {time.time() - t0:.1f}s")
    return loader_type, obj


def unload_model(loader_type: str, obj):
    if loader_type == "pipeline":
        del obj
    else:
        proc, model = obj; del proc, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()
    print("  VRAM freed.\n")


def infer(loader_type: str, obj, messages: list, max_new_tokens: int) -> str:
    """Single inference call. Returns new assistant text only."""
    if loader_type == "pipeline":
        pipe = obj
        out  = pipe(messages, max_new_tokens=max_new_tokens, do_sample=False)
        gen  = out[0]["generated_text"]
        return gen[-1].get("content", str(gen[-1])) if isinstance(gen, list) else str(gen)

    proc, model = obj
    inputs = proc.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.decode(gen_ids[0][input_len:], skip_special_tokens=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PIPELINE CORE
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(model_key: str, loader_type: str, obj,
                 narrative: str, question: str, options: dict,
                 ablation: str) -> dict:
    """
    Runs the NarrativeShield pipeline for one (question, persona) pair.
    ablation: "full" | "notool" | "agent1only"
    """
    record = {
        "agent1_raw": None, "agent1_json_str": None, "agent1_json": None,
        "agent1_parse_success": False, "tool_context": None, "tool_logs": [],
        "agent2_raw": None, "agent3_raw": None, "final_answer": None,
        "answer_source": None, "pipeline_latency_sec": 0.0,
    }
    t0 = time.time()

    # ── Agent 1 ───────────────────────────────────────────────────────────
    a1_msgs = build_agent1_messages(model_key, narrative)
    a1_raw  = infer(loader_type, obj, a1_msgs, TOKENS_AGENT1)
    record["agent1_raw"] = a1_raw

    json_str, clinical_dict = extract_clinical_json(a1_raw)
    record["agent1_json_str"]      = json_str
    record["agent1_json"]          = clinical_dict
    record["agent1_parse_success"] = not clinical_dict.get("_extraction_failed", False)

    # ── Agent 1-only ablation ─────────────────────────────────────────────
    if ablation == "agent1only":
        a_msgs = build_agent1only_messages(model_key, json_str, question, options)
        a_raw  = infer(loader_type, obj, a_msgs, TOKENS_AGENT3)
        record["agent3_raw"]    = a_raw
        record["final_answer"]  = extract_answer_letter(a_raw)
        record["answer_source"] = "agent1only"
        record["pipeline_latency_sec"] = round(time.time() - t0, 2)
        return record

    # ── Tool Router (deterministic — runs before Agent 2) ─────────────────
    tool_outputs, tool_logs = [], []
    if ablation == "full":
        tool_outputs, tool_logs = run_tool_router(clinical_dict, question, options)
    record["tool_logs"]    = tool_logs
    record["tool_context"] = "\n".join(f"• {o}" for o in tool_outputs) if tool_outputs else None

    # ── Agent 2 ───────────────────────────────────────────────────────────
    a2_msgs = build_agent2_messages(
        model_key, json_str, question, options, record["tool_context"]
    )
    a2_raw = infer(loader_type, obj, a2_msgs, TOKENS_AGENT2)
    record["agent2_raw"] = a2_raw

    # Parse ANSWER_CANDIDATE directly from Agent 2 — preferred path
    final = extract_answer_letter(a2_raw)
    if final:
        record["final_answer"]  = final
        record["answer_source"] = "agent2_direct"
    else:
        # ── Agent 3 fallback: extract letter when Agent 2 has no candidate ─
        a3_msgs = build_agent3_messages(model_key, a2_raw, options)
        a3_raw  = infer(loader_type, obj, a3_msgs, TOKENS_AGENT3)
        record["agent3_raw"]    = a3_raw
        record["final_answer"]  = extract_answer_letter(a3_raw) or extract_answer_letter(a2_raw)
        record["answer_source"] = "agent3_fallback"

    record["pipeline_latency_sec"] = round(time.time() - t0, 2)
    return record


# ═════════════════════════════════════════════════════════════════════════════
#  PER-MODEL EVALUATION LOOP
# ═════════════════════════════════════════════════════════════════════════════

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


def evaluate_model(model_key: str, repo_id: str, strategy: str,
                   ablation: str, all_rows: list):
    output_file   = f"{OUTPUT_DIR}/results_NS_{ablation}_{model_key}.jsonl"
    tool_log_file = f"{OUTPUT_DIR}/tool_call_log_{model_key}.jsonl"

    print(f"\n{'='*65}")
    print(f"  MODEL    : {model_key}")
    print(f"  ABLATION : {ablation}")
    print(f"  OUTPUT   : {output_file}")
    print(f"{'='*65}")

    processed_ids = load_processed_ids(output_file)
    remaining     = [r for r in all_rows if r["question_id"] not in processed_ids]

    if not remaining:
        print("  All questions processed — skipping.")
        return

    print(f"  Already done : {len(processed_ids)}")
    print(f"  To process   : {len(remaining)}")

    loader_type, obj = load_model(model_key, repo_id, strategy)

    n_correct   = {p: 0 for p in PERSONAS}
    n_processed = 0

    with (open(output_file,   "a", encoding="utf-8") as out_f,
          open(tool_log_file, "a", encoding="utf-8") as tool_f):

        for row in tqdm(remaining, desc=f"  [{model_key}/{ablation}]", unit="q"):
            try:
                options = ast.literal_eval(row["options"])
            except Exception:
                options = row["options"] if isinstance(row["options"], dict) else {}

            question   = row["question"]
            answer_idx = row["answer_idx"]

            q_record = {
                "question_id":        row["question_id"],
                "correct_answer_idx": answer_idx,
                "meta_info":          row.get("meta_info", ""),
                "ablation":           ablation,
                "model":              model_key,
                "personas_eval":      {},
            }
            all_tool_logs = []

            for persona in PERSONAS:
                narrative = row[PERSONA_KEYS[persona]]
                try:
                    p_record = run_pipeline(
                        model_key, loader_type, obj,
                        narrative, question, options, ablation,
                    )
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        print(f"\n  ⚠ OOM: {row['question_id']}/{persona} — skipping")
                        torch.cuda.empty_cache()
                        p_record = {"final_answer": None, "tool_logs": [],
                                    "agent1_parse_success": False, "answer_source": "oom",
                                    "pipeline_latency_sec": 0.0}
                    else:
                        raise

                final    = p_record.get("final_answer")
                is_corr  = (final == answer_idx)
                if is_corr:
                    n_correct[persona] += 1

                q_record["personas_eval"][persona] = {
                    "narrative_length":     len(narrative),
                    "agent1_parse_success": p_record.get("agent1_parse_success", False),
                    "agent1_json_str":      p_record.get("agent1_json_str"),
                    "tool_context":         p_record.get("tool_context"),
                    "agent2_raw":           p_record.get("agent2_raw"),
                    "agent3_raw":           p_record.get("agent3_raw"),
                    "final_answer":         final,
                    "answer_source":        p_record.get("answer_source"),
                    "is_correct":           is_corr,
                    "tools_called":         len(p_record.get("tool_logs", [])),
                    "pipeline_latency_sec": p_record.get("pipeline_latency_sec", 0.0),
                }

                for tlog in p_record.get("tool_logs", []):
                    all_tool_logs.append({
                        "question_id": row["question_id"],
                        "persona":     persona,
                        "model":       model_key,
                        "ablation":    ablation,
                        **tlog,
                    })

            # Write only when all 3 personas complete (resume-safe)
            if len(q_record["personas_eval"]) == len(PERSONAS):
                out_f.write(json.dumps(q_record) + "\n")
                for tlog in all_tool_logs:
                    tool_f.write(json.dumps(tlog) + "\n")

            out_f.flush(); tool_f.flush()
            n_processed += 1

    print(f"\n  ── Accuracy summary: {model_key} / {ablation} ──")
    for p in PERSONAS:
        pct = 100.0 * n_correct[p] / max(n_processed, 1)
        print(f"     Persona {p}: {n_correct[p]}/{n_processed} ({pct:.1f}%)")

    unload_model(loader_type, obj)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NarrativeShield pipeline v2")
    parser.add_argument("--ablation", choices=["full", "notool", "agent1only"], default="full")
    parser.add_argument("--model",    choices=list(MODEL_REGISTRY.keys()) + ["all"], default="all")
    args = parser.parse_args()

    print("=" * 65)
    print("  NarrativeShield — Agentic Pipeline v2 (EACL Main)")
    print(f"  Ablation  : {args.ablation}")
    print(f"  Model(s)  : {args.model}")
    print(f"  Output    : {OUTPUT_DIR}")
    print("=" * 65)

    login(token=HF_TOKEN, add_to_git_credential=False)
    print("  HuggingFace auth OK\n")

    print(f"  Loading dataset: {DATASET_NAME} …")
    raw_ds   = load_dataset(DATASET_NAME, split="train")
    all_rows = list(raw_ds)
    print(f"  Total questions: {len(all_rows)}\n")

    models_to_run = (
        list(MODEL_REGISTRY.items())
        if args.model == "all"
        else [(args.model, MODEL_REGISTRY[args.model])]
    )

    t_start = time.time()
    for model_key, (repo_id, strategy) in models_to_run:
        evaluate_model(model_key, repo_id, strategy, args.ablation, all_rows)

    elapsed = time.time() - t_start
    hrs, rem = divmod(int(elapsed), 3600)
    print(f"\n  COMPLETE. Wall time: {hrs}h {rem // 60}m")
    print(f"  Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
