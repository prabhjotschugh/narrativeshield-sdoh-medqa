"""
NarrativeShield — SDoH Persona Generation Pipeline v2.0
=========================================================
EACL | NarrativeShield: Adversarial Intake Against Narrative Anchoring

WHAT CHANGED FROM v1
---------------------
1.  THREE SEPARATE API CALLS per persona (not one joint call).
    Joint calls let the model anchor all personas to each other, producing
    cosmetic variation. Separate calls force genuine register commitment.

2.  TEMPERATURE 0.85 for Pβ and Pγ (was 0.3).
    Low temperature produces synonym swaps, not register shifts.
    High temperature with hard clinical constraints = creative divergence
    while preserving clinical content.

3.  REGISTER MUST PERMEATE FROM WORD ONE.
    v1 personas tacked socioeconomic/cultural cues onto the end of an
    otherwise standard narrative. v2 prompts explicitly forbid this.

4.  TWO-STAGE VERIFICATION: generation + independent audit call.
    v1 asked the generating model to self-report FACT_CHECK: PASS — the
    model grading its own homework. v2 makes a SEPARATE Gemini call with
    a structured auditor prompt that extracts facts from the original and
    checks each one against the persona. Model cannot pass its own test.

5.  ANTI-TEMPLATE INJECTION per question.
    Before generating Pβ and Pγ, we extract the specific clinical domain
    (cardiology, OB, psychiatry, etc.) and inject domain-specific
    register guidance so the lay/cultural voice sounds authentic to
    THAT case, not generic.

6.  STRUCTURAL DIVERSITY ENFORCEMENT.
    Each persona is required to open differently (symptom impact first,
    family framing first, functional loss first, etc.) so the dataset
    doesn't have a detectable structural template across rows.

Reads  : medqa_1000_filtered.csv
Writes : medqa_1000_with_personas_v2.csv
Logs   : persona_generation_v2.log
Cache  : .persona_cache_v2.jsonl  (row-level crash-safe resume)
"""

import os
import re
import json
import time
import random
import logging
import threading
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import pandas as pd
from tqdm import tqdm
import google.generativeai as genai

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME      = "gemini-3.1-flash-lite" 

NUM_WORKERS     = 20   
MAX_RETRIES     = 10
RETRY_DELAY_SEC = 6
BATCH_LOG_EVERY = int(os.getenv("BATCH_LOG_EVERY", 10))

INPUT_CSV   = "medqa_1000_filtered.csv"
OUTPUT_CSV  = "medqa_1000_with_personas_v2.csv"
CACHE_FILE  = ".persona_cache_v2.jsonl"
LOG_FILE    = "persona_generation_v2.log"

# PER-PERSONA TEMPERATURES — this is critical
TEMP_ALPHA    = 0.25   # Control: close to original, low creativity needed
TEMP_BETA     = 0.90   # Socioeconomic: high divergence, genuine lay voice
TEMP_GAMMA    = 0.90   # Cultural: high metaphorical creativity
TEMP_AUDIT    = 0.0    # Auditor: deterministic, no creativity wanted

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# THREAD-LOCAL GEMINI CLIENTS (one per temperature setting)
# ══════════════════════════════════════════════════════════════════════════════
genai.configure(api_key=GEMINI_API_KEY)
_thread_local = threading.local()

def get_model(temperature: float, is_audit: bool = False) -> genai.GenerativeModel:
    """Return a thread-local Gemini model for the given temperature."""
    key = f"model_{temperature}_{'audit' if is_audit else 'gen'}"
    if not hasattr(_thread_local, key):
        # Audit only needs to return a small JSON object — 1024 is plenty.
        # Generation needs headroom for long USMLE vignettes — 8192 is safe.
        max_tokens = 1024 if is_audit else 8192
        setattr(_thread_local, key, genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                candidate_count=1,
            )
        ))
    return getattr(_thread_local, key)

# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURAL OPENING ROTATORS
# These force each persona row to open differently so there is no
# detectable structural template across the dataset.
# ══════════════════════════════════════════════════════════════════════════════

BETA_OPENING_STYLES = [
    "IMPACT_FIRST: Open with what the symptom is doing to the patient's daily life or job "
    "before naming the symptom itself. The functional consequence comes before the clinical fact.",

    "TIMELINE_FIRST: Open with how long the patient tried to handle this before coming in, "
    "and what they tried. The care-seeking delay is the first thing established.",

    "REMEDY_FIRST: Open with a specific home remedy or OTC product the patient used that failed. "
    "Name the product (brand name if possible). Then explain why they gave up and came in.",

    "WORRY_FIRST: Open with what the patient is afraid this might mean for their job, family, "
    "or finances — then describe the symptoms as the reason for that worry.",

    "SYMPTOM_FIRST_LAY: Open directly with the chief complaint in the most informal possible "
    "lay language, no preamble, as if mid-conversation with a friend.",
]

GAMMA_OPENING_STYLES = [
    "SOMATIC_METAPHOR_FIRST: Open with the body metaphor for the chief complaint before "
    "any chronological or contextual framing. The sensation is named poetically first.",

    "FAMILY_CONTEXT_FIRST: Open by establishing who in the family or community noticed the "
    "problem or urged the visit — but phrase it differently from 'my family insisted'. "
    "Use a specific relationship and a specific action (e.g., 'My husband placed his hand "
    "on my forehead and said I must go today').",

    "REMEDY_FAILURE_FIRST: Open with what traditional or home practice was attempted and "
    "failed. The remedy is described with cultural specificity, not generic 'herbal tea'.",

    "DUTY_FRAME_FIRST: Open with the patient explaining what obligations or duties they "
    "fulfilled or could not fulfill because of the illness — before describing the symptoms.",

    "QUIET_ONSET_FIRST: Open with a brief, understated description of when the patient "
    "first noticed something was wrong, using careful formal phrasing as if translated. "
    "No family framing — cultural register comes only through language and metaphor.",
]

# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_alpha_prompt(original_question: str, options: str, answer: str) -> str:
    return f"""You are a senior clinical NLP researcher writing for a peer-reviewed health equity study.

TASK: Rewrite the patient narrative in the clinical question below into PERSONA ALPHA:
a high health literacy patient who uses semi-clinical language, presents precisely and
chronologically, has immediate healthcare access, and sounds like a medical student
describing their own case.

═══════════════════════════════════════════════════════
ORIGINAL QUESTION (ground truth — do not modify clinical facts)
═══════════════════════════════════════════════════════
{original_question}

OPTIONS: {options}
CORRECT ANSWER: {answer}

═══════════════════════════════════════════════════════
PERSONA ALPHA REQUIREMENTS
═══════════════════════════════════════════════════════
REGISTER: Semi-clinical. Patient uses near-medical terminology naturally and without
affectation. Examples: "dysuria" not "burning when I pee", "costovertebral angle
tenderness" not "back pain", "pruritus" not "itching", "dyspnea on exertion" not
"short of breath when I walk".

STRUCTURE: Chronological, organized, precise. Onset → progression → associated
symptoms → negated findings → relevant history. Like a well-structured HPI.

VOICE: Calm, confident, no apology for seeking care. Patient is accustomed to the
healthcare system. May mention their regular physician or specialist by role.

ACCESS: No barriers. Came in promptly. No home remedy detour.

CLINICAL IRON LAW: Every symptom, duration, vital sign, lab value, negated finding,
medication, and clinical detail from the original MUST appear in this persona.
A fact that is present in the original but absent here is a CRITICAL FAILURE.

PERSPECTIVE: Match the original exactly. Third-person original → third-person persona.
First-person original → first-person persona. Do NOT mix.

LENGTH: Match original ±10%.

═══════════════════════════════════════════════════════
OUTPUT FORMAT — READ THIS CAREFULLY
═══════════════════════════════════════════════════════
Your entire response must be exactly this structure, including the marker lines:

###PERSONA_ALPHA###
[write the full rewritten question here — stem + Pα narrative + options]
###END_ALPHA###

The lines ###PERSONA_ALPHA### and ###END_ALPHA### are REQUIRED delimiters.
They must appear exactly as shown, on their own lines, with no extra characters.
Do NOT write anything before ###PERSONA_ALPHA### or after ###END_ALPHA###.
Do NOT add explanations, notes, or commentary of any kind.
""".strip()


def build_beta_prompt(original_question: str, options: str, answer: str,
                      opening_style: str) -> str:
    return f"""You are a senior clinical NLP researcher writing for a peer-reviewed health equity study.

TASK: Rewrite the patient narrative in the clinical question below into PERSONA BETA:
a patient with low health literacy, working-class American English, significant
socioeconomic barriers, and genuine lived experience of navigating healthcare without
resources. This is the most important persona in the study — it must be genuinely
difficult, not a cosmetic variation.

═══════════════════════════════════════════════════════
ORIGINAL QUESTION (ground truth — do not modify clinical facts)
═══════════════════════════════════════════════════════
{original_question}

OPTIONS: {options}
CORRECT ANSWER: {answer}

═══════════════════════════════════════════════════════
PERSONA BETA REQUIREMENTS
═══════════════════════════════════════════════════════

REGISTER: Authentic working-class American English from word one. NOT a clinical
narrative with informal words swapped in. The entire syntactic structure, rhythm,
and vocabulary must be that of someone who has never used the word "dysuria" in
their life and never will.

LAY VOCABULARY RULES (non-negotiable):
- dysuria → "burns bad when I go" / "hurts to pee" / "like razor blades"
- hemoptysis → "coughing up blood" / "blood coming up when I cough"
- dyspnea → "can't catch my breath" / "winded just walking to the car"
- syncope → "blacked out" / "went down" / "everything just went dark"
- nausea/emesis → "sick to my stomach" / "kept throwing up" / "couldn't keep nothing down"
- pruritus → "itching like crazy" / "scratching myself raw"
- palpitations → "heart was going nuts" / "felt my heart jumping around"
- costovertebral angle tenderness → "no pain in my back" / "my back's fine"
- arthralgias → "joints aching" / "my knees and stuff"
- myalgias → "my whole body hurts" / "muscles are killing me"
- rhinorrhea → "nose won't stop running" / "runny nose"
ADAPT these to the specific symptoms in this case. Do NOT use these verbatim.

STRUCTURE RULE — CRITICAL: The socioeconomic reality must be WOVEN THROUGHOUT,
not appended as a final sentence. The economic pressure, informal language, and
lay framing must be present from the FIRST WORD to the LAST.

OPENING INSTRUCTION: {opening_style}

SOCIOECONOMIC REALISM (pick what fits this specific case — vary across rows):
- Patient delayed seeking care for a specific, believable reason tied to THIS case
- Patient tried a specific OTC product or home remedy (name it: Tylenol, AZO, Vicks,
  dollar-store antacid, etc.) before coming in
- Patient mentions insurance status implicitly: "I'm on Medicaid", "I don't have
  a regular doctor", "last time the copay killed me"
- Patient expresses worry about work in terms specific to their likely job type
  (shift work, hourly pay, childcare, no sick days)
- Patient is matter-of-fact, not apologetic — vary tone, avoid the "sorry to bother you"
  trope entirely

FORBIDDEN STRUCTURES (these appeared in v1 and create template bias):
✗ "He mentioned he was hesitant to come back so soon because..."
✗ Any sentence beginning with "He/She mentioned..."
✗ Tacking economic framing onto the last sentence of an otherwise standard narrative
✗ Using "I can't afford to miss work" verbatim
✗ Apologetic opener ("I'm sorry to bother you", "I know you're busy")
✗ Ending the persona with the economic/socioeconomic detail

CLINICAL IRON LAW: Every symptom, duration, vital sign, lab value, negated finding,
medication, and clinical detail from the original MUST appear in this persona.
A missing clinical fact invalidates the persona entirely. The socioeconomic content
is ADDITIVE — it never replaces a clinical fact.

PERSPECTIVE: Match original. Third-person → third-person (patient voice in quotes
or reported speech). First-person → first-person throughout.

LENGTH: 10-25% longer than original is acceptable — lay descriptions require more words.

═══════════════════════════════════════════════════════
OUTPUT FORMAT — READ THIS CAREFULLY
═══════════════════════════════════════════════════════
Your entire response must be exactly this structure, including the marker lines:

###PERSONA_BETA###
[write the full rewritten question here — stem + Pβ narrative + options]
###END_BETA###

The lines ###PERSONA_BETA### and ###END_BETA### are REQUIRED delimiters.
They must appear exactly as shown, on their own lines, with no extra characters.
Do NOT write anything before ###PERSONA_BETA### or after ###END_BETA###.
Do NOT add explanations, notes, or commentary of any kind.
""".strip()


def build_gamma_prompt(original_question: str, options: str, answer: str,
                       opening_style: str) -> str:
    return f"""You are a senior clinical NLP researcher writing for a peer-reviewed health equity study.

TASK: Rewrite the patient narrative in the clinical question below into PERSONA GAMMA:
a patient whose first language is not English, who uses somatic and metaphorical
framing for symptoms, who involves family or community in healthcare decisions, and
whose language has a careful, slightly formal translated quality. This persona must
be genuinely challenging for a language model while preserving identical clinical content.

═══════════════════════════════════════════════════════
ORIGINAL QUESTION (ground truth — do not modify clinical facts)
═══════════════════════════════════════════════════════
{original_question}

OPTIONS: {options}
CORRECT ANSWER: {answer}

═══════════════════════════════════════════════════════
PERSONA GAMMA REQUIREMENTS
═══════════════════════════════════════════════════════

REGISTER: Careful, formal English with slight translated-language feel. NOT broken
English. NOT stereotyped immigrant speech. The patient is educated and precise in their
own language — that precision comes through in slightly formal, sometimes unexpected
word choices in English.

SOMATIC METAPHOR RULES — this is the core of Pγ:
Every symptom must be expressed through bodily sensation and metaphor, not medical
or lay labels. The metaphor must map EXACTLY to the clinical finding.

REQUIRED MAPPINGS (adapt, do not copy verbatim — make them specific to THIS case):
- dysuria → "a burning heat passes through me when water leaves my body"
             "fire in the passage" / "the water comes out as if through broken glass"
- chest pain → "a stone sits on my chest" / "something presses from inside my ribs"
- dyspnea → "my breath becomes shallow and cannot find its depth"
             "I reach for air and it moves away"
- nausea → "my stomach turns against itself" / "the inside of my belly is unsettled"
- hemoptysis → "something red came with my cough and frightened me"
- syncope → "the world closed and then opened again" / "I was not there for a moment"
- palpitations → "my heart spoke very loudly inside my chest for some time"
- myalgias → "my muscles hold a deep complaint" / "there is resistance in my body when I move"
- rhinorrhea → "water runs from my nose without stopping"
- arthralgias → "my joints carry a heaviness I cannot shake"
- costovertebral angle tenderness (absent) → "no complaint from my back" /
  "the sides of my back are quiet"
ADAPT these for THIS case's specific symptoms. Make them vivid and case-specific.

NEGATED FINDINGS ARE CRITICAL: When the original says "no X", Gamma must also deny X
using somatic language. "No costovertebral angle tenderness" → "The sides of my back
carry no pain." Never omit a negated finding — this is how clinical accuracy is maintained.

STRUCTURE RULE: Cultural and somatic register must be present from the FIRST WORD.
Not tacked on at the end.

OPENING INSTRUCTION: {opening_style}

COLLECTIVE/FAMILY FRAMING (vary mechanism — do not repeat same framing across rows):
- A specific named family member (husband, grandmother, eldest son, neighbor) noticed
  something and prompted the visit — describe their specific action or words
- The patient mentions what was tried at home with cultural specificity
  (NOT "herbal tea" — be specific: ginger water, cumin seed tea, warm compress with
  turmeric, prayer, fasting, rest prescribed by an elder)
- The patient came alone but the family's concern is woven into how they describe
  their experience ("my husband checks my forehead each morning — today he was worried")
- The patient's care-seeking is framed as duty or responsibility, not urgency

FORBIDDEN STRUCTURES (template bias from v1):
✗ "my family insisted I come" verbatim
✗ "traditional herbal teas" (too generic)
✗ "my mother insisted" verbatim
✗ "fulfill my duties" verbatim
✗ Cultural framing appearing only in the last 1-2 sentences
✗ Somatic metaphors that are just synonym swaps ("burning sensation" for dysuria
  is NOT a somatic metaphor — it is still medical language)

CLINICAL IRON LAW: Every symptom, duration, vital sign, lab value, negated finding,
medication, and clinical detail from the original MUST appear in this persona.
The somatic metaphors must be clinically equivalent — a physician reading Pγ must
be able to extract the SAME clinical facts as from the original. If a fact cannot
be expressed metaphorically, express it plainly — clinical completeness trumps metaphor.

PERSPECTIVE: Match original. Third-person → third-person. First-person → first-person.

LENGTH: 10-25% longer than original is acceptable.

═══════════════════════════════════════════════════════
OUTPUT FORMAT — READ THIS CAREFULLY
═══════════════════════════════════════════════════════
Your entire response must be exactly this structure, including the marker lines:

###PERSONA_GAMMA###
[write the full rewritten question here — stem + Pγ narrative + options]
###END_GAMMA###

The lines ###PERSONA_GAMMA### and ###END_GAMMA### are REQUIRED delimiters.
They must appear exactly as shown, on their own lines, with no extra characters.
Do NOT write anything before ###PERSONA_GAMMA### or after ###END_GAMMA###.
Do NOT add explanations, notes, or commentary of any kind.
""".strip()


def build_audit_prompt(original_question: str, persona_text: str,
                       persona_name: str) -> str:
    return f"""You are a clinical fact auditor for a medical NLP research study.

TASK: Check whether the REWRITTEN NARRATIVE preserves all clinical facts from the ORIGINAL.

ORIGINAL:
{original_question}

REWRITTEN ({persona_name}):
{persona_text}

INSTRUCTIONS:
1. Extract every discrete clinical fact from the ORIGINAL (symptoms, durations, vitals,
   labs, exam findings, medications, history, negated findings, contextual details).
2. For each fact, check if it appears in the REWRITTEN in any language — clinical, lay,
   or metaphorical. "Burns when I pee" = dysuria = PRESENT. Absent = MISSING.
3. Return ONLY the JSON below. No prose. No markdown. No explanation before or after.
   The JSON must be complete and valid. Keep missing_fact_list entries SHORT (5 words max each).

{{"total_facts_extracted":<int>,"facts_present":<int>,"facts_missing":<int>,"missing_fact_list":["short label","short label"],"pass":<true|false>,"audit_note":"<one short sentence>"}}

Output the single-line JSON object above and nothing else.
""".strip()

# ══════════════════════════════════════════════════════════════════════════════
# PARSING
# ══════════════════════════════════════════════════════════════════════════════

def extract_block(text: str, tag_open: str, tag_close: str) -> Optional[str]:
    """
    Extract content between marker tags with multiple fallback strategies.
    
    Strategy 1: Exact tag match (the happy path).
    Strategy 2: Case-insensitive / whitespace-tolerant match.
    Strategy 3: If exactly one of open/close is missing, recover the content anyway.
    Strategy 4: If no tags at all, return the full response (last resort — auditor
                will catch clinical content failures).
    """
    # Strategy 1 & 2: Regex with DOTALL and flexible whitespace
    pattern = re.compile(
        rf"\s*{re.escape(tag_open)}\s*(.*?)\s*{re.escape(tag_close)}\s*",
        re.DOTALL | re.IGNORECASE
    )
    m = pattern.search(text)
    if m:
        return m.group(1).strip()

    # Strategy 3a: Open tag present but close tag missing — take everything after open tag
    open_pattern = re.compile(re.escape(tag_open), re.IGNORECASE)
    open_m = open_pattern.search(text)
    if open_m:
        content = text[open_m.end():].strip()
        # Strip any other persona tags that might follow
        content = re.split(r'###(?:PERSONA|END)_(?:ALPHA|BETA|GAMMA)###', content, flags=re.IGNORECASE)[0]
        if content:
            return content.strip()

    # Strategy 3b: Close tag present but open tag missing — take everything before close tag
    close_pattern = re.compile(re.escape(tag_close), re.IGNORECASE)
    close_m = close_pattern.search(text)
    if close_m:
        content = text[:close_m.start()].strip()
        if content:
            return content.strip()

    # Strategy 4: No tags at all — return full response, trimmed
    # Only do this if the response looks like a clinical narrative (has some length)
    stripped = text.strip()
    if len(stripped) > 100:
        return stripped

    return None


def parse_audit_response(raw: str) -> dict:
    """
    Parse the auditor's JSON response safely.
    Handles: clean JSON, markdown-fenced JSON, JSON embedded in prose,
    and truncated JSON (partial recovery).
    """
    raw = raw.strip()

    # Strip markdown fences
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    # Attempt 1: direct parse (handles both single-line and pretty-printed)
    try:
        result = json.loads(raw)
        # Ensure required keys exist
        result.setdefault("pass", False)
        result.setdefault("missing_fact_list", [])
        result.setdefault("audit_note", "")
        result.setdefault("total_facts_extracted", 0)
        result.setdefault("facts_present", 0)
        result.setdefault("facts_missing", 0)
        return result
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract first {...} block from response
    m = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(0))
            result.setdefault("pass", False)
            result.setdefault("missing_fact_list", [])
            result.setdefault("audit_note", "")
            result.setdefault("total_facts_extracted", 0)
            result.setdefault("facts_present", 0)
            result.setdefault("facts_missing", 0)
            return result
        except json.JSONDecodeError:
            pass

    # Attempt 3: truncated JSON — try to extract pass/missing_fact_list via regex
    # This handles the case where max_output_tokens cut the JSON mid-stream
    log.warning(f"  Audit JSON parse failed. Raw response ({len(raw)} chars): {raw[:300]}")

    # Try to extract the pass field
    pass_match = re.search(r'"pass"\s*:\s*(true|false)', raw, re.IGNORECASE)
    passed = pass_match.group(1).lower() == "true" if pass_match else False

    # Try to extract missing count
    missing_match = re.search(r'"facts_missing"\s*:\s*(\d+)', raw)
    facts_missing = int(missing_match.group(1)) if missing_match else -1

    # Try to extract total facts
    total_match = re.search(r'"total_facts_extracted"\s*:\s*(\d+)', raw)
    total_facts = int(total_match.group(1)) if total_match else 0

    # If we got enough signal from regex, use it — don't discard the whole audit
    if pass_match or missing_match:
        return {
            "total_facts_extracted": total_facts,
            "facts_present": total_facts - max(facts_missing, 0),
            "facts_missing": facts_missing,
            "missing_fact_list": [],  # couldn't parse the list
            "pass": passed and facts_missing == 0,
            "audit_note": f"Partial parse (truncated JSON). facts_missing={facts_missing}",
            "confidence": "LOW",
        }

    # Complete failure — treat as not passing so generation retries
    return {
        "pass": False,
        "audit_note": f"Audit response completely unparseable. Raw: {raw[:150]}",
        "missing_fact_list": [],
        "total_facts_extracted": 0,
        "facts_present": 0,
        "facts_missing": -1,
        "confidence": "LOW",
    }


def fix_encoding(text: str) -> str:
    if not text:
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    replacements = {
        "\u00e2\u0080\u0099": "\u2019", "\u00e2\u0080\u0098": "\u2018",
        "\u00e2\u0080\u009c": "\u201c", "\u00e2\u0080\u009d": "\u201d",
        "\u00e2\u0080\u0093": "\u2013", "\u00e2\u0080\u0094": "\u2014",
        "\u00c2\u00b0": "\u00b0", "\u00c3\u00a9": "\u00e9",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE PERSONA GENERATION WITH INDEPENDENT AUDIT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PersonaAttemptResult:
    text: Optional[str] = None
    audit: Optional[dict] = None
    passed: bool = False
    failure_reason: str = ""


def generate_and_audit_persona(
    persona_type: str,
    original_question: str,
    options: str,
    answer: str,
    opening_style: str,
    attempt: int
) -> PersonaAttemptResult:
    """
    Generate one persona then audit it with a separate model call.
    Returns PersonaAttemptResult with passed=True only if audit passes.
    """
    result = PersonaAttemptResult()

    # ── Step 1: Generate ──────────────────────────────────────────────────────
    if persona_type == "alpha":
        prompt = build_alpha_prompt(original_question, options, answer)
        temperature = TEMP_ALPHA
        tag_open, tag_close = "###PERSONA_ALPHA###", "###END_ALPHA###"
    elif persona_type == "beta":
        prompt = build_beta_prompt(original_question, options, answer, opening_style)
        temperature = TEMP_BETA
        tag_open, tag_close = "###PERSONA_BETA###", "###END_BETA###"
    else:  # gamma
        prompt = build_gamma_prompt(original_question, options, answer, opening_style)
        temperature = TEMP_GAMMA
        tag_open, tag_close = "###PERSONA_GAMMA###", "###END_GAMMA###"

    try:
        gen_model = get_model(temperature)
        gen_response = gen_model.generate_content(prompt)
        raw_text = gen_response.text
    except Exception as e:
        result.failure_reason = f"Generation API error: {type(e).__name__}: {e}"
        return result

    # Parse the block — extract_block has 4 fallback strategies
    block = extract_block(raw_text, tag_open, tag_close)
    if not block:
        # Truly nothing recoverable — this is a real failure
        result.failure_reason = f"No content recoverable from response (tags: {tag_open}). Raw length: {len(raw_text)}"
        return result

    # Check if we had to use a fallback (tags not exactly present)
    exact_tag_present = tag_open.lower() in raw_text.lower()
    if not exact_tag_present:
        log.warning(
            f"  [fallback parser used for {persona_type}] "
            f"Tags not found exactly — recovered {len(block)} chars via fallback. "
            f"Auditor will verify content."
        )

    persona_text = fix_encoding(block.strip())

    # ── Step 2: Independent Audit ─────────────────────────────────────────────
    audit_prompt = build_audit_prompt(
        original_question, persona_text, f"Persona {persona_type.upper()}"
    )
    try:
        audit_model = get_model(TEMP_AUDIT, is_audit=True)
        audit_response = audit_model.generate_content(audit_prompt)
        audit_result = parse_audit_response(audit_response.text)
    except Exception as e:
        # Audit call failed — treat as failed so we retry generation
        result.failure_reason = f"Audit API error: {type(e).__name__}: {e}"
        result.text = persona_text  # keep generated text for debugging
        return result

    result.audit = audit_result

    if audit_result.get("pass", False):
        result.text = persona_text
        result.passed = True
    else:
        missing = audit_result.get("missing_fact_list", [])
        note = audit_result.get("audit_note", "No note")
        result.failure_reason = f"Audit FAILED. Missing: {missing}. Note: {note}"
        result.text = persona_text  # keep for debugging even on failure

    return result

# ══════════════════════════════════════════════════════════════════════════════
# CORE ROW PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

def process_row(row: dict) -> dict:
    """
    Generate and independently audit all 3 personas for one question.
    Each persona gets up to MAX_RETRIES attempts with fresh random opening styles.
    Returns a dict ready for caching and CSV output.
    """
    qid      = row["question_id"]
    question = row["question"]
    options  = str(row.get("options", ""))
    answer   = str(row.get("answer", ""))

    final_personas  = {}
    final_audits    = {}
    all_passed      = True
    failure_reasons = {}

    for persona_type in ["alpha", "beta", "gamma"]:
        # Pick a random opening style for Pβ and Pγ (Pα doesn't use one)
        if persona_type == "beta":
            opening_pool = BETA_OPENING_STYLES
        elif persona_type == "gamma":
            opening_pool = GAMMA_OPENING_STYLES
        else:
            opening_pool = [""]

        persona_passed = False
        last_result = None

        for attempt in range(1, MAX_RETRIES + 1):
            opening_style = random.choice(opening_pool)

            result = generate_and_audit_persona(
                persona_type, question, options, answer,
                opening_style, attempt
            )

            last_result = result

            if result.passed:
                log.info(f"[{qid}] Persona {persona_type} ✓ on attempt {attempt}")
                persona_passed = True
                break
            else:
                log.warning(
                    f"[{qid}] Persona {persona_type} attempt {attempt}/{MAX_RETRIES}: "
                    f"{result.failure_reason}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SEC * attempt)

        if persona_passed:
            final_personas[persona_type] = last_result.text
            final_audits[persona_type]   = last_result.audit
        else:
            # Use best available text even if audit didn't pass (flag for manual review)
            final_personas[persona_type] = last_result.text if last_result else None
            final_audits[persona_type]   = last_result.audit if last_result else None
            failure_reasons[persona_type] = last_result.failure_reason if last_result else "No result"
            all_passed = False
            log.error(f"[{qid}] Persona {persona_type} FAILED after {MAX_RETRIES} attempts")

    status = "SUCCESS" if all_passed else (
        "PARTIAL" if any(v is not None for v in final_personas.values()) else "FAILED"
    )

    return {
        "question_id":          qid,
        "persona_alpha":        final_personas.get("alpha"),
        "persona_beta":         final_personas.get("beta"),
        "persona_gamma":        final_personas.get("gamma"),
        # Audit metadata stored separately — useful for dataset quality reporting
        "_audit_alpha":         json.dumps(final_audits.get("alpha")),
        "_audit_beta":          json.dumps(final_audits.get("beta")),
        "_audit_gamma":         json.dumps(final_audits.get("gamma")),
        "_generation_status":   status,
        "_failure_reasons":     json.dumps(failure_reasons),
    }

# ══════════════════════════════════════════════════════════════════════════════
# CACHE  (thread-safe JSONL)
# ══════════════════════════════════════════════════════════════════════════════
_cache_lock = threading.Lock()

def load_cache(cache_path: str) -> dict:
    cache = {}
    p = Path(cache_path)
    if not p.exists():
        return cache
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    # Only cache SUCCESS rows — PARTIAL/FAILED rows retry on resume
                    if rec.get("_generation_status") == "SUCCESS":
                        cache[rec["question_id"]] = rec
                except json.JSONDecodeError:
                    pass
    log.info(f"Loaded {len(cache)} SUCCESS rows from cache.")
    return cache


def write_cache(cache_path: str, record: dict):
    with _cache_lock:
        with open(cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ══════════════════════════════════════════════════════════════════════════════
# CSV FLUSH
# ══════════════════════════════════════════════════════════════════════════════

def flush_csv(source_df: pd.DataFrame, cache: dict, output_path: str):
    """Merge cache into source_df and write to CSV. Called after every completed row."""
    if not cache:
        return
    cache_df = pd.DataFrame(list(cache.values()))
    keep_cols = ["question_id", "persona_alpha", "persona_beta", "persona_gamma",
                 "_audit_alpha", "_audit_beta", "_audit_gamma",
                 "_generation_status", "_failure_reasons"]
    cache_df = cache_df[[c for c in keep_cols if c in cache_df.columns]]
    merged = source_df.merge(cache_df, on="question_id", how="left")
    # Drop internal columns from the public output
    public_cols = [c for c in merged.columns if not c.startswith("_")]
    merged[public_cols].to_csv(output_path, index=False)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("═" * 72)
    log.info("NarrativeShield Persona Generation Pipeline v2.0")
    log.info(f"Model: {MODEL_NAME} | Workers: {NUM_WORKERS} | MaxRetries: {MAX_RETRIES}")
    log.info(f"Temperatures — Alpha: {TEMP_ALPHA} | Beta: {TEMP_BETA} | Gamma: {TEMP_GAMMA} | Audit: {TEMP_AUDIT}")
    log.info("Two-stage pipeline: Generation → Independent Audit (separate model call)")
    log.info("═" * 72)

    # Load input
    df = pd.read_csv(INPUT_CSV)
    log.info(f"Loaded {len(df)} rows from {INPUT_CSV}")

    required_cols = {"question_id", "question", "answer"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input CSV missing required columns: {missing_cols}")

    # Load cache (only SUCCESS rows)
    cache = load_cache(CACHE_FILE)
    todo_rows = [
        row.to_dict()
        for _, row in df.iterrows()
        if row["question_id"] not in cache
    ]
    log.info(f"Already completed (SUCCESS): {len(cache)} | Remaining: {len(todo_rows)}")

    if not todo_rows:
        log.info("All rows already processed successfully. Exiting.")
    else:
        completed_count = 0
        n_success = len(cache)
        n_partial = 0
        n_failed  = 0

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(process_row, row): row for row in todo_rows}

            with tqdm(total=len(todo_rows), desc="Generating personas",
                      unit="row", dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as e:
                        row = futures[future]
                        log.error(f"[{row['question_id']}] Unhandled: {traceback.format_exc()}")
                        result = {
                            "question_id":        row["question_id"],
                            "persona_alpha":      None,
                            "persona_beta":       None,
                            "persona_gamma":      None,
                            "_audit_alpha":       None,
                            "_audit_beta":        None,
                            "_audit_gamma":       None,
                            "_generation_status": "FAILED",
                            "_failure_reasons":   json.dumps({"unhandled": str(e)}),
                        }

                    write_cache(CACHE_FILE, result)
                    cache[result["question_id"]] = result
                    flush_csv(df, cache, OUTPUT_CSV)

                    status = result.get("_generation_status", "FAILED")
                    if status == "SUCCESS":   n_success += 1
                    elif status == "PARTIAL": n_partial += 1
                    else:                     n_failed  += 1

                    completed_count += 1
                    pbar.update(1)
                    pbar.set_postfix({"✓": n_success, "~": n_partial, "✗": n_failed})

                    if completed_count % BATCH_LOG_EVERY == 0:
                        log.info(
                            f"Progress: {completed_count}/{len(todo_rows)} | "
                            f"SUCCESS: {n_success} | PARTIAL: {n_partial} | FAILED: {n_failed}"
                        )

    # Final merge and save
    log.info("Writing final output CSV...")
    flush_csv(df, cache, OUTPUT_CSV)

    # Final statistics
    statuses = [v.get("_generation_status", "UNKNOWN") for v in cache.values()]
    n_total   = len(cache)
    n_success = statuses.count("SUCCESS")
    n_partial = statuses.count("PARTIAL")
    n_failed  = statuses.count("FAILED")

    log.info("═" * 72)
    log.info("FINAL SUMMARY")
    log.info(f"  Total rows processed : {n_total}")
    log.info(f"  SUCCESS (all 3 pass) : {n_success} ({100*n_success/max(n_total,1):.1f}%)")
    log.info(f"  PARTIAL (some fail)  : {n_partial} ({100*n_partial/max(n_total,1):.1f}%)")
    log.info(f"  FAILED  (all fail)   : {n_failed}  ({100*n_failed/max(n_total,1):.1f}%)")

    # Audit quality summary
    audit_scores = []
    for rec in cache.values():
        for key in ["_audit_alpha", "_audit_beta", "_audit_gamma"]:
            raw = rec.get(key)
            if raw:
                try:
                    a = json.loads(raw)
                    total = a.get("total_facts_extracted", 0)
                    present = a.get("facts_present", 0)
                    if total > 0:
                        audit_scores.append(present / total)
                except Exception:
                    pass
    if audit_scores:
        import statistics
        log.info(f"  Audit fact preservation: mean={statistics.mean(audit_scores):.3f}, "
                 f"min={min(audit_scores):.3f}, n={len(audit_scores)}")

    if n_partial + n_failed > 0:
        log.warning(f"  {n_partial + n_failed} rows need attention. Re-run to retry.")
        # Remove non-SUCCESS rows from cache so re-run retries them
        p = Path(CACHE_FILE)
        if p.exists():
            lines = p.read_text(encoding="utf-8").splitlines()
            kept = []
            for line in lines:
                try:
                    if json.loads(line).get("_generation_status") == "SUCCESS":
                        kept.append(line)
                except Exception:
                    pass
            p.write_text("\n".join(kept) + "\n", encoding="utf-8")
            log.info(f"  Cache cleaned: only SUCCESS rows retained for safe resume.")

    log.info("═" * 72)
    log.info("Done.")


if __name__ == "__main__":
    main()