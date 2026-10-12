# retelling generation and fact audit

Retelling prompts (clinical alpha, socioeconomic beta, cultural gamma) and the independent fact-preservation auditor used during dataset construction.

The text below is the exact source of each prompt, extracted programmatically from the released code (f-string placeholders are filled at run time).


## `dataset/2. add_persona_code.py`

### `build_alpha_prompt`

```python
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
```

### `build_beta_prompt`

```python
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
```

### `build_gamma_prompt`

```python
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
```

### `build_audit_prompt`

```python
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
```
