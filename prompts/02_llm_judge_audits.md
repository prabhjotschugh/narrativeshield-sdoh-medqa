# llm judge audits

LLM-judge prompts: content added by retellings, fact recall of schema extractions, fact recall and invention in rewrites.

The text below is the exact source of each prompt, extracted programmatically from the released code (f-string placeholders are filled at run time).


## `codes/audit_added_facts.py`

### `build_prompt`

```python
def build_prompt(original: str, rewrite: str, name: str) -> str:
    return f"""You are a clinical auditor for a medical NLP study.

TASK: Find information in the REWRITTEN narrative that is NOT in the ORIGINAL case.

ORIGINAL:
{original}

REWRITTEN ({name}):
{rewrite}

INSTRUCTIONS:
1. List every discrete piece of information in the REWRITTEN that the ORIGINAL does not contain or imply
   (new symptoms, durations, exposures, behaviours, home remedies, medications, prior care, finances, family context, emotions).
   Ignore pure wording, tone and style changes.
2. For each, set "clinically_relevant" true ONLY if a clinician could use it to choose among diagnoses or management
   (new symptom, exposure, drug/remedy, timeline, risk factor, prior treatment). Social/financial/emotional framing that
   does not bear on diagnosis or management is false.
3. Return ONLY this single-line JSON, no markdown:
{{"added":[{{"item":"<5 words max>","clinically_relevant":<true|false>}}],"n_added":<int>,"n_clinically_relevant":<int>,"note":"<one short sentence>"}}
""".strip()
```


## `codes/audit_agent1_outputs.py`

### `build_audit_prompt`

```python
def build_audit_prompt(original_question: str, persona_text: str, persona_name: str) -> str:
    return f"""You are a clinical fact auditor for a medical NLP research study.

TASK: Check whether the REWRITTEN NARRATIVE preserves all clinical facts from the ORIGINAL.

ORIGINAL:
{original_question}

REWRITTEN ({persona_name}):
{persona_text}

INSTRUCTIONS:
1. Extract every discrete clinical fact from the ORIGINAL (symptoms, durations, vitals,
   labs, exam findings, medications, history, negated findings, contextual details).
2. For each fact, check if it appears in the REWRITTEN in any form (a structured JSON
   field value counts as present). Absent = MISSING.
3. Return ONLY the JSON below. No prose. No markdown. No explanation before or after.
   Keep missing_fact_list entries SHORT (5 words max each).

{{"total_facts_extracted":<int>,"facts_present":<int>,"facts_missing":<int>,"missing_fact_list":["short label","short label"],"pass":<true|false>,"audit_note":"<one short sentence>"}}

Output the single-line JSON object above and nothing else.
""".strip()
```


## `codes/audit_rewrites.py`

### `prompt`

```python
def prompt(narrative, rewrite):
    return f"""You are a clinical auditor. A patient's own account was rewritten by an AI as a clinical case vignette.
The rewrite was instructed to KEEP every clinical fact (age, sex, symptoms and denied symptoms, onset/duration/course, all numbers, history, medications, remedies or substances taken, exposures, exam/lab/imaging findings) and to DROP non-clinical content (finances, work or family pressure, who brought the patient, emotions, metaphor).

PATIENT NARRATIVE:
{narrative}

REWRITTEN VIGNETTE:
{rewrite}

1. List each CLINICAL fact in the narrative that is missing from or changed in the rewrite. Do not count dropped non-clinical content.
2. List each item in the rewrite that the narrative does not state or directly imply (invented content), and mark whether it is clinically relevant (could change diagnosis or management).
3. Return ONLY one line of JSON, no markdown:
{{"clinical_facts_in_narrative":<int>,"missing":["<5 words>"],"invented":[{{"item":"<5 words>","clinically_relevant":<true|false>}}]}}"""
```
