# rewrite then reason and story schema

Rewrite-then-Reason (rewrite + answer prompts) and Story+Schema.

The text below is the exact source of each prompt, extracted programmatically from the released code (f-string placeholders are filled at run time).


## `codes/run_rewrite.py`

### `REWRITE_SYSTEM`

```python
REWRITE_SYSTEM = """You are a clinician writing up a patient's own account as a standard clinical case vignette.

Rewrite the patient narrative as a concise, third-person clinical vignette in standard medical terminology, the way a USMLE case is written.

KEEP, exactly and completely:
- the patient's age and sex (if the age is given only approximately, state it as given)
- every symptom, with its onset, duration, timing and course, including symptoms the patient denies
- every number: vitals, lab values, doses, durations, measurements
- all past medical, surgical, family, medication, allergy, exposure and social history that bears on health (e.g. smoking, alcohol, travel, occupation exposures, substances taken, remedies or medicines the patient took)
- all physical examination, laboratory and imaging findings

DROP: financial worries, insurance, work or family pressures, who brought the patient in, emotions, cultural or metaphorical phrasing, and any other wording that is about how the patient speaks rather than what happened clinically.

Do not add, infer or interpret anything that is not in the narrative. Do not suggest a diagnosis. Output only the vignette."""
```

### `ANSWER_SYSTEM`

```python
ANSWER_SYSTEM = "Select the best answer letter (A, B, C, or D). Output exactly one letter."
```

## Inline user-message templates in `codes/run_rewrite.py` (`main`)

Rewrite-then-Reason, call 1 (system prompt = `REWRITE_SYSTEM`):

```
PATIENT NARRATIVE:
{narrative}

Write the clinical vignette.
```

Rewrite-then-Reason, call 2 (system prompt = `ANSWER_SYSTEM`; the narrative is not shown):

```
CLINICAL VIGNETTE:
{rewrite}

QUESTION:
{question sentence}

OPTIONS:
{options}

Answer letter:
```

Story+Schema (one call, system prompt = `ANSWER_SYSTEM`):

```
PATIENT NARRATIVE:
{narrative}

EXTRACTED CLINICAL FACTS (JSON):
{schema json from the Extract-then-Reason extractor}

QUESTION:
{question sentence}

OPTIONS:
{options}

Answer letter:
```
