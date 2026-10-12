# extract then reason agents

Extract-then-Reason: schema extractor (Agent 1), +Reason agent (Agent 2), answer-fallback agent (Agent 3), and the one-call answer prompt used for the schema-only condition.

The text below is the exact source of each prompt, extracted programmatically from the released code (f-string placeholders are filled at run time).


## `codes/run_ns.py`

### `AGENT1_SYSTEM`

```python
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
```

### `build_agent1_messages`

```python
def build_agent1_messages(model_key: str, narrative: str) -> list:
    content = f"PATIENT NARRATIVE:\n{narrative.strip()}\n\nExtract the clinical facts as JSON."
    if model_key in CONTENT_BLOCK_MODELS:
        # Manually-loaded Gemma-family models (gemma3_12b, medgemma_4b): content blocks
        combined = AGENT1_SYSTEM + "\n\n" + content
        return [{"role": "user", "content": [{"type": "text", "text": combined}]}]
    else:
        # All other models — standard system/user chat template
        return [
            {"role": "system", "content": AGENT1_SYSTEM},
            {"role": "user",   "content": content},
        ]
```

### `build_agent2_messages`

```python
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

    if model_key in CONTENT_BLOCK_MODELS:
        return [{"role": "user", "content": [{"type": "text", "text": system + "\n\n" + content}]}]
    else:
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": content},
        ]
```

### `build_agent3_messages`

```python
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

    if model_key in CONTENT_BLOCK_MODELS:
        return [{"role": "user", "content": [{"type": "text", "text": system + "\n\n" + content}]}]
    else:
        return [{"role": "system", "content": system}, {"role": "user", "content": content}]
```

### `build_agent1only_messages`

```python
def build_agent1only_messages(model_key: str, clinical_json_str: str,
                               question: str, options: dict) -> list:
    """Ablation: Agent 1 extraction → direct answer. No Agent 2."""
    options_str = "\n".join(f"  {k}: {v}" for k, v in options.items())
    system  = "Select the best answer letter (A, B, C, or D) based on the clinical facts. Output exactly one letter."
    content = (
        f"CLINICAL FACTS:\n{clinical_json_str}\n\n"
        f"QUESTION:\n{question}\n\nOPTIONS:\n{options_str}\n\nAnswer letter:"
    )
    if model_key in CONTENT_BLOCK_MODELS:
        return [{"role": "user", "content": [{"type": "text", "text": system + "\n\n" + content}]}]
    else:
        return [{"role": "system", "content": system}, {"role": "user", "content": content}]
```
