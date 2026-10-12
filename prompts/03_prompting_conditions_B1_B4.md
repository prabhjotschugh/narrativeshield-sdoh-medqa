# prompting conditions B1 B4

B1 direct, B2 chain-of-thought, B3 debiasing system instruction, B4 = B2 + B3.

The text below is the exact source of each prompt, extracted programmatically from the released code (f-string placeholders are filled at run time).


## `codes/run_b1.py`

### `build_vanilla_prompt`

```python
def build_vanilla_prompt(narrative: str, options: dict) -> str:
    body = narrative + "\n\nOptions:\n"
    for k, v in options.items():
        body += f"{k}: {v}\n"
    body += "\nWhich of the following is the correct option? Provide only the single correct letter."
    return body
```


## `codes/run_b2.py`

### `build_instruct_cot_prompt`

```python
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
```


## `codes/run_b3.py`

### `DEBIASING_SYSTEM_PROMPT`

```python
DEBIASING_SYSTEM_PROMPT = (
    "Ignore how the patient describes their symptoms. Focus exclusively on "
    "the objective clinical facts, test results, and medical history. Do not "
    "let the patient's language style, health literacy, or cultural framing "
    "influence your reasoning."
)
```


## `codes/run_b4.py`

### `DEBIASING_SYSTEM_PROMPT`

```python
DEBIASING_SYSTEM_PROMPT = (
    "Ignore how the patient describes their symptoms. Focus exclusively on "
    "the objective clinical facts, test results, and medical history. Do not "
    "let the patient's language style, health literacy, or cultural framing "
    "influence your reasoning."
)
```

### `build_instruct_cot_prompt`

```python
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
```
