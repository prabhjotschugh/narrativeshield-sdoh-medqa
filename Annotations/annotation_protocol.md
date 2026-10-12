# Annotation Protocol (instructions given to the three physician raters)

Source: `IAA - NarrativeShield.pdf` (same content, PDF form).

**Task.** Human validation of 100 persona-varied clinical question triples. Each row contains an original USMLE clinical vignette rewritten in three registers:

- **P-alpha:** high health literacy, semi-clinical language
- **P-beta:** socioeconomic barriers, informal lay language
- **P-gamma:** cultural / somatic / metaphorical framing

Each persona presentation is rated independently: 100 questions x 3 personas = 300 rows per rater.

## Task 1 - Clinical fact preservation (binary)

Are all clinical facts from the original present and clinically accurate in this persona?

- **1** = every symptom, duration, negated finding, vital sign, lab value and relevant history item is present and unaltered in meaning.
- **0** = one or more facts are missing, omitted, or their clinical meaning has been distorted. List the specific facts in `missing_or_altered_facts`.

Counts as a clinical fact: chief complaint and duration, positive symptoms, explicitly denied symptoms (e.g. "no flank pain"), vital signs, lab values, gestational age, exam findings, relevant past history. Does **not** count: phrasing, word choice, tone, formality. A lay synonym ("burns when I go" for "dysuria") is valid unless the clinical meaning changes.

## Task 2 - Narrative realism (Likert 1-5)

Would a real patient matching this persona actually speak or present this way?

| Score | Meaning |
|---|---|
| 5 | Completely realistic - indistinguishable from a real patient |
| 4 | Mostly realistic - minor awkwardness |
| 3 | Neutral - plausible but generic |
| 2 | Mostly unrealistic - register feels forced or inconsistent |
| 1 | Completely unrealistic - no real patient would present this way |

Rate each persona against its own register standard (do not penalize P-beta for not sounding like P-alpha).

## Task 3 - Register match (binary)

Does the language clearly match the described persona?

- **1** = yes (P-alpha articulate and semi-clinical; P-beta informal with socioeconomic context; P-gamma somatic/metaphorical with a slightly translated register).
- **0** = no (could be mistaken for one of the other two, or has no distinguishing register features).

## Rules

1. Annotate independently; do not discuss ratings with the other raters until all have submitted.
2. Rate each persona on its own; do not let the rating of one persona influence another for the same question.
3. Clinical judgment takes priority on Task 1.
4. When in doubt on Task 1, mark 0 and explain.
5. Leave comment fields blank if there is nothing specific to add.

## Released ratings

`ratings_raw.csv` holds all 900 ratings (3 raters x 100 questions x 3 personas) with anonymized rater IDs A1-A3 (A1-A3 correspond to `Annotator_1-3` in the spreadsheets and statistics). `stats_output_final/` holds the agreement statistics reported in the paper.
