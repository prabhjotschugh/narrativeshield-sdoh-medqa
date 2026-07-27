# NarrativeShield — Annotator Validation Statistics

## 1. Descriptive Statistics

| annotator               | persona                                |   n |   facts_preserved_rate |   register_match_rate |   realism_mean |   realism_median |   realism_sd |
|:------------------------|:---------------------------------------|----:|-----------------------:|----------------------:|---------------:|-----------------:|-------------:|
| Annotator_1             | alpha (pα) — high health literacy      | 100 |                      1 |                     1 |        4.68    |                5 |     0.601009 |
| Annotator_1             | beta (pβ) — socioeconomic barrier      | 100 |                      1 |                     1 |        4.22    |                4 |     0.675397 |
| Annotator_1             | gamma (pγ) — cultural/somatic register | 100 |                      1 |                     1 |        3.26    |                3 |     0.675995 |
| Annotator_2             | alpha (pα) — high health literacy      | 100 |                      1 |                     1 |        5       |                5 |     0        |
| Annotator_2             | beta (pβ) — socioeconomic barrier      | 100 |                      1 |                     1 |        5       |                5 |     0        |
| Annotator_2             | gamma (pγ) — cultural/somatic register | 100 |                      1 |                     1 |        5       |                5 |     0        |
| Annotator_3             | alpha (pα) — high health literacy      | 100 |                      1 |                     1 |        5       |                5 |     0        |
| Annotator_3             | beta (pβ) — socioeconomic barrier      | 100 |                      1 |                     1 |        4.71    |                5 |     0.607944 |
| Annotator_3             | gamma (pγ) — cultural/somatic register | 100 |                      1 |                     1 |        4.3     |                4 |     0.731679 |
| POOLED (all annotators) | alpha (pα) — high health literacy      | 300 |                      1 |                     1 |        4.89333 |                5 |     0.377399 |
| POOLED (all annotators) | beta (pβ) — socioeconomic barrier      | 300 |                      1 |                     1 |        4.64333 |                5 |     0.614313 |
| POOLED (all annotators) | gamma (pγ) — cultural/somatic register | 300 |                      1 |                     1 |        4.18667 |                4 |     0.91722  |

## 2. Variance Check

- Facts Preserved: constant for 3/3 annotators -> ALL constant (kappa undefined by construction)
- Register Match: constant for 3/3 annotators -> ALL constant (kappa undefined by construction)

## 3. Inter-Annotator Agreement

### Facts Preserved (binary)

> **Note:** All annotators rated Facts Preserved as constant (single value) across all 300 items. Cohen's kappa, Fleiss' kappa are mathematically undefined here (p_e = 1, division by zero) -- NOT reported as 1.0. Percent agreement = 1.000 (trivially, since there is no disagreement possible), but this reflects zero variance in the judgment task, not informative reliability. Recommended reporting: state raw agreement rate and explicitly flag that kappa is undefined by construction.

### Register Match (binary)

> **Note:** All annotators rated Register Match as constant (single value) across all 300 items. Cohen's kappa, Fleiss' kappa are mathematically undefined here (p_e = 1, division by zero) -- NOT reported as 1.0. Percent agreement = 1.000 (trivially, since there is no disagreement possible), but this reflects zero variance in the judgment task, not informative reliability. Recommended reporting: state raw agreement rate and explicitly flag that kappa is undefined by construction.

### Narrative Realism (ordinal 1-5)

**ALL annotators: Annotator_1, Annotator_2, Annotator_3**

- Krippendorff's alpha: -0.014
- ICC(2,1): 0.135

**EXCLUDING flagged straightliner(s): Annotator_1, Annotator_3**

- Krippendorff's alpha: 0.226
- ICC(2,1): 0.324

> Flagged straightliner(s): ['Annotator_2'] — rated realism as a constant value across all items; confirmed genuine on manual review.

## 4. Cross-Persona Statistical Tests

### McNemar's test — Facts Preserved

### McNemar's test — Register Match

### Chi-square test of independence (pooled)

### Kruskal-Wallis — Narrative Realism across personas (pooled)

- H=138.694, p=0.0000 (significant)

**Post-hoc Mann-Whitney U (Bonferroni-corrected):**

- alpha (pα) — high health literacy vs beta (pβ) — socioeconomic barrier: U=53912.5, p=0.0000, Bonferroni p=0.0000 (significant)

- alpha (pα) — high health literacy vs gamma (pγ) — cultural/somatic register: U=64435.0, p=0.0000, Bonferroni p=0.0000 (significant)

- beta (pβ) — socioeconomic barrier vs gamma (pγ) — cultural/somatic register: U=56876.5, p=0.0000, Bonferroni p=0.0000 (significant)
