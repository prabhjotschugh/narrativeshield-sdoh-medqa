# Data Card: NarrativeShield-SDoH

**What it is.** 1,000 MedQA-USMLE patient-vignette questions, each paired with three retellings that carry the same audited original clinical facts: a clinical voice (`persona_alpha`), a socioeconomic voice (`persona_beta`) and a cultural voice (`persona_gamma`). 3,000 retellings in total, in `dataset/medqa_1000.csv`.

**Source.** MedQA-USMLE (Jin et al., 2020; https://github.com/jind11/MedQA), 4-option version. Pooled train and test splits; 861 items come from the training split and 139 from the test split (recovered by text matching).

**Filtering and sampling.** Six filters applied as one boolean mask (seed 42): patient-vignette stem; at least 300 characters; at least one of 49 clinical signal terms; none of 21 mechanism, pathophysiology, anatomy, genetics or histology phrases; at least one of 24 diagnosis, treatment or investigation phrases; at least one of 27 narrative-richness markers. Of 2,921 eligible questions, 1,000 were drawn by proportional stratified sampling over exam step and answer letter. Code: `dataset/1. dataset_code.py`; term lists are in the code.

**Generation.** Each retelling was generated in its own call to a hosted Gemini model (`gemini-3.1-flash-lite`): clinical voice at temperature 0.25, marked voices at 0.90. Prompts forbid demographic labels and formulaic markers and rotate opening styles (`prompts/01_*`). A separate auditor at temperature 0, without the generation prompt, checks that every original fact appears; failures were regenerated. Hosted models change over time, so regenerating the retellings will not be bit-for-bit identical.

**Validation.** Three physicians rated 100 questions (300 retellings). No rater flagged an omitted or altered fact or a register mismatch in any of 900 ratings. Realism ratings are ordered clinical > socioeconomic > cultural, with low inter-rater agreement (Krippendorff's alpha 0.226 excluding a rater who gave every item 5; -0.014 with all three). Raw ratings: `Annotations/ratings_raw.csv`.

**No real patient data.** All narratives are synthetic and derive from public, de-identified exam questions. No IRB review was required.

**Properties that matter for use.** The retellings are not minimal pairs: marked voices are 2.2 to 2.6 times longer than the clinical voice, most add some clinically relevant detail, first-person narrators often give age in words or indirectly, and the clinical voice is third-person prose generated at a lower temperature. "Voice" therefore denotes this whole bundle of differences, not register alone.

**Limitations and risks.**
- Synthetic renderings of socioeconomic or cultural voice risk caricature. They were validated by clinicians for fact preservation, not by sociolinguists or community members for authenticity. They are controlled register templates, not portraits of real communities.
- One generator model; one question format (single-turn multiple choice).
- The MedQA test split is public, so memorization cannot be fully excluded.
- The finding that voice shifts diagnoses could be misused to craft narratives that steer a deployed model; we judge that documenting the vulnerability and its mitigation is more protective.

**Intended use.** Non-commercial research on robustness and fairness of clinical language models, and evaluation of input-representation interventions. Not for training or evaluating systems for unsupervised clinical use, and not a substitute for real patient narratives.

**License.** See `LICENSE`.
