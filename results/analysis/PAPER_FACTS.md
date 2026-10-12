> **WARNING (6 Oct 2026): every NarrativeShield number below (NS-full, NS-notool, NS-agent1only / Extract-then-Reason) is INVALID.** run_ns.py passed the full original clinical vignette to the reasoning call for every persona. Rerun with codes/rerun_ns_stem.py (results -> ns_results_v2) before using any mitigation result. B1 to B4, all confound analyses, judge audits and the Agent-1 extraction audit are unaffected.

# Facts collected for the rewrite (no new inference; dataset untouched)

## Controls for the register confound (zero-GPU, existing data) — files: ctrl_a.py, ctrl_split2.py
- Persona text is NOT 10-25% longer than alpha as the old plan assumed: beta is +131% (mean 261 vs 117 words), gamma +166% (301 words). State this plainly.
- Pooled over 12 models, gap ~ length change + edit-similarity + model FE (SE clustered by question):
  - B1: length coef -0.0015 (95% CI -0.016..+0.013, p=.83) -> gap is NOT explained by length. Edit-similarity coef -0.013 (CI -0.027..-0.0002, p=.047): the more a rewrite departs from alpha, the slightly larger the gap (marginal).
  - Gap by length tertile (B1): low .095 / mid .091 / high .102. Short-half .094 vs long-half .098.
  - NS-full / agent1only: no significant length or similarity effect.
- Memorization (MedQA split recovered by text-matching to GBaker/MedQA-USMLE-4-options: 861 train, 139 test; all 1000 matched):
  - B1 gap: test .090 vs train .097, diff -0.007 (95% CI -0.049..+0.034) -> no memorization effect. alpha acc .640 vs .634.
  - NS-full gap: test .014 vs train .003 (diff +.010, CI -.009..+.032, n.s.); agent1only: .031 vs .009 (diff +.022, CI +.005..+.039, small but significant; still far below B1).
- split labels: source_split.csv; per-model: split_results.csv; text features: persona_text_features.csv

## Pending (need GEMINI_API_KEY, no GPU)
- codes/audit_added_facts.py  -> added_facts_audit.csv, added_facts_summary.csv, clean_qids.txt (then rerun main table on clean subset)
- codes/audit_agent1_outputs.py -> Agent1 fact recall per model/persona

## Reporting
- master_table_no_base.csv is the table to use (llama31_8b_base rows moved to base_model_rows_excluded.csv).
- Core numbers (12-model means, NAG): B1 .109, B2 .096, B3 .093, B4 .086, NS-full .012, NS-notool .012, NS-agent1only .017.
- Gap reduction vs B1 (11 models, OpenBioLLM excluded): B2 .009 (0/11 sig), B3 .014 (3/11), B4 .022 (3/11), NS-full .093 / notool .093 / agent1only .086 (11/11 each).
- Accuracy cost of NS-full vs size: corr(size, change)=0.56; <5B -0.022, 5-9B -0.023, 9-30B +0.024. agent1only is +0.041 accuracy over full on average (8/11 sig).

## Added-facts audit (judge: gemini-3.8-flash, read-only; 3000 audits, 1000/persona) — files: added_facts_audit.csv, added_facts_summary.csv, gap_by_added_subset.csv, ctrl_added.py
- Marked personas DO add content: items added per question alpha 0.37 / beta 4.66 / gamma 3.18; clinically relevant additions 0.26 / 1.59 / 1.43; share with >=1 relevant addition 22.6% / 85.9% / 79.3%; >=2: 3.4% / 46.3% / 40.2%. (Judge flags 22.6% of alpha too, so it has a nonzero false-positive floor.) Admit this openly in the paper: the confound reviewers raised is real in the data.
- BUT the gap does not depend on it (pooled over 12 models, per-question gap, bootstrap CI over questions):
  - B1 gap, questions where neither marked persona adds a relevant fact (n=53): 0.127 (CI .066-.194); <=1 relevant (n=344): 0.094; >=2 relevant (n=656): 0.097; all: 0.096. No dose-response: regression coef on #relevant additions -0.005 (CI -.022..+.012, p=.58).
  - NS-full gap: .005 overall, .021 on the zero-added subset (n=53, too small to interpret), .003 on >=2.
  - So the register effect survives (and is not smaller) when the rewrite adds no clinically relevant facts. Caveat: zero-added subset is small and judge-defined; use the <=1 subset (n=344) as the main robustness table.

## Agent 1 fact-recall audit (judge gemini-3.8-flash; 100 questions x 3 personas x 12 models = 3600 audits; NS-full) — files: agent1_audit/, agent1_recall_summary_final.csv
- Agent 1 JSON parse rate 95.9% (11-model mean; OpenBioLLM .863, mistral7b .843, medgemma .927 lowest).
- Fact recall of the extracted JSON vs the narrative it came from (when parsed), 11 models excl. OpenBioLLM: mean 0.749 (range .67 ministral8b to .80 qwen3_8b); OpenBioLLM 0.568. By persona: alpha .797, beta .727, gamma .721.
- Strict pass rate (zero missing facts) only 1.9%: Agent 1 is lossy by this auditor's standard. Missing items are typically contextual (exposure details, negative findings, imaging modality, aggravating factors).
- Interpretation caveats to state in the paper: (1) the auditor counts every discrete fact in the narrative, and marked narratives contain the added social/financial content that extraction is MEANT to drop, so beta/gamma recall is under-measured; (2) alpha recall is also only ~.80, so losses are not mainly a register effect (alpha-vs-marked recall difference ~7 pts); (3) accuracy under agent1only is on average +0.04 vs B1 despite the loss, so the lost facts are mostly non-decisive. Report as a limitation, and as the reason agent1only/full lose accuracy on small models.

## Judge-reliability check for the added-facts audit — files: judge_gemini-3.1-pro-preview/, judge_gemini-3.1-flash-lite/, judge_agreement.csv, gap_by_subset_all_judges.csv, judge_agree.py
- Judges: primary gemini-3.8-flash; second judges gemini-3.1-pro-preview (3000/3000) and gemini-3.1-flash-lite (3000/3000). gemini-2.5-flash was stopped at ~600 items (rate limits; flagged ~88% of everything, over-sensitive) — do not report it.
- Agreement with the primary judge on "any clinically relevant addition" (pooled 3000 items): Pro 86.8% (kappa 0.70, Spearman on counts 0.76); flash-lite 78.9% (kappa 0.51, Spearman .58). Within persona, Pro kappa: alpha .58, beta .50, gamma .53 (pooled kappa is partly inflated by the persona difference).
- All judges agree marked personas add content: share with >=1 relevant addition — primary: alpha 22.6 / beta 85.9 / gamma 79.3; Pro: 37.5 / 94.2 / 90.9; flash-lite: 61.1 / 91.2 / 82.3. Pro is stricter on marked personas (mean relevant additions 2.75/2.48 vs 1.59/1.43), flash-lite over-flags alpha. So the count of added facts is judge-dependent but the ordering alpha << gamma < beta is stable.
- The conclusion is judge-independent. Per-question gap regressed on mean #relevant additions in the marked personas (model FE, clustered by question): B1 coef -0.005 (p=.57, primary), -0.003 (p=.63, Pro), +0.007 (p=.44, flash-lite); NS-full +0.001 / +0.000 / +0.001 (all p>.78). No dose-response under any judge.
- B1 gap by subset is stable: Pro: >=2 added (923 q) .095, <=1 added (77 q) .109; flash-lite: <=1 (277 q) .091, >=2 (723 q) .098; primary: <=1 (344 q) .094, >=2 (656 q) .097. Under the strict judge almost no rewrite is clean (only 8 zero-addition questions), so use the regression + the primary/flash-lite subsets, not the Pro zero subset.
- Caveat for the paper: judge-vs-clinician agreement was not measured (clinicians never rated additions), only judge-vs-judge.

## New checks found while drafting the paper (6 Oct 2026) — files: age_preserved.csv, gap_by_age_subset.csv, verbatim_leak.csv
- Explicit age (digits or number words in an age context) is stated in Pa 99.9%, Pb 50.7%, Pg 52.1%; both marked 29.9%. (Approximate phrasings like "in my seventies" not counted, so this is a lower bound on preservation.) Clinician raters marked facts preserved 900/900, so they accepted implicit/approximate age; say so.
- Age omission explains a small part of the gap: joint regression (B1, model FE, clustered) age-missing count +0.021 per marked persona (p=.03); relevant additions -0.006 (p=.49); length +0.010 (p=.55).
- B1 gap with age explicit in both marked personas (n=299): .085 [.061,.109] (89% of overall .096). Strict clean subset (age explicit in both AND <=1 relevant addition in each, n=106): B1 .065 [.026,.105], B4 .059, E-t-R .012 [-.010,.035], full .002.
- Pooled LPM with question x model FE, clustered by question, 12 models: marked-voice effect B1 -9.6 pts [-11.0,-8.2]; B2 -8.4; B3 -8.3; B4 -7.4; E-t-R -1.2 [-1.8,-0.6]; notool -0.5 (n.s.); full -0.5 (n.s.).
- Generation artifact = natural experiment: 84 Pb and 51 Pg rewrites (116 questions) append the original vignette verbatim after the persona narrative, so the marked input is a superset of the clinical text. B1 Pa-minus-Pb on those 84: .068 [.032,.106] (others .090); Pa-minus-Pg on 51: .033 [-.013,.078] (others .108). E-t-R: .009 / .015. So the voice lowers accuracy even when every original sentence is present (clear for Pb).
- Teaser example MQ_0066 (syringomyelia after MVC; age explicit in Pg): B1 correct Pa 11/12, Pg 5/12 and 7/12 models answer "conversion disorder" for the cultural voice; E-t-R Pg 11/12.

## Compute (sacct, gh partition, since 2026-09-01, all states)
- Before the NS v2 rerun (6 Oct 2026): 207 jobs, 786.9 GPU-hours. Re-run the same sacct after the 36 nsv2 jobs finish and use the final total in the paper (main.tex placeholder [GPU-HOURS]).
