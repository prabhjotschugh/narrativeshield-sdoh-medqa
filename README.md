# NarrativeShield-SDoH — Anonymous Code, Data and Results

Anonymous artifact for the paper **"Same Illness, Different Voice: Restating the Patient's Account Narrows Narrative Anchoring in Clinical Language Models"** (under review).

This repository contains the benchmark, the inference and evaluation code, every per-question model output, the human-validation data and the analysis scripts behind each table and figure. It contains no author information.

## What the paper does

- **Benchmark.** 1,000 MedQA-USMLE vignettes, each retold in three voices that carry the same audited original facts: clinical (`alpha`), socioeconomic (`beta`) and cultural (`gamma`). 3,000 retellings in total.
- **Evaluation.** 12 open instruction-tuned models (3B to 24B, two medical fine-tunes), every model on all 3,000 retellings, greedy decoding.
- **Finding.** All models are less accurate on the marked voices (pooled penalty -9.6 points, 95% CI [-11.0, -8.2]). Prompting barely helps. Having the model first rewrite the patient's account as a clinical vignette (*Rewrite-then-Reason*) halves the gap and raises marked-voice accuracy.
- **Caveat stated in the paper.** The clinical voice was generated at temperature 0.25 and the marked voices at 0.90, so "voice" means the whole bundle of differences between retellings, not register alone.

## Repository layout

```
dataset/        benchmark and its construction code
  medqa_1000.csv                 1,000 questions + the three retellings (main release)
  medqa_1000_filtered.jsonl      the filtered MedQA sample before retelling
  1. dataset_code.py             filtering and stratified sampling
  2. add_persona_code.py         retelling generation + independent fact-preservation audit
  3. clean_csv.ipynb             post-processing of generated text
  .persona_cache_v2.jsonl        resume cache of the generation run
codes/          inference, evaluation and audit scripts
slurm/          job matrix and SLURM templates used for the GPU runs
results/
  b1_results/ ... b4_results/    per-question outputs, prompting conditions B1-B4
  ns_results_v2/                 per-question outputs, representation conditions
  analysis/                      analysis scripts, audits, and final tables (analysis/final/)
Annotations/    human validation (three physicians), agreement statistics
figures/        the paper figures (PDF + PNG)
prompts/        all generation, audit, judge and pipeline prompts
LICENSE, DATA_CARD.md
```

### Dataset columns (`dataset/medqa_1000.csv`)

| column | content |
|---|---|
| `question_id` | `MQ_0001` ... `MQ_1000` |
| `question`, `options`, `answer`, `answer_idx` | original MedQA-USMLE item (4 options) |
| `meta_info` | exam step |
| `persona_alpha` | clinical voice (third person, temperature 0.25) |
| `persona_beta` | socioeconomic voice (first person, temperature 0.90) |
| `persona_gamma` | cultural voice (first person, temperature 0.90) |

## Conditions and where they live

| Paper name | Script | Output folder / file prefix |
|---|---|---|
| B1 direct | `codes/run_b1.py` | `results/b1_results/results_B1_*` |
| B2 chain-of-thought (1,280 new tokens) | `codes/run_b2.py` | `results/b2_results/` |
| B3 debias instruction | `codes/run_b3.py` | `results/b3_results/` |
| B4 = B2 + B3 | `codes/run_b4.py` | `results/b4_results/` |
| Extract-then-Reason (schema) | `codes/run_ns.py --ablation agent1only` (the reported v2 files were produced by `codes/rerun_ns_stem.py`, which re-answers from saved extractions with the question sentence only) | `results/ns_results_v2/results_NS_agent1only_*` |
| + Reason agent | same | `results_NS_notool_*` |
| + Tools | same | `results_NS_full_*` |
| Story+Schema | `codes/run_rewrite.py --method storyform` | `results_NS_storyform_*` |
| **Rewrite-then-Reason** | `codes/run_rewrite.py --method rewrite` | `results_NS_rewrite_*` |

Each result file is one JSON object per question with the three voices' raw output, parsed answer and correctness. All scripts are resume-safe.

Models: Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, medgemma-4b-it, Phi-4-mini-instruct, Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct, Qwen3-8B (thinking off), Mistral-7B-Instruct-v0.3, Ministral-8B-Instruct-2410, Mistral-Small-24B-Instruct-2501, gemma-3-12b-it, Llama3-OpenBioLLM-8B. Short keys used in file names: `llama32_3b`, `llama31_8b`, `medgemma_4b`, `phi4_mini`, `qwen25_7b`, `qwen25_14b`, `qwen3_8b`, `mistral7b`, `ministral8b`, `mistral_small_24b`, `gemma3_12b`, `openbiollm_8b`.

## Reproducing

### 1. Environment

```bash
pip install -r requirements.txt
export HF_TOKEN=...            # to download gated model weights (never hard-code it)
export GEMINI_API_KEY=...      # only for dataset generation and the LLM-judge audits
```

Cache and result directories default to `/path/to/scratch/...`; set `SCRATCH_BASE`, `RESULT_DIR`, `NS_RESULT_DIR` etc. for your machine. Scripts read the dataset from `dataset/medqa_1000.csv` by default (override with `DATASET_NAME`).

### 2. Inference (GPU)

```bash
cd codes
python run_b1.py --model llama31_8b                     # B1; likewise run_b2/b3/b4.py
python run_ns.py --model llama31_8b --ablation agent1only   # schema pipeline (question sentence only)
python run_rewrite.py --model llama31_8b --method rewrite
```

`slurm/job_matrix.py` lists every (model, condition) job and `slurm/submit_all.sh` submits them; adapt the partition and paths to your cluster.

### 3. Evaluation and analysis (CPU)

```bash
cd codes && python eval_b1.py && python eval_b2.py && python eval_b3.py && python eval_b4.py && python eval_ns.py
cd ../results/analysis && python final_analysis.py && python review_fixes.py
```

`results/analysis/load.py` locates the result folders (`NS_RESULTS_ROOT` overrides the default `results/`). The final tables are in `results/analysis/final/` (`condition_summary.csv`, `master_table_final.csv`, `rewrite_head_to_head.csv`, `subset_gaps.csv`, ...).

### 4. Audits (need `GEMINI_API_KEY`, no GPU)

| Audit | Script | Output |
|---|---|---|
| Content added by each retelling (3 judges) | `codes/audit_added_facts.py` | `results/analysis/added_facts_audit.csv`, `judge_gemini-*/` |
| Fact recall of schema extractions | `codes/audit_agent1_outputs.py` | `results/analysis/agent1_audit/` |
| Fact recall and invention in rewrites | `codes/audit_rewrites.py` | `results/analysis/rewrite_audit/` |

## Human validation

`Annotations/` holds the three physicians' ratings of 100 questions x 3 voices (`Annotator_1-3.xlsx`, and `ratings_raw.csv` with anonymized IDs A1-A3), the written instructions (`annotation_protocol.md`, `IAA - NarrativeShield.pdf`), and the agreement statistics (`stats_output_final/`). Annotator identities are withheld. Annotator_2 gave every realism item a 5, so agreement is reported with and without that rater, as in the paper.

## Notes on the released files

- **Paper-to-file map.** Table 2 and the per-condition numbers: `results/analysis/final/condition_summary.csv`. Per-model values: `master_table_final.csv`. Controlled subsets (Figure 5): `subset_gaps.csv`. Judge agreement: `judge_agreement.csv`. Rewrite audit: `rewrite_audit/`. Figures: `figures/`.
- **Means over 11 vs 12 models.** Llama3-OpenBioLLM-8B cannot produce usable intermediates (rewrite/schema outputs rarely parse), so mitigation means exclude it; `final/means_8_11_12.csv` gives both sets.
- **Superseded runs.** `master_table_final.csv` contains rows prefixed `LEAK-`: an early structured-pipeline run whose answering call also saw the original vignette. They are kept only as a leaky reference and are not reported as results. All reported pipeline numbers come from `ns_results_v2/` (question sentence only, no vignette).
- **Logs.** `results/eval_logs/` holds the evaluation printouts for B1-B4 and the pipelines.
- **Extra file.** `results/b2_results/results_B2_llama31_8b_base.jsonl` is a base-model check outside the 12-model study; it is excluded from all analyses (`results/analysis/base_model_rows_excluded.csv`).
- **Hardware.** GPU runs used single-GPU nodes (NVIDIA GH200). Judge audits use API calls only.

## Models (Hugging Face identifiers)

| Key | Hugging Face id |
|---|---|
| `llama32_3b` | `meta-llama/Llama-3.2-3B-Instruct` |
| `medgemma_4b` | `google/medgemma-4b-it` |
| `phi4_mini` | `microsoft/Phi-4-mini-instruct` |
| `qwen25_7b` | `Qwen/Qwen2.5-7B-Instruct` |
| `mistral7b` | `mistralai/Mistral-7B-Instruct-v0.3` |
| `qwen3_8b` | `Qwen/Qwen3-8B` (thinking disabled) |
| `ministral8b` | `mistralai/Ministral-8B-Instruct-2410` |
| `llama31_8b` | `meta-llama/Llama-3.1-8B-Instruct` |
| `openbiollm_8b` | `aaditya/Llama3-OpenBioLLM-8B` |
| `gemma3_12b` | `google/gemma-3-12b-it` |
| `qwen25_14b` | `Qwen/Qwen2.5-14B-Instruct` |
| `mistral_small_24b` | `mistralai/Mistral-Small-24B-Instruct-2501` |

## Compute

Every run (model x condition) is a single-GPU job on an NVIDIA GH200 node; total inference was about 1,113 GPU-hours. LLM-judge audits (about 17,000 calls) need no GPU. Package versions: see `requirements.txt`.

## Order to reproduce each table and figure

1. Inference: `codes/run_b1.py` to `run_b4.py`, then `codes/run_ns.py` / `codes/rerun_ns_stem.py` and `codes/run_rewrite.py` (see above). Pre-computed per-question outputs are already in `results/`, so steps 1 and 3 can be skipped.
2. Evaluation: `codes/eval_b1.py` to `eval_b4.py`, `eval_ns.py`, `eval_transitions.py`.
3. Audits (API key needed): `codes/audit_added_facts.py` (Table 1 judges, Section 5), `codes/audit_agent1_outputs.py`, `codes/audit_rewrites.py` (Section 7).
4. Tables and figures: `results/analysis/final_analysis.py` (Table 2 and the Section 5-7 numbers; writes `results/analysis/final/`), `results/analysis/review_fixes.py` (McNemar thresholds, 11- vs 12-model means, recall analyses), `ctrl_*.py` and `judge_agree.py` (confound controls). Figures are in `figures/`.

## Prompts

All prompts are in `prompts/` (retelling generation, fact-preservation auditor and judges, B1-B4, Extract-then-Reason agents including +Reason and +Tools, Rewrite-then-Reason, Story+Schema), extracted verbatim from the code.

## Regenerating the retellings

The retellings were produced through a hosted Gemini API whose models change over time, so regenerating them will not be bit-for-bit identical. Use the released `dataset/medqa_1000.csv` to reproduce the paper's results.

## License

Retellings, annotations and data: CC BY-NC 4.0, **non-commercial research use only**, with attribution to MedQA ([Jin et al., 2020](https://github.com/jind11/MedQA)). Code: non-commercial research use only. The retellings were generated with a Gemini model, so users should check its terms of use. See `LICENSE` and `DATA_CARD.md`.

## Intended use and ethics

All text is synthetic and derived from MedQA-USMLE, a public, de-identified exam corpus; no patient data are used. The voices are controlled register templates, not portraits of real communities, and the rewrite step can invent plausible details. Nothing here is a validated clinical tool or intended for unsupervised clinical use. The dataset is for non-commercial research with attribution to MedQA.
