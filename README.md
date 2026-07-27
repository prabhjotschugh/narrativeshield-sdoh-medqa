# NarrativeShield 🛡️

**Same Facts, Different Diagnosis: Measuring and Mitigating Narrative Anchoring in Clinical Language Models**

## 📖 Overview

Large language models used for clinical diagnostic reasoning are sensitive to sociolinguistic register, not just clinical content. We term this failure mode **Narrative Anchoring**: identical clinical facts expressed in different sociolinguistic registers cause model outputs to diverge despite clinically identical inputs.

This repository contains the code, data, and annotations for **NarrativeShield**, a three-agent pipeline that structurally extracts and verifies clinical facts before diagnostic reasoning begins, effectively reducing Narrative Anchoring to near-zero. 

## 📂 Repository Structure

- **`dataset/`**: Contains the **NarrativeShield-SDoH** dataset. 1,000 USMLE clinical vignettes rewritten into three sociolinguistically distinct personas (Control, Socioeconomic, Cultural) with independently audited fact-preservation guarantees.
- **`codes/`**: Contains the inference and evaluation scripts for the 4 conditions tested in the paper:
  - `B1`: Direct Prompting (Zero-shot floor)
  - `B2`: Chain-of-Thought
  - `B3`: Explicit Debiasing
  - `NS`: NarrativeShield Pipeline
- **`results/`**: Output logs and parsed evaluation metrics across 7 different LLMs (Llama-3.1-8B-Instruct, Llama-3.2-3B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, gemma-3-12b-it, gemma-4-E4B-it, BioMistral-7B).
- **`Annotations/`**: Human validation logs, inter-annotator agreement statistics, and summary reports verifying fact preservation and narrative realism.

## 🚀 Quick Start

### 1. Generating Model Outputs (Inference)
The `codes/` directory contains `run_*.py` scripts for each experimental condition. The scripts sequentially load and evaluate models (ensuring optimal VRAM usage) and are resume-safe (automatically skipping completed IDs).

```bash
cd codes
python run_b1.py  # Run Baseline 1 (Direct Prompting)
python run_b2.py  # Run Baseline 2 (Chain-of-Thought)
python run_b3.py  # Run Baseline 3 (Explicit Debiasing)
python run_ns.py  # Run NarrativeShield Pipeline
```

### 2. Evaluating Results
Once inference is complete, use the `eval_*.py` scripts to compute the metrics reported in the paper (Option Match Rate, Narrative Anchoring Gap, and Diagnostic Stability Score).

```bash
cd codes
python eval_b1.py
python eval_b2.py
python eval_b3.py
python eval_ns.py
```

## 📊 Key Findings

- **Narrative Anchoring is pervasive:** Models change their diagnostic predictions based merely on the socioeconomic or cultural phrasing of identical clinical facts.
- **Prompting is not enough:** Chain-of-thought (`B2`) and explicit debiasing instructions (`B3`) only partially mitigate the bias, and frequently confound it with accuracy collapse.
- **Structural intervention works:** NarrativeShield (`NS`) structurally decouples extraction from reasoning, dropping the Narrative Anchoring Gap to near-zero (`-0.004` to `0.037`) across all tested models while providing the highest diagnostic stability.