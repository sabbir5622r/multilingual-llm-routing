<div align="center">

# 🌐 Multilingual LLM Routing

### Who Pays for Efficient Inference? Confidence-Based Small-to-Large LLM Routing Across Tasks and Languages

**A reproducible evaluation of when confidence identifies useful escalations and how global routing allocates large-model access across languages.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E)](https://huggingface.co/docs/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

<p align="center">
  <img src="assets/00_introduction_teaser.png" alt="Task-dependent confidence routing" width="95%">
</p>

## Contents

1. [Overview](#overview)
2. [Key findings](#key-findings)
3. [Experimental design](#experimental-design)
4. [Reproduction](#reproduction)
5. [Repository structure](#repository-structure)
6. [Citation](#citation)

## Overview

This project evaluates confidence-based small-to-large routing across three model families, four languages, and three multilingual tasks. Each input is scored by a small model, and the lowest-confidence inputs are eligible for escalation to the larger model from the same family. Confidence is the probability margin between the two highest-scoring answer labels.

The study contains 72 model, dataset, language, and scale configurations and 42,288 individual predictions. It compares global confidence routing with random routing, equal-quota confidence routing, and an oracle diagnostic upper bound.

## Key findings

| Dataset | Task | Mean correction AUC | Gain over random at 50% escalation |
|---|---|---:|---:|
| Belebele | Reading comprehension | 0.612 | +1.40 to +4.67 points |
| MMLU-ProX-Lite | Knowledge and reasoning | 0.510 | No consistent gain |
| SIB-200 | Topic classification | 0.705 | +2.54 to +5.63 points |

Confidence is useful when uncertainty aligns with the benefit of escalation. That alignment holds for Belebele and SIB-200 but largely disappears on MMLU-ProX-Lite, where small models often remain confidently wrong. Global confidence routing also produces language escalation gaps of up to 30.33 percentage points. Equal-quota routing removes this allocation disparity with little aggregate accuracy cost.

Measured routed inference confirms that escalation rate usually translates into real savings. At 25% escalation, confidence routing reduces wall time by 27.5% to 58.2% and GPU energy by 26.3% to 62.6% relative to large-only inference on the evaluated hardware.

## Experimental design

| Family | Small model | Large model |
|---|---|---|
| Qwen3 | `Qwen/Qwen3-1.7B` | `Qwen/Qwen3-4B` |
| Gemma 3 | `google/gemma-3-1b-it` | `google/gemma-3-4b-it` |
| Llama 3.2 | `meta-llama/Llama-3.2-1B-Instruct` | `meta-llama/Llama-3.2-3B-Instruct` |

| Dataset | Task | Examples per language |
|---|---|---:|
| Belebele | Reading comprehension | 900 |
| MMLU-ProX-Lite | Knowledge and reasoning | 658 |
| SIB-200 | Topic classification | 204 |

The evaluated languages are English (`eng_Latn`), Bangla (`ben_Beng`), Hindi (`hin_Deva`), and Urdu (`urd_Arab`). Dataset sources, splits, and generated files are documented in [`data/README.md`](data/README.md).

## Reproduction

### 1. Install

```bash
git clone https://github.com/sabbir5622r/multilingual-llm-routing.git
cd multilingual-llm-routing
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Gemma and Llama may require accepting their Hugging Face licenses and setting `HF_TOKEN`. The corrected Gemma 3 4B configuration uses FP32 and requires two GPUs in the current implementation.

### 2. Prepare the datasets

```bash
python data/download_data.py --datasets belebele mmlu_prox_lite sib200
```

The downloader standardizes each task, checks row counts and parallel identifiers, and writes provenance reports under `data/processed_phase2/`.

### 3. Run a smoke test

```bash
python evaluation/score_models.py \
  --family qwen \
  --size small \
  --dataset belebele \
  --language eng_Latn \
  --limit 5
```

### 4. Run the full matrix

```bash
python experiments/run_all.py \
  --datasets belebele mmlu_prox_lite sib200 \
  --families qwen gemma llama \
  --skip-download
```

Evaluations are checkpointed every 25 examples and resume automatically. Limited runs use separate filenames and are excluded from final analysis.

### 5. Reproduce tables and figures

From newly generated raw predictions:

```bash
python analysis/analyze_results.py --source results/raw
python analysis/make_figures.py
```

From the frozen corrected result archive:

```bash
python analysis/analyze_results.py \
  --source results/canonical/multilingual_routing_corrected_full_results.zip
python analysis/make_figures.py
```

The analysis validates all 72 configurations, 21,144 paired comparisons, expected dataset sizes, unique identifiers, finite confidence values, and corrected Gemma 3 4B precision before producing the paper tables and figures.

### 6. Reproduce system efficiency

```bash
for family in qwen gemma llama; do
  python efficiency/benchmark_routing_system.py \
    --family "$family" \
    --paired-predictions results/paper_analysis/tables/paired_predictions.csv \
    --examples-per-group 100 \
    --repeats 3
done
```

The benchmark reports wall time, throughput, latency, GPU energy, and GPU-hours. Published measurements used sequential batch-size-one inference on two NVIDIA Tesla T4 GPUs; other hardware may produce different absolute values.

## Repository structure

```text
multilingual-llm-routing/
├── analysis/                 # validated tables, statistics, and figures
├── assets/                   # README figure
├── data/                     # dataset preparation and documentation
├── efficiency/               # routed-system efficiency benchmark
├── evaluation/               # deterministic model scoring
├── experiments/              # experiment orchestration
├── models/                   # model loading and precision handling
├── config.yaml               # datasets, models, languages, and settings
├── requirements.txt
└── LICENSE
```

Datasets, model weights, raw predictions, and generated outputs are excluded from Git. The repository preserves the complete code path from public datasets to paper tables and figures.



## 📚 Citation

If you use this repository or its experimental framework, please consider citing the associated paper:

```bibtex
@misc{hossen2026quantization,
      title={How Much Can We Compress Small LLMs? Quantization Trade-offs for Low-Resource Bangla Language Understanding}, 
      author={Md Sabbir Hossen and Anichur Rahman and Pabon Shaha and Andrew H. Sung and Md Shohel Rana},
      year={2026},
      eprint={2610.8184772},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.24615}, 
}
```

---

<div align="center">

**Built with love for reproducible research on efficient LLMs and low-resource language understanding.**  
*If this repository supports your research, consider giving it a ⭐ and citing the paper.*
**Multilingual LLM Routing Team**

</div>