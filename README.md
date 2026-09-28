# Multilingual Small-to-Large LLM Routing

This repository studies whether a small-to-large language-model cascade can reduce inference cost without increasing performance disparities across languages.

## Research setup

**Dataset:** `facebook/belebele` (CC BY-SA 4.0), using the parallel test sets below.

- `eng_Latn` — English
- `ben_Beng` — Bangla
- `hin_Deva` — Hindi
- `urd_Arab` — Urdu

**Model pairs:**

| Family | Small model | Large model |
|---|---|---|
| Qwen3 | `Qwen/Qwen3-1.7B` | `Qwen/Qwen3-4B` |
| Gemma 3 | `google/gemma-3-1b-it` | `google/gemma-3-4b-it` |

Gemma checkpoints require accepting Google's terms on Hugging Face. Qwen3 is run with thinking disabled. All models are evaluated in BF16 when the GPU supports it, otherwise FP16.

## Main idea

The small model scores answer labels A–D. Its confidence is the difference between the two highest normalized label probabilities. At a routing budget of 20%, for example, the 20% least-confident questions are escalated to the larger model.

This produces a cost–accuracy curve without using test labels to choose a threshold. Random routing at the same budget is the matched baseline. An oracle curve is reported only as an upper bound.

## Repository

```text
multilingual-llm-routing/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── download_data.py
│   └── dataset_manifest.json
├── models/
│   └── model_utils.py
├── evaluation/
│   └── score_models.py
├── routing/
│   └── simulate_routes.py
├── analysis/
│   ├── make_report.py
│   └── statistical_tests.py
└── experiments/
    └── run_all.py
```
---
