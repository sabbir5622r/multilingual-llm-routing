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

## 1. Local VS Code

Create a private repository and copy these files into it.

```powershell
git init
git add .
git commit -m "Initial multilingual routing pipeline"
git branch -M main
git remote add origin YOUR_PRIVATE_REPOSITORY_URL
git push -u origin main
```

Do not commit downloaded data or result files. The scripts recreate the dataset from its public source.

## 2. Kaggle

Enable a T4 GPU and internet access. Add your Hugging Face token as a Kaggle secret named `HF_TOKEN` only if Gemma access requires authentication.

```python
!git clone YOUR_PRIVATE_REPOSITORY_URL
%cd multilingual-llm-routing
!pip install -q -r requirements.txt
```

Download and validate the four parallel language subsets:

```python
!python data/download_data.py
```

Run a 25-example smoke test:

```python
!python experiments/run_all.py --limit 25 --families qwen gemma
```

Run the complete study:

```python
!python experiments/run_all.py --families qwen gemma
```

The evaluator checkpoints after every 25 examples. Re-running the command resumes completed files.

## Outputs

```text
results/
├── raw/                 # one row per model, language, and question
├── routing/             # policy results for every routing budget
├── summary/
│   ├── model_accuracy.csv
│   ├── routing_summary.csv
│   ├── language_disparity.csv
│   ├── correction_transitions.csv
│   └── statistical_tests.csv
└── figures/
    ├── cost_accuracy_curves.pdf
    ├── language_accuracy_loss.pdf
    └── escalation_comparison.pdf
```

Each raw row records the item ID, gold answer, prediction, A–D probabilities, confidence margin, token counts, latency, model revision, hardware, and package versions.

## Evaluation rules

- Primary task metric: accuracy.
- Primary routing coordinate: escalation rate, from 0% to 100%.
- Secondary efficiency measures: summed per-example latency and processed input tokens.
- Primary disparity outcome: maximum language-level accuracy loss relative to always-large.
- Confidence routing is compared with random routing at the same escalation rate.
- A 100% cascade runs both models and can be slower than running the large model alone; standalone large-model latency is reported separately.
- Results must be reported for each family separately before any pooled summary.
- Oracle routing is an unattainable upper bound and must not be described as a usable policy.

## Pilot decision gate

Run the smoke test before full evaluation. Continue only when the large model improves over the small model and there are enough `small wrong → large correct` cases to make routing meaningful. The full `--limit 25` run is for code validation; use a larger preregistered pilot sample for a research decision.

## Dataset license

Belebele is released under CC BY-SA 4.0. This repository stores only a manifest and download script. Cite the original Belebele paper and preserve its license when redistributing derived data.
