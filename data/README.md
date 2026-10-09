# 📚 Datasets

This repository does not redistribute dataset contents. `data/download_data.py` downloads the public sources from Hugging Face, records resolved revisions, standardizes their schemas, and validates the files used by the evaluation pipeline.

## Sources and splits

| Dataset | Hugging Face source | Subsets | Splits | Examples per language |
|---|---|---|---|---:|
| Belebele | [`facebook/belebele`](https://huggingface.co/datasets/facebook/belebele) | `eng_Latn`, `ben_Beng`, `hin_Deva`, `urd_Arab` | Test | 900 |
| MMLU-ProX-Lite | [`li-lab/MMLU-ProX-Lite`](https://huggingface.co/datasets/li-lab/MMLU-ProX-Lite) | `en`, `bn`, `hi`, `ur` | Validation and test | 658 |
| SIB-200 | [`Davlan/sib200`](https://huggingface.co/datasets/Davlan/sib200) | `eng_Latn`, `ben_Beng`, `hin_Deva`, `urd_Arab` | Test | 204 |

Users should review the dataset cards and comply with the original licenses and terms of use.

## Preparation

From the repository root, run:

```bash
python data/download_data.py --datasets belebele mmlu_prox_lite sib200
```

To rebuild only selected datasets:

```bash
python data/download_data.py --datasets belebele sib200
```

Existing processed files are reused after validation. Add `--force` to download and rebuild them.

## Standardized schema

Every generated JSONL record contains:

| Field | Description |
|---|---|
| `dataset` | Internal dataset key |
| `language_code` | Language and script identifier |
| `language_name` | Display name |
| `example_id` | Stable identifier used for pairing |
| `task` | Reading comprehension, reasoning, or classification |
| `context` | Optional passage or empty string |
| `question` | Question or input text |
| `choices` | Ordered candidate answers |
| `choice_labels` | Symbolic answer labels |
| `correct_label` | Gold symbolic label |
| `category` | Task category or topic |
| `source_split` | Original dataset split |

## Generated structure

```text
data/processed_phase2/
├── belebele/
│   ├── eng_Latn.jsonl
│   ├── ben_Beng.jsonl
│   ├── hin_Deva.jsonl
│   ├── urd_Arab.jsonl
│   └── download_report.json
├── mmlu_prox_lite/
│   └── ...
├── sib200/
│   └── ...
└── download_manifest.json
```

The preparation script checks expected row counts, unique example identifiers, valid answer labels, and agreement of parallel identifiers across languages. Each report records file hashes and the resolved Hugging Face revision.

## Version control

Generated JSONL files are excluded from Git. For exact reproduction, preserve `download_manifest.json` and the per-dataset `download_report.json` files with the experiment artifact, then pin the recorded revisions in `config.yaml`.
