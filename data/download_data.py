import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset
from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
LETTERS = list("ABCDEFGHIJ")

SIB_CATEGORIES = [
    "entertainment",
    "geography",
    "health",
    "politics",
    "science/technology",
    "sports",
    "travel",
]


def file_sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_splits(name, subset, splits, revision):
    frames = []

    for split in splits:
        frame = load_dataset(
            name,
            subset,
            split=split,
            revision=revision,
        ).to_pandas()

        frame["source_split"] = split
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def standardize_belebele(frame):
    rows = []

    for _, row in frame.iterrows():
        choices = [
            row["mc_answer1"],
            row["mc_answer2"],
            row["mc_answer3"],
            row["mc_answer4"],
        ]

        answer = str(row["correct_answer_num"]).replace(".0", "")

        rows.append(
            {
                "example_id": f"{row['link']}::{row['question_number']}",
                "task": "reading_comprehension",
                "context": row["flores_passage"],
                "question": row["question"],
                "choices": choices,
                "choice_labels": LETTERS[:4],
                "correct_label": LETTERS[int(answer) - 1],
                "category": "reading_comprehension",
                "source_split": "test",
            }
        )

    return pd.DataFrame(rows)


def standardize_mmlu(frame):
    rows = []

    for _, row in frame.iterrows():
        choices = []
        choice_labels = []

        for index, label in enumerate(LETTERS):
            value = row[f"option_{index}"]

            if pd.isna(value):
                continue

            if str(value).strip().upper() == "N/A":
                continue

            choices.append(str(value))
            choice_labels.append(label)

        correct_label = str(row["answer"]).strip().upper()

        if correct_label not in choice_labels:
            raise ValueError(
                f"MMLU answer {correct_label} is not among "
                f"the available labels {choice_labels}"
            )

        split = row["source_split"]

        if "question_id_src" in row:
            source_id = row["question_id_src"]
        else:
            source_id = row["question_id"]

        rows.append(
            {
                "example_id": f"{split}:{source_id}",
                "task": "multitask_reasoning",
                "context": "",
                "question": row["question"],
                "choices": choices,
                "choice_labels": choice_labels,
                "correct_label": correct_label,
                "category": row["category"],
                "source_split": split,
            }
        )

    return pd.DataFrame(rows)


def standardize_sib(frame):
    rows = []

    category_to_label = {
        category: LETTERS[index]
        for index, category in enumerate(SIB_CATEGORIES)
    }

    for _, row in frame.iterrows():
        category = str(row["category"])

        if category not in category_to_label:
            raise ValueError(
                f"Unknown SIB-200 category: {category}"
            )

        rows.append(
            {
                "example_id": f"test:{row['index_id']}",
                "task": "topic_classification",
                "context": "",
                "question": row["text"],
                "choices": SIB_CATEGORIES,
                "choice_labels": LETTERS[:len(SIB_CATEGORIES)],
                "correct_label": category_to_label[category],
                "category": category,
                "source_split": "test",
            }
        )

    return pd.DataFrame(rows)


STANDARDIZERS = {
    "belebele": standardize_belebele,
    "mmlu_prox_lite": standardize_mmlu,
    "sib200": standardize_sib,
}


def validate_dataset(
    standardized,
    dataset_key,
    language_code,
    expected_rows,
):
    required_columns = {
        "dataset",
        "language_code",
        "language_name",
        "example_id",
        "task",
        "question",
        "choices",
        "choice_labels",
        "correct_label",
        "category",
    }

    missing_columns = required_columns - set(standardized.columns)

    if missing_columns:
        raise ValueError(
            f"{dataset_key}/{language_code} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(standardized) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows for "
            f"{dataset_key}/{language_code}, "
            f"but found {len(standardized)}"
        )

    if standardized["example_id"].duplicated().any():
        raise ValueError(
            f"Duplicate example IDs found in "
            f"{dataset_key}/{language_code}"
        )

    valid_answers = standardized.apply(
        lambda row: row["correct_label"] in row["choice_labels"],
        axis=1,
    )

    if not valid_answers.all():
        raise ValueError(
            f"Invalid correct labels found in "
            f"{dataset_key}/{language_code}"
        )


def validate_parallel_ids(output_dir, language_codes):
    id_sets = []

    for language_code in language_codes:
        path = output_dir / f"{language_code}.jsonl"
        frame = pd.read_json(path, lines=True)
        id_sets.append(set(frame["example_id"].astype(str)))

    reference_ids = id_sets[0]

    for current_ids in id_sets[1:]:
        if current_ids != reference_ids:
            raise ValueError(
                f"Language subsets in {output_dir.name} "
                "do not contain parallel example IDs"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=str(ROOT / "config.yaml"),
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if "all" in args.datasets:
        selected_datasets = list(config["datasets"])
    else:
        selected_datasets = args.datasets

    unknown_datasets = (
        set(selected_datasets) - set(config["datasets"])
    )

    if unknown_datasets:
        raise ValueError(
            f"Unknown datasets: {sorted(unknown_datasets)}"
        )

    output_root = ROOT / config["paths"]["processed_data"]
    output_root.mkdir(parents=True, exist_ok=True)

    reports = []

    for dataset_key in selected_datasets:
        dataset_config = config["datasets"][dataset_key]

        output_dir = output_root / dataset_key
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_info = HfApi().dataset_info(
            dataset_config["name"],
            revision=dataset_config["revision"],
        )

        resolved_revision = dataset_info.sha
        dataset_rows = []

        for language_code, subset in dataset_config["subsets"].items():
            output_path = output_dir / f"{language_code}.jsonl"

            if output_path.exists() and not args.force:
                standardized = pd.read_json(
                    output_path,
                    lines=True,
                )

            else:
                raw_data = load_splits(
                    dataset_config["name"],
                    subset,
                    dataset_config["splits"],
                    resolved_revision,
                )

                standardized = STANDARDIZERS[dataset_key](
                    raw_data
                )

                standardized.insert(
                    0,
                    "dataset",
                    dataset_key,
                )

                standardized.insert(
                    1,
                    "language_code",
                    language_code,
                )

                standardized.insert(
                    2,
                    "language_name",
                    config["languages"][language_code],
                )

                standardized.to_json(
                    output_path,
                    orient="records",
                    lines=True,
                    force_ascii=False,
                )

            validate_dataset(
                standardized,
                dataset_key,
                language_code,
                dataset_config["expected_rows"],
            )

            dataset_rows.append(
                {
                    "language_code": language_code,
                    "language_name": config["languages"][
                        language_code
                    ],
                    "rows": len(standardized),
                    "sha256": file_sha256(output_path),
                    "path": str(
                        output_path.relative_to(ROOT)
                    ),
                }
            )

        validate_parallel_ids(
            output_dir,
            config["languages"],
        )

        report = {
            "dataset_key": dataset_key,
            "dataset_name": dataset_config["name"],
            "requested_revision": dataset_config["revision"],
            "resolved_revision": resolved_revision,
            "files": dataset_rows,
        }

        report_path = output_dir / "download_report.json"

        with open(
            report_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                report,
                file,
                ensure_ascii=False,
                indent=2,
            )

        reports.append(report)

        print()
        print(f"Validated dataset: {dataset_key}")
        print(pd.DataFrame(dataset_rows).to_string(index=False))

    manifest_path = output_root / "download_manifest.json"

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            reports,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        f"Saved and validated {len(selected_datasets)} "
        f"datasets in {output_root}"
    )


if __name__ == "__main__":
    main()