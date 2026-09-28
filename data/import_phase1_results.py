import argparse
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_COLUMNS = {
    "example_id",
    "language_code",
    "language_name",
    "family",
    "model_size",
    "model_name",
    "true_label",
    "is_correct",
    "predicted_label",
    "confidence_margin",
    "entropy",
    "input_tokens",
    "latency_ms",
}


def is_full_result(filename):
    path = Path(filename)

    return (
        path.suffix == ".csv"
        and len(path.parts) >= 2
        and path.parts[-2] == "raw"
        and "_limit" not in path.stem
    )


def read_phase1_files(source):
    if (
        source.is_file()
        and source.suffix.lower() == ".zip"
    ):
        with zipfile.ZipFile(source) as archive:
            result_files = sorted(
                filename
                for filename in archive.namelist()
                if is_full_result(filename)
            )

            for filename in result_files:
                file_bytes = archive.read(filename)

                frame = pd.read_csv(
                    io.BytesIO(file_bytes)
                )

                yield Path(filename).name, frame

        return

    if (source / "results" / "raw").is_dir():
        raw_directory = (
            source / "results" / "raw"
        )
    else:
        raw_directory = source

    if not raw_directory.is_dir():
        raise FileNotFoundError(
            "Expected a Phase 1 results ZIP file "
            f"or raw-results directory: {source}"
        )

    for path in sorted(
        raw_directory.glob("*.csv")
    ):
        if "_limit" not in path.stem:
            yield path.name, pd.read_csv(path)


def upgrade_result(frame):
    missing_columns = (
        REQUIRED_COLUMNS - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Phase 1 result is missing columns: "
            f"{sorted(missing_columns)}"
        )

    probability_columns = [
        "prob_A",
        "prob_B",
        "prob_C",
        "prob_D",
    ]

    missing_probabilities = (
        set(probability_columns)
        - set(frame.columns)
    )

    if missing_probabilities:
        raise ValueError(
            "Phase 1 result is missing probability "
            f"columns: {sorted(missing_probabilities)}"
        )

    upgraded = frame.copy()

    probabilities = upgraded[
        probability_columns
    ].astype(float)

    upgraded.insert(
        0,
        "dataset",
        "belebele",
    )

    upgraded.insert(
        1,
        "task",
        "reading_comprehension",
    )

    upgraded.insert(
        2,
        "category",
        "reading_comprehension",
    )

    upgraded["choice_probabilities"] = (
        probabilities.apply(
            lambda row: json.dumps(
                {
                    label: row[f"prob_{label}"]
                    for label in "ABCD"
                }
            ),
            axis=1,
        )
    )

    upgraded["max_choice_probability"] = (
        probabilities.max(axis=1)
    )

    upgraded["num_choices"] = 4
    upgraded["imported_from_phase1"] = True

    upgraded["phase1_latency_ms_original"] = (
        upgraded["latency_ms"]
    )

    upgraded["latency_ms"] = float("nan")
    upgraded["latency_measurement_valid"] = False

    return upgraded


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Import completed Phase 1 Qwen and Gemma "
            "Belebele predictions without rerunning them."
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help=(
            "Path to the Phase 1 results ZIP file "
            "or extracted results directory"
        ),
    )

    parser.add_argument(
        "--config",
        default=str(ROOT / "config.yaml"),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    with open(
        args.config,
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    output_directory = (
        ROOT
        / config["paths"]["raw_results"]
        / "belebele"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    pending_files = []

    for filename, frame in read_phase1_files(
        args.source
    ):
        family = str(frame["family"].iloc[0])

        if family not in {"qwen", "gemma"}:
            continue

        output_path = (
            output_directory / filename
        )

        if (
            output_path.exists()
            and not args.overwrite
        ):
            raise FileExistsError(
                f"Refusing to overwrite {output_path}. "
                "Use --overwrite if this is intended."
            )

        upgraded = upgrade_result(frame)

        pending_files.append(
            (output_path, upgraded)
        )

    expected_files = 16

    if len(pending_files) != expected_files:
        raise ValueError(
            "Expected 16 full Qwen/Gemma files "
            "(2 families × 2 sizes × 4 languages), "
            f"but found {len(pending_files)}"
        )

    total_predictions = sum(
        len(frame)
        for _, frame in pending_files
    )

    expected_predictions = 14_400

    if total_predictions != expected_predictions:
        raise ValueError(
            f"Expected {expected_predictions:,} "
            "Phase 1 predictions, but found "
            f"{total_predictions:,}"
        )

    for output_path, frame in pending_files:
        frame.to_csv(
            output_path,
            index=False,
        )

    print()
    print(
        f"Imported {len(pending_files)} files "
        f"and {total_predictions:,} predictions."
    )

    print(
        f"Output directory: {output_directory}"
    )


if __name__ == "__main__":
    main()