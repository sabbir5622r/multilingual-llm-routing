import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

COMPLETE_DATASETS = {
    "belebele",
    "mmlu_prox_lite",
    "sib200",
}

COMPLETE_FAMILIES = {
    "qwen",
    "gemma",
    "llama",
}


def run(command):
    print(
        "Running:",
        " ".join(map(str, command)),
        flush=True,
    )

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run multilingual model scoring and "
            "the final paper-analysis pipeline."
        )
    )

    parser.add_argument(
        "--config",
        default=str(ROOT / "config.yaml"),
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "belebele",
            "mmlu_prox_lite",
            "sib200",
        ],
    )

    parser.add_argument(
        "--families",
        nargs="+",
        default=[
            "qwen",
            "gemma",
            "llama",
        ],
    )

    parser.add_argument(
        "--limit",
        type=int,
        help=(
            "Evaluate only the first N examples. "
            "Limited runs are excluded from final analysis."
        ),
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use already prepared datasets.",
    )

    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Run model scoring without final analysis.",
    )

    args = parser.parse_args()

    with open(
        args.config,
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    unknown_datasets = (
        set(args.datasets)
        - set(config["datasets"])
    )

    if unknown_datasets:
        raise ValueError(
            "Unknown datasets: "
            f"{sorted(unknown_datasets)}"
        )

    available_families = {
        key
        for key, value in config["models"].items()
        if isinstance(value, dict)
        and "small" in value
        and "large" in value
    }

    unknown_families = (
        set(args.families)
        - available_families
    )

    if unknown_families:
        raise ValueError(
            "Unknown model families: "
            f"{sorted(unknown_families)}"
        )

    python = sys.executable

    if not args.skip_download:
        run(
            [
                python,
                "data/download_data.py",
                "--config",
                args.config,
                "--datasets",
                *args.datasets,
            ]
        )

    for dataset_key in args.datasets:
        for family in args.families:
            for size in ("small", "large"):
                for language_code in config["languages"]:
                    command = [
                        python,
                        "evaluation/score_models.py",
                        "--family",
                        family,
                        "--size",
                        size,
                        "--dataset",
                        dataset_key,
                        "--language",
                        language_code,
                        "--config",
                        args.config,
                    ]

                    if args.limit is not None:
                        command.extend(
                            [
                                "--limit",
                                str(args.limit),
                            ]
                        )

                    run(command)

    if args.skip_analysis:
        print(
            "Model scoring completed. "
            "Final analysis was skipped."
        )
        return

    is_complete_experiment = (
        args.limit is None
        and set(args.datasets)
        == COMPLETE_DATASETS
        and set(args.families)
        == COMPLETE_FAMILIES
    )

    if not is_complete_experiment:
        print(
            "Model scoring completed. "
            "Final paper analysis was skipped "
            "because this was not the complete "
            "72-configuration experiment."
        )
        return

    raw_results = (
        ROOT
        / config["paths"]["raw_results"]
    )

    run(
        [
            python,
            "analysis/analyze_results.py",
            "--source",
            str(raw_results),
        ]
    )

    run(
        [
            python,
            "analysis/make_figures.py",
        ]
    )

    print()
    print(
        "Complete experiment and paper analysis "
        "finished successfully."
    )


if __name__ == "__main__":
    main()