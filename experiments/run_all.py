import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=str(ROOT / "config.yaml"),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["belebele", "mmlu_prox_lite", "sib200"],
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=["qwen", "gemma", "llama"],
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--skip-download",
        action="store_true",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
    )

    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)

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
                            ["--limit", str(args.limit)]
                        )

                    run(command)

    if args.skip_analysis:
        return

    shared_args = [
        "--config",
        args.config,
        "--datasets",
        *args.datasets,
        "--families",
        *args.families,
    ]

    commands = [
        [
            python,
            "routing/simulate_routes.py",
            *shared_args,
        ],
        [
            python,
            "routing/advanced_routes.py",
            *shared_args,
        ],
        [
            python,
            "analysis/make_report.py",
            *shared_args,
        ],
        [
            python,
            "analysis/statistical_tests.py",
            *shared_args,
        ],
    ]

    if args.limit is not None:
        for command in commands:
            command.extend(
                ["--limit", str(args.limit)]
            )

    for command in commands:
        run(command)


if __name__ == "__main__":
    main()