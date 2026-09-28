import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.score_models import score_one
from models.model_utils import clear_model, hardware_info, load_model


def percentile(values, fraction):
    ordered = sorted(values)
    position = int(
        round((len(ordered) - 1) * fraction)
    )

    return ordered[position]


def benchmark(
    model_name,
    family,
    size,
    dataset_key,
    language,
    config_path,
    measured=None,
):
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_path = (
        ROOT
        / config["paths"]["processed_data"]
        / dataset_key
        / f"{language}.jsonl"
    )

    data = pd.read_json(
        data_path,
        lines=True,
    )

    warmup_count = config["efficiency"][
        "warmup_examples"
    ]

    measured_count = (
        measured
        if measured is not None
        else config["efficiency"]["measured_examples"]
    )

    if measured_count <= 0:
        raise ValueError(
            "The measured example count must be positive."
        )

    if len(data) < warmup_count + measured_count:
        raise ValueError(
            "The dataset does not contain enough examples "
            "for this benchmark."
        )

    loaded_model = load_model(model_name)

    try:
        for _, row in data.head(
            warmup_count
        ).iterrows():
            score_one(
                loaded_model,
                family,
                row,
                config["generation"]["max_input_tokens"],
            )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        latencies = []
        token_counts = []

        started = time.perf_counter()

        measured_data = data.iloc[
            warmup_count:
            warmup_count + measured_count
        ]

        for _, row in measured_data.iterrows():
            result = score_one(
                loaded_model,
                family,
                row,
                config["generation"]["max_input_tokens"],
            )

            latencies.append(
                result["latency_ms"]
            )
            token_counts.append(
                result["input_tokens"]
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - started

        report = {
            "dataset": dataset_key,
            "language_code": language,
            "family": family,
            "model_size": size,
            "model_name": model_name,
            "model_revision": loaded_model.revision,
            "dtype": loaded_model.dtype,
            "warmup_examples": warmup_count,
            "measured_examples": measured_count,
            "mean_input_tokens": statistics.mean(
                token_counts
            ),
            "mean_latency_ms": statistics.mean(
                latencies
            ),
            "median_latency_ms": statistics.median(
                latencies
            ),
            "p95_latency_ms": percentile(
                latencies,
                0.95,
            ),
            "examples_per_second": (
                measured_count / elapsed
            ),
            "input_tokens_per_second": (
                sum(token_counts) / elapsed
            ),
            "model_footprint_gb": (
                loaded_model.model.get_memory_footprint()
                / (1024 ** 3)
            ),
            "peak_gpu_memory_gb": (
                torch.cuda.max_memory_allocated()
                / (1024 ** 3)
                if torch.cuda.is_available()
                else None
            ),
            **hardware_info(),
        }

        output_dir = ROOT / "results" / "efficiency"
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = output_dir / (
            f"{dataset_key}_{family}_{size}_{language}.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                report,
                file,
                indent=2,
            )

        print(
            json.dumps(
                report,
                indent=2,
            )
        )
        print(f"Saved {output_path}")

        return output_path

    finally:
        clear_model(loaded_model)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--family",
        required=True,
        choices=["qwen", "gemma", "llama"],
    )
    parser.add_argument(
        "--size",
        required=True,
        choices=["small", "large"],
    )
    parser.add_argument(
        "--dataset",
        default="belebele",
    )
    parser.add_argument(
        "--language",
        required=True,
    )
    parser.add_argument(
        "--measured",
        type=int,
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config.yaml"),
    )

    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model_name = config["models"][
        args.family
    ][args.size]

    benchmark(
        model_name,
        args.family,
        args.size,
        args.dataset,
        args.language,
        args.config,
        args.measured,
    )


if __name__ == "__main__":
    main()