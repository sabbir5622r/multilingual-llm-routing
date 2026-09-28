import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def safe_ratio(numerator, denominator):
    numerator = np.asarray(
        numerator,
        dtype=float,
    )

    denominator = np.asarray(
        denominator,
        dtype=float,
    )

    if not np.isfinite(numerator).all():
        return np.nan

    if not np.isfinite(denominator).all():
        return np.nan

    denominator_total = denominator.sum()

    if denominator_total == 0:
        return np.nan

    return (
        numerator.sum()
        / denominator_total
    )


def find_result(
    raw_root,
    dataset_key,
    family,
    model_size,
    language_code,
    limit,
):
    suffix = (
        f"_limit{limit}"
        if limit is not None
        else ""
    )

    dataset_directory = (
        raw_root / dataset_key
    )

    matches = list(
        dataset_directory.glob(
            f"{family}_{model_size}_*"
            f"_{language_code}{suffix}.csv"
        )
    )

    if limit is None:
        matches = [
            path
            for path in matches
            if "_limit" not in path.stem
        ]

    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected exactly one result file for "
            f"{dataset_key}/{family}/{model_size}/"
            f"{language_code}, but found "
            f"{len(matches)}"
        )

    return matches[0]


def merge_pair(
    small_result_path,
    large_result_path,
):
    small_results = pd.read_csv(
        small_result_path
    )

    large_results = pd.read_csv(
        large_result_path
    )

    prediction_columns = [
        "example_id",
        "predicted_label",
        "is_correct",
        "confidence_margin",
        "max_choice_probability",
        "entropy",
        "latency_ms",
        "input_tokens",
    ]

    metadata_columns = [
        "dataset",
        "task",
        "category",
        "language_code",
        "language_name",
        "family",
    ]

    paired_results = small_results[
        prediction_columns + metadata_columns
    ].merge(
        large_results[prediction_columns],
        on="example_id",
        suffixes=("_small", "_large"),
        validate="one_to_one",
    )

    return paired_results


def route_metrics(frame, use_large, policy, budget, repeat=None):
    use_large = np.asarray(use_large, dtype=bool)
    correct = np.where(use_large, frame["is_correct_large"], frame["is_correct_small"]).astype(float)
    latency = frame["latency_ms_small"].to_numpy(float) + np.where(
        use_large, frame["latency_ms_large"].to_numpy(float), 0.0
    )
    large_only_latency = frame["latency_ms_large"].sum()
    return {
        "family": frame["family"].iloc[0],
        "language_code": frame["language_code"].iloc[0],
        "language_name": frame["language_name"].iloc[0],
        "policy": policy,
        "budget": budget,
        "repeat": repeat,
        "examples": len(frame),
        "escalation_rate": use_large.mean(),
        "accuracy": correct.mean(),
        "total_latency_ms": latency.sum(),
        "latency_ratio_vs_large": latency.sum() / large_only_latency,
    }


def simulate_language(frame, budgets, repeats, seed):
    results = []
    n = len(frame)
    for budget_index, budget in enumerate(budgets):
        count = int(round(n * budget))
        confidence_order = np.argsort(frame["confidence_margin_small"].to_numpy())
        confidence_mask = np.zeros(n, dtype=bool)
        confidence_mask[confidence_order[:count]] = True
        results.append(route_metrics(frame, confidence_mask, "confidence", budget))

        oracle_gain = frame["is_correct_large"].astype(int) - frame["is_correct_small"].astype(int)
        oracle_order = np.argsort(-oracle_gain.to_numpy(), kind="stable")
        oracle_mask = np.zeros(n, dtype=bool)
        oracle_mask[oracle_order[:count]] = True
        results.append(route_metrics(frame, oracle_mask, "oracle_upper_bound", budget))

        for repeat in range(repeats):
            rng = np.random.default_rng(np.random.SeedSequence([seed, budget_index, repeat]))
            random_mask = np.zeros(n, dtype=bool)
            if count:
                random_mask[rng.choice(n, size=count, replace=False)] = True
            results.append(route_metrics(frame, random_mask, "random", budget, repeat))

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--families", nargs="+", default=["qwen", "gemma"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    raw_dir = ROOT / cfg["paths"]["raw_results"]
    output_dir = ROOT / cfg["paths"]["routing_results"]
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for family in args.families:
        for language in cfg["dataset"]["languages"]:
            small = find_result(raw_dir, family, "small", language, args.limit)
            large = find_result(raw_dir, family, "large", language, args.limit)
            frame = merge_pair(small, large)
            all_results.append(simulate_language(
                frame,
                cfg["routing"]["budgets"],
                cfg["routing"]["random_repeats"],
                cfg["seed"],
            ))

    results = pd.concat(all_results, ignore_index=True)
    suffix = f"_limit{args.limit}" if args.limit else ""
    output_path = output_dir / f"routing_results{suffix}.csv"
    results.to_csv(output_path, index=False)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
