import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def find_result(raw_dir, family, size, language, limit):
    suffix = f"_limit{limit}" if limit else ""
    matches = list(raw_dir.glob(f"{family}_{size}_*_{language}{suffix}.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {family}/{size}/{language} result, found {len(matches)} in {raw_dir}"
        )
    return matches[0]


def merge_pair(small_path, large_path):
    small = pd.read_csv(small_path)
    large = pd.read_csv(large_path)
    keep = ["example_id", "predicted_label", "is_correct", "confidence_margin", "latency_ms", "input_tokens"]
    merged = small[keep + ["language_code", "language_name", "family"]].merge(
        large[keep], on="example_id", suffixes=("_small", "_large"), validate="one_to_one"
    )
    return merged


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
