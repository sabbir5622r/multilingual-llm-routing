import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routing.simulate_routes import find_result, merge_pair, safe_ratio


def load_family(raw_root, dataset_key, family, languages, limit):
    pairs = []

    for language in languages:
        small = find_result(
            raw_root,
            dataset_key,
            family,
            "small",
            language,
            limit,
        )
        large = find_result(
            raw_root,
            dataset_key,
            family,
            "large",
            language,
            limit,
        )

        pairs.append(merge_pair(small, large))

    return pd.concat(pairs, ignore_index=True)


def choose_lowest(values, count):
    mask = np.zeros(len(values), dtype=bool)
    order = np.argsort(np.asarray(values), kind="stable")
    mask[order[:count]] = True

    return mask

def local_equal_quota(frame, budget):
    mask = np.zeros(len(frame), dtype=bool)

    for indices in frame.groupby("language_code").groups.values():
        positions = np.asarray(list(indices))
        count = int(round(len(positions) * budget))

        local_mask = choose_lowest(
            frame.loc[positions, "confidence_margin_small"],
            count,
        )

        mask[positions[local_mask]] = True

    return mask


def language_percentile(frame, budget):
    normalized = frame.groupby("language_code")[
        "confidence_margin_small"
    ].rank(
        method="first",
        pct=True,
    )

    count = int(round(len(frame) * budget))

    return choose_lowest(normalized, count)


def quota_floor_confidence(frame, budget, minimum_quota_fraction):
    total_count = int(round(len(frame) * budget))
    mask = np.zeros(len(frame), dtype=bool)

    groups = list(
        frame.groupby("language_code").groups.values()
    )

    quota_counts = []

    for indices in groups:
        proportional = total_count * len(indices) / len(frame)
        quota_counts.append(
            int(np.floor(proportional * minimum_quota_fraction))
        )

    for indices, count in zip(groups, quota_counts):
        positions = np.asarray(list(indices))

        local_mask = choose_lowest(
            frame.loc[positions, "confidence_margin_small"],
            count,
        )

        mask[positions[local_mask]] = True

    remaining = total_count - int(mask.sum())

    if remaining:
        candidates = np.flatnonzero(~mask)

        selected = choose_lowest(
            frame.loc[candidates, "confidence_margin_small"],
            remaining,
        )

        mask[candidates[selected]] = True

    return mask

def global_oracle(frame, budget):
    gain = (
        frame["is_correct_large"].astype(int)
        - frame["is_correct_small"].astype(int)
    )

    count = int(round(len(frame) * budget))
    mask = np.zeros(len(frame), dtype=bool)

    order = np.argsort(
        -gain.to_numpy(),
        kind="stable",
    )
    mask[order[:count]] = True

    return mask


def english_threshold(frame, budget):
    if budget <= 0:
        return np.zeros(len(frame), dtype=bool)

    if budget >= 1:
        return np.ones(len(frame), dtype=bool)

    english_confidence = frame.loc[
        frame["language_code"] == "eng_Latn",
        "confidence_margin_small",
    ]

    threshold = english_confidence.quantile(
        budget,
        interpolation="higher",
    )

    return (
        frame["confidence_margin_small"].to_numpy()
        <= threshold
    )

def summarize(frame, mask, policy, budget, repeat=None):
    mask = np.asarray(mask, dtype=bool)

    routed_correctness = np.where(
        mask,
        frame["is_correct_large"],
        frame["is_correct_small"],
    ).astype(float)

    routed_latency = (
        frame["latency_ms_small"].to_numpy(float)
        + np.where(
            mask,
            frame["latency_ms_large"].to_numpy(float),
            0.0,
        )
    )

    large_accuracy = frame.groupby("language_code")[
        "is_correct_large"
    ].mean()

    rows = []

    for language_code, indices in frame.groupby(
        "language_code"
    ).groups.items():
        positions = np.asarray(list(indices))
        accuracy = routed_correctness[positions].mean()

        rows.append(
            {
                "dataset": frame["dataset"].iloc[0],
                "family": frame["family"].iloc[0],
                "scope": "language",
                "language_code": language_code,
                "policy": policy,
                "budget": budget,
                "repeat": repeat,
                "examples": len(positions),
                "escalation_rate": mask[positions].mean(),
                "accuracy": accuracy,
                "accuracy_loss_vs_large": (
                    large_accuracy[language_code] - accuracy
                ),
                "total_latency_ms": routed_latency[
                    positions
                ].sum(),
                "latency_ratio_vs_large": safe_ratio(
                    routed_latency[positions],
                    frame.loc[
                        positions,
                        "latency_ms_large",
                    ],
                ),
            }
        )

    language_rows = pd.DataFrame(rows)

    rows.append(
        {
            "dataset": frame["dataset"].iloc[0],
            "family": frame["family"].iloc[0],
            "scope": "overall",
            "language_code": "all",
            "policy": policy,
            "budget": budget,
            "repeat": repeat,
            "examples": len(frame),
            "escalation_rate": mask.mean(),
            "accuracy": routed_correctness.mean(),
            "accuracy_loss_vs_large": (
                frame["is_correct_large"].mean()
                - routed_correctness.mean()
            ),
            "total_latency_ms": routed_latency.sum(),
            "latency_ratio_vs_large": safe_ratio(
                routed_latency,
                frame["latency_ms_large"],
            ),
            "escalation_gap": (
                language_rows["escalation_rate"].max()
                - language_rows["escalation_rate"].min()
            ),
            "max_language_loss": language_rows[
                "accuracy_loss_vs_large"
            ].max(),
            "language_loss_gap": (
                language_rows["accuracy_loss_vs_large"].max()
                - language_rows["accuracy_loss_vs_large"].min()
            ),
            "worst_language_accuracy": language_rows[
                "accuracy"
            ].min(),
        }
    )

    return rows

def simulate(
    frame,
    budgets,
    repeats,
    seed,
    minimum_quota_fraction,
):
    results = []

    for budget_index, budget in enumerate(budgets):
        count = int(round(len(frame) * budget))

        policies = {
            "global_confidence": choose_lowest(
                frame["confidence_margin_small"],
                count,
            ),
            "equal_quota_confidence": local_equal_quota(
                frame,
                budget,
            ),
            "language_percentile": language_percentile(
                frame,
                budget,
            ),
            "quota_floor_confidence": quota_floor_confidence(
                frame,
                budget,
                minimum_quota_fraction,
            ),
            "english_threshold_transfer": english_threshold(
                frame,
                budget,
            ),
            "global_oracle_upper_bound": global_oracle(
                frame,
                budget,
            ),
        }

        for policy, mask in policies.items():
            results.extend(
                summarize(frame, mask, policy, budget)
            )

        for repeat in range(repeats):
            random_generator = np.random.default_rng(
                np.random.SeedSequence(
                    [seed, budget_index, repeat]
                )
            )

            random_mask = np.zeros(
                len(frame),
                dtype=bool,
            )

            if count:
                selected = random_generator.choice(
                    len(frame),
                    size=count,
                    replace=False,
                )
                random_mask[selected] = True

            results.extend(
                summarize(
                    frame,
                    random_mask,
                    "global_random",
                    budget,
                    repeat,
                )
            )

    return pd.DataFrame(results)

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
    parser.add_argument(
        "--limit",
        type=int,
    )

    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    raw_root = ROOT / config["paths"]["raw_results"]

    for dataset_key in args.datasets:
        family_results = []

        for family in args.families:
            paired = load_family(
                raw_root,
                dataset_key,
                family,
                config["languages"],
                args.limit,
            )

            family_results.append(
                simulate(
                    paired,
                    config["routing"]["budgets"],
                    config["routing"]["random_repeats"],
                    config["seed"],
                    config["routing"]["minimum_quota_fraction"],
                )
            )

        results = pd.concat(
            family_results,
            ignore_index=True,
        )

        suffix = (
            f"_limit{args.limit}"
            if args.limit is not None
            else ""
        )

        output_dir = (
            ROOT
            / config["paths"]["routing_results"]
            / dataset_key
        )
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_dir / f"advanced_routes{suffix}.csv"
        )
        results.to_csv(output_path, index=False)

        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()