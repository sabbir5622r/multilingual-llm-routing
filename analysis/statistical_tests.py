import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routing.simulate_routes import find_result, merge_pair


def route(frame, budget):
    count = int(round(len(frame) * budget))
    mask = np.zeros(len(frame), dtype=bool)

    order = np.argsort(
        frame["confidence_margin_small"].to_numpy(),
        kind="stable",
    )
    mask[order[:count]] = True

    return np.where(
        mask,
        frame["is_correct_large"],
        frame["is_correct_small"],
    ).astype(float)


def random_mean(frame, budget, repeats, seed, budget_index):
    count = int(round(len(frame) * budget))
    outcomes = np.empty(
        (repeats, len(frame)),
        dtype=float,
    )

    for repeat in range(repeats):
        random_generator = np.random.default_rng(
            np.random.SeedSequence(
                [seed, budget_index, repeat]
            )
        )

        mask = np.zeros(len(frame), dtype=bool)

        if count:
            selected = random_generator.choice(
                len(frame),
                size=count,
                replace=False,
            )
            mask[selected] = True

        outcomes[repeat] = np.where(
            mask,
            frame["is_correct_large"],
            frame["is_correct_small"],
        )

    return outcomes.mean(axis=0)


def paired_bootstrap(first, second, repeats, seed):
    random_generator = np.random.default_rng(seed)

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)

    differences = np.empty(repeats)

    for index in range(repeats):
        sample = random_generator.integers(
            0,
            len(first),
            size=len(first),
        )

        differences[index] = (
            first[sample] - second[sample]
        ).mean()

    return (
        (first - second).mean(),
        *np.quantile(
            differences,
            [0.025, 0.975],
        ),
    )


def mcnemar(first, second):
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)

    first_only = int(
        np.sum(first & ~second)
    )
    second_only = int(
        np.sum(~first & second)
    )

    discordant = first_only + second_only

    if discordant == 0:
        p_value = 1.0
    else:
        p_value = binomtest(
            min(first_only, second_only),
            discordant,
            0.5,
        ).pvalue

    return first_only, second_only, p_value


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)

    adjusted = np.empty_like(values)
    running = 0.0

    for rank, index in enumerate(order):
        running = max(
            running,
            min(
                1.0,
                (len(values) - rank) * values[index],
            ),
        )
        adjusted[index] = running

    return adjusted


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
        "--bootstrap-repeats",
        type=int,
        default=5000,
    )

    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    raw_root = ROOT / config["paths"]["raw_results"]
    rows = []
    tested_budgets = [0.2, 0.5, 0.8]

    for dataset_key in args.datasets:
        for family in args.families:
            for language_code in config["languages"]:
                small_path = find_result(
                    raw_root,
                    dataset_key,
                    family,
                    "small",
                    language_code,
                    args.limit,
                )
                large_path = find_result(
                    raw_root,
                    dataset_key,
                    family,
                    "large",
                    language_code,
                    args.limit,
                )

                pair = merge_pair(
                    small_path,
                    large_path,
                )

                large_correct = pair[
                    "is_correct_large"
                ].to_numpy(float)

                for budget in tested_budgets:
                    routed = route(pair, budget)

                    random_correct = random_mean(
                        pair,
                        budget,
                        config["routing"]["random_repeats"],
                        config["seed"],
                        config["routing"]["budgets"].index(
                            budget
                        ),
                    )

                    vs_random = paired_bootstrap(
                        routed,
                        random_correct,
                        args.bootstrap_repeats,
                        config["seed"],
                    )

                    vs_large = paired_bootstrap(
                        routed,
                        large_correct,
                        args.bootstrap_repeats,
                        config["seed"],
                    )

                    routed_only, large_only, p_value = (
                        mcnemar(
                            routed,
                            large_correct,
                        )
                    )

                    rows.append(
                        {
                            "dataset": dataset_key,
                            "family": family,
                            "language_code": language_code,
                            "budget": budget,
                            "routed_accuracy": routed.mean(),
                            "random_mean_accuracy": (
                                random_correct.mean()
                            ),
                            "large_accuracy": (
                                large_correct.mean()
                            ),
                            "confidence_minus_random": (
                                vs_random[0]
                            ),
                            "random_ci_low": vs_random[1],
                            "random_ci_high": vs_random[2],
                            "confidence_minus_large": (
                                vs_large[0]
                            ),
                            "large_ci_low": vs_large[1],
                            "large_ci_high": vs_large[2],
                            "routed_only_correct": routed_only,
                            "large_only_correct": large_only,
                            "mcnemar_p": p_value,
                        }
                    )

    results = pd.DataFrame(rows)
    results["mcnemar_p_holm"] = np.nan

    for _, indices in results.groupby(
        ["dataset", "family", "language_code"]
    ).groups.items():
        results.loc[
            indices,
            "mcnemar_p_holm",
        ] = holm(
            results.loc[indices, "mcnemar_p"]
        )

    suffix = (
        f"_limit{args.limit}"
        if args.limit is not None
        else ""
    )

    output_path = (
        ROOT
        / config["paths"]["summary"]
        / f"statistical_tests{suffix}.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()