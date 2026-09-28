import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml


ROOT = Path(__file__).resolve().parents[1]


def expected_calibration_error(frame, bins=10):
    confidence = frame[
        "max_choice_probability"
    ].to_numpy(float)

    correct = frame[
        "is_correct"
    ].to_numpy(float)

    edges = np.linspace(0, 1, bins + 1)
    error = 0.0

    for low, high in zip(edges[:-1], edges[1:]):
        if high == 1:
            selected = (
                (confidence >= low)
                & (confidence <= high)
            )
        else:
            selected = (
                (confidence >= low)
                & (confidence < high)
            )

        if selected.any():
            error += selected.mean() * abs(
                correct[selected].mean()
                - confidence[selected].mean()
            )

    return error


def finite_median(series):
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    return values.median()


def finite_mean(series):
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    return values.mean()

def raw_summaries(raw_root, dataset_key, families, limit):
    suffix = (
        f"_limit{limit}"
        if limit is not None
        else ""
    )

    paths = list(
        (raw_root / dataset_key).glob(
            f"*{suffix}.csv"
        )
    )

    if limit is None:
        paths = [
            path
            for path in paths
            if "_limit" not in path.stem
        ]

    frames = [
        pd.read_csv(path)
        for path in paths
    ]

    if not frames:
        raise FileNotFoundError(
            f"No raw results found for {dataset_key}"
        )

    raw = pd.concat(
        frames,
        ignore_index=True,
    )

    raw = raw[
        raw["family"].isin(families)
    ]

    if raw.empty:
        raise ValueError(
            f"No results for the selected families in {dataset_key}"
        )

    model_rows = []

    group_columns = [
        "dataset",
        "family",
        "model_size",
        "model_name",
        "language_code",
        "language_name",
    ]

    for keys, group in raw.groupby(group_columns):
        model_rows.append(
            {
                "dataset": keys[0],
                "family": keys[1],
                "model_size": keys[2],
                "model_name": keys[3],
                "language_code": keys[4],
                "language_name": keys[5],
                "examples": len(group),
                "accuracy": group["is_correct"].mean(),
                "mean_choice_confidence": group[
                    "max_choice_probability"
                ].mean(),
                "choice_ece_10bin": (
                    expected_calibration_error(group)
                ),
                "mean_input_tokens": group[
                    "input_tokens"
                ].mean(),
                "median_latency_ms": finite_median(
                    group["latency_ms"]
                ),
            }
        )

    transitions = []

    for (family, language), group in raw.groupby(
        ["family", "language_code"]
    ):
        small = group[
            group["model_size"] == "small"
        ][["example_id", "is_correct"]]

        large = group[
            group["model_size"] == "large"
        ][["example_id", "is_correct"]]

        pair = small.merge(
            large,
            on="example_id",
            suffixes=("_small", "_large"),
            validate="one_to_one",
        )

        if pair.empty:
            raise ValueError(
                f"No matching examples for "
                f"{dataset_key}/{family}/{language}"
            )

        names = {
            (True, True): "correct_to_correct",
            (False, True): "wrong_to_correct",
            (True, False): "correct_to_wrong",
            (False, False): "wrong_to_wrong",
        }

        counts = pair.groupby(
            ["is_correct_small", "is_correct_large"]
        ).size()

        for state, name in names.items():
            count = int(counts.get(state, 0))

            transitions.append(
                {
                    "dataset": dataset_key,
                    "family": family,
                    "language_code": language,
                    "transition": name,
                    "count": count,
                    "rate": count / len(pair),
                }
            )

    return (
        pd.DataFrame(model_rows),
        pd.DataFrame(transitions),
    )

def summarize_routes(frame):
    keys = [
        "dataset",
        "family",
        "scope",
        "language_code",
        "policy",
        "budget",
    ]

    numeric_columns = [
        "escalation_rate",
        "accuracy",
        "accuracy_loss_vs_large",
        "latency_ratio_vs_large",
        "escalation_gap",
        "max_language_loss",
        "language_loss_gap",
        "worst_language_accuracy",
    ]

    aggregations = {
        column: (
            column,
            finite_mean
            if column == "latency_ratio_vs_large"
            else "mean",
        )
        for column in numeric_columns
        if column in frame.columns
    }

    return frame.groupby(
        keys,
        as_index=False,
        dropna=False,
    ).agg(**aggregations)


def make_figures(routes, output_dir, dataset_key):
    sns.set_theme(
        style="whitegrid",
        context="paper",
    )

    overall = routes[
        routes["scope"] == "overall"
    ]

    selected = overall[
        overall["policy"].isin(
            [
                "global_confidence",
                "equal_quota_confidence",
                "quota_floor_confidence",
                "global_random",
            ]
        )
    ]

    grid = sns.relplot(
        data=selected,
        x="escalation_rate",
        y="accuracy",
        hue="policy",
        col="family",
        kind="line",
        marker="o",
        height=3.5,
        aspect=1.15,
    )

    grid.set_axis_labels(
        "Escalation rate",
        "Accuracy",
    )

    grid.figure.savefig(
        output_dir / f"{dataset_key}_policy_accuracy.pdf",
        bbox_inches="tight",
    )
    plt.close(grid.figure)

    language = routes[
        (routes["scope"] == "language")
        & (routes["policy"] == "global_confidence")
    ]

    grid = sns.relplot(
        data=language,
        x="budget",
        y="escalation_rate",
        hue="language_code",
        col="family",
        kind="line",
        marker="o",
        height=3.5,
        aspect=1.15,
    )

    grid.set_axis_labels(
        "Global budget",
        "Language escalation rate",
    )

    grid.figure.savefig(
        output_dir / f"{dataset_key}_allocation_disparity.pdf",
        bbox_inches="tight",
    )
    plt.close(grid.figure)

    fairness = overall[
        overall["policy"].isin(
            [
                "global_confidence",
                "equal_quota_confidence",
                "quota_floor_confidence",
            ]
        )
    ]

    grid = sns.relplot(
        data=fairness,
        x="budget",
        y="max_language_loss",
        hue="policy",
        col="family",
        kind="line",
        marker="o",
        height=3.5,
        aspect=1.15,
    )

    grid.set_axis_labels(
        "Target budget",
        "Maximum language loss vs. always-large",
    )

    grid.figure.savefig(
        output_dir / f"{dataset_key}_worst_language_loss.pdf",
        bbox_inches="tight",
    )
    plt.close(grid.figure)


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
    summary_root = ROOT / config["paths"]["summary"]
    figure_root = ROOT / config["paths"]["figures"]

    suffix = (
        f"_limit{args.limit}"
        if args.limit is not None
        else ""
    )

    for dataset_key in args.datasets:
        summary_dir = summary_root / dataset_key
        figure_dir = figure_root / dataset_key

        summary_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        figure_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        models, transitions = raw_summaries(
            raw_root,
            dataset_key,
            args.families,
            args.limit,
        )

        route_path = (
            ROOT
            / config["paths"]["routing_results"]
            / dataset_key
            / f"advanced_routes{suffix}.csv"
        )

        routes = summarize_routes(
            pd.read_csv(route_path)
        )

        models.to_csv(
            summary_dir / f"model_summary{suffix}.csv",
            index=False,
        )
        transitions.to_csv(
            summary_dir
            / f"correction_transitions{suffix}.csv",
            index=False,
        )
        routes.to_csv(
            summary_dir
            / f"advanced_routing_summary{suffix}.csv",
            index=False,
        )

        make_figures(
            routes,
            figure_dir,
            dataset_key,
        )

        print(f"Saved report for {dataset_key}")


if __name__ == "__main__":
    main()