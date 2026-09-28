import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


ROOT = Path(__file__).resolve().parents[1]


def summarize_random(frame):
    keys = ["family", "language_code", "language_name", "policy", "budget"]
    return frame.groupby(keys, as_index=False).agg(
        accuracy=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        escalation_rate=("escalation_rate", "mean"),
        total_latency_ms=("total_latency_ms", "mean"),
        latency_ratio_vs_large=("latency_ratio_vs_large", "mean"),
    )


def raw_model_summary(raw_dir, limit):
    suffix = f"_limit{limit}" if limit else ""
    rows = []
    transition_rows = []
    paths = list(raw_dir.glob(f"*{suffix}.csv"))
    if limit is None:
        paths = [path for path in paths if "_limit" not in path.stem]
    for path in paths:
        frame = pd.read_csv(path)
        rows.append({
            "family": frame["family"].iloc[0],
            "model_size": frame["model_size"].iloc[0],
            "model_name": frame["model_name"].iloc[0],
            "language_code": frame["language_code"].iloc[0],
            "language_name": frame["language_name"].iloc[0],
            "examples": len(frame),
            "accuracy": frame["is_correct"].mean(),
            "mean_latency_ms": frame["latency_ms"].mean(),
            "median_latency_ms": frame["latency_ms"].median(),
            "mean_input_tokens": frame["input_tokens"].mean(),
        })

    summary = pd.DataFrame(rows)
    for (family, language), group in summary.groupby(["family", "language_code"]):
        raw_files = {}
        for size in ["small", "large"]:
            matches = list(raw_dir.glob(f"{family}_{size}_*_{language}{suffix}.csv"))
            if len(matches) == 1:
                raw_files[size] = pd.read_csv(matches[0])[["example_id", "is_correct"]]
        if len(raw_files) == 2:
            pair = raw_files["small"].merge(raw_files["large"], on="example_id", suffixes=("_small", "_large"))
            labels = {
                (True, True): "correct_to_correct",
                (False, True): "wrong_to_correct",
                (True, False): "correct_to_wrong",
                (False, False): "wrong_to_wrong",
            }
            counts = pair.groupby(["is_correct_small", "is_correct_large"]).size()
            for state, name in labels.items():
                transition_rows.append({
                    "family": family,
                    "language_code": language,
                    "transition": name,
                    "count": int(counts.get(state, 0)),
                    "rate": counts.get(state, 0) / len(pair),
                })
    return summary, pd.DataFrame(transition_rows)


def make_figures(summary, figure_dir):
    sns.set_theme(style="whitegrid", context="paper")
    confidence = summary[summary["policy"] == "confidence"]

    grid = sns.relplot(
        data=confidence,
        x="escalation_rate",
        y="accuracy",
        hue="language_name",
        col="family",
        kind="line",
        marker="o",
        height=3.5,
        aspect=1.15,
    )
    grid.set_axis_labels("Escalation rate", "Accuracy")
    grid.figure.savefig(figure_dir / "cost_accuracy_curves.pdf", bbox_inches="tight")
    plt.close(grid.figure)

    large = confidence[confidence["budget"] == 1.0][["family", "language_code", "accuracy"]].rename(
        columns={"accuracy": "large_accuracy"}
    )
    losses = confidence.merge(large, on=["family", "language_code"])
    losses["accuracy_loss_vs_large"] = losses["large_accuracy"] - losses["accuracy"]
    grid = sns.relplot(
        data=losses,
        x="budget",
        y="accuracy_loss_vs_large",
        hue="language_name",
        col="family",
        kind="line",
        marker="o",
        height=3.5,
        aspect=1.15,
    )
    grid.set_axis_labels("Escalation rate", "Accuracy loss vs. always-large")
    grid.figure.savefig(figure_dir / "language_accuracy_loss.pdf", bbox_inches="tight")
    plt.close(grid.figure)

    comparison = summary[summary["policy"].isin(["confidence", "random"])]
    grid = sns.relplot(
        data=comparison,
        x="budget",
        y="accuracy",
        hue="policy",
        style="language_name",
        col="family",
        kind="line",
        height=3.7,
        aspect=1.2,
    )
    grid.set_axis_labels("Escalation rate", "Accuracy")
    grid.figure.savefig(figure_dir / "escalation_comparison.pdf", bbox_inches="tight")
    plt.close(grid.figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    suffix = f"_limit{args.limit}" if args.limit else ""
    routing_path = ROOT / cfg["paths"]["routing_results"] / f"routing_results{suffix}.csv"
    routing = pd.read_csv(routing_path)
    summary = summarize_random(routing)
    summary_dir = ROOT / cfg["paths"]["summary"]
    figure_dir = ROOT / cfg["paths"]["figures"]
    summary_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    model_summary, transitions = raw_model_summary(ROOT / cfg["paths"]["raw_results"], args.limit)
    model_summary.to_csv(summary_dir / f"model_accuracy{suffix}.csv", index=False)
    transitions.to_csv(summary_dir / f"correction_transitions{suffix}.csv", index=False)
    summary.to_csv(summary_dir / f"routing_summary{suffix}.csv", index=False)

    large = summary[(summary["policy"] == "confidence") & (summary["budget"] == 1.0)][
        ["family", "language_code", "accuracy"]
    ].rename(columns={"accuracy": "large_accuracy"})
    disparity = summary[summary["policy"] == "confidence"].merge(large, on=["family", "language_code"])
    disparity["accuracy_loss_vs_large"] = disparity["large_accuracy"] - disparity["accuracy"]
    disparity.to_csv(summary_dir / f"language_disparity{suffix}.csv", index=False)
    make_figures(summary, figure_dir)
    print(f"Saved summaries to {summary_dir} and figures to {figure_dir}")


if __name__ == "__main__":
    main()
