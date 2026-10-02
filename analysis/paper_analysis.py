import argparse
import math
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = (
    ROOT
    / "results"
    / "canonical"
    / "multilingual_routing_corrected_full_results.zip"
)
DEFAULT_OUTPUT = ROOT / "results" / "paper_analysis"

DATASETS = ("belebele", "mmlu_prox_lite", "sib200")
FAMILIES = ("qwen", "gemma", "llama")
LANGUAGES = ("eng_Latn", "ben_Beng", "hin_Deva", "urd_Arab")
EXPECTED_ROWS = {
    "belebele": 900,
    "mmlu_prox_lite": 658,
    "sib200": 204,
}
DISPLAY_DATASETS = {
    "belebele": "Belebele",
    "mmlu_prox_lite": "MMLU-ProX-Lite",
    "sib200": "SIB-200",
}
DISPLAY_FAMILIES = {
    "qwen": "Qwen3",
    "gemma": "Gemma 3",
    "llama": "Llama 3.2",
}
DISPLAY_POLICIES = {
    "confidence": "Confidence",
    "equal_quota": "Equal quota",
    "random": "Random",
    "oracle": "Oracle",
}
COLORS = {
    "confidence": "#1f77b4",
    "equal_quota": "#2ca02c",
    "random": "#7f7f7f",
    "oracle": "#d62728",
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--random-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_raw_results(archive_path):
    if not archive_path.is_file():
        raise FileNotFoundError(f"Canonical archive not found: {archive_path}")

    frames = []

    with ZipFile(archive_path) as archive:
        broken = archive.testzip()
        if broken is not None:
            raise ValueError(f"Corrupted file inside archive: {broken}")

        names = [
            name
            for name in archive.namelist()
            if name.startswith("results/raw/")
            and name.endswith(".csv")
            and "_limit" not in name
        ]

        if len(names) != 72:
            raise ValueError(f"Found {len(names)} raw files; expected 72")

        for name in sorted(names):
            frame = pd.read_csv(archive.open(name))
            frame["source_file"] = name
            frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    validate_raw_results(raw)
    return raw


def validate_raw_results(raw):
    keys = ["dataset", "family", "model_size", "language_code"]
    groups = raw.groupby(keys, sort=False)

    if groups.ngroups != 72:
        raise ValueError(f"Found {groups.ngroups} configurations; expected 72")

    for key, frame in groups:
        dataset, family, size, language = key

        if dataset not in DATASETS:
            raise ValueError(f"Unexpected dataset: {dataset}")
        if family not in FAMILIES:
            raise ValueError(f"Unexpected family: {family}")
        if size not in ("small", "large"):
            raise ValueError(f"Unexpected model size: {size}")
        if language not in LANGUAGES:
            raise ValueError(f"Unexpected language: {language}")

        expected = EXPECTED_ROWS[dataset]
        if len(frame) != expected:
            raise ValueError(f"{key}: found {len(frame)} rows; expected {expected}")
        if frame["example_id"].nunique() != expected:
            raise ValueError(f"Duplicate or missing example IDs: {key}")

        for column in ("confidence_margin", "max_choice_probability"):
            values = pd.to_numeric(frame[column], errors="coerce")
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite {column}: {key}")

        if frame["is_correct"].isna().any():
            raise ValueError(f"Missing correctness values: {key}")

        if (
            dataset in ("mmlu_prox_lite", "sib200")
            and family == "gemma"
            and size == "large"
            and not (frame["dtype"] == "float32").all()
        ):
            raise ValueError(f"Corrected Gemma 3 4B file is not float32: {key}")


def make_pairs(raw):
    keys = ["dataset", "family", "language_code", "example_id"]

    small = raw[raw["model_size"] == "small"][
        keys + ["is_correct", "confidence_margin", "max_choice_probability"]
    ].rename(
        columns={
            "is_correct": "small_correct",
            "confidence_margin": "small_margin",
            "max_choice_probability": "small_probability",
        }
    )

    large = raw[raw["model_size"] == "large"][
        keys + ["is_correct"]
    ].rename(columns={"is_correct": "large_correct"})

    pairs = small.merge(large, on=keys, how="inner", validate="one_to_one")
    expected = sum(EXPECTED_ROWS.values()) * len(FAMILIES) * len(LANGUAGES)

    if len(pairs) != expected:
        raise ValueError(f"Found {len(pairs)} model pairs; expected {expected}")

    pairs["small_correct"] = pairs["small_correct"].astype(bool)
    pairs["large_correct"] = pairs["large_correct"].astype(bool)
    pairs["correction"] = ~pairs["small_correct"] & pairs["large_correct"]
    pairs["gain"] = (
        pairs["large_correct"].astype(int)
        - pairs["small_correct"].astype(int)
    )
    return pairs


def model_accuracy_table(raw):
    table = (
        raw.groupby(
            ["dataset", "family", "model_size", "language_code"],
            as_index=False,
            sort=False,
        )
        .agg(examples=("is_correct", "size"), accuracy=("is_correct", "mean"))
    )
    return table.sort_values(
        ["dataset", "family", "model_size", "language_code"]
    ).reset_index(drop=True)


def choose_lowest(values, count):
    mask = np.zeros(len(values), dtype=bool)
    if count:
        order = np.argsort(np.asarray(values), kind="stable")
        mask[order[:count]] = True
    return mask


def confidence_mask(frame, budget):
    count = int(round(len(frame) * budget))
    return choose_lowest(frame["small_margin"].to_numpy(), count)


def equal_quota_mask(frame, budget):
    mask = np.zeros(len(frame), dtype=bool)

    for indices in frame.groupby("language_code", sort=False).indices.values():
        positions = np.asarray(indices)
        count = int(round(len(positions) * budget))
        local = choose_lowest(
            frame.iloc[positions]["small_margin"].to_numpy(),
            count,
        )
        mask[positions[local]] = True

    return mask


def oracle_mask(frame, budget):
    count = int(round(len(frame) * budget))
    mask = np.zeros(len(frame), dtype=bool)
    if count:
        order = np.argsort(-frame["gain"].to_numpy(), kind="stable")
        mask[order[:count]] = True
    return mask


def route_outcomes(frame, mask):
    return np.where(
        mask,
        frame["large_correct"].to_numpy(float),
        frame["small_correct"].to_numpy(float),
    )


def deterministic_policy(frame, policy, budget):
    if policy == "confidence":
        mask = confidence_mask(frame, budget)
    elif policy == "equal_quota":
        mask = equal_quota_mask(frame, budget)
    elif policy == "oracle":
        mask = oracle_mask(frame, budget)
    else:
        raise ValueError(f"Unknown deterministic policy: {policy}")

    return mask.astype(float), route_outcomes(frame, mask)


def random_policy(frame, budget, repeats, seed):
    count = int(round(len(frame) * budget))
    escalations = np.zeros((repeats, len(frame)), dtype=float)
    outcomes = np.zeros((repeats, len(frame)), dtype=float)

    for repeat in range(repeats):
        rng = np.random.default_rng(np.random.SeedSequence([seed, repeat]))
        mask = np.zeros(len(frame), dtype=bool)
        if count:
            selected = rng.choice(len(frame), size=count, replace=False)
            mask[selected] = True
        escalations[repeat] = mask
        outcomes[repeat] = route_outcomes(frame, mask)

    return escalations.mean(axis=0), outcomes.mean(axis=0)


def policy_outputs(frame, policy, budget, random_repeats, seed):
    if policy == "random":
        return random_policy(frame, budget, random_repeats, seed)
    return deterministic_policy(frame, policy, budget)


def summarize_policy(frame, escalation, outcome, policy, budget):
    language_rows = []

    for language, indices in frame.groupby("language_code", sort=False).indices.items():
        positions = np.asarray(indices)
        language_rows.append(
            {
                "language_code": language,
                "escalation_rate": escalation[positions].mean(),
                "accuracy": outcome[positions].mean(),
            }
        )

    language = pd.DataFrame(language_rows)
    return {
        "policy": policy,
        "budget": budget,
        "escalation_rate": escalation.mean(),
        "accuracy": outcome.mean(),
        "worst_language_accuracy": language["accuracy"].min(),
        "language_accuracy_gap": (
            language["accuracy"].max() - language["accuracy"].min()
        ),
        "escalation_gap": (
            language["escalation_rate"].max()
            - language["escalation_rate"].min()
        ),
    }


def routing_tables(pairs, budgets, random_repeats, seed):
    policies = ("confidence", "equal_quota", "random", "oracle")
    overall_rows = []
    language_rows = []

    for dataset_index, dataset in enumerate(DATASETS):
        for family_index, family in enumerate(FAMILIES):
            frame = pairs[
                (pairs["dataset"] == dataset)
                & (pairs["family"] == family)
            ].reset_index(drop=True)

            for budget_index, budget in enumerate(budgets):
                for policy_index, policy in enumerate(policies):
                    local_seed = (
                        seed
                        + dataset_index * 10000
                        + family_index * 1000
                        + budget_index * 100
                        + policy_index
                    )
                    escalation, outcome = policy_outputs(
                        frame,
                        policy,
                        budget,
                        random_repeats,
                        local_seed,
                    )

                    row = summarize_policy(
                        frame,
                        escalation,
                        outcome,
                        policy,
                        budget,
                    )
                    row.update({"dataset": dataset, "family": family})
                    overall_rows.append(row)

                    for language, indices in frame.groupby(
                        "language_code", sort=False
                    ).indices.items():
                        positions = np.asarray(indices)
                        language_rows.append(
                            {
                                "dataset": dataset,
                                "family": family,
                                "language_code": language,
                                "policy": policy,
                                "budget": budget,
                                "escalation_rate": escalation[positions].mean(),
                                "accuracy": outcome[positions].mean(),
                            }
                        )

    return pd.DataFrame(overall_rows), pd.DataFrame(language_rows)


def correction_auc_table(pairs):
    rows = []

    for (dataset, family, language), frame in pairs.groupby(
        ["dataset", "family", "language_code"], sort=False
    ):
        target = frame["correction"].astype(int).to_numpy()
        if np.unique(target).size < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(target, -frame["small_margin"].to_numpy())

        rows.append(
            {
                "dataset": dataset,
                "family": family,
                "language_code": language,
                "examples": len(frame),
                "corrections": int(target.sum()),
                "correction_rate": target.mean(),
                "correction_detection_auc": auc,
            }
        )

    detail = pd.DataFrame(rows)
    mean = (
        detail.groupby(["dataset", "family"], as_index=False)
        .agg(
            examples=("examples", "sum"),
            corrections=("corrections", "sum"),
            correction_rate=("correction_rate", "mean"),
            correction_detection_auc=("correction_detection_auc", "mean"),
            auc_std_across_languages=("correction_detection_auc", "std"),
        )
        .assign(language_code="mean")
    )
    return pd.concat([detail, mean], ignore_index=True)


def mmlu_quintile_table(pairs):
    frame = pairs[pairs["dataset"] == "mmlu_prox_lite"].copy()
    frame["confidence_rank"] = frame.groupby(
        ["family", "language_code"], sort=False
    )["small_margin"].rank(method="first", pct=True)
    frame["confidence_quintile"] = np.ceil(
        frame["confidence_rank"] * 5
    ).clip(1, 5).astype(int)

    rows = []
    for (family, language, quintile), group in frame.groupby(
        ["family", "language_code", "confidence_quintile"], sort=False
    ):
        errors = ~group["small_correct"]
        conditional = (
            group.loc[errors, "large_correct"].mean()
            if errors.any()
            else np.nan
        )
        rows.append(
            {
                "family": family,
                "language_code": language,
                "confidence_quintile": quintile,
                "examples": len(group),
                "mean_confidence_margin": group["small_margin"].mean(),
                "small_accuracy": group["small_correct"].mean(),
                "small_error_rate": errors.mean(),
                "large_accuracy": group["large_correct"].mean(),
                "net_large_gain": group["gain"].mean(),
                "correction_rate_all_examples": group["correction"].mean(),
                "correction_rate_given_small_error": conditional,
            }
        )

    detail = pd.DataFrame(rows)
    mean = (
        detail.groupby(["family", "confidence_quintile"], as_index=False)
        .agg(
            examples=("examples", "sum"),
            mean_confidence_margin=("mean_confidence_margin", "mean"),
            small_accuracy=("small_accuracy", "mean"),
            small_error_rate=("small_error_rate", "mean"),
            large_accuracy=("large_accuracy", "mean"),
            net_large_gain=("net_large_gain", "mean"),
            correction_rate_all_examples=("correction_rate_all_examples", "mean"),
            correction_rate_given_small_error=(
                "correction_rate_given_small_error",
                "mean",
            ),
        )
        .assign(language_code="mean")
    )
    return pd.concat([detail, mean], ignore_index=True)


def cluster_bootstrap_difference(frame, first, second, repeats, seed):
    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    clustered = (
        pd.DataFrame(
            {
                "example_id": frame["example_id"].to_numpy(),
                "difference": differences,
            }
        )
        .groupby("example_id", sort=False)["difference"]
        .mean()
        .to_numpy()
    )

    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats, dtype=float)

    for repeat in range(repeats):
        sample = rng.integers(0, len(clustered), size=len(clustered))
        estimates[repeat] = clustered[sample].mean()

    low, high = np.quantile(estimates, [0.025, 0.975])
    estimate = clustered.mean()
    return estimate, low, high


def bootstrap_table(pairs, budgets, repeats, random_repeats, seed):
    rows = []
    comparison_index = 0

    for dataset in DATASETS:
        for family in FAMILIES:
            frame = pairs[
                (pairs["dataset"] == dataset)
                & (pairs["family"] == family)
            ].reset_index(drop=True)

            for budget in budgets:
                confidence_escalation, confidence_outcome = policy_outputs(
                    frame, "confidence", budget, random_repeats, seed
                )
                del confidence_escalation

                for comparator in ("random", "equal_quota"):
                    comparator_escalation, comparator_outcome = policy_outputs(
                        frame,
                        comparator,
                        budget,
                        random_repeats,
                        seed + comparison_index + 1,
                    )
                    del comparator_escalation

                    estimate, low, high = cluster_bootstrap_difference(
                        frame,
                        confidence_outcome,
                        comparator_outcome,
                        repeats,
                        seed + 1000 + comparison_index,
                    )

                    rows.append(
                        {
                            "dataset": dataset,
                            "family": family,
                            "budget": budget,
                            "comparison": f"confidence_minus_{comparator}",
                            "accuracy_difference": estimate,
                            "ci_95_low": low,
                            "ci_95_high": high,
                            "ci_excludes_zero": bool(low > 0 or high < 0),
                            "bootstrap_unit": "example_id",
                            "bootstrap_repeats": repeats,
                        }
                    )
                    comparison_index += 1

    return pd.DataFrame(rows)


def save_table(frame, path):
    frame.to_csv(path, index=False)


def markdown_table(frame):
    display = frame.copy()
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for row in display.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                text = ""
            else:
                text = str(value)
            values.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def plot_accuracy_curves(routes, figure_dir):
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)

    for row, dataset in enumerate(DATASETS):
        for column, family in enumerate(FAMILIES):
            ax = axes[row, column]
            selected = routes[
                (routes["dataset"] == dataset)
                & (routes["family"] == family)
            ]

            for policy in ("confidence", "equal_quota", "random", "oracle"):
                line = selected[selected["policy"] == policy].sort_values("budget")
                ax.plot(
                    line["escalation_rate"],
                    line["accuracy"],
                    marker="o",
                    markersize=3,
                    linewidth=1.7,
                    color=COLORS[policy],
                    label=DISPLAY_POLICIES[policy],
                )

            ax.grid(alpha=0.25)
            ax.set_xlim(-0.02, 1.02)
            if row == 0:
                ax.set_title(DISPLAY_FAMILIES[family])
            if column == 0:
                ax.set_ylabel(f"{DISPLAY_DATASETS[dataset]}\nAccuracy")
            if row == 2:
                ax.set_xlabel("Escalation rate")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.955))

    for extension in ("pdf", "png"):
        fig.savefig(
            figure_dir / f"accuracy_escalation_curves.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_auc(auc_table, figure_dir):
    mean = auc_table[auc_table["language_code"] == "mean"].copy()
    detail = auc_table[auc_table["language_code"] != "mean"].copy()
    x = np.arange(len(DATASETS))
    width = 0.23
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    for index, family in enumerate(FAMILIES):
        family_mean = mean[mean["family"] == family].set_index("dataset")
        values = [family_mean.loc[dataset, "correction_detection_auc"] for dataset in DATASETS]
        errors = [family_mean.loc[dataset, "auc_std_across_languages"] for dataset in DATASETS]
        positions = x + (index - 1) * width

        ax.bar(
            positions,
            values,
            width,
            yerr=errors,
            capsize=3,
            color=colors[index],
            alpha=0.82,
            label=DISPLAY_FAMILIES[family],
        )

        family_detail = detail[detail["family"] == family]
        for dataset_index, dataset in enumerate(DATASETS):
            points = family_detail[family_detail["dataset"] == dataset][
                "correction_detection_auc"
            ].to_numpy()
            jitter = np.linspace(-0.035, 0.035, len(points))
            ax.scatter(
                np.full(len(points), positions[dataset_index]) + jitter,
                points,
                s=14,
                color="black",
                alpha=0.6,
                zorder=3,
            )

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance")
    ax.set_xticks(x, [DISPLAY_DATASETS[dataset] for dataset in DATASETS])
    ax.set_ylabel("Correction-detection AUC")
    ax.set_ylim(0.35, 0.85)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    for extension in ("pdf", "png"):
        fig.savefig(
            figure_dir / f"correction_detection_auc.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_mmlu_quintiles(quintiles, figure_dir):
    mean = quintiles[quintiles["language_code"] == "mean"].copy()
    colors = {"qwen": "#1f77b4", "gemma": "#ff7f0e", "llama": "#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)

    metrics = [
        ("small_error_rate", "Small-model error rate"),
        (
            "correction_rate_given_small_error",
            "Large corrects small-model error",
        ),
    ]

    for ax, (metric, label) in zip(axes, metrics):
        for family in FAMILIES:
            line = mean[mean["family"] == family].sort_values(
                "confidence_quintile"
            )
            ax.plot(
                line["confidence_quintile"],
                line[metric],
                marker="o",
                linewidth=2,
                color=colors[family],
                label=DISPLAY_FAMILIES[family],
            )
        ax.set_xlabel("Confidence quintile (Q1 low, Q5 high)")
        ax.set_ylabel(label)
        ax.set_xticks(range(1, 6))
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)

    axes[0].legend(frameon=False)
    fig.tight_layout()

    for extension in ("pdf", "png"):
        fig.savefig(
            figure_dir / f"mmlu_confidence_quintiles.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def write_summary(path, model_table, routes, auc_table, bootstrap):
    model_mean = (
        model_table.groupby(["dataset", "family", "model_size"])["accuracy"]
        .mean()
        .reset_index()
    )
    auc_mean = auc_table[auc_table["language_code"] == "mean"]
    route_50 = routes[
        (routes["budget"] == 0.5)
        & (routes["policy"].isin(["confidence", "random", "oracle"]))
    ]

    lines = [
        "# Paper Analysis Summary",
        "",
        "All analyses use the canonical corrected result archive.",
        "Gemma 3 4B MMLU-ProX-Lite and SIB-200 predictions are the float32 reruns.",
        "",
        "## Mean model accuracy across four languages",
        "",
        markdown_table(model_mean.round(4)),
        "",
        "## Correction-detection AUC",
        "",
        markdown_table(
            auc_mean[
                ["dataset", "family", "correction_detection_auc"]
            ].round(4)
        ),
        "",
        "## Routing accuracy at 50% escalation",
        "",
        markdown_table(
            route_50[
                ["dataset", "family", "policy", "accuracy"]
            ].round(4)
        ),
        "",
        "## Bootstrap comparisons",
        "",
        markdown_table(bootstrap.round(4)),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_arguments()
    table_dir = args.output / "tables"
    figure_dir = args.output / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading canonical archive: {args.archive}")
    raw = load_raw_results(args.archive)
    pairs = make_pairs(raw)
    print("Validated 72 configurations and created small/large model pairs")

    model_table = model_accuracy_table(raw)
    curve_budgets = np.round(np.linspace(0, 1, 11), 2).tolist()
    routes, language_routes = routing_tables(
        pairs,
        curve_budgets,
        args.random_repeats,
        args.seed,
    )
    requested_routes, requested_language = routing_tables(
        pairs,
        [0.25, 0.5],
        args.random_repeats,
        args.seed,
    )
    auc_table = correction_auc_table(pairs)
    quintiles = mmlu_quintile_table(pairs)
    bootstrap = bootstrap_table(
        pairs,
        [0.25, 0.5],
        args.bootstrap_repeats,
        args.random_repeats,
        args.seed,
    )

    save_table(model_table, table_dir / "model_accuracy_by_language.csv")
    save_table(routes, table_dir / "routing_curves.csv")
    save_table(language_routes, table_dir / "routing_curves_by_language.csv")
    save_table(requested_routes, table_dir / "routing_at_25_50.csv")
    save_table(requested_language, table_dir / "routing_at_25_50_by_language.csv")
    save_table(
        requested_routes[
            [
                "dataset",
                "family",
                "policy",
                "budget",
                "worst_language_accuracy",
                "language_accuracy_gap",
                "escalation_gap",
            ]
        ],
        table_dir / "language_disparity_at_25_50.csv",
    )
    save_table(auc_table, table_dir / "correction_detection_auc.csv")
    save_table(quintiles, table_dir / "mmlu_confidence_quintiles.csv")
    save_table(bootstrap, table_dir / "bootstrap_comparisons.csv")

    plot_accuracy_curves(routes, figure_dir)
    plot_auc(auc_table, figure_dir)
    plot_mmlu_quintiles(quintiles, figure_dir)
    write_summary(
        args.output / "paper_analysis_summary.md",
        model_table,
        requested_routes,
        auc_table,
        bootstrap,
    )

    print(f"Saved paper analysis to: {args.output}")


if __name__ == "__main__":
    main()
