import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLES = ROOT / "results" / "paper_analysis" / "tables"
DEFAULT_FIGURES = ROOT / "results" / "paper_analysis" / "figures"

DATASETS = ("belebele", "mmlu_prox_lite", "sib200")
FAMILIES = ("qwen", "gemma", "llama")
LANGUAGES = ("eng_Latn", "ben_Beng", "hin_Deva", "urd_Arab")
POLICIES = ("confidence", "equal_quota", "random", "oracle")

DATASET_NAMES = {
    "belebele": "Belebele",
    "mmlu_prox_lite": "MMLU-ProX-Lite",
    "sib200": "SIB-200",
}
FAMILY_NAMES = {"qwen": "Qwen3", "gemma": "Gemma 3", "llama": "Llama 3.2"}
LANGUAGE_NAMES = {
    "eng_Latn": "English",
    "ben_Beng": "Bangla",
    "hin_Deva": "Hindi",
    "urd_Arab": "Urdu",
}
POLICY_NAMES = {
    "confidence": "Confidence",
    "equal_quota": "Equal quota",
    "random": "Random",
    "oracle": "Oracle",
}
FAMILY_COLORS = {"qwen": "#0072B2", "gemma": "#E69F00", "llama": "#009E73"}
POLICY_COLORS = {
    "confidence": "#0072B2",
    "equal_quota": "#009E73",
    "random": "#7F7F7F",
    "oracle": "#D55E00",
}
SIZE_COLORS = {"small": "#56B4E9", "large": "#CC79A7"}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURES)
    return parser.parse_args()


def read_table(table_dir, name):
    path = table_dir / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing analysis table: {path}\nRun python analysis/analyze_results.py first."
        )
    return pd.read_csv(path)


def save_figure(fig, output_dir, name):
    pdf_path = output_dir / f"{name}.pdf"
    png_path = output_dir / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")

    for attempt in range(3):
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        ending = png_path.read_bytes()[-12:] if png_path.exists() else b""
        if ending == b"\x00\x00\x00\x00IEND\xaeB`\x82":
            break
        if attempt == 2:
            raise OSError(f"Could not write a complete PNG file: {png_path}")

    plt.close(fig)


def set_style():
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "legend.title_fontsize": 18,
            "figure.titlesize": 18,
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_model_accuracy_overall(table, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), sharey=True)
    x = np.arange(len(FAMILIES))
    width = 0.34

    for ax, dataset in zip(axes, DATASETS):
        selected = table[table["dataset"] == dataset]
        for offset, size in ((-width / 2, "small"), (width / 2, "large")):
            values = [
                selected[
                    (selected["family"] == family)
                    & (selected["model_size"] == size)
                ]["mean_language_accuracy"].iloc[0]
                for family in FAMILIES
            ]
            ax.bar(
                x + offset,
                values,
                width,
                color=SIZE_COLORS[size],
                label=size.title(),
            )
        ax.set_title(DATASET_NAMES[dataset])
        ax.set_xticks(x, [FAMILY_NAMES[family] for family in FAMILIES])
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean accuracy across languages")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir, "01_model_accuracy_overall")


def plot_language_accuracy(table, output_dir):
    fig, axes = plt.subplots(3, 3, figsize=(13, 9.5), sharey=True)
    x = np.arange(len(LANGUAGES))
    width = 0.34

    for row, dataset in enumerate(DATASETS):
        for column, family in enumerate(FAMILIES):
            ax = axes[row, column]
            selected = table[
                (table["dataset"] == dataset) & (table["family"] == family)
            ]
            for offset, size in ((-width / 2, "small"), (width / 2, "large")):
                values = [
                    selected[
                        (selected["language_code"] == language)
                        & (selected["model_size"] == size)
                    ]["accuracy"].iloc[0]
                    for language in LANGUAGES
                ]
                ax.bar(
                    x + offset,
                    values,
                    width,
                    color=SIZE_COLORS[size],
                    label=size.title(),
                )
            if row == 0:
                ax.set_title(FAMILY_NAMES[family])
            if column == 0:
                ax.set_ylabel(f"{DATASET_NAMES[dataset]}\nAccuracy")
            ax.set_xticks(x, [LANGUAGE_NAMES[x] for x in LANGUAGES], rotation=25)
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.25)
    axes[0, -1].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir, "02_model_accuracy_by_language")


def plot_small_large_scatter(table, output_dir):
    pivot = table.pivot_table(
        index=["dataset", "family", "language_code"],
        columns="model_size",
        values="accuracy",
    ).reset_index()
    markers = {
        "eng_Latn": "o",
        "ben_Beng": "s",
        "hin_Deva": "^",
        "urd_Arab": "D",
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharex=True, sharey=True)

    for ax, dataset in zip(axes, DATASETS):
        selected = pivot[pivot["dataset"] == dataset]
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
        for family in FAMILIES:
            for language in LANGUAGES:
                row = selected[
                    (selected["family"] == family)
                    & (selected["language_code"] == language)
                ].iloc[0]
                ax.scatter(
                    row["small"],
                    row["large"],
                    color=FAMILY_COLORS[family],
                    marker=markers[language],
                    s=45,
                    alpha=0.85,
                )
        ax.set_title(DATASET_NAMES[dataset])
        ax.set_xlabel("Small-model accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Large-model accuracy")

    family_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=FAMILY_COLORS[f], label=FAMILY_NAMES[f])
        for f in FAMILIES
    ]
    language_handles = [
        plt.Line2D([0], [0], marker=markers[l], linestyle="", color="black", label=LANGUAGE_NAMES[l])
        for l in LANGUAGES
    ]
    fig.legend(
        handles=family_handles + language_handles,
        loc="upper center",
        ncol=7,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_figure(fig, output_dir, "03_small_vs_large_accuracy")


def plot_accuracy_escalation(table, output_dir):
    dataset_colors = {
        "belebele": "#4C78A8",
        "mmlu_prox_lite": "#F58518",
        "sib200": "#54A24B",
    }

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(13, 8),
        sharex=True,
    )

    for row, dataset in enumerate(DATASETS):
        dataset_color = dataset_colors[dataset]

        for column, family in enumerate(FAMILIES):
            ax = axes[row, column]

            ax.set_box_aspect(6 / 10)

            ax.set_facecolor(
                to_rgba(dataset_color, alpha=0.07)
            )

            for spine in ax.spines.values():
                spine.set_color(dataset_color)
                spine.set_linewidth(1.2)

            selected = table[
                (table["dataset"] == dataset)
                & (table["family"] == family)
            ]

            for policy in POLICIES:
                line = selected[
                    selected["policy"] == policy
                ].sort_values("budget")

                ax.plot(
                    line["escalation_rate"],
                    line["accuracy"],
                    marker="o",
                    markersize=4,
                    linewidth=2.5,
                    color=POLICY_COLORS[policy],
                    label=POLICY_NAMES[policy],
                )

            if row == 0:
                ax.set_title(
                    FAMILY_NAMES[family],
                    fontsize=18,
                    pad=6,
                )

            ax.set_xlim(-0.02, 1.02)

            
            ax.xaxis.set_major_formatter(
                FormatStrFormatter("%.1f")
            )

            ax.yaxis.set_major_formatter(
                FormatStrFormatter("%.2f")
            )

            ax.tick_params(
                axis="both",
                labelsize=14,
                pad=3,
            )

            ax.grid(
                alpha=0.25,
                color="#888888",
            )

    
    fig.supxlabel(
        "Escalation rate",
        fontsize=18,
        x=0.535,
        y=0.035,
    )

    fig.supylabel(
        "Accuracy",
        fontsize=18,
        x=0.025,
        y=0.45,
    )

    policy_handles, policy_labels = (
        axes[0, 0].get_legend_handles_labels()
    )

    policy_legend = fig.legend(
        policy_handles,
        policy_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=False,
        fontsize=18,
        columnspacing=1.8,
        handletextpad=0.7,
    )

    fig.add_artist(policy_legend)

    dataset_handles = [
        Patch(
            facecolor=to_rgba(
                dataset_colors[dataset],
                alpha=0.15,
            ),
            edgecolor=dataset_colors[dataset],
            linewidth=1.5,
            label=DATASET_NAMES[dataset],
        )
        for dataset in DATASETS
    ]

    fig.legend(
        handles=dataset_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        frameon=False,
        fontsize=18,
        columnspacing=2.2,
        handletextpad=0.7,
    )

   
    fig.subplots_adjust(
        left=0.083,
        right=0.985,
        bottom=0.11,
        top=0.80,
        wspace=0.10,
        hspace=0.15,
    )

    save_figure(
        fig,
        output_dir,
        "04_accuracy_escalation_curves",
    )


def policy_difference(table, first, second, value="accuracy"):
    left = table[table["policy"] == first][
        ["dataset", "family", "budget", value]
    ].rename(columns={value: "first"})
    right = table[table["policy"] == second][
        ["dataset", "family", "budget", value]
    ].rename(columns={value: "second"})
    result = left.merge(right, on=["dataset", "family", "budget"], validate="one_to_one")
    result["difference"] = result["first"] - result["second"]
    return result


def plot_policy_difference(table, output_dir, first, second, name, ylabel):
    difference = policy_difference(table, first, second)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        selected = difference[difference["dataset"] == dataset]
        for family in FAMILIES:
            line = selected[selected["family"] == family].sort_values("budget")
            ax.plot(
                line["budget"],
                line["difference"],
                marker="o",
                color=FAMILY_COLORS[family],
                label=FAMILY_NAMES[family],
            )
        ax.axhline(0, linestyle="--", color="black", linewidth=1)
        ax.set_title(DATASET_NAMES[dataset])
        ax.set_xlabel("Escalation budget")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir, name)


def plot_correction_auc(table, output_dir):
    mean = table[table["language_code"] == "mean"]
    detail = table[table["language_code"] != "mean"]
    x = np.arange(len(DATASETS))
    width = 0.23
    fig, ax = plt.subplots(figsize=(8.2, 4.7))

    for index, family in enumerate(FAMILIES):
        selected = mean[mean["family"] == family].set_index("dataset")
        values = [selected.loc[d, "correction_detection_auc"] for d in DATASETS]
        errors = [selected.loc[d, "auc_std_across_languages"] for d in DATASETS]
        positions = x + (index - 1) * width
        ax.bar(
            positions,
            values,
            width,
            yerr=errors,
            capsize=3,
            color=FAMILY_COLORS[family],
            alpha=0.82,
            label=FAMILY_NAMES[family],
        )
        for dataset_index, dataset in enumerate(DATASETS):
            points = detail[
                (detail["dataset"] == dataset) & (detail["family"] == family)
            ]["correction_detection_auc"].to_numpy()
            jitter = np.linspace(-0.035, 0.035, len(points))
            ax.scatter(
                np.full(len(points), positions[dataset_index]) + jitter,
                points,
                color="black",
                s=14,
                alpha=0.6,
                zorder=3,
            )
    ax.axhline(0.5, linestyle="--", color="black", linewidth=1, label="Chance")
    ax.set_xticks(x, [DATASET_NAMES[d] for d in DATASETS])
    ax.set_ylabel("Correction-detection AUC")
    ax.set_ylim(0.35, 0.85)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, output_dir, "07_correction_detection_auc")


def plot_confidence_error_curves(table, output_dir):
    mean = (
        table.groupby(["dataset", "family", "confidence_bin"], as_index=False)
        .agg(
            small_error_rate=("small_error_rate", "mean"),
            correction_rate_given_error=("correction_rate_given_error", "mean"),
        )
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        selected = mean[mean["dataset"] == dataset]
        for family in FAMILIES:
            line = selected[selected["family"] == family].sort_values("confidence_bin")
            ax.plot(
                line["confidence_bin"],
                line["small_error_rate"],
                marker="o",
                color=FAMILY_COLORS[family],
                label=FAMILY_NAMES[family],
            )
        ax.set_title(DATASET_NAMES[dataset])
        ax.set_xlabel("Confidence decile (1 low, 10 high)")
        ax.set_xticks([1, 3, 5, 7, 10])
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Small-model error rate")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir, "08_error_rate_by_confidence_decile")


def plot_mmlu_quintiles(table, output_dir):
    mean = (
        table.groupby(["family", "confidence_quintile"], as_index=False)
        .agg(
            small_error_rate=("small_error_rate", "mean"),
            correction_rate_given_error=("correction_rate_given_error", "mean"),
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    metrics = [
        ("small_error_rate", "Small-model error rate"),
        ("correction_rate_given_error", "Large corrects small-model error"),
    ]
    for ax, (metric, label) in zip(axes, metrics):
        for family in FAMILIES:
            line = mean[mean["family"] == family].sort_values("confidence_quintile")
            ax.plot(
                line["confidence_quintile"],
                line[metric],
                marker="o",
                linewidth=2,
                color=FAMILY_COLORS[family],
                label=FAMILY_NAMES[family],
            )
        ax.set_xlabel("Confidence quintile (Q1 low, Q5 high)")
        ax.set_ylabel(label)
        ax.set_xticks(range(1, 6))
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir, "09_mmlu_confidence_quintiles")


def plot_high_confidence_errors(table, output_dir):
    mean = (
        table[table["threshold"] == 0.9]
        .groupby(["dataset", "family"], as_index=False)
        .agg(
            error_rate=("error_rate_among_high_confidence", "mean"),
            high_confidence_share=("high_confidence_share", "mean"),
        )
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    x = np.arange(len(FAMILIES))
    for ax, dataset in zip(axes, DATASETS):
        selected = mean[mean["dataset"] == dataset].set_index("family")
        values = [selected.loc[f, "error_rate"] for f in FAMILIES]
        ax.bar(x, values, color=[FAMILY_COLORS[f] for f in FAMILIES])
        ax.set_xticks(x, [FAMILY_NAMES[f] for f in FAMILIES])
        ax.set_title(DATASET_NAMES[dataset])
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Error rate when margin ≥ 0.90")
    fig.tight_layout()
    save_figure(fig, output_dir, "10_high_confidence_error_rate")


def plot_calibration(table, output_dir):
    mean = (
        table.groupby(["dataset", "family", "model_size"], as_index=False)
        .agg(ece_10bin=("ece_10bin", "mean"))
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    x = np.arange(len(FAMILIES))
    width = 0.34
    for ax, dataset in zip(axes, DATASETS):
        selected = mean[mean["dataset"] == dataset]
        for offset, size in ((-width / 2, "small"), (width / 2, "large")):
            values = [
                selected[
                    (selected["family"] == family)
                    & (selected["model_size"] == size)
                ]["ece_10bin"].iloc[0]
                for family in FAMILIES
            ]
            ax.bar(x + offset, values, width, color=SIZE_COLORS[size], label=size.title())
        ax.set_xticks(x, [FAMILY_NAMES[f] for f in FAMILIES])
        ax.set_title(DATASET_NAMES[dataset])
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("ECE (10 bins; lower is better)")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir, "11_calibration_ece")


def plot_transitions(table, output_dir):
    mean = (
        table.groupby(["dataset", "family", "transition"], as_index=False)
        .agg(rate=("rate", "mean"))
    )
    transitions = (
        "small_correct_large_correct",
        "small_wrong_large_correct",
        "small_correct_large_wrong",
        "small_wrong_large_wrong",
    )
    names = ("Both correct", "Large fixes", "Large breaks", "Both wrong")
    colors = ("#009E73", "#56B4E9", "#E69F00", "#D55E00")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)
    x = np.arange(len(FAMILIES))

    for ax, dataset in zip(axes, DATASETS):
        selected = mean[mean["dataset"] == dataset]
        bottom = np.zeros(len(FAMILIES))
        for transition, label, color in zip(transitions, names, colors):
            values = [
                selected[
                    (selected["family"] == family)
                    & (selected["transition"] == transition)
                ]["rate"].iloc[0]
                for family in FAMILIES
            ]
            ax.bar(x, values, bottom=bottom, color=color, label=label)
            bottom += np.asarray(values)
        ax.set_xticks(x, [FAMILY_NAMES[f] for f in FAMILIES])
        ax.set_title(DATASET_NAMES[dataset])
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Fraction of examples")
    axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save_figure(fig, output_dir, "12_small_large_correction_transitions")


def plot_disparity_heatmap(table, output_dir, metric, name, title, budget=0.5):
    selected = table[table["budget"] == budget].copy()
    rows = [(dataset, family) for dataset in DATASETS for family in FAMILIES]
    matrix = np.array(
        [
            [
                selected[
                    (selected["dataset"] == dataset)
                    & (selected["family"] == family)
                    & (selected["policy"] == policy)
                ][metric].iloc[0]
                for policy in POLICIES
            ]
            for dataset, family in rows
        ]
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(POLICIES)), [POLICY_NAMES[p] for p in POLICIES], rotation=20)
    ax.set_yticks(
        range(len(rows)),
        [f"{DATASET_NAMES[d]} — {FAMILY_NAMES[f]}" for d, f in rows],
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"{title} at {int(budget * 100)}% escalation")
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    save_figure(fig, output_dir, name)


def plot_policy_bars(table, output_dir, metric, name, ylabel, budget=0.5):
    selected = table[table["budget"] == budget]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    x = np.arange(len(FAMILIES))
    width = 0.19

    for ax, dataset in zip(axes, DATASETS):
        local = selected[selected["dataset"] == dataset]
        for index, policy in enumerate(POLICIES):
            values = [
                local[
                    (local["family"] == family) & (local["policy"] == policy)
                ][metric].iloc[0]
                for family in FAMILIES
            ]
            ax.bar(
                x + (index - 1.5) * width,
                values,
                width,
                color=POLICY_COLORS[policy],
                label=POLICY_NAMES[policy],
            )
        ax.set_xticks(x, [FAMILY_NAMES[f] for f in FAMILIES])
        ax.set_title(DATASET_NAMES[dataset])
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save_figure(fig, output_dir, name)

def plot_intro_teaser(auc_table, routes, output_dir):
    auc = auc_table[
        auc_table["language_code"] == "mean"
    ].copy()

    selected_routes = routes[
        (routes["budget"] == 0.5)
        & (routes["policy"].isin(["confidence", "random"]))
    ]

    routing = selected_routes.pivot_table(
        index=["dataset", "family"],
        columns="policy",
        values="accuracy",
    ).reset_index()

    routing["gain_pp"] = (
        routing["confidence"] - routing["random"]
    ) * 100

    x = np.arange(len(DATASETS))

    offsets = {
        "qwen": -0.13,
        "gemma": 0.0,
        "llama": 0.13,
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6.5),
        sharex=True,
    )

    for family in FAMILIES:
        family_auc = auc[
            auc["family"] == family
        ].set_index("dataset")

        family_gain = routing[
            routing["family"] == family
        ].set_index("dataset")

        axes[0].scatter(
            x + offsets[family],
            [
                family_auc.loc[
                    dataset,
                    "correction_detection_auc",
                ]
                for dataset in DATASETS
            ],
            s=170,
            color=FAMILY_COLORS[family],
            label=FAMILY_NAMES[family],
            zorder=3,
        )

        axes[1].scatter(
            x + offsets[family],
            [
                family_gain.loc[dataset, "gain_pp"]
                for dataset in DATASETS
            ],
            s=170,
            color=FAMILY_COLORS[family],
            zorder=3,
        )

    auc_mean = auc.groupby(
        "dataset"
    )["correction_detection_auc"].mean()

    gain_mean = routing.groupby(
        "dataset"
    )["gain_pp"].mean()

    axes[0].plot(
        x,
        [auc_mean.loc[dataset] for dataset in DATASETS],
        marker="D",
        markersize=13,
        linewidth=2.2,
        color="black",
        label="Mean",
        zorder=4,
    )

    axes[1].plot(
        x,
        [gain_mean.loc[dataset] for dataset in DATASETS],
        marker="D",
        markersize=13,
        linewidth=2.2,
        color="black",
        zorder=4,
    )

    for index, dataset in enumerate(DATASETS):
        axes[0].annotate(
            f"{auc_mean.loc[dataset]:.2f}",
            (x[index], auc_mean.loc[dataset]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=22,
        )

        axes[1].annotate(
            f"{gain_mean.loc[dataset]:+.1f}",
            (x[index], gain_mean.loc[dataset]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=22,
        )

    axes[0].axhline(
        0.5,
        linestyle="--",
        color="#666666",
        linewidth=1,
    )

    axes[1].axhline(
        0,
        linestyle="--",
        color="#666666",
        linewidth=1,
    )

    axes[0].set_ylabel(
        "Correction-detection AUC",
        fontsize=24,
    )

    axes[1].set_ylabel(
        "Gain over random (points)\nat 50% escalation",
        fontsize=24,
    )

    axes[0].set_ylim(0.45, 0.84)
    axes[1].set_ylim(-2, 7)

    dataset_labels = [
        DATASET_NAMES[dataset]
        for dataset in DATASETS
    ]

    for ax in axes:
        ax.set_xticks(
            x,
            dataset_labels,
            rotation=12,
        )

        ax.tick_params(
            axis="both",
            labelsize=22,
        )

        ax.grid(
            axis="y",
            alpha=0.2,
        )

    for label, ax in zip(("(a)", "(b)"), axes):
        ax.text(
            0.01,
            0.96,
            label,
            transform=ax.transAxes,
            va="top",
            fontweight="bold",
            fontsize=22,
        )

    handles, labels = axes[0].get_legend_handles_labels()

    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=22,
        markerscale=1.15,
    )

    for text in legend.get_texts():
        if text.get_text() in {
            FAMILY_NAMES["qwen"],
            FAMILY_NAMES["gemma"],
            FAMILY_NAMES["llama"],
        }:
            text.set_fontweight("bold")

    fig.tight_layout(
        rect=(0, 0, 1, 0.84),
        w_pad=2.5,
    )

    save_figure(
        fig,
        output_dir,
        "00_introduction_teaser",
    )
    
def main():
    args = parse_arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    set_style()

    model_language = read_table(args.tables, "model_accuracy_by_language.csv")
    model_overall = read_table(args.tables, "model_accuracy_overall.csv")
    routes = read_table(args.tables, "routing_curves.csv")
    auc = read_table(args.tables, "correction_detection_auc.csv")
    confidence_bins = read_table(args.tables, "confidence_bins.csv")
    mmlu_quintiles = read_table(args.tables, "mmlu_confidence_quintiles.csv")
    high_confidence = read_table(args.tables, "high_confidence_errors.csv")
    calibration = read_table(args.tables, "calibration_summary.csv")
    transitions = read_table(args.tables, "correction_transitions.csv")
    disparity = read_table(args.tables, "language_disparity_at_25_50.csv")

    plot_intro_teaser(auc, routes, args.output)
    plot_model_accuracy_overall(model_overall, args.output)
    plot_language_accuracy(model_language, args.output)
    plot_small_large_scatter(model_language, args.output)
    plot_accuracy_escalation(routes, args.output)
    plot_policy_difference(
        routes,
        args.output,
        "confidence",
        "random",
        "05_confidence_gain_over_random",
        "Accuracy: confidence − random",
    )
    plot_policy_difference(
        routes,
        args.output,
        "oracle",
        "confidence",
        "06_oracle_headroom",
        "Accuracy: oracle − confidence",
    )
    plot_correction_auc(auc, args.output)
    plot_confidence_error_curves(confidence_bins, args.output)
    plot_mmlu_quintiles(mmlu_quintiles, args.output)
    plot_high_confidence_errors(high_confidence, args.output)
    plot_calibration(calibration, args.output)
    plot_transitions(transitions, args.output)
    plot_disparity_heatmap(
        disparity,
        args.output,
        "escalation_gap",
        "13_language_escalation_gap",
        "Language escalation-rate gap",
    )
    plot_policy_bars(
        disparity,
        args.output,
        "worst_language_accuracy",
        "14_worst_language_accuracy",
        "Worst-language accuracy",
    )
    plot_policy_bars(
        disparity,
        args.output,
        "language_accuracy_gap",
        "15_language_accuracy_gap",
        "Best–worst language accuracy gap",
    )

    print(f"Saved 16 figures as PDF and PNG to: {args.output}")


if __name__ == "__main__":
    main()
