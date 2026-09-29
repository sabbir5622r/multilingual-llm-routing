import argparse
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "results" / "canonical" / "multilingual_routing_corrected_full_results.zip"
DEFAULT_OUTPUT = ROOT / "results" / "paper_analysis" / "tables"

DATASETS = ("belebele", "mmlu_prox_lite", "sib200")
FAMILIES = ("qwen", "gemma", "llama")
LANGUAGES = ("eng_Latn", "ben_Beng", "hin_Deva", "urd_Arab")
EXPECTED_ROWS = {"belebele": 900, "mmlu_prox_lite": 658, "sib200": 204}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--random-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_results(archive_path):
    if not archive_path.is_file():
        raise FileNotFoundError(f"Canonical archive not found: {archive_path}")

    frames = []
    with ZipFile(archive_path) as archive:
        broken = archive.testzip()
        if broken is not None:
            raise ValueError(f"Corrupted archive member: {broken}")

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
    validate_results(raw)
    return raw


def validate_results(raw):
    keys = ["dataset", "family", "model_size", "language_code"]
    groups = raw.groupby(keys, sort=False)

    if groups.ngroups != 72:
        raise ValueError(f"Found {groups.ngroups} configurations; expected 72")

    for key, frame in groups:
        dataset, family, size, language = key

        if dataset not in DATASETS or family not in FAMILIES:
            raise ValueError(f"Unexpected configuration: {key}")
        if size not in ("small", "large") or language not in LANGUAGES:
            raise ValueError(f"Unexpected configuration: {key}")

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
        keys + ["is_correct", "confidence_margin", "max_choice_probability"]
    ].rename(
        columns={
            "is_correct": "large_correct",
            "confidence_margin": "large_margin",
            "max_choice_probability": "large_probability",
        }
    )

    pairs = small.merge(large, on=keys, how="inner", validate="one_to_one")
    expected = sum(EXPECTED_ROWS.values()) * len(FAMILIES) * len(LANGUAGES)
    if len(pairs) != expected:
        raise ValueError(f"Found {len(pairs)} model pairs; expected {expected}")

    pairs["small_correct"] = pairs["small_correct"].astype(bool)
    pairs["large_correct"] = pairs["large_correct"].astype(bool)
    pairs["correction"] = ~pairs["small_correct"] & pairs["large_correct"]
    pairs["regression"] = pairs["small_correct"] & ~pairs["large_correct"]
    pairs["gain"] = pairs["large_correct"].astype(int) - pairs["small_correct"].astype(int)
    return pairs


def model_tables(raw):
    language = (
        raw.groupby(
            ["dataset", "family", "model_size", "language_code"],
            as_index=False,
            sort=False,
        )
        .agg(
            examples=("is_correct", "size"),
            accuracy=("is_correct", "mean"),
            mean_confidence=("max_choice_probability", "mean"),
            mean_margin=("confidence_margin", "mean"),
        )
    )

    overall = (
        language.groupby(["dataset", "family", "model_size"], as_index=False)
        .agg(
            examples=("examples", "sum"),
            mean_language_accuracy=("accuracy", "mean"),
            worst_language_accuracy=("accuracy", "min"),
            best_language_accuracy=("accuracy", "max"),
            mean_confidence=("mean_confidence", "mean"),
            mean_margin=("mean_margin", "mean"),
        )
    )
    overall["language_accuracy_gap"] = (
        overall["best_language_accuracy"] - overall["worst_language_accuracy"]
    )
    return language, overall


def expected_calibration_error(confidence, correct, bins=10):
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0

    for low, high in zip(edges[:-1], edges[1:]):
        if high == 1:
            selected = (confidence >= low) & (confidence <= high)
        else:
            selected = (confidence >= low) & (confidence < high)
        if selected.any():
            value += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return value


def calibration_table(raw):
    rows = []
    keys = ["dataset", "family", "model_size", "language_code"]
    for key, frame in raw.groupby(keys, sort=False):
        rows.append(
            {
                "dataset": key[0],
                "family": key[1],
                "model_size": key[2],
                "language_code": key[3],
                "examples": len(frame),
                "accuracy": frame["is_correct"].mean(),
                "mean_confidence": frame["max_choice_probability"].mean(),
                "confidence_bias": frame["max_choice_probability"].mean() - frame["is_correct"].mean(),
                "ece_10bin": expected_calibration_error(
                    frame["max_choice_probability"], frame["is_correct"], 10
                ),
            }
        )
    return pd.DataFrame(rows)


def choose_lowest(values, count):
    mask = np.zeros(len(values), dtype=bool)
    if count:
        order = np.argsort(np.asarray(values), kind="stable")
        mask[order[:count]] = True
    return mask


def policy_mask(frame, policy, budget):
    count = int(round(len(frame) * budget))

    if policy == "confidence":
        return choose_lowest(frame["small_margin"].to_numpy(), count)

    if policy == "oracle":
        mask = np.zeros(len(frame), dtype=bool)
        if count:
            order = np.argsort(-frame["gain"].to_numpy(), kind="stable")
            mask[order[:count]] = True
        return mask

    if policy == "equal_quota":
        mask = np.zeros(len(frame), dtype=bool)
        for indices in frame.groupby("language_code", sort=False).indices.values():
            positions = np.asarray(indices)
            local_count = int(round(len(positions) * budget))
            local = choose_lowest(
                frame.iloc[positions]["small_margin"].to_numpy(), local_count
            )
            mask[positions[local]] = True
        return mask

    raise ValueError(f"Unknown policy: {policy}")


def route_outcomes(frame, mask):
    return np.where(
        mask,
        frame["large_correct"].to_numpy(float),
        frame["small_correct"].to_numpy(float),
    )


def random_policy(frame, budget, repeats, seed):
    count = int(round(len(frame) * budget))
    escalations = np.zeros((repeats, len(frame)), dtype=float)
    outcomes = np.zeros((repeats, len(frame)), dtype=float)

    for repeat in range(repeats):
        rng = np.random.default_rng(np.random.SeedSequence([seed, repeat]))
        mask = np.zeros(len(frame), dtype=bool)
        if count:
            mask[rng.choice(len(frame), size=count, replace=False)] = True
        escalations[repeat] = mask
        outcomes[repeat] = route_outcomes(frame, mask)

    return escalations.mean(axis=0), outcomes.mean(axis=0)


def get_policy_output(frame, policy, budget, repeats, seed):
    if policy == "random":
        return random_policy(frame, budget, repeats, seed)
    mask = policy_mask(frame, policy, budget)
    return mask.astype(float), route_outcomes(frame, mask)


def summarize_route(frame, escalation, outcome):
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
        "escalation_rate": escalation.mean(),
        "accuracy": outcome.mean(),
        "worst_language_accuracy": language["accuracy"].min(),
        "best_language_accuracy": language["accuracy"].max(),
        "language_accuracy_gap": language["accuracy"].max() - language["accuracy"].min(),
        "escalation_gap": language["escalation_rate"].max() - language["escalation_rate"].min(),
    }


def routing_tables(pairs, budgets, repeats, seed):
    policies = ("confidence", "equal_quota", "random", "oracle")
    overall_rows = []
    language_rows = []

    for dataset_index, dataset in enumerate(DATASETS):
        for family_index, family in enumerate(FAMILIES):
            frame = pairs[
                (pairs["dataset"] == dataset) & (pairs["family"] == family)
            ].reset_index(drop=True)

            for budget_index, budget in enumerate(budgets):
                for policy_index, policy in enumerate(policies):
                    local_seed = seed + dataset_index * 10000 + family_index * 1000 + budget_index * 100 + policy_index
                    escalation, outcome = get_policy_output(
                        frame, policy, budget, repeats, local_seed
                    )
                    row = summarize_route(frame, escalation, outcome)
                    row.update(
                        {
                            "dataset": dataset,
                            "family": family,
                            "policy": policy,
                            "budget": budget,
                        }
                    )
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


def correction_table(pairs):
    rows = []
    keys = ["dataset", "family", "language_code"]
    for key, frame in pairs.groupby(keys, sort=False):
        states = {
            "small_correct_large_correct": frame["small_correct"] & frame["large_correct"],
            "small_wrong_large_correct": ~frame["small_correct"] & frame["large_correct"],
            "small_correct_large_wrong": frame["small_correct"] & ~frame["large_correct"],
            "small_wrong_large_wrong": ~frame["small_correct"] & ~frame["large_correct"],
        }
        for transition, selected in states.items():
            rows.append(
                {
                    "dataset": key[0],
                    "family": key[1],
                    "language_code": key[2],
                    "transition": transition,
                    "count": int(selected.sum()),
                    "rate": selected.mean(),
                }
            )
    return pd.DataFrame(rows)


def correction_auc_table(pairs):
    rows = []
    for key, frame in pairs.groupby(
        ["dataset", "family", "language_code"], sort=False
    ):
        target = frame["correction"].astype(int).to_numpy()
        auc = (
            roc_auc_score(target, -frame["small_margin"].to_numpy())
            if np.unique(target).size == 2
            else np.nan
        )
        rows.append(
            {
                "dataset": key[0],
                "family": key[1],
                "language_code": key[2],
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


def confidence_bin_table(pairs, bins=10):
    frame = pairs.copy()
    frame["confidence_rank"] = frame.groupby(
        ["dataset", "family", "language_code"], sort=False
    )["small_margin"].rank(method="first", pct=True)
    frame["confidence_bin"] = np.ceil(frame["confidence_rank"] * bins).clip(1, bins).astype(int)

    rows = []
    keys = ["dataset", "family", "language_code", "confidence_bin"]
    for key, group in frame.groupby(keys, sort=False):
        errors = ~group["small_correct"]
        rows.append(
            {
                "dataset": key[0],
                "family": key[1],
                "language_code": key[2],
                "confidence_bin": key[3],
                "examples": len(group),
                "mean_confidence_margin": group["small_margin"].mean(),
                "small_accuracy": group["small_correct"].mean(),
                "small_error_rate": errors.mean(),
                "large_accuracy": group["large_correct"].mean(),
                "net_large_gain": group["gain"].mean(),
                "correction_rate_all": group["correction"].mean(),
                "correction_rate_given_error": (
                    group.loc[errors, "large_correct"].mean() if errors.any() else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def mmlu_quintile_table(pairs):
    frame = pairs[pairs["dataset"] == "mmlu_prox_lite"].copy()
    frame["confidence_rank"] = frame.groupby(
        ["family", "language_code"], sort=False
    )["small_margin"].rank(method="first", pct=True)
    frame["confidence_quintile"] = np.ceil(frame["confidence_rank"] * 5).clip(1, 5).astype(int)

    rows = []
    keys = ["family", "language_code", "confidence_quintile"]
    for key, group in frame.groupby(keys, sort=False):
        errors = ~group["small_correct"]
        rows.append(
            {
                "family": key[0],
                "language_code": key[1],
                "confidence_quintile": key[2],
                "examples": len(group),
                "mean_confidence_margin": group["small_margin"].mean(),
                "small_accuracy": group["small_correct"].mean(),
                "small_error_rate": errors.mean(),
                "large_accuracy": group["large_correct"].mean(),
                "net_large_gain": group["gain"].mean(),
                "correction_rate_all": group["correction"].mean(),
                "correction_rate_given_error": (
                    group.loc[errors, "large_correct"].mean() if errors.any() else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def high_confidence_error_table(pairs):
    rows = []
    for key, frame in pairs.groupby(
        ["dataset", "family", "language_code"], sort=False
    ):
        for threshold in (0.8, 0.9, 0.95):
            selected = frame["small_margin"] >= threshold
            high = frame[selected]
            errors = ~high["small_correct"]
            rows.append(
                {
                    "dataset": key[0],
                    "family": key[1],
                    "language_code": key[2],
                    "threshold": threshold,
                    "examples": len(frame),
                    "high_confidence_examples": int(selected.sum()),
                    "high_confidence_share": selected.mean(),
                    "error_rate_among_high_confidence": (
                        errors.mean() if len(high) else np.nan
                    ),
                    "correction_rate_among_high_confidence_errors": (
                        high.loc[errors, "large_correct"].mean() if errors.any() else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def cluster_bootstrap(frame, first, second, repeats, seed):
    difference = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    clustered = (
        pd.DataFrame(
            {"example_id": frame["example_id"].to_numpy(), "difference": difference}
        )
        .groupby("example_id", sort=False)["difference"]
        .mean()
        .to_numpy()
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats)
    for repeat in range(repeats):
        sample = rng.integers(0, len(clustered), size=len(clustered))
        estimates[repeat] = clustered[sample].mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return clustered.mean(), low, high


def bootstrap_table(pairs, budgets, bootstrap_repeats, random_repeats, seed):
    rows = []
    test_index = 0
    for dataset in DATASETS:
        for family in FAMILIES:
            frame = pairs[
                (pairs["dataset"] == dataset) & (pairs["family"] == family)
            ].reset_index(drop=True)
            for budget in budgets:
                _, confidence = get_policy_output(
                    frame, "confidence", budget, random_repeats, seed
                )
                for comparator in ("random", "equal_quota"):
                    _, baseline = get_policy_output(
                        frame,
                        comparator,
                        budget,
                        random_repeats,
                        seed + test_index + 1,
                    )
                    estimate, low, high = cluster_bootstrap(
                        frame,
                        confidence,
                        baseline,
                        bootstrap_repeats,
                        seed + 1000 + test_index,
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
                            "bootstrap_repeats": bootstrap_repeats,
                        }
                    )
                    test_index += 1
    return pd.DataFrame(rows)


def save(frame, output_dir, name):
    frame.to_csv(output_dir / name, index=False)


def write_summary(output_dir, model_overall, routes, auc, bootstrap):
    auc_mean = auc[auc["language_code"] == "mean"]
    route_50 = routes[(routes["budget"] == 0.5) & routes["policy"].isin(["confidence", "random", "oracle"])]
    lines = [
        "# Multilingual Routing Analysis",
        "",
        "Canonical corrected configurations: 72",
        "Bootstrap unit: parallel example/question ID",
        "Bootstrap repetitions: 5,000 by default",
        "",
        "## Mean model accuracy",
        model_overall.round(4).to_string(index=False),
        "",
        "## Mean correction-detection AUC",
        auc_mean[["dataset", "family", "correction_detection_auc"]].round(4).to_string(index=False),
        "",
        "## Routing accuracy at 50% escalation",
        route_50[["dataset", "family", "policy", "accuracy"]].round(4).to_string(index=False),
        "",
        "## Bootstrap comparisons",
        bootstrap.round(4).to_string(index=False),
        "",
    ]
    (output_dir.parent / "analysis_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_arguments()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {args.archive}")
    raw = load_results(args.archive)
    pairs = make_pairs(raw)
    print("Validated 72 configurations")

    model_language, model_overall = model_tables(raw)
    calibration = calibration_table(raw)
    transitions = correction_table(pairs)
    auc = correction_auc_table(pairs)
    confidence_bins = confidence_bin_table(pairs)
    mmlu_quintiles = mmlu_quintile_table(pairs)
    high_confidence = high_confidence_error_table(pairs)

    curve_budgets = np.round(np.linspace(0, 1, 11), 2).tolist()
    routes, language_routes = routing_tables(
        pairs, curve_budgets, args.random_repeats, args.seed
    )
    routes_25_50, language_25_50 = routing_tables(
        pairs, [0.25, 0.5], args.random_repeats, args.seed
    )
    bootstrap = bootstrap_table(
        pairs,
        [0.25, 0.5],
        args.bootstrap_repeats,
        args.random_repeats,
        args.seed,
    )

    pair_columns = [
        "dataset", "family", "language_code", "example_id",
        "small_correct", "large_correct", "small_margin", "large_margin",
        "small_probability", "large_probability", "correction", "regression", "gain",
    ]
    save(pairs[pair_columns], args.output, "paired_predictions.csv")
    save(model_language, args.output, "model_accuracy_by_language.csv")
    save(model_overall, args.output, "model_accuracy_overall.csv")
    save(calibration, args.output, "calibration_summary.csv")
    save(transitions, args.output, "correction_transitions.csv")
    save(auc, args.output, "correction_detection_auc.csv")
    save(confidence_bins, args.output, "confidence_bins.csv")
    save(mmlu_quintiles, args.output, "mmlu_confidence_quintiles.csv")
    save(high_confidence, args.output, "high_confidence_errors.csv")
    save(routes, args.output, "routing_curves.csv")
    save(language_routes, args.output, "routing_curves_by_language.csv")
    save(routes_25_50, args.output, "routing_at_25_50.csv")
    save(language_25_50, args.output, "routing_at_25_50_by_language.csv")
    save(
        routes_25_50[
            [
                "dataset", "family", "policy", "budget",
                "worst_language_accuracy", "best_language_accuracy",
                "language_accuracy_gap", "escalation_gap",
            ]
        ],
        args.output,
        "language_disparity_at_25_50.csv",
    )
    save(bootstrap, args.output, "bootstrap_comparisons.csv")
    write_summary(args.output, model_overall, routes_25_50, auc, bootstrap)

    print(f"Saved 15 analysis tables to: {args.output}")


if __name__ == "__main__":
    main()
