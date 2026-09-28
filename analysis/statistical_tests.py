import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]


def find_result(raw_dir, family, size, language, limit):
    suffix = f"_limit{limit}" if limit else ""
    matches = list(raw_dir.glob(f"{family}_{size}_*_{language}{suffix}.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one result for {family}/{size}/{language}, found {len(matches)}")
    return matches[0]


def load_pair(raw_dir, family, language, limit):
    small = pd.read_csv(find_result(raw_dir, family, "small", language, limit))
    large = pd.read_csv(find_result(raw_dir, family, "large", language, limit))
    columns = ["example_id", "is_correct", "confidence_margin"]
    return small[columns].merge(
        large[["example_id", "is_correct"]],
        on="example_id",
        suffixes=("_small", "_large"),
        validate="one_to_one",
    )


def confidence_decisions(frame, budget):
    count = int(round(len(frame) * budget))
    order = np.argsort(frame["confidence_margin"].to_numpy())
    escalate = np.zeros(len(frame), dtype=bool)
    escalate[order[:count]] = True
    routed = np.where(escalate, frame["is_correct_large"], frame["is_correct_small"]).astype(bool)
    return routed


def mean_random_correctness(frame, budget, repeats, seed, budget_index):
    count = int(round(len(frame) * budget))
    outcomes = np.empty((repeats, len(frame)), dtype=float)
    for repeat in range(repeats):
        rng = np.random.default_rng(np.random.SeedSequence([seed, budget_index, repeat]))
        escalate = np.zeros(len(frame), dtype=bool)
        if count:
            escalate[rng.choice(len(frame), size=count, replace=False)] = True
        outcomes[repeat] = np.where(
            escalate, frame["is_correct_large"], frame["is_correct_small"]
        ).astype(float)
    return outcomes.mean(axis=0)


def paired_bootstrap(a, b, repeats, seed):
    rng = np.random.default_rng(seed)
    differences = np.empty(repeats)
    n = len(a)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    for idx in range(repeats):
        sample = rng.integers(0, n, size=n)
        differences[idx] = (a[sample] - b[sample]).mean()
    return float((a - b).mean()), *np.quantile(differences, [0.025, 0.975]).tolist()


def mcnemar_exact(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    discordant = a_only + b_only
    p_value = 1.0 if discordant == 0 else binomtest(min(a_only, b_only), discordant, 0.5).pvalue
    return a_only, b_only, p_value


def holm_adjust(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--families", nargs="+", default=["qwen", "gemma"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    raw_dir = ROOT / cfg["paths"]["raw_results"]
    output_dir = ROOT / cfg["paths"]["summary"]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for family in args.families:
        for language in cfg["dataset"]["languages"]:
            frame = load_pair(raw_dir, family, language, args.limit)
            large = frame["is_correct_large"].astype(bool).to_numpy()
            for budget in [0.2, 0.5, 0.8]:
                routed = confidence_decisions(frame, budget)
                versus_large, large_ci_low, large_ci_high = paired_bootstrap(
                    routed, large, args.bootstrap_repeats, cfg["seed"]
                )
                budget_index = cfg["routing"]["budgets"].index(budget)
                random_mean = mean_random_correctness(
                    frame, budget, cfg["routing"]["random_repeats"], cfg["seed"], budget_index
                )
                versus_random, random_ci_low, random_ci_high = paired_bootstrap(
                    routed, random_mean, args.bootstrap_repeats, cfg["seed"]
                )
                routed_only, large_only, p_value = mcnemar_exact(routed, large)
                rows.append({
                    "family": family,
                    "language_code": language,
                    "budget": budget,
                    "routed_accuracy": routed.mean(),
                    "large_accuracy": large.mean(),
                    "confidence_minus_large": versus_large,
                    "versus_large_ci_low": large_ci_low,
                    "versus_large_ci_high": large_ci_high,
                    "random_mean_accuracy": random_mean.mean(),
                    "confidence_minus_random_mean": versus_random,
                    "versus_random_ci_low": random_ci_low,
                    "versus_random_ci_high": random_ci_high,
                    "routed_correct_large_wrong": routed_only,
                    "routed_wrong_large_correct": large_only,
                    "mcnemar_p": p_value,
                })

    results = pd.DataFrame(rows)
    results["mcnemar_p_holm"] = np.nan
    for (_, _), indices in results.groupby(["family", "language_code"]).groups.items():
        results.loc[indices, "mcnemar_p_holm"] = holm_adjust(results.loc[indices, "mcnemar_p"])
    suffix = f"_limit{args.limit}" if args.limit else ""
    output = output_dir / f"statistical_tests{suffix}.csv"
    results.to_csv(output, index=False)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
