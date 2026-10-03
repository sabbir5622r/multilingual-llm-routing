import argparse
import gc
import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

try:
    import pynvml
except ImportError as error:
    raise ImportError(
        "Install NVIDIA ML bindings with: pip install nvidia-ml-py"
    ) from error


ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))

from evaluation.score_models import apply_chat_template, build_prompt
from models.model_utils import clear_model, load_model


KEYS = ["dataset", "language_code", "example_id"]
POLICIES = ("confidence", "equal_quota", "random")
BUDGETS = (0.25, 0.50)


class PowerMonitor:
    def __init__(self, interval_seconds=0.05):
        self.interval_seconds = interval_seconds
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None
        self.handles = []

    def start(self):
        pynvml.nvmlInit()
        self.handles = [
            pynvml.nvmlDeviceGetHandleByIndex(index)
            for index in range(pynvml.nvmlDeviceGetCount())
        ]
        self.samples = []
        self.stop_event.clear()
        self._sample()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _sample(self):
        total_watts = 0.0
        for handle in self.handles:
            total_watts += pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        self.samples.append((time.perf_counter(), total_watts))

    def _run(self):
        while not self.stop_event.wait(self.interval_seconds):
            self._sample()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self._sample()
        pynvml.nvmlShutdown()

    def energy_joules(self):
        if len(self.samples) < 2:
            return math.nan
        energy = 0.0
        for (time_a, power_a), (time_b, power_b) in zip(
            self.samples[:-1], self.samples[1:]
        ):
            energy += (time_b - time_a) * (power_a + power_b) / 2.0
        return energy

    def mean_power_watts(self):
        if not self.samples:
            return math.nan
        return float(np.mean([power for _, power in self.samples]))


def stable_seed(*parts, base_seed=42):
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (base_seed + int(digest[:8], 16)) % (2**32)


def load_benchmark_data(config, family, paired_path, examples_per_group):
    paired = pd.read_csv(paired_path)
    paired["example_id"] = paired["example_id"].astype(str)
    paired = paired[paired["family"] == family].copy()

    rows = []
    processed_root = ROOT / config["paths"]["processed_data"]

    for dataset in config["datasets"]:
        for language in config["languages"]:
            path = processed_root / dataset / f"{language}.jsonl"
            if not path.exists():
                raise FileNotFoundError(path)

            frame = pd.read_json(path, lines=True)
            frame["example_id"] = frame["example_id"].astype(str)
            frame["dataset"] = dataset
            frame["language_code"] = language

            available = paired[
                (paired["dataset"] == dataset)
                & (paired["language_code"] == language)
            ][["example_id", "small_margin"]]

            frame = frame.merge(available, on="example_id", how="inner")
            expected = min(examples_per_group, len(frame))
            if len(frame) < expected:
                raise ValueError(f"Insufficient paired examples for {dataset}/{language}")

            frame = frame.sample(
                n=expected,
                random_state=stable_seed(dataset, language),
            )
            rows.append(frame)

    benchmark = pd.concat(rows, ignore_index=True)
    benchmark = benchmark.sort_values(KEYS).reset_index(drop=True)
    benchmark["benchmark_index"] = np.arange(len(benchmark))
    return benchmark


def lowest_mask(values, count):
    mask = np.zeros(len(values), dtype=bool)
    order = np.argsort(np.asarray(values), kind="stable")
    mask[order[:count]] = True
    return mask


def build_policy_mask(frame, policy, budget, seed=42):
    started = time.perf_counter()
    mask = np.zeros(len(frame), dtype=bool)

    for dataset, dataset_indices in frame.groupby("dataset").groups.items():
        positions = np.asarray(list(dataset_indices))
        dataset_frame = frame.loc[positions]

        if policy == "confidence":
            count = int(round(len(positions) * budget))
            selected = lowest_mask(dataset_frame["small_margin"], count)
            mask[positions[selected]] = True

        elif policy == "equal_quota":
            for language, language_indices in dataset_frame.groupby(
                "language_code"
            ).groups.items():
                local_positions = np.asarray(list(language_indices))
                count = int(round(len(local_positions) * budget))
                selected = lowest_mask(
                    frame.loc[local_positions, "small_margin"], count
                )
                mask[local_positions[selected]] = True

        elif policy == "random":
            count = int(round(len(positions) * budget))
            generator = np.random.default_rng(
                stable_seed(dataset, policy, budget, base_seed=seed)
            )
            selected = generator.choice(
                positions,
                size=count,
                replace=False,
            )
            mask[selected] = True

        else:
            raise ValueError(policy)

    ranking_seconds = time.perf_counter() - started
    return mask, ranking_seconds


def prepare_input(loaded, family, row, max_input_tokens):
    prompt = build_prompt(row)
    formatted = apply_chat_template(loaded.tokenizer, prompt, family)
    encoded = loaded.tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
        add_special_tokens=False,
    )
    return {
        key: value.to(loaded.model.device)
        for key, value in encoded.items()
    }


@torch.inference_mode()
def forward_once(loaded, family, row, max_input_tokens):
    encoded = prepare_input(loaded, family, row, max_input_tokens)
    options = (
        {"logits_to_keep": 1}
        if loaded.model_name == "google/gemma-3-4b-it"
        else {}
    )
    output = loaded.model(
        **encoded,
        use_cache=False,
        **options,
    )
    input_tokens = int(encoded["input_ids"].shape[1])
    del output, encoded
    return input_tokens


def warm_up(loaded, family, frame, max_input_tokens, warmup_examples):
    if frame.empty:
        return
    warmup = frame.head(min(warmup_examples, len(frame)))
    for _, row in warmup.iterrows():
        forward_once(loaded, family, row, max_input_tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_stage(
    loaded,
    family,
    frame,
    stage,
    repeat,
    max_input_tokens,
    power_interval,
):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    monitor = PowerMonitor(power_interval)
    monitor.start()
    wall_start = time.perf_counter()
    per_example = []

    for _, row in tqdm(
        frame.iterrows(),
        total=len(frame),
        desc=f"{family}-{stage}-repeat{repeat}",
    ):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        example_start = time.perf_counter()
        input_tokens = forward_once(
            loaded,
            family,
            row,
            max_input_tokens,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - example_start) * 1000.0
        per_example.append(
            {
                "family": family,
                "stage": stage,
                "repeat": repeat,
                "dataset": row["dataset"],
                "language_code": row["language_code"],
                "example_id": str(row["example_id"]),
                "benchmark_index": int(row["benchmark_index"]),
                "input_tokens": input_tokens,
                "latency_ms": latency_ms,
            }
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    monitor.stop()

    energy_joules = monitor.energy_joules()
    stage_summary = {
        "family": family,
        "stage": stage,
        "repeat": repeat,
        "examples": len(frame),
        "wall_seconds": wall_seconds,
        "energy_joules": energy_joules,
        "energy_wh": energy_joules / 3600.0,
        "mean_power_watts": monitor.mean_power_watts(),
        "mean_latency_ms": float(
            np.mean([row["latency_ms"] for row in per_example])
        ),
        "p50_latency_ms": float(
            np.quantile([row["latency_ms"] for row in per_example], 0.50)
        ),
        "p95_latency_ms": float(
            np.quantile([row["latency_ms"] for row in per_example], 0.95)
        ),
        "throughput_examples_s": len(frame) / wall_seconds,
        "dtype": loaded.dtype,
        "model_name": loaded.model_name,
        "model_revision": loaded.revision,
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": json.dumps(
            [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
        ),
    }
    return stage_summary, per_example


def append_rows(path, rows):
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        mode="a" if path.exists() else "w",
        header=not path.exists(),
        index=False,
    )


def completed_stages(stage_path):
    if not stage_path.exists():
        return set()
    frame = pd.read_csv(stage_path)
    return set(zip(frame["stage"], frame["repeat"]))


def benchmark_model(
    model_name,
    family,
    size,
    benchmark,
    masks,
    output_dir,
    repeats,
    warmup_examples,
    max_input_tokens,
    power_interval,
):
    stage_path = output_dir / "stage_measurements.csv"
    example_path = output_dir / "per_example_latency.csv"
    completed = completed_stages(stage_path)

    if size == "small":
        stage_frames = {"small_all": benchmark}
    else:
        stage_frames = {"large_only": benchmark}
        for stage, mask in masks.items():
            stage_frames[stage] = benchmark.loc[mask].copy()

    pending = [
        (stage, repeat)
        for stage in stage_frames
        for repeat in range(1, repeats + 1)
        if (stage, repeat) not in completed
    ]
    if not pending:
        print(f"All {size} stages already completed.")
        return

    loaded = load_model(model_name)
    try:
        warm_up(
            loaded,
            family,
            benchmark,
            max_input_tokens,
            warmup_examples,
        )
        for stage, repeat in pending:
            summary, per_example = measure_stage(
                loaded,
                family,
                stage_frames[stage],
                stage,
                repeat,
                max_input_tokens,
                power_interval,
            )
            append_rows(stage_path, [summary])
            append_rows(example_path, per_example)
            print("Saved:", stage, "repeat", repeat)
    finally:
        clear_model(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def summarize_results(
    output_dir,
    family,
    benchmark,
    ranking_times,
    gpu_hour_price,
):
    stages = pd.read_csv(output_dir / "stage_measurements.csv")
    examples = pd.read_csv(output_dir / "per_example_latency.csv")
    rows = []
    gpu_count = int(stages["gpu_count"].max())

    policies = ["small_only", "large_only"] + [
        f"{policy}_{int(budget * 100)}"
        for policy in POLICIES
        for budget in BUDGETS
    ]

    for repeat in sorted(stages["repeat"].unique()):
        small_stage = stages[
            (stages["stage"] == "small_all")
            & (stages["repeat"] == repeat)
        ].iloc[0]
        large_stage = stages[
            (stages["stage"] == "large_only")
            & (stages["repeat"] == repeat)
        ].iloc[0]

        small_latency = examples[
            (examples["stage"] == "small_all")
            & (examples["repeat"] == repeat)
        ][KEYS + ["latency_ms"]].rename(
            columns={"latency_ms": "small_latency_ms"}
        )
        large_latency = examples[
            (examples["stage"] == "large_only")
            & (examples["repeat"] == repeat)
        ][KEYS + ["latency_ms"]].rename(
            columns={"latency_ms": "large_latency_ms"}
        )

        for policy in policies:
            if policy == "small_only":
                wall = float(small_stage["wall_seconds"])
                energy = float(small_stage["energy_joules"])
                latency = small_latency["small_latency_ms"].to_numpy()
                escalation_rate = 0.0
            elif policy == "large_only":
                wall = float(large_stage["wall_seconds"])
                energy = float(large_stage["energy_joules"])
                latency = large_latency["large_latency_ms"].to_numpy()
                escalation_rate = 1.0
            else:
                routed_stage = stages[
                    (stages["stage"] == policy)
                    & (stages["repeat"] == repeat)
                ].iloc[0]
                routed_latency = examples[
                    (examples["stage"] == policy)
                    & (examples["repeat"] == repeat)
                ][KEYS + ["latency_ms"]].rename(
                    columns={"latency_ms": "large_latency_ms"}
                )
                combined = small_latency.merge(
                    routed_latency,
                    on=KEYS,
                    how="left",
                )
                combined["large_latency_ms"] = combined[
                    "large_latency_ms"
                ].fillna(0.0)
                latency = (
                    combined["small_latency_ms"]
                    + combined["large_latency_ms"]
                ).to_numpy()
                wall = (
                    float(small_stage["wall_seconds"])
                    + float(routed_stage["wall_seconds"])
                    + ranking_times[policy]
                )
                energy = (
                    float(small_stage["energy_joules"])
                    + float(routed_stage["energy_joules"])
                )
                escalation_rate = float(routed_stage["examples"]) / len(
                    benchmark
                )

            gpu_hours = wall * gpu_count / 3600.0
            cost = (
                gpu_hours * gpu_hour_price
                if gpu_hour_price is not None
                else math.nan
            )
            baseline_wall = float(large_stage["wall_seconds"])
            baseline_energy = float(large_stage["energy_joules"])

            rows.append(
                {
                    "family": family,
                    "policy": policy,
                    "repeat": repeat,
                    "examples": len(benchmark),
                    "escalation_rate": escalation_rate,
                    "mean_latency_ms": float(np.mean(latency)),
                    "p50_latency_ms": float(np.quantile(latency, 0.50)),
                    "p95_latency_ms": float(np.quantile(latency, 0.95)),
                    "wall_seconds": wall,
                    "throughput_examples_s": len(benchmark) / wall,
                    "energy_joules": energy,
                    "energy_wh": energy / 3600.0,
                    "gpu_hours": gpu_hours,
                    "estimated_cost_usd": cost,
                    "wall_time_saving": 1.0 - wall / baseline_wall,
                    "energy_saving": 1.0 - energy / baseline_energy,
                    "gpu_hour_price_usd": gpu_hour_price,
                }
            )

    repeated = pd.DataFrame(rows)
    repeated.to_csv(output_dir / "system_efficiency_repeats.csv", index=False)

    metrics = [
        "escalation_rate",
        "mean_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "wall_seconds",
        "throughput_examples_s",
        "energy_joules",
        "energy_wh",
        "gpu_hours",
        "estimated_cost_usd",
        "wall_time_saving",
        "energy_saving",
    ]
    summary = repeated.groupby(["family", "policy"])[metrics].agg(
        ["mean", "std"]
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output_dir / "system_efficiency_summary.csv", index=False)
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=["qwen", "gemma", "llama"])
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--paired-predictions", required=True)
    parser.add_argument("--examples-per-group", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-examples", type=int, default=5)
    parser.add_argument("--power-interval", type=float, default=0.05)
    parser.add_argument("--gpu-hour-price", type=float)
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "results" / "system_efficiency"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    benchmark = load_benchmark_data(
        config,
        args.family,
        Path(args.paired_predictions),
        args.examples_per_group,
    )
    output_dir = Path(args.output_root) / args.family
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark.to_json(
        output_dir / "benchmark_examples.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )

    masks = {}
    ranking_times = {}
    for policy in POLICIES:
        for budget in BUDGETS:
            stage = f"{policy}_{int(budget * 100)}"
            mask, ranking_seconds = build_policy_mask(
                benchmark,
                policy,
                budget,
                seed=config["seed"],
            )
            masks[stage] = mask
            ranking_times[stage] = ranking_seconds

    with open(output_dir / "routing_overhead.json", "w", encoding="utf-8") as file:
        json.dump(ranking_times, file, indent=2)

    models = config["models"][args.family]
    benchmark_model(
        models["small"],
        args.family,
        "small",
        benchmark,
        masks,
        output_dir,
        args.repeats,
        args.warmup_examples,
        config["generation"]["max_input_tokens"],
        args.power_interval,
    )
    benchmark_model(
        models["large"],
        args.family,
        "large",
        benchmark,
        masks,
        output_dir,
        args.repeats,
        args.warmup_examples,
        config["generation"]["max_input_tokens"],
        args.power_interval,
    )
    summarize_results(
        output_dir,
        args.family,
        benchmark,
        ranking_times,
        args.gpu_hour_price,
    )


if __name__ == "__main__":
    main()
