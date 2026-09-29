import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.model_utils import clear_model, hardware_info, load_model


def build_prompt(row):
    options = "\n".join(
        f"{label}. {choice}"
        for label, choice in zip(
            row["choice_labels"],
            row["choices"],
        )
    )

    allowed_labels = ", ".join(row["choice_labels"])

    if row["task"] == "reading_comprehension":
        return (
            "Read the passage and answer the multiple-choice question. "
            f"Reply with only one of: {allowed_labels}.\n\n"
            f"Passage:\n{row['context']}\n\n"
            f"Question:\n{row['question']}\n\n"
            f"{options}\n\n"
            "Answer:"
        )

    if row["task"] == "multitask_reasoning":
        return (
            "Answer the multiple-choice question. "
            f"Reply with only one of: {allowed_labels}.\n\n"
            f"Question:\n{row['question']}\n\n"
            f"{options}\n\n"
            "Answer:"
        )

    if row["task"] == "topic_classification":
        return (
            "Classify the text into one topic. "
            f"Reply with only one of: {allowed_labels}.\n\n"
            f"Text:\n{row['question']}\n\n"
            f"Topics:\n{options}\n\n"
            "Answer:"
        )

    raise ValueError(
        f"Unknown task type: {row['task']}"
    )


def apply_chat_template(tokenizer, prompt, family):
    template_options = {
        "tokenize": False,
        "add_generation_prompt": True,
    }

    if family == "qwen":
        template_options["enable_thinking"] = False

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        **template_options,
    )


def label_token_ids(tokenizer, labels):
    token_ids = {}

    for label in labels:
        pieces = tokenizer.encode(
            label,
            add_special_tokens=False,
        )

        if len(pieces) != 1:
            pieces = tokenizer.encode(
                " " + label,
                add_special_tokens=False,
            )

        if len(pieces) != 1:
            raise ValueError(
                f"Answer label {label!r} is not represented "
                f"by one token for {tokenizer.name_or_path}"
            )

        token_ids[label] = pieces[0]

    return token_ids



@torch.inference_mode()
def score_one(
    loaded_model,
    family,
    row,
    max_input_tokens,
):
    labels = list(row["choice_labels"])

    prompt = build_prompt(row)

    formatted_prompt = apply_chat_template(
        loaded_model.tokenizer,
        prompt,
        family,
    )

    encoded = loaded_model.tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
        add_special_tokens=False,
    )

    encoded = {
        key: value.to(loaded_model.model.device)
        for key, value in encoded.items()
    }

    answer_token_ids = label_token_ids(
        loaded_model.tokenizer,
        labels,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    output = loaded_model.model(
        **encoded,
        use_cache=False,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    next_token_logits = output.logits[0, -1].float()

    choice_logits = torch.stack(
        [
            next_token_logits[answer_token_ids[label]]
            for label in labels
        ]
    )

    if not torch.isfinite(choice_logits).all():
        raise FloatingPointError(
            f"Nonfinite answer logits for "
            f"{loaded_model.model_name}. "
            "Stop evaluation and check model precision."
        )   

    probabilities = torch.softmax(
        choice_logits,
        dim=0,
    ).cpu().tolist()

    ranking = sorted(
        range(len(probabilities)),
        key=lambda index: probabilities[index],
        reverse=True,
    )

    predicted_label = labels[ranking[0]]

    confidence_margin = (
        probabilities[ranking[0]]
        - probabilities[ranking[1]]
    )

    entropy = -sum(
        probability
        * math.log(max(probability, 1e-12))
        for probability in probabilities
    )

    probability_dictionary = dict(
        zip(labels, probabilities)
    )

    return {
        "predicted_label": predicted_label,
        "choice_probabilities": json.dumps(
            probability_dictionary
        ),
        "confidence_margin": confidence_margin,
        "max_choice_probability": probabilities[
            ranking[0]
        ],
        "entropy": entropy,
        "input_tokens": int(
            encoded["input_ids"].shape[1]
        ),
        "latency_ms": latency_ms,
    }


def evaluate(
    model_name,
    family,
    model_size,
    dataset_key,
    language_code,
    config_path,
    limit=None,
    resume=True,
):
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_path = (
        ROOT
        / config["paths"]["processed_data"]
        / dataset_key
        / f"{language_code}.jsonl"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset file was not found: {data_path}\n"
            "Run data/download_data.py first."
        )

    data = pd.read_json(
        data_path,
        lines=True,
    )

    if limit is not None:
        data = data.head(limit)

    if data.empty:
        raise ValueError(
            f"No examples found in {data_path}"
        )

    suffix = (
        f"_limit{limit}"
        if limit is not None
        else ""
    )

    safe_model_name = model_name.replace("/", "__")

    output_directory = (
        ROOT
        / config["paths"]["raw_results"]
        / dataset_key
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_directory / (
        f"{family}_{model_size}_"
        f"{safe_model_name}_{language_code}"
        f"{suffix}.csv"
    )

    if resume and output_path.exists():
        previous_results = pd.read_csv(output_path)
    else:
        previous_results = pd.DataFrame()

    if not previous_results.empty:
        completed_ids = set(
            previous_results["example_id"].astype(str)
        )
    else:
        completed_ids = set()

    loaded_model = load_model(model_name)
    environment = hardware_info()

    new_rows = []

    checkpoint_every = config["evaluation"][
        "checkpoint_every"
    ]

    try:
        pending_data = data[
            ~data["example_id"]
            .astype(str)
            .isin(completed_ids)
        ]

        if not pending_data.empty:
            score_one(
                loaded_model,
                family,
                pending_data.iloc[0],
                config["generation"]["max_input_tokens"],
            )

        progress = tqdm(
            data.iterrows(),
            total=len(data),
            desc=(
                f"{dataset_key}-"
                f"{family}-"
                f"{model_size}-"
                f"{language_code}"
            ),
        )

        for _, row in progress:
            example_id = str(row["example_id"])

            if example_id in completed_ids:
                continue

            scored = score_one(
                loaded_model,
                family,
                row,
                config["generation"]["max_input_tokens"],
            )

            result = {
                "dataset": dataset_key,
                "task": row["task"],
                "category": row["category"],
                "example_id": row["example_id"],
                "language_code": language_code,
                "language_name": config["languages"][
                    language_code
                ],
                "family": family,
                "model_size": model_size,
                "model_name": model_name,
                "model_revision": loaded_model.revision,
                "dtype": loaded_model.dtype,
                "true_label": row["correct_label"],
                "is_correct": (
                    scored["predicted_label"]
                    == row["correct_label"]
                ),
                "num_choices": len(
                    row["choice_labels"]
                ),
                "seed": config["seed"],
                **scored,
                **environment,
            }

            new_rows.append(result)

            if (
                len(new_rows) % checkpoint_every
                == 0
            ):
                checkpoint = pd.concat(
                    [
                        previous_results,
                        pd.DataFrame(new_rows),
                    ],
                    ignore_index=True,
                )

                checkpoint.to_csv(
                    output_path,
                    index=False,
                )

        combined_results = pd.concat(
            [
                previous_results,
                pd.DataFrame(new_rows),
            ],
            ignore_index=True,
        )

        combined_results = (
            combined_results
            .drop_duplicates(
                "example_id",
                keep="last",
            )
        )

        combined_results.to_csv(
            output_path,
            index=False,
        )

        return output_path

    finally:
        clear_model(loaded_model)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--family",
        required=True,
        choices=[
            "qwen",
            "gemma",
            "llama",
        ],
    )

    parser.add_argument(
        "--size",
        required=True,
        choices=[
            "small",
            "large",
        ],
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "belebele",
            "mmlu_prox_lite",
            "sib200",
        ],
    )

    parser.add_argument(
        "--language",
        required=True,
    )

    parser.add_argument(
        "--config",
        default=str(ROOT / "config.yaml"),
    )

    parser.add_argument(
        "--limit",
        type=int,
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
    )

    args = parser.parse_args()

    with open(
        args.config,
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if args.language not in config["languages"]:
        raise ValueError(
            f"Unknown language code: {args.language}"
        )

    model_name = config["models"][
        args.family
    ][args.size]

    output_path = evaluate(
        model_name=model_name,
        family=args.family,
        model_size=args.size,
        dataset_key=args.dataset,
        language_code=args.language,
        config_path=args.config,
        limit=args.limit,
        resume=not args.no_resume,
    )

    print()
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()