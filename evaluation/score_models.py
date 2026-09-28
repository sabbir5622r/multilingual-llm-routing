import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from models.model_utils import clear_model, hardware_info, load_model


def build_prompt(row):
    return (
        "Read the passage and answer the multiple-choice question. "
        "Reply with only A, B, C, or D.\n\n"
        f"Passage:\n{row['flores_passage']}\n\n"
        f"Question:\n{row['question']}\n\n"
        f"A. {row['mc_answer1']}\n"
        f"B. {row['mc_answer2']}\n"
        f"C. {row['mc_answer3']}\n"
        f"D. {row['mc_answer4']}\n\n"
        "Answer:"
    )


def apply_chat(tokenizer, prompt, family):
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if family == "qwen":
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)


def label_token_ids(tokenizer, labels):
    ids = {}
    for label in labels:
        pieces = tokenizer.encode(label, add_special_tokens=False)
        if len(pieces) != 1:
            spaced = tokenizer.encode(" " + label, add_special_tokens=False)
            if len(spaced) != 1:
                raise ValueError(f"Label {label!r} is not a single token for {tokenizer.name_or_path}")
            pieces = spaced
        ids[label] = pieces[0]
    return ids


def correct_label(value):
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text in {"1", "2", "3", "4"}:
        return "ABCD"[int(text) - 1]
    if text in {"A", "B", "C", "D"}:
        return text
    raise ValueError(f"Unknown gold answer: {value!r}")


@torch.inference_mode()
def score_one(loaded, family, row, labels, max_input_tokens):
    prompt = apply_chat(loaded.tokenizer, build_prompt(row), family)
    encoded = loaded.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
        add_special_tokens=False,
    )
    encoded = {key: value.to(loaded.model.device) for key, value in encoded.items()}
    token_ids = label_token_ids(loaded.tokenizer, labels)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    output = loaded.model(**encoded, use_cache=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - started) * 1000

    next_logits = output.logits[0, -1].float()
    choice_logits = torch.tensor([next_logits[token_ids[label]].item() for label in labels])
    probs = torch.softmax(choice_logits, dim=0).cpu().tolist()
    ranking = sorted(range(len(probs)), key=lambda idx: probs[idx], reverse=True)
    prediction = labels[ranking[0]]
    margin = probs[ranking[0]] - probs[ranking[1]]
    entropy = -sum(prob * math.log(max(prob, 1e-12)) for prob in probs)

    return {
        "predicted_label": prediction,
        "prob_A": probs[0],
        "prob_B": probs[1],
        "prob_C": probs[2],
        "prob_D": probs[3],
        "confidence_margin": margin,
        "entropy": entropy,
        "input_tokens": int(encoded["input_ids"].shape[1]),
        "latency_ms": latency_ms,
    }


def evaluate(model_name, family, size, language_code, config_path, limit=None, resume=True):
    with open(config_path, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    data_path = ROOT / cfg["paths"]["processed_data"] / f"{language_code}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"Run data/download_data.py first: {data_path}")
    data = pd.read_json(data_path, lines=True)
    if limit:
        data = data.head(limit)

    suffix = f"_limit{limit}" if limit else ""
    safe_name = model_name.replace("/", "__")
    output_dir = ROOT / cfg["paths"]["raw_results"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{family}_{size}_{safe_name}_{language_code}{suffix}.csv"

    previous = pd.read_csv(output_path) if resume and output_path.exists() else pd.DataFrame()
    completed = set(previous["example_id"].astype(str)) if not previous.empty else set()
    loaded = load_model(model_name)
    metadata = hardware_info()
    new_rows = []
    checkpoint_every = cfg["evaluation"]["checkpoint_every"]

    try:
        pending = data[~data["example_id"].astype(str).isin(completed)]
        if not pending.empty:
            score_one(
                loaded,
                family,
                pending.iloc[0],
                cfg["generation"]["answer_labels"],
                cfg["generation"]["max_input_tokens"],
            )
        for _, row in tqdm(data.iterrows(), total=len(data), desc=f"{family}-{size}-{language_code}"):
            if str(row["example_id"]) in completed:
                continue
            scored = score_one(
                loaded,
                family,
                row,
                cfg["generation"]["answer_labels"],
                cfg["generation"]["max_input_tokens"],
            )
            gold = correct_label(row["correct_answer_num"])
            record = {
                "example_id": row["example_id"],
                "language_code": language_code,
                "language_name": cfg["dataset"]["languages"][language_code],
                "family": family,
                "model_size": size,
                "model_name": model_name,
                "model_revision": loaded.revision,
                "dtype": loaded.dtype,
                "true_label": gold,
                "is_correct": scored["predicted_label"] == gold,
                "seed": cfg["seed"],
                **scored,
                **metadata,
            }
            new_rows.append(record)
            if len(new_rows) % checkpoint_every == 0:
                combined = pd.concat([previous, pd.DataFrame(new_rows)], ignore_index=True)
                combined.to_csv(output_path, index=False)

        combined = pd.concat([previous, pd.DataFrame(new_rows)], ignore_index=True)
        combined = combined.drop_duplicates("example_id", keep="last")
        combined.to_csv(output_path, index=False)
        return output_path
    finally:
        clear_model(loaded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=["qwen", "gemma"])
    parser.add_argument("--size", required=True, choices=["small", "large"])
    parser.add_argument("--language", required=True)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    model_name = cfg["models"][args.family][args.size]
    path = evaluate(
        model_name, args.family, args.size, args.language,
        args.config, limit=args.limit, resume=not args.no_resume,
    )
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
