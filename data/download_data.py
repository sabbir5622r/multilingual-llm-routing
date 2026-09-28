import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset
from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    output_dir = ROOT / cfg["paths"]["processed_data"]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    resolved_revision = HfApi().dataset_info(
        cfg["dataset"]["name"], revision=cfg["dataset"]["revision"]
    ).sha

    for code, name in cfg["dataset"]["languages"].items():
        output_path = output_dir / f"{code}.jsonl"
        if output_path.exists() and not args.force:
            frame = pd.read_json(output_path, lines=True)
        else:
            dataset = load_dataset(
                cfg["dataset"]["name"],
                code,
                split="test",
                revision=resolved_revision,
            )
            frame = dataset.to_pandas()
            frame.insert(0, "language_code", code)
            frame.insert(1, "language_name", name)
            frame["example_id"] = frame["link"].astype(str) + "::" + frame["question_number"].astype(str)
            frame.to_json(output_path, orient="records", lines=True, force_ascii=False)

        required = {
            "example_id", "flores_passage", "question", "mc_answer1",
            "mc_answer2", "mc_answer3", "mc_answer4", "correct_answer_num"
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{code} is missing columns: {sorted(missing)}")
        if frame["example_id"].duplicated().any():
            raise ValueError(f"Duplicate example IDs found in {code}")
        if len(frame) != 900:
            raise ValueError(f"Expected 900 rows for {code}, found {len(frame)}")

        rows.append({
            "language_code": code,
            "language_name": name,
            "rows": len(frame),
            "sha256": file_sha256(output_path),
            "path": str(output_path.relative_to(ROOT)),
        })

    id_sets = []
    for code in cfg["dataset"]["languages"]:
        frame = pd.read_json(output_dir / f"{code}.jsonl", lines=True)
        id_sets.append(set(frame["example_id"]))
    if any(ids != id_sets[0] for ids in id_sets[1:]):
        raise ValueError("The selected language subsets do not contain identical parallel item IDs")

    with open(output_dir / "download_report.json", "w", encoding="utf-8") as file:
        json.dump({
            "dataset": cfg["dataset"],
            "resolved_revision": resolved_revision,
            "files": rows,
        }, file, ensure_ascii=False, indent=2)

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved validated data to {output_dir}")


if __name__ == "__main__":
    main()
