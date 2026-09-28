import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def run(command):
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--families", nargs="+", default=["qwen", "gemma"], choices=["qwen", "gemma"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    python = sys.executable

    if not args.skip_download:
        run([python, "data/download_data.py", "--config", args.config])

    for family in args.families:
        for size in ["small", "large"]:
            for language in cfg["dataset"]["languages"]:
                command = [
                    python, "evaluation/score_models.py",
                    "--family", family,
                    "--size", size,
                    "--language", language,
                    "--config", args.config,
                ]
                if args.limit:
                    command += ["--limit", str(args.limit)]
                run(command)

    route_command = [python, "routing/simulate_routes.py", "--config", args.config, "--families", *args.families]
    report_command = [python, "analysis/make_report.py", "--config", args.config]
    stats_command = [python, "analysis/statistical_tests.py", "--config", args.config, "--families", *args.families]
    if args.limit:
        route_command += ["--limit", str(args.limit)]
        report_command += ["--limit", str(args.limit)]
        stats_command += ["--limit", str(args.limit)]
    run(route_command)
    run(report_command)
    run(stats_command)


if __name__ == "__main__":
    main()
