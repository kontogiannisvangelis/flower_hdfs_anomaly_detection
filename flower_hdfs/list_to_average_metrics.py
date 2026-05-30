import json
from pathlib import Path
from typing import Any

"""
Convert evaluation logs by replacing metric lists with macro-averaged values.
"""

METRIC_TYPES = {"metrics", "tuned_metrics"}
MACRO_AVG_METRICS = {"precision", "recall", "f1"}


def macro_average(values: list[Any]) -> float:
    if not values:
        raise ValueError("Cannot compute macro average for an empty list")

    return sum(float(value) for value in values) / len(values)


def to_macro_avg(
    log_dir: Path,
    log_filename: str = "eval_logs.jsonl",
    avg_log_filename: str = "eval_logs_avg.jsonl",
) -> Path:
    input_path = log_dir / log_filename
    output_path = log_dir / avg_log_filename

    with input_path.open("r", encoding="utf-8") as in_file, output_path.open(
        "w", encoding="utf-8"
    ) as out_file:
        for line in in_file:
            if not line.strip():
                continue

            record = json.loads(line)

            if record.get("type") in METRIC_TYPES:
                for metric_name in MACRO_AVG_METRICS:
                    value = record.get(metric_name)
                    if isinstance(value, list):
                        record[metric_name] = macro_average(value)

            out_file.write(json.dumps(record) + "\n")

    return output_path


def convert_all_runs(
    base_path: Path,
    log_filename: str = "eval_logs.jsonl",
    avg_log_filename: str = "eval_logs_avg.jsonl",
) -> list[Path]:
    output_paths = []

    for log_path in sorted(base_path.glob(f"*/{log_filename}")):
        print(log_path)
        output_paths.append(to_macro_avg(log_path.parent, log_filename, avg_log_filename))

    return output_paths

if __name__ == "__main__":
    base_path = Path("../experiments/rank_16/2_clients")
    print(base_path.exists())
    filename = "eval_logs.jsonl"
    avg_filename = "eval_logs_avg.jsonl"

    outputs = convert_all_runs(base_path, filename, avg_filename)
    print(f"Created {len(outputs)} averaged log file(s).")
