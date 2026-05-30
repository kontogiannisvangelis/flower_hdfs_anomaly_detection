import json
from pathlib import Path

import matplotlib.pyplot as plt


LOG_DIR = Path("../experiments/rank_16/2_clients/logs")
OUT_DIR = LOG_DIR / "plots"
METRICS = ["accuracy", "precision", "recall", "f1"]


def read_metrics(path):
    rounds = []
    values = {metric: [] for metric in METRICS}

    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("type") != "metrics":
                continue

            rounds.append(row["round"])
            for metric in METRICS:
                values[metric].append(row[metric])

    return rounds, values


def main():
    OUT_DIR.mkdir(exist_ok=True)
    log_files = sorted(LOG_DIR.glob("*.jsonl"))

    for metric in METRICS:
        plt.figure(figsize=(9, 5))

        for log_file in log_files:
            rounds, values = read_metrics(log_file)
            plt.plot(rounds, values[metric], label=log_file.name)

        plt.title(metric.title())
        plt.xlabel("Round")
        plt.ylabel(metric.title())
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{metric}.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
