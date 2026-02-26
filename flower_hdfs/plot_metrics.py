import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt

def plot(base_path, eval_log_filename, plot_save_path):
    records = []

    with open(os.path.join(base_path, eval_log_filename), "r") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("type") == "metrics":
                records.append(obj)

    print(records)


    records.sort(key=lambda x: x["round"])
    rounds = [r["round"] for r in records]

    list_metrics = defaultdict(lambda: defaultdict(list))
    scalar_metrics = defaultdict(list)

    for r in records:
        for k, v in r.items():
            if k in ("type", "round"):
                continue
            if isinstance(v, list):
                for ci, val in enumerate(v):
                    list_metrics[k][ci].append(val)
            else:
                scalar_metrics[k].append(v)

    class_colors = {0: "blue", 1: "orange"}

    for metric_name, per_class in list_metrics.items():
        plt.figure()
        for ci, vals in sorted(per_class.items()):
            color = class_colors.get(ci)
            plt.plot(rounds, vals, marker="o", label=f"class {ci}", color=color)
        plt.title(metric_name)
        plt.xlabel("round")
        plt.ylabel(metric_name)
        plt.xticks(rounds)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(plot_save_path, f"{metric_name}.png"), dpi=150)
        plt.close()

    for metric_name, vals in scalar_metrics.items():
        plt.figure()
        plt.plot(rounds, vals, marker="o")
        plt.title(metric_name)
        plt.xlabel("round")
        plt.ylabel(metric_name)
        plt.xticks(rounds)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(os.path.join(plot_save_path, f"{metric_name}.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    base_path = "/home/ekontogi/Diploma-Thesis/flower_hdfs_anomaly_detection/results/2026-02-26_01-31-54"
    filename = "eval_logs.jsonl"
    plot_save_path = os.path.join(base_path, "metrics")
    os.makedirs(plot_save_path, exist_ok=True)
    plot(base_path, filename, plot_save_path)