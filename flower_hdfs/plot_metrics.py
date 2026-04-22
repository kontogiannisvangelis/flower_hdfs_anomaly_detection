import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import ticker
from PIL import Image

LABELS = {
    "eval_loss": "Evaluation Loss",
    "training_params": "Training Parameters",
    "accuracy": "Accuracy",
    "f1": "F1 Score",
    "precision": "Precision",
    "recall": "Recall",
    "num_clients": "Number of Clients",
    "learning_rate": "Learning Rate",
    "fraction_train": "Fraction Train",
    "batch_size": "Batch Size",
    "num_train_epochs": "Number of Training Epochs",
    "dirichlet_alpha": "Dirichlet Alpha",
    "num_server_rounds": "Number of Server Rounds",
    "round": "Round",
    "class": "Class",
}

def pretty_label(name):
    return LABELS.get(name, name.replace("_", " ").title())


def plot_params(params, plot_save_path):
    cfg = params[-1]["config"]

    train_args = cfg.get("train", {}).get("training_arguments", {})
    train_cfg = cfg.get("train", {})
    strategy_cfg = cfg.get("strategy", {})

    summary_params = {
        "num_clients": cfg.get(
            "num_clients", "2"
        ),
        "fraction_train": strategy_cfg.get(
            "fraction_train", "N/A"
        ),
        "learning_rate": train_cfg.get(
            "learning_rate", train_args.get("learning_rate", "N/A")
        ),
        "gamma": train_cfg.get(
            "focal_gamma", "N/A"
        ),
        "alpha": train_cfg.get(
            "focal_alpha", "N/A"
        ),
        "batch_size": train_args.get("per_device_train_batch_size", "N/A"),
        "num_train_epochs": train_args.get("num_train_epochs", "N/A"),
        "dirichlet_alpha": cfg.get("dirichlet_alpha", "N/A"),
        "num_server_rounds": cfg.get("num_server_rounds", "N/A"),
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")

    title = "Training Parameters"
    lines = [f"{pretty_label(k)}: {v}" for k, v in summary_params.items()]
    text = "\n".join(lines)

    ax.text(
        0.5,
        0.95,
        title,
        va="top",
        ha="center",
        fontsize=16,
        fontweight="bold",
        transform=ax.transAxes,
    )

    ax.text(
        0.5,
        0.82,
        text,
        va="top",
        ha="center",
        fontsize=13,
        family="monospace",
        multialignment="center",
        transform=ax.transAxes,
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_save_path, "training_params.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def create_grid_image(plot_save_path):
    names = [
        "eval_loss.png",
        "training_params.png",
        "accuracy.png",
        "f1.png",
        "precision.png",
        "recall.png",
    ]

    imgs = []
    for n in names:
        path = os.path.join(plot_save_path, n)
        img = Image.open(path)
        imgs.append(img)
    w, h = imgs[0].size
    out = Image.new("RGB", (3 * w, 2 * h), "white")

    for i, im in enumerate(imgs):
        out.paste(im, ((i % 3) * w, (i // 3) * h))

    out.save(os.path.join(plot_save_path, "grid.png"))


def plot_list_metrics(list_metrics, rounds, plot_save_path):
    class_colors = {0: "blue", 1: "orange"}

    for metric_name, per_class in list_metrics.items():
        plt.figure(figsize=(8, 4.5))
        for ci, vals in sorted(per_class.items()):
            color = class_colors.get(ci)
            plt.plot(
                rounds, vals, linewidth=1.5, alpha=0.8, label=f"class {ci}", color=color
            )
        plt.title(pretty_label(metric_name))
        plt.xlabel("Round")
        plt.ylabel(pretty_label(metric_name))
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10, integer=True))
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_save_path, f"{metric_name}.png"), dpi=150)
        plt.close()


def plot_scalar_metrics(scalar_metrics, rounds, plot_save_path):
    for metric_name, vals in scalar_metrics.items():
        plt.figure(figsize=(8, 4.5))
        plt.plot(rounds, vals, linewidth=1.5, alpha=0.8)
        plt.title(pretty_label(metric_name))
        plt.xlabel("Round")
        plt.ylabel(pretty_label(metric_name))
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10, integer=True))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(os.path.join(plot_save_path, f"{metric_name}.png"), dpi=150)
        plt.close()


def plot(log_dir, eval_log_filename, plot_save_path):
    records = []
    params = []

    with open(os.path.join(log_dir, eval_log_filename), "r") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("type") == "metrics":
                records.append(obj)
            if obj.get("type") == "params":
                params.append(obj)

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

    plot_params(params, plot_save_path)
    plot_list_metrics(list_metrics, rounds, plot_save_path)
    plot_scalar_metrics(scalar_metrics, rounds, plot_save_path)
    create_grid_image(plot_save_path)


if __name__ == "__main__":
    base_path = "/home/ekontogi/Diploma-Thesis/flower_hdfs_anomaly_detection/results"
    filename = "eval_logs.jsonl"
    for entry in os.listdir(base_path):
        subdir = os.path.join(base_path, entry)
        plot_save_path = os.path.join(subdir, "metrics")
        # if os.path.exists(plot_save_path):
        #    continue
        os.makedirs(plot_save_path, exist_ok=True)
        plot(subdir, filename, plot_save_path)
