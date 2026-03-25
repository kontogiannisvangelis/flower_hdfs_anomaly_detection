import evaluate
import numpy as np

from flower_hdfs.dataset_split import DatasetSplit
from flower_hdfs.utils import softmax_np

accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")


def compute_metrics(preds):
    logits, labels = preds
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "precision": precision.compute(
            predictions=preds, references=labels, average="binary"
        )["precision"],
        "recall": recall.compute(
            predictions=preds, references=labels, average="binary"
        )["recall"],
        "f1": f1.compute(predictions=preds, references=labels, average="binary")["f1"],
    }


def compute_metrics_per_class(preds):
    logits, labels = preds
    preds = np.argmax(logits, axis=-1)

    acc = accuracy.compute(predictions=preds, references=labels)["accuracy"]
    prec = precision.compute(predictions=preds, references=labels, average=None)[
        "precision"
    ]
    rec = recall.compute(predictions=preds, references=labels, average=None)["recall"]
    f1s = f1.compute(predictions=preds, references=labels, average=None)["f1"]

    return {
        "accuracy": float(acc),
        "precision": [float(x) for x in prec],
        "recall": [float(x) for x in rec],
        "f1": [float(x) for x in f1s],
    }


def compute_metrics_per_class_with_threshold(
    logits: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    dataset_split: DatasetSplit,
):
    probs = softmax_np(logits, axis=1)
    probs_pos = probs[:, 1]
    preds = (probs_pos >= threshold).astype(int)

    acc = accuracy.compute(predictions=preds, references=labels)["accuracy"]
    prec = precision.compute(predictions=preds, references=labels, average=None)[
        "precision"
    ]
    rec = recall.compute(predictions=preds, references=labels, average=None)["recall"]
    f1s = f1.compute(predictions=preds, references=labels, average=None)["f1"]

    return {
        "accuracy": float(acc),
        "precision": [float(x) for x in prec],
        "recall": [float(x) for x in rec],
        "f1": [float(x) for x in f1s],
        "threshold": float(threshold),
        "dataset_split": dataset_split.name,
    }
