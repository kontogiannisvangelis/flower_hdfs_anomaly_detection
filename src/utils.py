import evaluate
import numpy as np
import torch

accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "precision": precision.compute(
            predictions=preds, references=labels, average="binary"
        )["precision"],
        "recall": recall.compute(
            predictions=preds, references=labels, average="binary"
        )["recall"],
        "f1": f1.compute(predictions=preds, references=labels, average="binary")[
            "f1"
        ],
    }


def compute_class_weights(dataset, label_column="label", num_classes=None):
    labels = np.array(dataset[label_column])
    if num_classes is None:
        num_classes = int(labels.max()) + 1 if labels.size else 0
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    weights = total / (len(counts) * counts)
    weights = np.where(counts == 0, 0.0, weights)
    return torch.tensor(weights, dtype=torch.float32)
