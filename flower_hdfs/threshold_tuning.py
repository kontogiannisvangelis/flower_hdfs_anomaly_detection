import evaluate
import numpy as np

from flower_hdfs.utils import softmax_np

f1 = evaluate.load("f1")


def find_best_threshold(logits: np.ndarray, labels: np.ndarray):
    probs = softmax_np(logits)[:, 1]

    best_thr = 0.5
    best_f1 = -1.0

    for t in np.linspace(0.0, 1.0, 201):
        preds = (probs >= t).astype(int)

        score = f1.compute(predictions=preds, references=labels)["f1"]

        if score > best_f1:
            best_f1 = score
            best_thr = t

    return best_thr, best_f1
