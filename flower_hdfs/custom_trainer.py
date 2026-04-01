from logging import INFO

import torch
import torch.nn.functional as F
from transformers import Trainer
from flwr.common.logger import log


class CustomTrainer(Trainer):
    def __init__(
        self, class_weights=None, focal_gamma=None, focal_alpha=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]

        weight = (
            self.class_weights.to(logits.device)
            if self.class_weights is not None
            else None
        )
        if self.focal_gamma is not None and self.focal_gamma > 0:
            ce_loss = F.cross_entropy(logits, labels, reduction="none")
            pt = torch.exp(-ce_loss)
            if self.focal_alpha is not None:
                alpha = self.focal_alpha.to(logits.device)
                alpha_t = alpha[labels]
                loss = (alpha_t * ((1 - pt) ** self.focal_gamma) * ce_loss).mean()
            else:
                loss = ((1 - pt) ** self.focal_gamma * ce_loss).mean()
        else:
            ce_loss = F.cross_entropy(logits, labels, weight=weight, reduction="none")
            loss = ce_loss.mean()

        return (loss, outputs) if return_outputs else loss
