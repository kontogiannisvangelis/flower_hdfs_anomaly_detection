import torch.nn.functional as F
from transformers import Trainer


class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

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
        ce_loss = F.cross_entropy(logits, labels, weight=weight, reduction="none")
        loss = ce_loss.mean()

        return (loss, outputs) if return_outputs else loss
