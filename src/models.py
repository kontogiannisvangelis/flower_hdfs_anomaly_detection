from omegaconf import DictConfig
from peft import (
    LoraConfig,
    get_peft_model
)
from transformers import AutoModelForSequenceClassification


def get_model(model_cfg: DictConfig):

    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg.name,
        num_labels=2,
        id2label={0: "Normal", 1: "Anomaly"},
        label2id={"Normal": 0, "Anomaly": 1},
    )

    lora_cfg = LoraConfig(
        r=model_cfg.lora.peft_lora_r,
        lora_alpha=model_cfg.lora.peft_lora_alpha,
        target_modules=["query", "value"],
        lora_dropout=0.01,
        bias="none",
        task_type="SEQ_CLS",
    )

    return get_peft_model(model, lora_cfg)
