from flwr.app import ArrayRecord, Message, Context
from omegaconf import DictConfig
from peft import set_peft_model_state_dict
from transformers import TrainingArguments

from flower_hdfs.compute_metrics import compute_metrics_per_class, compute_metrics
from flower_hdfs.custom_trainer import CustomTrainer
from flower_hdfs.dataset import (
    load_eval_data,
    load_test_data,
)

from flower_hdfs.dataset_split import DatasetSplit
from flower_hdfs.models import get_model
from flower_hdfs.utils import compute_class_weights
from flwr.common.config import unflatten_dict
from flower_hdfs.dataset import (
    get_tokenizer_and_data_collator,
    load_data,
    replace_keys,
)


def fit(msg: Message, context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    training_arguments = TrainingArguments(**cfg.train.training_arguments)
    trainset = load_data(
        partition_id, num_partitions, cfg.dataset.dirichlet.alpha, cfg.dataset.path
    )
    tokenizer, data_collator = get_tokenizer_and_data_collator(cfg.model.name)
    class_weights = compute_class_weights(trainset, 2)

    model = get_model(cfg.model)
    set_peft_model_state_dict(model, msg.content["arrays"].to_torch_state_dict())

    training_arguments.learning_rate = cfg.train.learning_rate
    training_arguments.output_dir = msg.content["config"]["save_path"]

    focal_gamma = cfg.train.focal_gamma
    focal_alpha = None
    if cfg.train.focal_alpha is not None and cfg.train.focal_alpha:
        focal_alpha = class_weights

    trainer = CustomTrainer(
        model=model,
        args=training_arguments,
        train_dataset=trainset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
    )
    return trainer.train(), model, len(trainset)


def score(cfg: DictConfig, save_path: str, arrays: ArrayRecord):
    training_arguments = TrainingArguments(**cfg.train.training_arguments)
    training_arguments.output_dir = save_path
    dataset = load_eval_data(cfg.dataset.path)
    tokenizer, data_collator = get_tokenizer_and_data_collator(cfg.model.name)

    model = get_model(cfg.model)
    set_peft_model_state_dict(model, arrays.to_torch_state_dict())

    trainer = CustomTrainer(
        model=model,
        args=training_arguments,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_per_class,
    )

    eval_results = trainer.evaluate()

    metrics = {
        "eval_loss": eval_results.get("eval_loss"),
        "accuracy": eval_results.get("eval_accuracy"),
        "precision": eval_results.get("eval_precision"),
        "recall": eval_results.get("eval_recall"),
        "f1": eval_results.get("eval_f1"),
    }
    return metrics


def predict(
    cfg: DictConfig, save_path: str, arrays: ArrayRecord, dataset_split: DatasetSplit
):
    training_arguments = TrainingArguments(**cfg.train.training_arguments)
    training_arguments.output_dir = save_path
    if dataset_split == DatasetSplit.EVAL:
        dataset = load_eval_data(cfg.dataset.path)
    elif dataset_split == DatasetSplit.TEST:
        dataset = load_test_data(cfg.dataset.path)
    else:
        raise ValueError(f"Unknown dataset split: {dataset_split}")
    tokenizer, data_collator = get_tokenizer_and_data_collator(cfg.model.name)

    model = get_model(cfg.model)
    set_peft_model_state_dict(model, arrays.to_torch_state_dict())

    trainer = CustomTrainer(
        model=model,
        args=training_arguments,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_per_class,
    )

    return trainer.predict(dataset)
