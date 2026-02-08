import os
import warnings

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict
from omegaconf import DictConfig
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import TrainingArguments

from flower_hdfs.dataset import (
    get_tokenizer_and_data_collator,
    load_data,
    load_eval_data,
    replace_keys,
)
from flower_hdfs.custom_trainer import CustomTrainer
from flower_hdfs.models import get_model
from flower_hdfs.utils import compute_class_weights, compute_metrics

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    # Parse config
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
    training_arguments = TrainingArguments(**cfg.train.training_arguments)

    # Let's get the client partition
    trainset = load_data(partition_id, num_partitions, cfg.dataset.path)
    tokenizer, data_collator = get_tokenizer_and_data_collator(cfg.model.name)
    class_weights = compute_class_weights(trainset)

    # Load the model and initialize it with the received weights
    model = get_model(cfg.model)
    set_peft_model_state_dict(model, msg.content["arrays"].to_torch_state_dict())

    training_arguments.learning_rate = cfg.train.learning_rate
    training_arguments.output_dir = msg.content["config"]["save_path"]

    # Construct trainer
    trainer = CustomTrainer(
        model=model,
        args=training_arguments,
        train_dataset=trainset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    # Do local training
    results = trainer.train()

    # Construct and return reply Message
    model_record = ArrayRecord(get_peft_model_state_dict(model))
    metrics = {
        "train_loss": results.training_loss,
        "num-examples": len(trainset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
    training_arguments = TrainingArguments(**cfg.train.training_arguments)

    evalset = load_eval_data(cfg.dataset.path)
    tokenizer, data_collator = get_tokenizer_and_data_collator(cfg.model.name)

    # Load the model and initialize it with the received weights
    model = get_model(cfg.model)
    set_peft_model_state_dict(model, msg.content["arrays"].to_torch_state_dict())

    training_arguments.learning_rate = cfg.train.learning_rate
    training_arguments.output_dir = msg.content["config"]["save_path"]

    trainer = CustomTrainer(
        model=model,
        args=training_arguments,
        eval_dataset=evalset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Evaluate (this calls compute_metrics internally)
    eval_results = trainer.evaluate()

    # Pull metrics out (Trainer prefixes with "eval_")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n" * 10)
    print(eval_results)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n" * 10)
    metrics = {
        "eval_loss": eval_results.get("eval_loss"),
        "accuracy": eval_results.get("eval_accuracy"),
        "precision": eval_results.get("eval_precision"),
        "recall": eval_results.get("eval_recall"),
        "f1": eval_results.get("eval_f1"),
        "num-examples": len(evalset),
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    return Message(content=content, reply_to=msg)
