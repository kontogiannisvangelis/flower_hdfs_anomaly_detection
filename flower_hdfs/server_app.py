import logging
import os
from datetime import datetime

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedAdam
from omegaconf import DictConfig
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import TrainingArguments

from flower_hdfs.custom_trainer import CustomTrainer
from flower_hdfs.dataset import replace_keys, get_tokenizer_and_data_collator, load_eval_data
from flower_hdfs.models import get_model
from flower_hdfs.utils import compute_metrics, compute_metrics_per_class

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(os.getcwd(), f"results/{folder_name}")
    os.makedirs(save_path, exist_ok=True)

    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    init_model = get_model(cfg.model)
    arrays = ArrayRecord(get_peft_model_state_dict(init_model))

    strategy = FedAdam(
        fraction_train=cfg.strategy.fraction_train,
        fraction_evaluate=cfg.strategy.fraction_evaluate,
    )

    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"save_path": save_path}),
        #evaluate_config=ConfigRecord({"save_path": save_path}),
        num_rounds=num_rounds,
        evaluate_fn=get_evaluate_fn(cfg.model, cfg.train.training_arguments, cfg.dataset.path),
    )


def get_evaluate_fn(model_cfg, training_args_cfg, dataset_path_cfg):

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        if server_round > 0:
            training_arguments = TrainingArguments(**training_args_cfg)

            eval_dataset = load_eval_data(dataset_path_cfg)
            tokenizer, data_collator = get_tokenizer_and_data_collator(model_cfg.name)

            model = get_model(model_cfg)
            set_peft_model_state_dict(model, arrays.to_torch_state_dict())

            trainer = CustomTrainer(
                model=model,
                args=training_arguments,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics_per_class,
            )

            print("CENTRAL EVAL START")
            eval_results = trainer.evaluate()
            print("CENTRAL EVAL END")
            metrics = {
                "eval_loss": eval_results.get("eval_loss"),
                "accuracy": eval_results.get("eval_accuracy"),
                "precision": eval_results.get("eval_precision"),
                "recall": eval_results.get("eval_recall"),
                "f1": eval_results.get("eval_f1"),
            }
            return MetricRecord(metrics)
        return MetricRecord()

    return evaluate
