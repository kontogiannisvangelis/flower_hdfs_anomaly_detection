import os
from datetime import datetime
from logging import INFO

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedAdam
from flwr.common.logger import log
from omegaconf import DictConfig
from peft import get_peft_model_state_dict

from flower_hdfs.central_eval_logger import (
    write_header,
    write_metrics,
    write_tuned_metrics,
)
from flower_hdfs.compute_metrics import compute_metrics_per_class_with_threshold
from flower_hdfs.dataset import replace_keys
from flower_hdfs.dataset_split import DatasetSplit
from flower_hdfs.models import get_model
from flower_hdfs.model_ops import score, predict
from flower_hdfs.threshold_tuning import find_best_threshold

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

    if cfg.strategy.type == "avg":
        strategy_cls = FedAvg
    elif cfg.strategy.type == "adam":
        strategy_cls = FedAdam
    else:
        raise ValueError(f"Unknown strategy type: {cfg.type}")

    strategy = strategy_cls(
        fraction_train=cfg.strategy.fraction_train,
        fraction_evaluate=cfg.strategy.fraction_evaluate,
    )

    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"save_path": save_path}),
        num_rounds=num_rounds,
        evaluate_fn=get_evaluate_fn(cfg, save_path),
    )


def get_evaluate_fn(cfg: DictConfig, save_path: str):

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        os.makedirs(save_path, exist_ok=True)
        eval_log_path = os.path.join(save_path, "eval_logs.jsonl")

        if server_round == 0:
            write_header(cfg, eval_log_path)
            return MetricRecord()
        else:
            metrics = score(cfg, save_path, arrays)
            write_metrics(metrics, server_round, eval_log_path)

            # if last round perform threshold tuning
            if server_round == cfg.num_server_rounds:
                log(INFO, "THRESHOLD TUNING")
                preds = predict(cfg, save_path, arrays, DatasetSplit.EVAL)
                logits = preds.predictions
                labels = preds.label_ids

                best_thr, best_f1 = find_best_threshold(logits, labels)

                # evaluate new threshold on EVAL split
                metrics_tuned = compute_metrics_per_class_with_threshold(
                    logits, labels, best_thr, DatasetSplit.EVAL
                )
                write_tuned_metrics(metrics_tuned, server_round, eval_log_path)

                # evaluate new threshold on TEST split
                preds = predict(cfg, save_path, arrays, DatasetSplit.TEST)
                logits = preds.predictions
                labels = preds.label_ids
                metrics_tuned = compute_metrics_per_class_with_threshold(
                    logits, labels, best_thr, DatasetSplit.TEST
                )
                write_tuned_metrics(metrics_tuned, server_round, eval_log_path)
            return MetricRecord(metrics)

    return evaluate
