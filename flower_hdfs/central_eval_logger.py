import json

from omegaconf import OmegaConf

TYPE_KEY = "type"
ROUND_KEY = "round"

def write_header(cfg, eval_log_path):
    subset_cfg = OmegaConf.create({
        "model": cfg.model,
        "train": cfg.train,
        "strategy": cfg.strategy,
        "dirichlet_alpha": cfg.dataset.dirichlet.alpha,
        "num_server_rounds": cfg.num_server_rounds
    })
    header = {
        "type": "params",
        "config": OmegaConf.to_container(subset_cfg, resolve=True)
    }

    with open(eval_log_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")

def write_metrics(metrics, server_round, eval_log_path):
    metrics[TYPE_KEY] = "metrics"
    metrics[ROUND_KEY] = server_round

    with open(eval_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")

    del metrics[TYPE_KEY]
    del metrics[ROUND_KEY]