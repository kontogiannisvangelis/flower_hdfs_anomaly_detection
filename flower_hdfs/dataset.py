from flwr_datasets.partitioner import DirichletPartitioner
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_from_disk

FDS = None
DS_EVAL = None
DS_TEST = None


def get_tokenizer_and_data_collator(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side="right"
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return tokenizer, data_collator


def load_data(
    partition_id: int, num_partitions: int, dirichlet_alpa: float, dataset_path: str
):
    """Load partition data."""
    global FDS
    if FDS is None:
        ds = load_from_disk(dataset_path)
        train_ds = ds["train"]
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions, partition_by="label", alpha=dirichlet_alpa
        )
        partitioner.dataset = train_ds
        FDS = partitioner
    return FDS.load_partition(partition_id)


def load_eval_data(dataset_path: str):
    """Load Evaluation data."""
    global DS_EVAL
    if DS_EVAL is None:
        ds = load_from_disk(dataset_path)
        DS_EVAL = ds["val"]
    return DS_EVAL


def load_test_data(dataset_path: str):
    """Load Test data."""
    global DS_TEST
    if DS_TEST is None:
        ds = load_from_disk(dataset_path)
        DS_TEST = ds["test"]
    return DS_TEST


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
