from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_from_disk

FDS = None  # Cache FederatedDataset


def get_tokenizer_and_data_collator(model_name: str):
    # From: https://huggingface.co/docs/trl/en/sft_trainer
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


def load_data(partition_id: int, num_partitions: int, dataset_path: str):
    """Load partition data."""
    # Only initialize `FederatedDataset` once
    global FDS
    if FDS is None:
        ds = load_from_disk(dataset_path)
        train_ds = ds['train']
        partitioner = IidPartitioner(num_partitions=num_partitions)
        partitioner.dataset = train_ds
        FDS = partitioner
    client_trainset = FDS.load_partition(partition_id)

    return client_trainset


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
