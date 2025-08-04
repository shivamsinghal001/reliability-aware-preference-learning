import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_dataset  # type: ignore
from transformers import AutoTokenizer  # type: ignore
from trl.trainer.utils import RewardDataCollatorWithPadding  # type: ignore

from .training_utils import calculate_beta, calculate_margin, calculate_p

script_dir = os.path.dirname(__file__)


# LIE dataset
LIE_TRAIN_DATA_PATH = os.path.join(
    script_dir, "../../datasets/LIE/LIE_dataset_train.csv"
)
LIE_VAL_DATA_PATH = os.path.join(script_dir, "../../datasets/LIE/folds/LIE_val_0.csv")
LIE_TEST_DATA_PATH = os.path.join(script_dir, "../../datasets/LIE/LIE_dataset_test.csv")

# TRUE dataset
TRUE_TRAIN_DATA_PATH = os.path.join(
    script_dir, "../../datasets/TRUE/repeated_orders/double_full_dataset_train.csv"
)
TRUE_VAL_DATA_PATH = os.path.join(
    script_dir, "../../datasets/TRUE/repeated_orders/double_full_dataset_cal.csv"
)
TRUE_TEST_DATA_PATH = os.path.join(
    script_dir, "../../datasets/TRUE/repeated_orders/double_full_dataset_cal.csv"
)


class RLHFPreprocessor(object):
    def __init__(self, tokenizer, **tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs

    def __call__(self, examples):
        new_examples: dict = {}
        if "beta" in examples:
            new_examples["beta"] = examples["beta"]
        elif "p" in examples:
            new_examples["p"] = examples["p"]
        elif "margin" in examples:
            new_examples["margin"] = examples["margin"]

        if "chosen" in examples and "rejected" in examples:
            new_examples["input_ids_chosen"] = []
            new_examples["attention_mask_chosen"] = []
            new_examples["input_ids_rejected"] = []
            new_examples["attention_mask_rejected"] = []
            for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
                tokenized_chosen = self.tokenizer(chosen, **self.tokenizer_kwargs)
                tokenized_rejected = self.tokenizer(rejected, **self.tokenizer_kwargs)
                new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
                new_examples["attention_mask_chosen"].append(
                    tokenized_chosen["attention_mask"]
                )
                new_examples["input_ids_rejected"].append(
                    tokenized_rejected["input_ids"]
                )
                new_examples["attention_mask_rejected"].append(
                    tokenized_rejected["attention_mask"]
                )

        return new_examples


class TRUEPreprocessor(object):
    def __init__(self, tokenizer, **tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs

    def __call__(self, examples):
        new_examples: dict = {}
        if "correct_chosen" in examples:
            new_examples["correct_chosen"] = examples["correct_chosen"]

        if "prompt_response_group" in examples:
            new_examples["input_ids_prompt_response_group"] = []
            new_examples["attention_mask_prompt_response_group"] = []
            for prompt_response_group in examples["prompt_response_group"]:
                tokenized_prompt_response_group = self.tokenizer(
                    prompt_response_group, **self.tokenizer_kwargs
                )
                new_examples["input_ids_prompt_response_group"].append(
                    tokenized_prompt_response_group["input_ids"]
                )
                new_examples["attention_mask_prompt_response_group"].append(
                    tokenized_prompt_response_group["attention_mask"]
                )

        return new_examples


class LengthStudyRLHFPreprocessor(object):
    def __init__(self, tokenizer, **tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs

    def __call__(self, examples):
        new_examples: dict = {
            "input_ids_correct_concise": [],
            "attention_mask_correct_concise": [],
            "input_ids_correct_detailed": [],
            "attention_mask_correct_detailed": [],
            "input_ids_incorrect_concise": [],
            "attention_mask_incorrect_concise": [],
            "input_ids_incorrect_detailed": [],
            "attention_mask_incorrect_detailed": [],
        }

        for c_detailed, c_concise, i_detailed, i_concise in zip(
            examples["correct_detailed"],
            examples["correct_concise"],
            examples["incorrect_detailed"],
            examples["incorrect_concise"],
        ):
            tokenized_c_detailed = self.tokenizer(c_detailed, **self.tokenizer_kwargs)
            tokenized_c_concise = self.tokenizer(c_concise, **self.tokenizer_kwargs)
            tokenized_i_detailed = self.tokenizer(i_detailed, **self.tokenizer_kwargs)
            tokenized_i_concise = self.tokenizer(i_concise, **self.tokenizer_kwargs)

            new_examples["input_ids_correct_detailed"].append(
                tokenized_c_detailed["input_ids"]
            )
            new_examples["attention_mask_correct_detailed"].append(
                tokenized_c_detailed["attention_mask"]
            )
            new_examples["input_ids_correct_concise"].append(
                tokenized_c_concise["input_ids"]
            )
            new_examples["attention_mask_correct_concise"].append(
                tokenized_c_concise["attention_mask"]
            )
            new_examples["input_ids_incorrect_detailed"].append(
                tokenized_i_detailed["input_ids"]
            )
            new_examples["attention_mask_incorrect_detailed"].append(
                tokenized_i_detailed["attention_mask"]
            )
            new_examples["input_ids_incorrect_concise"].append(
                tokenized_i_concise["input_ids"]
            )
            new_examples["attention_mask_incorrect_concise"].append(
                tokenized_i_concise["attention_mask"]
            )

        return new_examples


# TODO: easy data threshold
def rlhf_format_dataset_creation(
    tokenizer: AutoTokenizer,
    num_proc: int = 24,
    max_length: int = 1024,
    split: str = "train",
    dataset_size: int = 0,
    data_path: str = LIE_TRAIN_DATA_PATH,
    data_seed: int = 42,
    chosen_col_name: str = "chosen",
    rejected_col_name: str = "rejected",
    prompt_col_name: str = "prompt",
    rapl_col_id: Optional[str] = None,
    rapl_method: str = "beta",
    is_rapl_score_correctness_prob: Optional[bool] = False,
    reward_bench_eval: bool = False,
    other_cols_to_keep: Optional[List[str]] = [],
) -> Dataset:
    assert split in [
        "train",
        "val",
        "test",
    ], "You must choose one of the train, val, or test sets!"
    df = pd.read_csv(data_path)
    if split == "test":
        cols_to_keep = [
            "correct_detailed",
            "correct_concise",
            "incorrect_detailed",
            "incorrect_concise",
        ]
        preprocess_fn = LengthStudyRLHFPreprocessor
    else:
        preprocess_fn = RLHFPreprocessor
        assert prompt_col_name in df.columns, f"Missing {prompt_col_name} column!"
        df.rename(columns={prompt_col_name: "prompt"}, inplace=True)
        assert chosen_col_name in df.columns, f"Missing {chosen_col_name} column!"
        df.rename(columns={chosen_col_name: "chosen"}, inplace=True)
        assert rejected_col_name in df.columns, f"Missing {rejected_col_name} column!"
        df.rename(columns={rejected_col_name: "rejected"}, inplace=True)
        cols_to_keep = ["prompt", "chosen", "rejected"]

        if rapl_col_id:
            assert rapl_col_id in df.columns, f"Missing {rapl_col_id} column!"
            assert rapl_method in [
                "beta",
                "p",
                "margin",
            ], "Invalid RAPL method! Must be either 'beta', 'p', or 'margin'."
            rapl_label = (
                "beta"
                if rapl_method == "beta"
                else "p" if rapl_method == "p" else "margin"
            )
            df.rename(columns={rapl_col_id: rapl_label}, inplace=True)
            rapl_function = (
                calculate_beta
                if rapl_method == "beta"
                else calculate_p if rapl_method == "p" else calculate_margin
            )
            if is_rapl_score_correctness_prob:
                df[rapl_label] = df[rapl_label].apply(lambda x: rapl_function(x))
            cols_to_keep.append(rapl_label)

        if reward_bench_eval:
            assert "subset" in df.columns, "Missing subset column!"
            cols_to_keep.append("subset")

    for col in other_cols_to_keep or []:
        if col in df.columns:
            cols_to_keep.append(col)
        else:
            warnings.warn(f"{col} is not present in the specified set!", UserWarning)

    df = df.sample(frac=1, random_state=data_seed).reset_index(drop=True)

    if dataset_size:
        df = df.iloc[:dataset_size]

    df = df[cols_to_keep]
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(
        preprocess_fn(tokenizer),
        batched=True,
        num_proc=num_proc,
        # remove_columns=cols_to_keep,
    )
    if split == "test":
        dataset = dataset.filter(
            lambda x: len(x["input_ids_correct_concise"]) <= max_length
            and len(x["input_ids_incorrect_concise"]) <= max_length
            and len(x["input_ids_correct_detailed"]) <= max_length
            and len(x["input_ids_incorrect_detailed"]) <= max_length
        )
    else:
        dataset = dataset.filter(
            lambda x: len(x["input_ids_chosen"]) <= max_length
            and len(x["input_ids_rejected"]) <= max_length
        )
    return dataset


def true_format_dataset_creation(
    tokenizer: AutoTokenizer,
    num_proc: int = 24,
    max_length: int = 1024,
    split: str = "train",
    dataset_size: int = 0,
    data_path: str = TRUE_TRAIN_DATA_PATH,
    data_seed: int = 42,
    prompt_response_group_col: str = "prompt_response_group",
    correctness_col_name: str = "correct_chosen",
    other_cols_to_keep: Optional[List[str]] = None,
) -> Dataset:
    assert split in [
        "train",
        "val",
        "test",
    ], "You must choose one of the train, val, or test sets!"
    df = pd.read_csv(data_path)
    assert (
        prompt_response_group_col in df.columns
    ), f"Missing {prompt_response_group_col} column!"
    df.rename(
        columns={prompt_response_group_col: "prompt_response_group"}, inplace=True
    )
    cols_to_keep = ["prompt_response_group"]

    if split == "train" or split == "val":
        assert (
            correctness_col_name in df.columns
        ), f"Missing {correctness_col_name} column!"
        df["correct_chosen"] = df[correctness_col_name]
        if correctness_col_name != "correct_chosen":
            df.drop(correctness_col_name, axis=1, inplace=True)
        cols_to_keep.append("correct_chosen")

    for col in other_cols_to_keep or []:
        if col in df.columns:
            cols_to_keep.append(col)
        else:
            warnings.warn(f"{col} is not present in the specified set!", UserWarning)

    df = df.sample(frac=1, random_state=data_seed).reset_index(drop=True)

    if dataset_size:
        df = df.iloc[:dataset_size]

    df = df[cols_to_keep]
    dataset = Dataset.from_pandas(df)

    dataset = dataset.map(
        TRUEPreprocessor(tokenizer, padding="longest", truncation=False),
        batched=True,
        num_proc=num_proc,
    )

    dataset = dataset.filter(
        lambda x: len(x["input_ids_prompt_response_group"]) <= max_length
    )

    return dataset


def huggingface_dataset_creation(
    split: str = "train",
    dataset_size: int = 0,
    data_path: str = "Anthropic/hh-rlhf",
    data_subset: str = "both",
) -> Dataset:
    datasets: List[Dataset] = []
    split = "test" if split == "val" else split
    if data_path == "Anthropic/hh-rlhf":
        assert data_subset in ["both", "harmless", "helpful"]
        if data_subset == "harmless" or data_subset == "both":
            datasets.append(
                load_dataset(
                    "Anthropic/hh-rlhf", data_dir="harmless-base", split=split
                ).map(lambda data: {"data_subset": "harmless"})
            )
        if data_subset == "helpful" or data_subset == "both":
            datasets.append(
                load_dataset(
                    "Anthropic/hh-rlhf", data_dir="helpful-base", split=split
                ).map(lambda data: {"data_subset": "helpful"})
            )
    elif data_path == "allenai/reward-bench":
        assert data_subset in ["raw", "filtered", "both"]
        if data_subset == "raw" or data_subset == "both":
            datasets.append(
                load_dataset("allenai/reward-bench", split="raw").map(
                    lambda data: {"data_subset": "raw"}
                )
            )
        if data_subset == "filtered" or data_subset == "both":
            datasets.append(
                load_dataset("allenai/reward-bench", split="filtered").map(
                    lambda data: {"data_subset": "filtered"}
                )
            )
    else:
        raise ValueError(
            "Must be either 'Anthropic/hh-rlhf' or 'allenai/reward-bench'. No other huggingface dataset is currently supported!"
        )

    if dataset_size:
        datasets = [
            dataset.select(range(dataset_size // len(datasets))) for dataset in datasets
        ]

    return concatenate_datasets(datasets)


@dataclass
class RAPLCollatorWithPadding(RewardDataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch: dict[str, Any] = super().__call__(features)

        beta = [feature["beta"] for feature in features if "beta" in feature]
        p = [feature["p"] for feature in features if "p" in feature]

        if len(beta) > 0:
            batch["beta"] = torch.tensor(beta, dtype=torch.float)
        elif len(p) > 0:
            batch["p"] = torch.tensor(p, dtype=torch.float)

        return batch


@dataclass
class TRUECollatorWithPadding(RewardDataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch: Dict[str, Union[bool, torch.Tensor, Optional[list]]] = {}
        correct_chosen = [
            float(feature["correct_chosen"])
            for feature in features
            if "correct_chosen" in feature
        ]

        features_prompt_response_group = [
            {
                "input_ids": feature["input_ids_prompt_response_group"],
                "attention_mask": feature["attention_mask_prompt_response_group"],
            }
            for feature in features
            if "input_ids_prompt_response_group" in feature
            and "attention_mask_prompt_response_group" in feature
        ]

        batch_prompts = self.tokenizer.pad(
            features_prompt_response_group,
            padding=self.padding,
            # pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if len(correct_chosen) > 0:
            batch["labels"] = torch.tensor(correct_chosen, dtype=torch.float32)

        if len(features_prompt_response_group) > 0:
            batch["input_ids"] = batch_prompts["input_ids"]
            batch["attention_mask"] = batch_prompts["attention_mask"]
        # batch["return_loss"] = True
        return batch
