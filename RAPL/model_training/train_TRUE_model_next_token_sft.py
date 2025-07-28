import os

import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import login
from sacred import Experiment
from sacred.observers import FileStorageObserver
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

from .configs_and_utils.model_training_config import make_model_training_config

ex = Experiment("next_token_classification")
make_model_training_config(ex)


@ex.config_hook
def setup_observer(config, command_name, logger):
    ex.observers.append(FileStorageObserver(config["out_dir"]))
    return config


PROMPT = (
    "Task: Predict whether human annotators will correctly identify the more helpful and honest AI assistant response.\n\n"
    "A human annotator was shown the following conversation between a user and an AI assistant:\n"
    "\n{conv}\n\n"
    "The annotator was also shown the following possible AI responses to the final query:\n"
    "Possible Assistant Response 1: {r1}\n"
    "Possible Assistant Response 2: {r2}\n\n"
    "Complete the following sentence with either 'c orrect' or 'incorrect'.\n"
    "Based on the provided information, the human annotator's identification of the more helpful and honest AI response will be"
)


def load_data(csv_file, text_column, label_column):
    df = pd.read_csv(csv_file)

    pat1 = r"\n\nPossible Assistant Response 1: "
    pat2 = r"\n\nPossible Assistant Response 2: "

    conv_and_r1_r2 = df[text_column].str.split(pat1, n=1, expand=True)
    conv = conv_and_r1_r2[0]

    r1_and_r2 = conv_and_r1_r2[1].str.split(pat2, n=1, expand=True)
    r1 = r1_and_r2[0]
    r2 = r1_and_r2[1]

    label_map = {True: " correct", False: " incorrect", 1: " correct", 0: " incorrect"}
    label = df[label_column].map(label_map)

    return pd.DataFrame({"conv": conv, "r1": r1, "r2": r2, "label": label})


def to_text(example, tokenizer):
    example["text"] = (
        PROMPT.format(conv=example["conv"], r1=example["r1"], r2=example["r2"])
        + example["label"]
        + tokenizer.eos_token
    )
    return example


def tokenize(ex, tokenizer, max_length):
    enc = tokenizer(
        ex["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    seq_len = sum(attention_mask)

    answer_ids = tokenizer(ex["label"], add_special_tokens=False).input_ids
    n_answer = len(answer_ids)

    labels = [-100] * len(input_ids)
    answer_start = seq_len - n_answer - 1
    labels[answer_start : answer_start + n_answer] = answer_ids
    enc["labels"] = labels
    return enc


@ex.automain
def main(
    _config,
    _run,
):
    observer = _run.observers[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if _config["auth_token"]:
        login(token=_config["auth_token"])

    tokenizer = AutoTokenizer.from_pretrained(_config["model_name"])
    # Need to do this for GPT2 and Llama because they doesn't have official pad tokens.
    if (
        tokenizer.pad_token is None
        and "<|finetune_right_pad_id|>" in tokenizer.get_vocab()
    ):
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = tokenizer.get_vocab()[tokenizer.pad_token]
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    torch.set_anomaly_enabled(True)

    model = AutoModelForCausalLM.from_pretrained(
        _config["model_name"],
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="eager",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    model.to(device)

    train_dataset_raw = load_data(
        _config["train_data_path"],
        _config["prompt_response_group_col_name"],
        _config["correctness_label_col_name"],
    )
    train_dataset = Dataset.from_pandas(train_dataset_raw)
    train_dataset = train_dataset.map(
        to_text, remove_columns=["conv", "r1", "r2"], fn_kwargs={"tokenizer": tokenizer}
    )
    train_dataset = train_dataset.map(
        tokenize,
        remove_columns=["label", "text"],
        fn_kwargs={"tokenizer": tokenizer, "max_length": _config["max_length"]},
    )
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids"]) <= _config["max_length"]
    )
    print(f"Training examples: {len(train_dataset_raw)}")

    val_dataset_raw = load_data(
        _config["eval_data_path"],
        _config["prompt_response_group_col_name"],
        _config["correctness_label_col_name"],
    )
    val_dataset = Dataset.from_pandas(val_dataset_raw)
    val_dataset = val_dataset.map(
        to_text, remove_columns=["conv", "r1", "r2"], fn_kwargs={"tokenizer": tokenizer}
    )
    val_dataset = val_dataset.map(
        tokenize,
        remove_columns=["label", "text"],
        fn_kwargs={"tokenizer": tokenizer, "max_length": _config["max_length"]},
    )
    val_dataset = val_dataset.filter(
        lambda x: len(x["input_ids"]) <= _config["max_length"]
    )
    print(f"Validation examples: {len(val_dataset_raw)}")

    gradient_checkpointing_kwargs = {}
    training_args = TrainingArguments(
        output_dir=os.path.join(observer.dir, "checkpoints"),
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="no",
        save_only_model=True,
        per_device_train_batch_size=_config["per_device_train_batch_size"],
        per_device_eval_batch_size=_config["per_device_eval_batch_size"],
        learning_rate=_config["learning_rate"],
        weight_decay=_config["weight_decay"],
        num_train_epochs=_config["num_train_epochs"],
        lr_scheduler_type=_config["lr_scheduler_type"],
        warmup_steps=_config["lr_warmup_steps"],
        warmup_ratio=_config["lr_warmup_ratio"],
        logging_dir=os.path.join(observer.dir, "logs"),
        logging_steps=10,
        report_to=["tensorboard"],
        save_total_limit=2,
        fp16=False,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=_config["gradient_checkpointing"],
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        gradient_accumulation_steps=_config["gradient_accumulation_steps"],
        push_to_hub=False,
        logging_first_step=True,
        save_safetensors=False,
        seed=_config["seed"],
        data_seed=_config["seed"],
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()
    breakpoint()
    saved_model_dir = f"{observer.dir}-last_checkpoint"
    trainer.save_model(saved_model_dir)
    print(f"Saved last checkpoint of the model to {saved_model_dir}")
