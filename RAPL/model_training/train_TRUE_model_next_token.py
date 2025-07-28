import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from datasets import Dataset
from huggingface_hub import login
from sacred import Experiment
from sacred.observers import FileStorageObserver
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from .configs_and_utils.model_training_config import make_model_training_config

ex = Experiment("next_token_classification")
make_model_training_config(ex)


@ex.config_hook
def setup_observer(config, command_name, logger):
    ex.observers.append(FileStorageObserver(config["out_dir"]))
    return config


# prompt_v1
PROMPT = (
    "Task: Predict whether human annotators will correctly identify the more helpful and honest AI assistant response.\n\n"
    "A human annotator was shown the following conversation between a user and an AI assistant:\n"
    "\n{}\n\n"
    "The annotator was also shown the following possible AI responses to the final query:\n"
    "Possible Assistant Response 1: {}\n"
    "Possible Assistant Response 2: {}\n\n"
    "Complete the following sentence with either 'Correct' or 'Incorrect'.\n"
    "Based on the provided information, the human annotator's identification of the more helpful and honest AI response will be "
)

# PROMPT = (
#     "<task>Predict whether human annotators will correctly or incorrectly identify the more helpful AI response.</task>\n\n"
#     "<conversation>\n{}\n</conversation>\n\n"
#     "<possible_response_1>\n{}\n</possible_response_1>\n\n"
#     "<possible_response_2>\n{}\n</possible_response_2>\n\n"
#     "Context: Human annotators evaluated these responses based on helpfulness, honesty, accuracy, and clarity.\n\n"
#     "<prediction>"
# )


class CustomEvalCallback(TrainerCallback):
    def __init__(self, model, raw_eval_data, eval_steps=10):
        self.model = model
        self.raw_eval_data = raw_eval_data
        self.eval_steps = eval_steps
        self.best_metric = -float("inf")
        # Create the compute_metrics function
        self.compute_metrics_fn = augmented_compute_metrics(model, raw_eval_data)

    def on_step_end(self, args, state, control, **kwargs):
        # Run evaluation at specified steps
        if state.global_step % self.eval_steps == 0:
            print(f"\nRunning custom evaluation at step {state.global_step}")

            # Use your existing compute_metrics function
            # We pass None for eval_pred since your function doesn't use it
            metrics = self.compute_metrics_fn(None)
            print(metrics)
            # Log all metrics
            for key, value in metrics.items():
                state.log_history.append(
                    {"step": state.global_step, f"eval_{key}": value}
                )

            print(f"Step {state.global_step} evaluation: {metrics}")

            # Track best model if desired
            if metrics.get("accuracy", 0) > self.best_metric:
                self.best_metric = metrics.get("accuracy", 0)
                control.should_save = True
                print(f"New best model with accuracy: {self.best_metric:.4f}")

        return control


def load_data(csv_file, text_column, label_column):
    df = pd.read_csv(csv_file)
    dataset = Dataset.from_dict(
        {
            "text": df[text_column],
            "label": df[label_column].astype(int),
        }
    )
    return dataset


class NextTokenClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, is_train_val=True, max_length=2048):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_additions = float("inf")
        self.is_train_val = is_train_val

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        text = example["text"]
        breakpoint()
        splits = text.split("\n\nPossible Assistant Response 1: ")
        question = splits[0]
        response_1 = splits[1].split("\n\nPossible Assistant Response 2: ")[0]
        response_2 = splits[1].split("\n\nPossible Assistant Response 2: ")[1]
        label_idx = example["label"]
        prediction = "Correct" if label_idx == 1 else "Incorrect"
        prompt = PROMPT.format(question, response_1, response_2)
        if self.is_train_val:
            prompt = prompt + prediction

        encodings = self.tokenizer(
            prompt,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        input_ids_tensor = torch.tensor(input_ids)
        attention_mask_tensor = torch.tensor(attention_mask)

        labels = torch.ones_like(input_ids_tensor) * -100

        if self.is_train_val:
            base_prompt = PROMPT.format(question, response_1, response_2)
            base_tokens = self.tokenizer(base_prompt)["input_ids"]
            prediction_start_idx = len(base_tokens)
            labels[prediction_start_idx:] = input_ids_tensor[prediction_start_idx:]

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": labels,
        }


def log_p_of_continuations(model, tokenizer, prefix, continuation):
    input_ids = torch.cat([prefix, continuation[:, :-1]], dim=1)

    with torch.no_grad():
        outputs = model(input_ids)

    # seq_len = outputs.logits.size(1)
    cont_len = continuation.size(1)

    logits_for_cont = outputs.logits[
        :, -cont_len:, :
    ]  # shape [1, cont_len, vocab_size]

    log_probs = F.log_softmax(logits_for_cont, dim=-1)  # [1, cont_len, vocab_size]

    gold_ids = continuation.view(-1)  # shape [cont_len]
    cont_log_probs = log_probs[0, range(cont_len), gold_ids]

    return cont_log_probs.sum().item()


class ConstrainedHeadModel(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return outputs

    def get_normalized_probs(self, conversation, response_1, response_2):
        # 1) Build the prompt (without the final label substring)
        #    up to <prediction>
        prompt_text = PROMPT.format(conversation, response_1, response_2)
        prefix_enc = self.tokenizer(prompt_text, return_tensors="pt")
        prefix_ids = prefix_enc["input_ids"].to(self.model.device)

        # 2) Build the label continuations
        #    e.g. "Correct</prediction>" or "Incorrect</prediction>"
        correct_label_str = "Correct</prediction>"
        incorrect_label_str = "Incorrect</prediction>"

        correct_ids = self.tokenizer(
            correct_label_str, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(self.model.device)
        incorrect_ids = self.tokenizer(
            incorrect_label_str, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(self.model.device)

        # 3) Compute log P( "Correct</prediction>" | prompt )
        #    and log P( "Incorrect</prediction>" | prompt ),
        #    using teacher forcing
        logp_correct = log_p_of_continuations(
            self.model, self.tokenizer, prefix_ids, correct_ids
        )
        logp_incorrect = log_p_of_continuations(
            self.model, self.tokenizer, prefix_ids, incorrect_ids
        )

        # 4) Softmax
        scores = torch.tensor([logp_correct, logp_incorrect])
        probs = torch.softmax(scores, dim=0)
        correct_prob, incorrect_prob = probs.tolist()

        return {
            "Correct": correct_prob,
            "Incorrect": incorrect_prob,
        }

    def generate_prediction(self, conversation, response_1, response_2):
        prompt_text = PROMPT.format(conversation, response_1, response_2)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                num_beams=1,
            )

        # decode
        generated_ids = outputs[0]
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[prompt_len:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # if you want to parse out the substring before "</prediction>"
        if "</prediction>" in generated_text:
            return generated_text.split("</prediction>")[0].strip()
        else:
            return generated_text.strip()


def augmented_compute_metrics(model, raw_eval_data):
    def compute_metrics(eval_pred):
        correct_count = 0
        total = 0

        correct_probs_list = []
        incorrect_probs_list = []
        gold_labels_list = []
        pred_labels_list = []

        for ex in raw_eval_data:
            text = ex["text"]
            gold_label = ex["label"]

            splits = text.split("\n\nPossible Assistant Response 1: ")
            question = splits[0]
            response_1 = splits[1].split("\n\nPossible Assistant Response 2: ")[0]
            response_2 = splits[1].split("\n\nPossible Assistant Response 2: ")[1]

            probs = model.get_normalized_probs(question, response_1, response_2)
            p_correct = probs["Correct"]
            p_incorrect = probs["Incorrect"]

            pred_label = 1 if p_correct > p_incorrect else 0

            correct_count += pred_label == gold_label
            total += 1

            correct_probs_list.append(p_correct)
            incorrect_probs_list.append(p_incorrect)
            gold_labels_list.append(gold_label)
            pred_labels_list.append(pred_label)

        accuracy = correct_count / total if total > 0 else 0.0
        avg_correct_prob = sum(correct_probs_list) / total if total > 0 else 0.0
        avg_incorrect_prob = sum(incorrect_probs_list) / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "avg_correct_prob": avg_correct_prob,
            "avg_incorrect_prob": avg_incorrect_prob,
        }

    return compute_metrics


@ex.automain
def main(
    _config,
    _run,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if _config["auth_token"]:
        login(token=_config["auth_token"])

    tokenizer = AutoTokenizer.from_pretrained(_config["model_name"])

    # special_tokens = {
    #     "additional_special_tokens": [
    #         "<task>",
    #         "</task>",
    #         "<conversation>",
    #         "</conversation>",
    #         "<possible_response_1>",
    #         "</possible_response_1>",
    #         "<possible_response_2>",
    #         "</possible_response_2>",
    #         "<prediction>",
    #         "</prediction>",
    #     ]
    # }

    # tokenizer.add_special_tokens(special_tokens)

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
    model.config.use_cache = False  # not gradient_checkpointing
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    print(f"Loading training data from: {_config['train_data_path']}")
    train_dataset_raw = load_data(
        _config["train_data_path"],
        _config["prompt_response_group_col_name"],
        _config["correctness_label_col_name"],
    )
    print(f"Training examples: {len(train_dataset_raw)}")

    print(f"Loading validation data from: {_config['eval_data_path']}")
    val_dataset_raw = load_data(
        _config["eval_data_path"],
        _config["prompt_response_group_col_name"],
        _config["correctness_label_col_name"],
    )
    print(f"Validation examples: {len(val_dataset_raw)}")

    train_dataset = NextTokenClassificationDataset(
        train_dataset_raw,
        tokenizer,
        is_train_val=True,
        max_length=_config["max_length"],
    )
    breakpoint()
    train_dataset.__getitem__(0)
    val_dataset = NextTokenClassificationDataset(
        val_dataset_raw, tokenizer, is_train_val=True, max_length=_config["max_length"]
    )

    constrained_model = ConstrainedHeadModel(model, tokenizer)
    constrained_model.to(device)

    gradient_checkpointing_kwargs = {}
    if _config["gradient_checkpointing"]:
        gradient_checkpointing_kwargs = {
            "use_reentrant": False,
        }

    training_args = TrainingArguments(
        output_dir=os.path.join(_run.observers[0].dir, "checkpoints"),
        evaluation_strategy="no",  # Turn off built-in evaluation
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
        logging_dir=os.path.join(_run.observers[0].dir, "logs"),
        logging_steps=10,
        report_to=["tensorboard"],
        save_total_limit=2,
        fp16=False,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=_config["gradient_checkpointing"],
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        gradient_accumulation_steps=_config["gradient_accumulation_steps"],
        eval_accumulation_steps=10,
        push_to_hub=False,
        logging_first_step=True,
        prediction_loss_only=True,
        save_safetensors=False,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=_config["padding_strategy"],
        pad_to_multiple_of=_config["pad_to_multiple_of"],
        return_tensors="pt",
    )

    trainer = Trainer(
        model=constrained_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=augmented_compute_metrics(constrained_model, val_dataset_raw),
    )

    trainer.add_callback(
        CustomEvalCallback(
            model=constrained_model,
            raw_eval_data=val_dataset_raw,
            eval_steps=10,
        )
    )
    print("Starting training")
    trainer.train()

    final_model_path = os.path.join(_run.observers[0].dir, "final_model")
    trainer.save_model(final_model_path)
    print(f"Model saved to {final_model_path}")
