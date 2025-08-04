import gc
import os
from functools import partial
from typing import Any, Dict, Optional

import pandas as pd
import torch
from datasets import concatenate_datasets  # type: ignore
from peft import LoraConfig, TaskType, get_peft_model
from sacred import Experiment
from sacred.observers import FileStorageObserver
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from transformers import (  # type: ignore
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import EvalPrediction  # type: ignore

from .configs_and_utils.data_utils import (
    TRUECollatorWithPadding,
    true_format_dataset_creation,
)
from .configs_and_utils.eval_utils import (
    apply_calibration,
    train_calibration,
)
from .configs_and_utils.model_training_config import make_model_training_config
from .configs_and_utils.training_utils import (
    EMA,
    EMACallback,
    get_cosine_decay_lr_lambda,
    get_step_decay_lr_lambda,
)

script_dir = os.path.dirname(__file__)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.cuda.empty_cache()
gc.collect()

ex = Experiment("train_value_head_TRUE_models")

make_model_training_config(ex)


@ex.config_hook
def setup_observer(config, command_name, logger):
    ex.observers.append(FileStorageObserver(config["out_dir"]))
    return config


class TRUETrainer(Trainer):
    def __init__(self, *args, lr_lambda=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_lambda = lr_lambda
        # self.score_scale = score_scale

    # def compute_loss(
    #     self, model, inputs, return_outputs=False, num_items_in_batch=None
    # ):
    #     outputs = model(
    #         input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
    #     )
    #     logits = outputs.logits * self.score_scale
    #     loss = nn.functional.cross_entropy(logits, inputs["labels"], reduction="mean")

    #     if return_outputs:
    #         return loss, outputs
    #     return loss

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        outputs = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        logits = outputs.logits.flatten()
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, inputs["labels"])
        if return_outputs:
            return loss, outputs
        return loss

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_lambda is not None:
            lr_lambda = partial(
                self.lr_lambda,
                num_training_steps=num_training_steps,
            )
            self.lr_scheduler = LambdaLR(optimizer, lr_lambda)
            return self.lr_scheduler
        else:
            return super().create_scheduler(num_training_steps, optimizer)

    @classmethod
    def augmented_compute_metrics(cls, **kwargs):
        # score_scale = kwargs["score_scale"]
        use_calibration = kwargs.get("use_calibration", False)
        if use_calibration:
            print(
                "The second half of the evaluation predictions will be used for calibration."
            )
        out_dir = kwargs.get("out_dir")

        def compute_metrics(eval_prediction: EvalPrediction):
            val_logits = torch.from_numpy(eval_prediction.predictions).squeeze(1)
            val_labels = torch.tensor(eval_prediction.label_ids).float()
            metrics = {}
            if use_calibration:
                mid = len(val_logits) // 2
                cal_logits, cal_labels = val_logits[:mid], val_labels[:mid]
                val_logits, val_labels = val_logits[mid:], val_labels[mid:]
                calibrator = train_calibration(cal_logits, cal_labels)
                calibrated_val_probs = apply_calibration(calibrator, val_logits)
                pd.DataFrame(
                    {"logit": cal_logits.cpu(), "label": cal_labels.cpu()}
                ).to_csv(f"{out_dir}/calibration_data.csv", index=False)

                calibrated_loss = nn.BCELoss()(calibrated_val_probs, val_labels)

                metrics["calibration_val_loss"] = calibrated_loss
                if calibrator.kind == "logistic":
                    metrics["calibration_scale"] = calibrator.coef_[0][0]
                    metrics["calibration_bias"] = calibrator.intercept_[0]

            loss = nn.BCEWithLogitsLoss()(val_logits, val_labels)
            metrics["loss"] = loss.item()

            probabilities = torch.sigmoid(val_logits)
            predicted_labels = (probabilities > 0.5).int()
            accuracy = (predicted_labels == val_labels).float().mean().item()
            metrics["accuracy"] = accuracy

            val_df = pd.DataFrame(
                {
                    "logit": val_logits.cpu(),
                    "prob": probabilities.cpu(),
                    "label": val_labels.cpu(),
                }
            )
            val_df.to_csv(os.path.join(out_dir, "val_data.csv"), index=False)

            return metrics

        return compute_metrics


@ex.automain
def main(_config, _run):
    print("TRUETrainer is being used.")

    observer = _run.observers[0]
    set_seed(_config["seed"])

    tokenizer_name = (
        _config["tokenizer_name"]
        if _config["tokenizer_name"] is not None
        else _config["model_name"]
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, token=_config["auth_token"]
    )

    # Llama 3.1
    if (
        tokenizer.pad_token is None
        and "<|finetune_right_pad_id|>" in tokenizer.get_vocab()
    ):
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = tokenizer.get_vocab()[tokenizer.pad_token]
    # Llama and GPT
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    torch.set_anomaly_enabled(True)

    train_dataset = true_format_dataset_creation(
        tokenizer=tokenizer,
        num_proc=_config["num_proc"],
        max_length=_config["max_length"],
        split="train",
        dataset_size=_config["train_dataset_size"],
        data_path=_config["train_data_path"],
        data_seed=_config["data_seed"],
        prompt_response_group_col=_config["prompt_response_group_col_name"],
        correctness_col_name=_config["correctness_label_col_name"],
        other_cols_to_keep=_config["other_cols_to_keep"],
    )

    eval_dataset = true_format_dataset_creation(
        tokenizer=tokenizer,
        num_proc=_config["num_proc"],
        max_length=_config["max_length"],
        split="val",
        dataset_size=_config["eval_dataset_size"],
        data_path=_config["eval_data_path"],
        data_seed=_config["data_seed"],
        prompt_response_group_col=_config["prompt_response_group_col_name"],
        correctness_col_name=_config["correctness_label_col_name"],
        other_cols_to_keep=_config["other_cols_to_keep"],
    )
    cal_dataset = None
    if _config["cal_data_path"] is not None:
        cal_dataset = true_format_dataset_creation(
            tokenizer=tokenizer,
            num_proc=_config["num_proc"],
            max_length=_config["max_length"],
            split="val",
            dataset_size=_config["eval_dataset_size"],
            data_path=_config["cal_data_path"],
            data_seed=_config["data_seed"],
            prompt_response_group_col=_config["prompt_response_group_col_name"],
            correctness_col_name=_config["correctness_label_col_name"],
            other_cols_to_keep=_config["other_cols_to_keep"],
        )

        eval_dataset = concatenate_datasets([eval_dataset, cal_dataset])
    trainer_kwargs: Dict[str, Any] = {}
    lr_scheduler_type = _config["lr_scheduler_type"]
    lr_scheduler_kwargs = {}
    if lr_scheduler_type == "step_decay":
        lr_scheduler_type = "constant"
        trainer_kwargs["lr_lambda"] = get_step_decay_lr_lambda
    elif lr_scheduler_type == "cosine_w_floor":
        lr_scheduler_type = "constant"
        trainer_kwargs["lr_lambda"] = get_cosine_decay_lr_lambda
    elif lr_scheduler_type == "cosine_with_restarts":
        lr_scheduler_kwargs["num_cycles"] = _config["lr_num_cycles"]

    # trainer_kwargs["score_scale"] = _config["score_scale"]

    gradient_checkpointing_kwargs = None
    if _config["gradient_checkpointing"]:
        gradient_checkpointing_kwargs = {
            "use_reentrant": False,
        }
    training_args = TrainingArguments(
        output_dir=observer.dir,
        report_to="tensorboard",
        learning_rate=_config["learning_rate"],
        per_device_train_batch_size=_config["per_device_train_batch_size"],
        per_device_eval_batch_size=_config["per_device_eval_batch_size"],
        num_train_epochs=_config["num_train_epochs"],
        weight_decay=_config["weight_decay"],
        eval_strategy=_config["eval_strategy"],
        save_strategy="no",
        save_only_model=False,
        gradient_accumulation_steps=_config["gradient_accumulation_steps"],
        gradient_checkpointing=_config["gradient_checkpointing"],
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        fsdp_config={"activation_checkpointing": _config["activation_checkpointing"]},
        local_rank=_config["local_rank"],
        remove_unused_columns=False,
        bf16=_config["bf16"],
        fp16=False,
        logging_strategy="steps",
        logging_steps=10,
        optim=_config["optim"],
        lr_scheduler_type=lr_scheduler_type,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        warmup_ratio=_config["lr_warmup_ratio"],
        seed=_config["seed"],
        data_seed=_config["data_seed"],
        full_determinism=_config["full_determinism"],
        max_grad_norm=_config["max_grad_norm"],
        torch_empty_cache_steps=_config["torch_empty_cache_steps"],
        label_names=(["labels"]),
    )

    callbacks = []
    # callbacks.append(GradientLoggingCallback())

    def model_init():
        model_to_use = (
            _config["model_checkpoint"]
            if _config["model_checkpoint"] is not None
            else _config["model_name"]
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_to_use,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            token=_config["auth_token"],
        )
        model.score.weight.data *= _config["weight_scale"]

        # weights = model.score.weight.data.numpy()

        # plt.figure(figsize=(10, 6))
        # plt.hist(weights.flatten(), bins=50, density=True)
        # plt.title('Distribution of Classifier Weights')
        # plt.xlabel('Weight Value')
        # plt.ylabel('Density')
        # plt.grid(True)
        # plt.savefig('weight_distribution.png')
        # plt.close()

        if _config["use_peft"]:
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=_config["lora_rank"],
                lora_alpha=_config["lora_alpha"],
                lora_dropout=_config["lora_dropout"],
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        if _config["ema_decay"] > 0:
            ema_handler = EMA(
                model,
                decay=_config["ema_decay"],
                path=f"{observer.dir}-ema-weights",
            )
            callbacks.append(EMACallback(ema_handler))

        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = not _config["gradient_checkpointing"]
        return model

    trainer = TRUETrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=TRUECollatorWithPadding(
            tokenizer=tokenizer,
            padding=_config["padding_strategy"],
            pad_to_multiple_of=_config["pad_to_multiple_of"],
        ),
        compute_metrics=TRUETrainer.augmented_compute_metrics(
            # score_scale=_config["score_scale"],
            use_calibration=cal_dataset is not None,
            out_dir=observer.dir,
        ),
        callbacks=callbacks,
        **trainer_kwargs,
    )

    trainer.train(_config["resume_from_checkpoint"])
    # saved_model_dir = f"{observer.dir}-last_checkpoint"
    # trainer.save_model(saved_model_dir)
    # print(f"Saved last checkpoint of the model to {saved_model_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = trainer.model
    model = model.to(device).eval()

    del trainer
    del train_dataset, eval_dataset
    gc.collect()
    torch.cuda.empty_cache()

    if _config["eval_true_model"]:
        print("Evaluating TRUE model on LIE dataset")
        from .configs_and_utils.data_utils import LIE_TRAIN_DATA_PATH
        from .configs_and_utils.eval_utils import eval_true_model

        val_dataset_1 = true_format_dataset_creation(
            tokenizer=tokenizer,
            data_seed=_config["data_seed"],
            data_path=LIE_TRAIN_DATA_PATH,
            prompt_response_group_col="prompt_response_group_1",
            split="val",
            max_length=_config["max_length"],
            num_proc=_config["num_proc"],
        )
        val_dataset_2 = true_format_dataset_creation(
            tokenizer=tokenizer,
            data_seed=_config["data_seed"],
            data_path=LIE_TRAIN_DATA_PATH,
            prompt_response_group_col="prompt_response_group_2",
            split="val",
            max_length=_config["max_length"],
            num_proc=_config["num_proc"],
        )

        val_dataset1, val_dataset2, loss1, loss2 = eval_true_model(
            val_dataset_1=val_dataset_1,
            val_dataset_2=val_dataset_2,
            cal_dataset=cal_dataset,
            model=model,
            tokenizer=tokenizer,
            batch_size=_config["per_device_eval_batch_size"],
            # score_scale=_config["score_scale"],
        )
        loss = (loss1 + loss2) / 2
        new_df = pd.DataFrame(
            [
                {
                    "eval_results_tag": _config["eval_results_key"],
                    "avg_loss": loss,
                    "loss1": loss1,
                    "loss2": loss2,
                }
            ],
        )
        new_df.to_csv(
            f"{_config['eval_results_dir']}/true_model_lie_results.csv",
            mode="a",
            header=False,
            index=False,
        )

        val_dataset1.to_csv(f"{observer.dir}/lie_train_true_eval_group_1.csv")
        val_dataset2.to_csv(f"{observer.dir}/lie_train_true_eval_group_2.csv")
