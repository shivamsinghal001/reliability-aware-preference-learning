import gc
import os
from functools import partial
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import log_loss  # type: ignore
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from transformers import (  # type: ignore
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)
from transformers.trainer_pt_utils import nested_detach  # type: ignore
from transformers.trainer_utils import EvalPrediction  # type: ignore
from trl import RewardConfig, RewardTrainer  # type: ignore

from .configs_and_utils.data_utils import (
    RAPLCollatorWithPadding,
    rlhf_format_dataset_creation,
)
from .configs_and_utils.model_training_config import make_model_training_config
from .configs_and_utils.training_utils import (
    EMA,
    EMACallback,
    get_cosine_decay_lr_lambda,
    get_step_decay_lr_lambda,
)

# import matplotlib.pyplot as plt

script_dir = os.path.dirname(__file__)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.cuda.empty_cache()
gc.collect()

ex = Experiment("train_reward_models")

make_model_training_config(ex)


@ex.config_hook
def setup_observer(config, command_name, logger):
    ex.observers.append(FileStorageObserver(config["out_dir"]))
    return config


class RAPLTrainer(RewardTrainer):
    def __init__(
        self,
        *args,
        lr_lambda=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lr_lambda = lr_lambda

    @classmethod
    def standard_loss(cls, rewards_chosen, rewards_rejected):
        return -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

    @classmethod
    def beta_adjustment_loss(cls, rewards_chosen, rewards_rejected, beta):
        return -nn.functional.logsigmoid(
            beta * (rewards_chosen - rewards_rejected)
        ).mean()

    @classmethod
    def prob_adjustment_loss(cls, rewards_chosen, rewards_rejected, p):
        boltzmann_act_prob = p * (
            1 / (1 + torch.exp(-(rewards_chosen - rewards_rejected)))
        )
        rand_act_prob = (1 - p) * 0.5
        return -torch.log(boltzmann_act_prob + rand_act_prob).mean()

    @classmethod
    def margin_loss(cls, rewards_chosen, rewards_rejected, margin):
        return -nn.functional.logsigmoid((rewards_chosen - rewards_rejected - margin))

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        loss, rewards = super().compute_loss(model, inputs, True, num_items_in_batch)
        rewards_chosen = rewards["rewards_chosen"]
        rewards_rejected = rewards["rewards_rejected"]
        if "beta" in inputs:
            loss = self.beta_adjustment_loss(
                rewards_chosen, rewards_rejected, inputs["beta"]
            )

        elif "p" in inputs:
            loss = self.prob_adjustment_loss(
                rewards_chosen, rewards_rejected, inputs["p"]
            )

        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
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

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, logits_dict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()
        logits_tuple = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        detached_tuple = nested_detach(logits_tuple)
        logits: torch.Tensor = torch.stack(detached_tuple).mean(dim=2).T

        labels = torch.zeros(logits.shape[0])
        labels = self._prepare_inputs(labels)

        return loss, logits, labels

    @classmethod
    def augmented_compute_metrics(cls, **kwargs):
        eval_dataset = kwargs["eval_dataset"]
        eval_betas = (
            torch.tensor(eval_dataset["beta"])
            if "beta" in eval_dataset.features
            else None
        )
        eval_p = (
            torch.tensor(eval_dataset["p"]) if "p" in eval_dataset.features else None
        )
        eval_margins = (
            torch.tensor(eval_dataset["margin"])
            if "margin" in eval_dataset.features
            else None
        )
        out_dir = kwargs["out_dir"]

        def compute_metrics(
            eval_prediction: EvalPrediction,
            beta=eval_betas,
            p=eval_p,
            margin=eval_margins,
            eval_dataset=eval_dataset,
            out_dir=out_dir,
        ):
            rewards_chosen, rewards_rejected = eval_prediction.predictions.T
            rewards_chosen = torch.from_numpy(rewards_chosen)
            rewards_rejected = torch.from_numpy(rewards_rejected)
            eval_dataset_df = eval_dataset.to_pandas()
            eval_dataset_df["rewards_chosen"] = rewards_chosen
            eval_dataset_df["rewards_rejected"] = rewards_rejected
            eval_dataset_df.to_csv(
                os.path.join(out_dir, "eval_dataset_rewards.csv"),
            )
            if beta is not None:
                loss = cls.beta_adjustment_loss(rewards_chosen, rewards_rejected, beta)
            elif p is not None:
                loss = cls.prob_adjustment_loss(rewards_chosen, rewards_rejected, p)
            elif margin is not None:
                loss = cls.margin_loss(rewards_chosen, rewards_rejected, margin)
            else:
                loss = cls.standard_loss(rewards_chosen, rewards_rejected)

            accuracy = torch.mean((loss < np.log(2)).float())

            # calibrated loss
            x1 = rewards_chosen - rewards_rejected
            y1 = torch.ones(x1.shape[0])
            x2 = rewards_rejected - rewards_chosen
            y2 = torch.zeros(x2.shape[0])
            x = torch.cat((x1, x2), dim=0)
            x = x.reshape(-1, 1)
            y = torch.cat((y1, y2), dim=0)
            model = LogisticRegression()
            model.fit(x, y)
            probabilities = model.predict_proba(x)
            calibrated_loss = log_loss(y, probabilities)
            return {
                "loss": loss.item(),
                "calibrated_loss": calibrated_loss,
                "accuracy": accuracy.item(),
            }

        return compute_metrics


@ex.automain
def main(_config, _run):
    print("RAPLTrainer is being used.")
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

    train_dataset = rlhf_format_dataset_creation(
        tokenizer=tokenizer,
        num_proc=_config["num_proc"],
        max_length=_config["max_length"],
        split="train",
        dataset_size=_config["train_dataset_size"],
        data_path=_config["train_data_path"],
        data_seed=_config["data_seed"],
        chosen_col_name=_config["chosen_col_name"],
        rejected_col_name=_config["rejected_col_name"],
        prompt_col_name=_config["prompt_col_name"],
        rapl_col_id=_config["rapl_col_id"],
        rapl_method=_config["rapl_method"],
        is_rapl_score_correctness_prob=_config["is_rapl_score_correctness_prob"],
        reward_bench_eval=_config["reward_bench_eval"],
        other_cols_to_keep=_config["other_cols_to_keep"],
    )

    eval_dataset = rlhf_format_dataset_creation(
        tokenizer=tokenizer,
        num_proc=_config["num_proc"],
        max_length=_config["max_length"],
        split="val",
        dataset_size=_config["eval_dataset_size"],
        data_path=_config["eval_data_path"],
        data_seed=_config["data_seed"],
        chosen_col_name=_config["chosen_col_name"],
        rejected_col_name=_config["rejected_col_name"],
        prompt_col_name=_config["prompt_col_name"],
        rapl_col_id=_config["rapl_col_id"],
        rapl_method=_config["rapl_method"],
        is_rapl_score_correctness_prob=_config["is_rapl_score_correctness_prob"],
        reward_bench_eval=_config["reward_bench_eval"],
        other_cols_to_keep=_config["other_cols_to_keep"],
    )

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

    gradient_checkpointing_kwargs = None
    if _config["gradient_checkpointing"]:
        gradient_checkpointing_kwargs = {
            "use_reentrant": False,
        }
    training_args = RewardConfig(
        output_dir=observer.dir,
        report_to="tensorboard",
        learning_rate=_config["learning_rate"],
        per_device_train_batch_size=_config["per_device_train_batch_size"],
        per_device_eval_batch_size=_config["per_device_eval_batch_size"],
        num_train_epochs=_config["num_train_epochs"],
        weight_decay=_config["weight_decay"],
        eval_strategy=_config["eval_strategy"],
        save_strategy="no",
        save_only_model=True,
        gradient_accumulation_steps=_config["gradient_accumulation_steps"],
        gradient_checkpointing=_config["gradient_checkpointing"],
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        fsdp_config={"activation_checkpointing": _config["activation_checkpointing"]},
        local_rank=_config["local_rank"],
        remove_unused_columns=False,
        bf16=_config["bf16"],
        fp16=False,
        logging_strategy="steps",
        logging_steps=1,
        optim=_config["optim"],
        lr_scheduler_type=lr_scheduler_type,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        warmup_ratio=_config["lr_warmup_ratio"],
        seed=_config["seed"],
        data_seed=_config["data_seed"],
        full_determinism=_config["full_determinism"],
        dataset_num_proc=_config["num_proc"],
        max_grad_norm=_config["max_grad_norm"],
        center_rewards_coefficient=_config["center_rewards_coefficient"],
        torch_empty_cache_steps=_config["torch_empty_cache_steps"],
    )

    callbacks = []

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

    trainer = RAPLTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=RAPLCollatorWithPadding(
            tokenizer=tokenizer,
            padding=_config["padding_strategy"],
            pad_to_multiple_of=_config["pad_to_multiple_of"],
        ),
        compute_metrics=RAPLTrainer.augmented_compute_metrics(
            eval_dataset=eval_dataset, out_dir=observer.dir
        ),
        callbacks=callbacks,
        **trainer_kwargs,
    )

    trainer.train(_config["resume_from_checkpoint"])
    saved_model_dir = f"{observer.dir}-last_checkpoint"
    trainer.save_model(saved_model_dir)
    print(f"Saved last checkpoint of the model to {saved_model_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = trainer.model
    model = model.to(device).eval()

    del trainer
    del train_dataset, eval_dataset
    gc.collect()
    torch.cuda.empty_cache()

    if _config["eval_on_lie_test"]:
        from .configs_and_utils.data_utils import LIE_TEST_DATA_PATH
        from .configs_and_utils.eval_utils import eval_lie_test

        print("Evaluating on LIE test")

        lie_dataset = rlhf_format_dataset_creation(
            tokenizer=tokenizer,
            data_seed=_config["data_seed"],
            data_path=LIE_TEST_DATA_PATH,
            reward_bench_eval=False,
            split="test",
            max_length=_config["max_length"],
            num_proc=_config["num_proc"],
        )
        length_weight, correctness_weight, ratio, lie_dataset_results = eval_lie_test(
            lie_dataset=lie_dataset,
            tokenizer=tokenizer,
            model=model,
            batch_size=_config["per_device_eval_batch_size"],
        )
        new_df = pd.DataFrame(
            [
                {
                    "eval_results_tag": _config["eval_results_key"],
                    "length_weight": length_weight,
                    "correctness_weight": correctness_weight,
                    "ratio": ratio,
                }
            ]
        )
        new_df.to_csv(
            f"{_config['eval_results_dir']}/lie_test_results.csv",
            mode="a",
            header=False,
            index=False,
        )
        lie_dataset_results.to_csv(f"{observer.dir}/lie_test_rewards.csv")

        del (lie_dataset, new_df)
        gc.collect()
        torch.cuda.empty_cache()

    if _config["eval_on_reward_bench"]:
        from .configs_and_utils.eval_utils import (  # REWARD_BENCH_PRIOR_DATA_PATH,
            REWARD_BENCH_LEADERBOARD_DATA_PATH,
            eval_reward_bench,
        )

        print("Evaluating on Reward Bench")
        leaderboard_dataset = rlhf_format_dataset_creation(
            tokenizer=tokenizer,
            data_seed=_config["data_seed"],
            data_path=REWARD_BENCH_LEADERBOARD_DATA_PATH,
            reward_bench_eval=True,
            max_length=_config["max_length"],
            num_proc=_config["num_proc"],
        )
        # prior_dataset = rlhf_format_dataset_creation(
        #     tokenizer=tokenizer,
        #     data_seed=_config["data_seed"],
        #     data_path=REWARD_BENCH_PRIOR_DATA_PATH,
        #     max_length=_config["max_length"],
        #     num_proc=_config["num_proc"],
        # )
        results, dataset = eval_reward_bench(
            leaderboard_dataset=leaderboard_dataset,
            # prior_dataset=prior_dataset,
            tokenizer=tokenizer,
            model=model,
            batch_size=_config["per_device_eval_batch_size"],
        )
        new_df = pd.DataFrame(
            [{"eval_results_tag": _config["eval_results_key"], **results}]
        )
        new_df.to_csv(
            f"{_config['eval_results_dir']}/reward_bench_results.csv",
            mode="a",
            header=False,
            index=False,
        )
        dataset.to_csv(f"{observer.dir}/reward_bench_rewards.csv")
        del (leaderboard_dataset, new_df)  # prior_dataset
        gc.collect()
        torch.cuda.empty_cache()

    if _config["eval_on_anthropic_bias"]:
        from .configs_and_utils.eval_utils import (
            ANTHROPIC_BIAS_CATEGORY_COLS,
            ANTHROPIC_BIAS_DATA_PATH,
            eval_anthropic_bias,
        )

        print("Evaluating on the Anthropic bias dataset")
        dataset = rlhf_format_dataset_creation(
            tokenizer=tokenizer,
            data_seed=_config["data_seed"],
            data_path=ANTHROPIC_BIAS_DATA_PATH,
            other_cols_to_keep=ANTHROPIC_BIAS_CATEGORY_COLS + ["chosen_choice"],
            max_length=_config["max_length"],
            num_proc=_config["num_proc"],
        )
        results, anthropic_dataset = eval_anthropic_bias(
            dataset=dataset,
            tokenizer=tokenizer,
            model=model,
            batch_size=_config["per_device_eval_batch_size"],
        )
        new_df = pd.DataFrame(
            [{"eval_results_tag": _config["eval_results_key"], **results}]
        )
        new_df.to_csv(
            f"{_config['eval_results_dir']}/anthropic_bias_results.csv",
            mode="a",
            header=False,
            index=False,
        )
        anthropic_dataset.to_csv(f"{observer.dir}/anthropic_bias_rewards.csv")
        del (dataset, new_df)
        gc.collect()
        torch.cuda.empty_cache()

    if _config["eval_on_rm_bench"]:
        from .configs_and_utils.eval_utils import RM_BENCH_DATA_PATH, eval_rm_bench

        print("Evaluating on RM-bench")
        rm_bench_dataset = rlhf_format_dataset_creation(
            tokenizer=tokenizer,
            data_seed=_config["data_seed"],
            data_path=RM_BENCH_DATA_PATH,
            max_length=_config["max_length"],
            num_proc=_config["num_proc"],
            other_cols_to_keep=["type", "id"],
        )
        results = eval_rm_bench(
            rm_bench_dataset=rm_bench_dataset,
            tokenizer=tokenizer,
            model=model,
            batch_size=_config["per_device_eval_batch_size"],
        )
        new_df = pd.DataFrame(
            [{"eval_results_tag": _config["eval_results_key"], **results}]
        )
        new_df.to_csv(
            f"{_config['eval_results_dir']}/rm_bench_results.csv",
            mode="a",
            header=False,
            index=False,
        )
