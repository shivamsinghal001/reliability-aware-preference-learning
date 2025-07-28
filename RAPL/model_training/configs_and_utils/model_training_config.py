import os
from typing import List, Optional

import pandas as pd

from .data_utils import LIE_TRAIN_DATA_PATH, LIE_VAL_DATA_PATH
from .eval_utils import ANTHROPIC_BIAS_CATEGORY_COLS


# @config_ingredient.config
def make_model_training_config(ex):
    @ex.config
    def default_config():
        # model setup
        model_name: str = (
            "meta-llama/Meta-Llama-3-8B"  # must be a valid model either on the HuggingFace hub or a local download from HF
        )
        auth_token: Optional[str] = None  # noqa: F841
        tokenizer_name: Optional[str] = model_name  # noqa: F841
        resume_from_checkpoint: Optional[str] = None  # noqa: F841
        model_checkpoint: Optional[str] = None  # noqa: F841
        ema_decay: float = 0  # noqa: F841

        # data / batching
        local_rank: int = -1  # noqa: F841
        per_device_train_batch_size: int = 8
        per_device_eval_batch_size: int = 8  # noqa: F841
        gradient_accumulation_steps: int = 1
        gradient_checkpointing: bool = False  # noqa: F841
        activation_checkpointing: bool = False  # noqa: F841
        data_seed: int = 42  # noqa: F841
        num_proc: int = 24  # noqa: F841

        train_data_path: str = LIE_TRAIN_DATA_PATH  # noqa: F841
        eval_data_path: str = LIE_VAL_DATA_PATH  # noqa: F841
        cal_data_path: Optional[str] = None  # noqa: F841
        train_dataset_size: int = 0  # noqa: F841
        eval_dataset_size: int = 0  # noqa: F841
        test_dataset_size: int = 0  # noqa: F841

        # train / val
        prompt_col_name: str = "prompt"  # noqa: F841
        chosen_col_name: str = "chosen"  # noqa: F841
        rejected_col_name: str = "rejected"  # noqa: F841
        correctness_label_col_name: str = "correct_chosen"  # noqa: F841
        prompt_response_group_col_name: str = "prompt_response_group"  # noqa: F841

        # test
        prompt_col_name_test: str = "prompt"  # noqa: F841
        chosen_col_name_test: str = "chosen"  # noqa: F841
        rejected_col_name_test: str = "rejected"  # noqa: F841
        correctness_label_col_name_test: str = "correct_chosen"  # noqa: F841
        prompt_response_group_col_name_test: str = "prompt_response_group"  # noqa: F841

        rapl_col_id: Optional[str] = None  # noqa: F841
        rapl_method: str = "beta"  # noqa: F841
        is_rapl_score_correctness_prob: bool = False  # noqa: F841
        reward_bench_eval: bool = False  # noqa: F841
        other_cols_to_keep: Optional[List[str]] = []  # noqa: F841
        bf16: bool = True  # noqa: F841
        max_length: int = 1024  # noqa: F841
        padding_strategy: str = "longest"  # noqa: F841
        pad_to_multiple_of: int = 64  # noqa: F841
        eval_strategy: str = "epoch"  # noqa: F841

        # optimizer / model training
        learning_rate: float = 1e-6
        lr_scheduler_type: str = "cosine"
        lr_warmup_ratio: float = 0
        lr_warmup_steps: int = 0  # noqa: F841
        lr_num_cycles: int = 1  # noqa: F841
        weight_decay: float = 0.0
        optim: str = "adamw_torch"  # noqa: F841
        num_train_epochs: int = 3
        weight_scale: float = 0.1
        torch_empty_cache_steps: Optional[int] = None  # noqa: F841
        max_grad_norm: float = 1.0  # noqa: F841
        center_rewards_coefficient: Optional[float] = None  # noqa: F841
        seed: int = 42  # noqa: F841
        full_determinism: bool = False  # noqa: F841
        score_scale: float = 1.0

        # PEFT
        use_peft: bool = False  # noqa: F841
        lora_alpha: int = 16  # noqa: F841
        lora_rank: int = 8  # noqa: F841
        lora_dropout = 0.1  # noqa: F841

        model_name_split = (
            model_name.split("/")[-1] if not model_checkpoint else "RELOAD"
        )
        experiment_tag = None
        dir_name = "reward_models"
        out_dir: str = f"data/{dir_name}/{model_name_split}"
        if experiment_tag:
            out_dir = f"{out_dir}/{experiment_tag}"
        out_dir = f"{out_dir}/lr_{learning_rate}_{lr_scheduler_type}"
        if lr_warmup_ratio > 0:
            out_dir = f"{out_dir}_warmup_{lr_warmup_ratio}"
        out_dir = f"{out_dir}/epochs_{num_train_epochs}/batch_{per_device_train_batch_size*gradient_accumulation_steps}/ws_{weight_scale}"
        if weight_decay > 0:
            out_dir = f"{out_dir}/wd_{weight_decay}"
        if score_scale != 0:
            out_dir = f"{out_dir}/score_scale_{score_scale}"

        # evaluation
        reward_model_checkpoint: str = None  # noqa: F841
        eval_on_lie_test: bool = False  # noqa: F841
        eval_on_reward_bench: bool = False  # noqa: F841
        eval_on_anthropic_bias: bool = False  # noqa: F841
        eval_true_model: bool = False  # noqa: F841
        eval_on_rm_bench: bool = False  # noqa: F841
        any_eval: bool = (
            eval_on_lie_test
            or eval_on_reward_bench
            or eval_on_anthropic_bias
            or eval_true_model
            or eval_on_rm_bench
        )  # noqa: F841
        if any_eval:
            eval_results_tag: str = "eval_results"  # noqa: F841
            eval_results_key = f"lr_{learning_rate}-epochs_{num_train_epochs}"
            if reward_model_checkpoint:
                lr = reward_model_checkpoint.split("lr_")[1].split("_")[0]
                epoch = reward_model_checkpoint.split("epochs_")[1].split("/")[0]
                fold = (
                    reward_model_checkpoint.split("fold_")[1].split("/")[0]
                    if "fold_" in reward_model_checkpoint
                    else None
                )
                eval_results_key = f"lr_{lr}_epochs_{epoch}"
                if fold:
                    eval_results_key = f"{eval_results_key}_fold_{fold}"
            eval_results_dir: str = (
                f"data/{dir_name}/{model_name_split}/{eval_results_tag}"
            )

            os.makedirs(eval_results_dir, exist_ok=True)

        if eval_on_lie_test:
            lie_cols = [
                "eval_results_tag",
                "length_weight",
                "correctness_weight",
                "ratio",
            ]
            lie_test = f"{eval_results_dir}/lie_test_results.csv"
            if not os.path.exists(lie_test):
                pd.DataFrame(columns=lie_cols).to_csv(lie_test, index=False)

        if eval_on_reward_bench:
            reward_bench_cols = [
                "eval_results_tag",
                "Chat",
                "Chat Hard",
                "Safety",
                "Reasoning",
            ]
            reward_bench = f"{eval_results_dir}/reward_bench_results.csv"
            if not os.path.exists(reward_bench):
                pd.DataFrame(columns=reward_bench_cols).to_csv(
                    reward_bench, index=False
                )

        if eval_on_anthropic_bias:
            anthropic_bias_cols = ["eval_results_tag"] + ANTHROPIC_BIAS_CATEGORY_COLS
            anthropic_bias = f"{eval_results_dir}/anthropic_bias_results.csv"
            if not os.path.exists(anthropic_bias):
                pd.DataFrame(columns=anthropic_bias_cols).to_csv(
                    anthropic_bias, index=False
                )

        if eval_true_model:
            true_model_cols = [
                "eval_results_tag",
                "avg_loss",
                "loss1",
                "loss2",
            ]
            true_model = f"{eval_results_dir}/true_model_lie_results.csv"
            if not os.path.exists(true_model):
                pd.DataFrame(columns=true_model_cols).to_csv(true_model, index=False)

        if eval_on_rm_bench:
            rm_bench_cols = ["eval_results_tag", "hard_acc", "normal_acc", "easy_acc"]
            rm_bench = f"{eval_results_dir}/rm_bench_results.csv"
            if not os.path.exists(rm_bench):
                pd.DataFrame(columns=rm_bench_cols).to_csv(rm_bench, index=False)
