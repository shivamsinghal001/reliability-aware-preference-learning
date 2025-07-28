import gc

import pandas as pd
import torch
from peft import PeftModel  # , LoraConfig
from sacred import Experiment
from sacred.observers import FileStorageObserver
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .configs_and_utils.data_utils import (
    rlhf_format_dataset_creation,
    true_format_dataset_creation,
)
from .configs_and_utils.model_training_config import make_model_training_config

ex = Experiment("evaluate_llm_preference_model")

make_model_training_config(ex)


@ex.config_hook
def setup_observer(config, command_name, logger):
    ex.observers.append(FileStorageObserver("/nas/ucb/shivamsinghal/eval_trials"))
    return config


@ex.automain
def main(_config, _run):
    tokenizer_name = (
        _config["tokenizer_name"]
        if _config["tokenizer_name"] is not None
        else _config["model_name"]
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, token=_config["auth_token"]
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    model_kwargs = {}
    if _config["bf16"]:
        model_kwargs["torch_dtype"] = torch.bfloat16

    if _config["use_peft"]:
        # peft_config = LoraConfig.from_pretrained(_config["reward_model_checkpoint"])
        model = AutoModelForSequenceClassification.from_pretrained(
            _config["model_name"],
            num_labels=1,
            token=_config["auth_token"],
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            model, _config["reward_model_checkpoint"], is_trainable=False
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            _config["reward_model_checkpoint"],
            num_labels=1,
            token=_config["auth_token"],
            **model_kwargs,
        )
    model = model.to(device).eval()

    model.config.pad_token_id = tokenizer.pad_token_id

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
        lie_dataset_results.to_csv(
            f"{_config['reward_model_checkpoint']}/lie_test_rewards.csv"
        )

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
        dataset.to_csv(f"{_config['reward_model_checkpoint']}/reward_bench_rewards.csv")
        del (leaderboard_dataset, new_df)  # prior_dataset
        gc.collect()
        torch.cuda.empty_cache()

    if _config["eval_on_anthropic_bias"]:
        print("Evaluating on the Anthropic bias dataset")
        from .configs_and_utils.eval_utils import (
            ANTHROPIC_BIAS_CATEGORY_COLS,
            ANTHROPIC_BIAS_DATA_PATH,
            eval_anthropic_bias,
        )

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
        anthropic_dataset.to_csv(
            f"{_config['reward_model_checkpoint']}/anthropic_bias_rewards.csv"
        )
        del (dataset, new_df)
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

        val_dataset1, val_dataset2, loss1, loss2 = eval_true_model(
            val_dataset_1=val_dataset_1,
            val_dataset_2=val_dataset_2,
            cal_dataset=cal_dataset,
            model=model,
            tokenizer=tokenizer,
            batch_size=_config["per_device_eval_batch_size"],
            score_scale=_config["score_scale"],
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
        val_dataset1.to_csv(
            f"{_config['reward_model_checkpoint']}/lie_train_true_eval_group_1.csv"
        )
        val_dataset2.to_csv(
            f"{_config['reward_model_checkpoint']}/lie_train_true_eval_group_2.csv"
        )

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
