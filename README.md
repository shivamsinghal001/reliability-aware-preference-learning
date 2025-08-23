# Reliability-Aware Preference Learning
This repository contains the code and datasets for the paper "Reliability-Aware Preference Learning for LLM Reward Models".

## Requirements

All Python code is under the `RAPL` package. Run

    pip install -e .

to install dependencies. 

## Datasets
- The Length-Incentivized Evaluations (LIE) dataset is included in the RAPL/LIE directory.
- The Testing Reasoning and Understanding Errors (of annotators) (TRUE) dataset is in the RAPL/TRUE directory.

Note that all folds for the five-fold cross validation are present in the directory, so that our results can be replicated.

## Training Reward Models
- Train a standard preference learning model on LIE by running the following command:

        python -m RAPL.model_training.train_reward_model with model_name=$MODEL_NAME experiment_tag=$TAG/fold_$FOLD auth_token=$AUTH seed=42 data_seed=42 learning_rate=$LR num_train_epochs=$EPOCHS weight_scale=0.1 per_device_train_batch_size=2 train_data_path=~/reliability-aware-preference-learning/RAPL/datasets/LIE/folds/LIE_train_$FOLD.csv eval_data_path=~/reliability-aware-preference-learning/RAPL/datasets/LIE/folds/LIE_val_$FOLD.csv gradient_accumulation_steps=16 eval_on_lie_test=True eval_on_reward_bench=True eval_results_tag=$TAG eval_results_key=lr_$LR-epochs_$EPOCHS-fold_$FOLD

- Train a RAPL model on LIE by running the following command:
  
        python -m RAPL.model_training.train_reward_model with model_name=$MODEL_NAME auth_token=$AUTH experiment_tag=$TAG/$METHOD/uncalibrated/$COL/fold_$FOLD seed=42 data_seed=42 learning_rate=$LR num_train_epochs=$EPOCHS weight_scale=0.1 per_device_train_batch_size=2 gradient_accumulation_steps=16 train_data_path=~/reliability-aware-preference-learning/RAPL/datasets/LIE/folds/LIE_train_$FOLD.csv eval_data_path=~/reliability-aware-preference-learning/RAPL/datasets/LIE/folds/LIE_val_$FOLD.csv rapl_col_id=$COL rapl_method=$METHOD eval_on_lie_test=True eval_on_reward_bench=True is_rapl_score_correctness_prob=$IS_PROB eval_results_tag=$TAG/$METHOD/uncalibrated/$COL eval_results_key=lr_$LR-epochs_$EPOCHS-fold_$FOLD

- Fine-tune a model on TRUE by running the following command:

        python -m RAPL.model_training.train_TRUE_value_head with model_name=$MODEL_NAME experiment_tag=$TAG/fold_$FOLD auth_token=$AUTH seed=42 data_seed=42 learning_rate=$LR num_train_epochs=$EPOCHS weight_scale=0.1 per_device_train_batch_size=2 train_data_path=~/reliability-aware-preference-learning/RAPL/datasets/TRUE/repeated_orders/train_double_$FOLD.csv eval_data_path=~/reliability-aware-preference-learning/RAPL/datasets/TRUE/repeated_orders/val_double_$FOLD.csv cal_data_path=~/reliability-aware-preference-learning/RAPL/datasets/TRUE/repeated_orders/cal_double_$FOLD.csv gradient_accumulation_steps=16 lr_warmup_ratio=0.05 weight_decay=0.01 eval_true_model=True eval_results_tag=$TAG eval_results_key=lr_$LR-epochs_$EPOCHS-fold_$FOLD

Notes about parameter settings:
- AUTH = <token from huggingface>
- We set MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
- eval_on_{x} = True for automatic evals on various datasets.
- METHOD = "beta" (\beta-RAPL) or "p" (\eps-RAPL)
- COL = "confidence_scores", "decision_time", etc. (the name of the ARMs column in the dataset)
- IS_PROB = True if running with calibrated ARMs / TRUE model scores and False otherwise
- Check RAPL/model_training/configs_and_utils/model_training_config.py for other hyperparameter settings.

Notes about the various other data directories that are present in the repo:
- LLM_ARM_generation: As mentioned in Appendix A, we have outputs from prompting various LLMs (o3, gemini-2.5-pro, and Claude-4-Opus) to determine ARMs. We hope this data could be used by someone in the future.
- RM-bench: We allow for evaluation on RM-bench (https://arxiv.org/abs/2410.16184). We found that results on this were not consistent, so we decided to omit it from our paper.
- Anthropic dataset: We allow for evaluation on a dataset out of Anthropic that measured various other good and bad attributes of text other than length and correctness. This is based on this paper: https://arxiv.org/abs/2310.13548.
- helpsteer / hh-rlhf: We allow for training on the HelpSteer2 (https://arxiv.org/abs/2410.01257) and HH-RLHF (https://arxiv.org/abs/2204.05862) datasets.
- scale_LIE / scale_TRUE: These datasets contain the same prompts and respnses as the TRUE dataset, but the ratings are aggregated across multiple annotators. We leave this data here in case useful for anyone.
