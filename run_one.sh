#!/usr/bin/env bash
# run_one.sh – safely grab an idle GPU and launch one training job
set -euo pipefail

############################
# CONFIG
############################
FREE_MEM_THRESHOLD=1000   # MiB
UTIL_THRESHOLD=5          # %
LOCK_DIR=/tmp

############################
# GPU-lock helpers
############################
reserve_gpu() {
  # $1 might be unset – check without expanding it
  if [[ $# -ne 1 || -z "${1:-}" ]]; then
    echo "reserve_gpu: missing id" >&2
    return 1
  fi
  local id=$1
  local lock="${LOCK_DIR}/gpu${id}.lock"
  ( set -o noclobber; echo $$ >"$lock" ) 2>/dev/null
}

release_gpu() {
  [[ $# -eq 1 && -n "${1:-}" ]] || return 0
  rm -f "${LOCK_DIR}/gpu${1}.lock"
}

############################
# Pick the first idle GPU
############################
first_free_gpu() {
  while true; do
    mapfile -t stats < <(
      nvidia-smi --query-gpu=memory.used,utilization.gpu \
                 --format=csv,noheader,nounits
    )

    for raw_idx in "${!stats[@]}"; do
      idx=${raw_idx// }                            # strip whitespace
      IFS=',' read -r mem util <<<"${stats[$raw_idx]}"
      mem=${mem// }  util=${util// }

      [[ $mem -lt $FREE_MEM_THRESHOLD && $util -lt $UTIL_THRESHOLD ]] || continue
      if reserve_gpu "$idx"; then
        echo "$idx"
        return
      fi
    done
    sleep 5
  done
}

############################
# MAIN
############################
fold=$1; lr=$2; epoch=$3
gpu=$(first_free_gpu)

trap 'release_gpu "$gpu"' EXIT

echo "[$(date +%T)] fold=$fold  lr=$lr  epoch=$epoch  →  GPU $gpu"

CUDA_VISIBLE_DEVICES="$gpu" \


# calibrated confidence - p
# python -m RAPL.model_training.train_reward_model with model_name=meta-llama/Llama-3.1-8B \
#   auth_token=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 experiment_tag="llama_scale_p/calibrated/confidence_scores/fold_${fold}" \
#   seed=42 data_seed=42 learning_rate=$lr num_train_epochs=$epoch \
#   weight_scale=0.1 per_device_train_batch_size=2 \
#   gradient_accumulation_steps=16 \
#   train_data_path=RAPL/datasets/scale_LIE/folds/scale_LIE_train_$fold.csv \
#   eval_data_path=RAPL/datasets/scale_LIE/folds/scale_LIE_val_$fold.csv \
#   rapl_col_id=confidence_scores rapl_method=p eval_on_lie_test=True \
#   eval_on_reward_bench=True eval_on_anthropic_bias=True \
#   eval_on_rm_bench=True is_rapl_score_correctness_prob=True \
#   eval_results_tag=llama_scale_p/calibrated/confidence_scores \
#   eval_results_key="lr_${lr}-epochs_${epoch}-fold_${fold}"

# calibrated confidence - p - plus weight decay + warmup ratio
python -m RAPL.model_training.train_reward_model with model_name=meta-llama/Llama-3.1-8B \
  auth_token=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 experiment_tag="llama_scale_p/calibrated/confidence_scores/fold_${fold}" \
  seed=42 data_seed=42 learning_rate=$lr num_train_epochs=$epoch \
  weight_scale=0.1 per_device_train_batch_size=2 \
  gradient_accumulation_steps=16 \
  train_data_path=RAPL/datasets/scale_LIE/folds/scale_LIE_train_$fold.csv \
  eval_data_path=RAPL/datasets/scale_LIE/folds/scale_LIE_val_$fold.csv \
  rapl_col_id=confidence_scores rapl_method=p eval_on_lie_test=True \
  eval_on_reward_bench=True eval_on_anthropic_bias=True \
  eval_on_rm_bench=True is_rapl_score_correctness_prob=True \
  eval_results_tag=llama_scale_p/calibrated/confidence_scores \
  eval_results_key="lr_${lr}-epochs_${epoch}-fold_${fold}" \
  lr_warmup_ratio=0.05 weight_decay=0.01


# calibrated decision time - p
# python -m RAPL.model_training.train_reward_model with model_name=meta-llama/Llama-3.1-8B \
#   auth_token=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 experiment_tag="llama_scale_p/calibrated/decision_time/fold_${fold}" \
#   seed=42 data_seed=42 learning_rate=$lr num_train_epochs=$epoch \
#   weight_scale=0.1 per_device_train_batch_size=2 \
#   gradient_accumulation_steps=16 \
#   train_data_path=RAPL/datasets/scale_LIE/folds/scale_LIE_train_$fold.csv \
#   eval_data_path=RAPL/datasets/scale_LIE/folds/scale_LIE_val_$fold.csv \
#   rapl_col_id=decision_time rapl_method=p eval_on_lie_test=True \
#   eval_on_reward_bench=True eval_on_anthropic_bias=True \
#   eval_on_rm_bench=True is_rapl_score_correctness_prob=True \
#   eval_results_tag=llama_scale_p/calibrated/decision_time \
#   eval_results_key="lr_${lr}-epochs_${epoch}-fold_${fold}"


# TRUE
# python -m RAPL.model_training.train_TRUE_value_head with \
#   model_name=meta-llama/Llama-3.1-8B experiment_tag="final_llama_true/fold_${fold}" \
#   auth_token=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 seed=42 data_seed=42 \
#   learning_rate=$lr num_train_epochs=$epoch weight_scale=0.1 \
#   per_device_train_batch_size=2 \
#   train_data_path=RAPL/datasets/scale_TRUE/repeated_orders/scale_train_double_${fold}.csv \
#   eval_data_path=RAPL/datasets/scale_TRUE/repeated_orders/scale_val_double_${fold}.csv \
#   cal_data_path=RAPL/datasets/scale_TRUE/repeated_orders/scale_cal_double_${fold}.csv \
#   gradient_accumulation_steps=16 lr_warmup_ratio=0.05 weight_decay=0.01 \
#   eval_true_model=True eval_results_tag=final_llama_true \
#   eval_results_key="lr_${lr}-epochs_${epoch}-fold_${fold}" \
#   correctness_label_col_name=accuracy 


# regular
# python -m RAPL.model_training.train_reward_model with \
#   model_name=meta-llama/Llama-3.1-8B \
#   experiment_tag="final_llama_regular_scale/fold_${fold}" \
#   auth_token=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 \
#   seed=42 data_seed=42 \
#   learning_rate="$lr" \
#   num_train_epochs="$epoch" \
#   weight_scale=0.1 \
#   per_device_train_batch_size=2 \
#   train_data_path="RAPL/datasets/scale_LIE/folds/scale_LIE_train_${fold}.csv" \
#   eval_data_path="RAPL/datasets/scale_LIE/folds/scale_LIE_val_${fold}.csv" \
#   gradient_accumulation_steps=16 \
#   eval_on_lie_test=True \
#   eval_on_reward_bench=True \
#   eval_on_anthropic_bias=True \
#   eval_on_rm_bench=True \
#   eval_results_tag=final_llama_regular_scale \
#   eval_results_key="lr_${lr}-epochs_${epoch}-fold_${fold}" \ lr_warmup_ratio=0.05 weight_decay=0.01
