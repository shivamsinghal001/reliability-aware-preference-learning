import os

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss

# from RewardBench codebase
EXAMPLE_COUNTS = {
    "alpacaeval-easy": 100,
    "alpacaeval-length": 95,
    "alpacaeval-hard": 95,
    "mt-bench-easy": 28,
    "mt-bench-med": 40,
    "mt-bench-hard": 37,
    "math-prm": 984,  # actual length 447, upweighting to be equal to code
    "refusals-dangerous": 100,
    "refusals-offensive": 100,
    "llmbar-natural": 100,
    "llmbar-adver-neighbor": 134,
    "llmbar-adver-GPTInst": 92,
    "llmbar-adver-GPTOut": 47,
    "llmbar-adver-manual": 46,
    "xstest-should-refuse": 154,
    "xstest-should-respond": 250,
    "donotanswer": 136,
    "hep-cpp": 164,
    "hep-go": 164,
    "hep-java": 164,
    "hep-js": 164,
    "hep-python": 164,
    "hep-rust": 164,
}

SUBSET_MAPPING = {
    "Chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "Reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}

def calculate_scores_per_section(example_counts, subset_mapping, metrics):
    """
    Helper function for immediately logging RewardBench scores.
    """
    section_scores = {}
    for section, tests in subset_mapping.items():
        total_weighted_score = 0
        total_examples = 0
        for test in tests:
            if test in metrics:
                total_weighted_score += metrics[test] * example_counts[test]
                total_examples += example_counts[test]
        if total_examples > 0:
            section_scores[section] = total_weighted_score / total_examples
        else:
            section_scores[section] = 0
    return section_scores



device = "cuda" if torch.cuda.is_available() else "cpu"

script_dir = os.path.dirname(__file__)
REWARD_BENCH_LEADERBOARD_DATA_PATH = os.path.join(
    script_dir, "../../datasets/reward_bench/reward_bench_leaderboard.csv"
)
REWARD_BENCH_PRIOR_DATA_PATH = os.path.join(
    script_dir, "../../datasets/reward_bench/reward_bench_prior.csv"
)
ANTHROPIC_BIAS_DATA_PATH = os.path.join(
    script_dir, "../../datasets/anthropic_dataset/filtered_anthropic_dataset.csv"
)
ANTHROPIC_BIAS_CATEGORY_COLS = [
    "authoritative",
    "agree_human_explicit",
    "agree_human_implicit",
    "grammatically_sound",
    "well_written",
    "entertaining",
    "truthful",
    "higher_reading_age",
    "empathetic",
    "funny",
    "better_supported",
    "polite",
    "matches_human_style",
    "optimistic",
    "structured",
    "informative",
    "engaging",
    "friendly",
    "motivating",
    "concise",
    "persuasive",
    "rigorous",
    "logically_sound",
    "relevant",
    "longer",
]
RM_BENCH_DATA_PATH = os.path.join(script_dir, "../../datasets/RM-bench/rm_bench.csv")


def compute_predictions_length_dataset(example, keys, tokenizer, model):
    output = {}
    for key in keys:
        batch = tokenizer.pad(
            {
                "input_ids": example[f"input_ids_{key}"],
            },
            padding="longest",
            pad_to_multiple_of=64,
            return_tensors="pt",
        )
        with torch.no_grad():
            output[f"reward_output_{key}"] = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )[0].tolist()
    return output


def calculate_num_correct(example):
    num_correct = {}
    chosen_reward = example["reward_output_chosen"][0]
    rejected_reward = example["reward_output_rejected"][0]
    correct = 1 if chosen_reward > rejected_reward else 0
    num_correct["results"] = correct
    return num_correct


def eval_reward_bench(leaderboard_dataset, tokenizer, model, batch_size=1):
    # Leaderboard dataset
    rew_bench_leaderboard = leaderboard_dataset.map(
        lambda ex: compute_predictions_length_dataset(
            ex,
            ["chosen", "rejected"],
            tokenizer=tokenizer,
            model=model,
        ),
        remove_columns=[
            "input_ids_chosen",
            "attention_mask_chosen",
            "input_ids_rejected",
            "attention_mask_rejected",
        ],
        batched=True,
        batch_size=batch_size,
    )
    present_subsets = np.unique(rew_bench_leaderboard["subset"])
    rew_bench_leaderboard = rew_bench_leaderboard.map(calculate_num_correct)
    results_grouped = {}
    for subset in present_subsets:
        subset_dataset = rew_bench_leaderboard.filter(
            lambda example: example["subset"] == subset
        )
        num_correct = sum(subset_dataset["results"])
        num_total = len(subset_dataset["results"])
        print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
        results_grouped[subset] = num_correct / num_total

    results_leaderboard = calculate_scores_per_section(
        EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped
    )

    # Prior dataset
    # rew_bench_prior = prior_dataset.map(
    #     lambda ex: compute_predictions_length_dataset(
    #         ex,
    #         ["chosen", "rejected"],
    #         tokenizer=tokenizer,
    #         model=model,
    #     ),
    #     remove_columns=[
    #         "input_ids_chosen",
    #         "attention_mask_chosen",
    #         "input_ids_rejected",
    #         "attention_mask_rejected",
    #     ],
    #     batched=True,
    #     batch_size=batch_size,
    # )

    # rew_bench_prior = rew_bench_prior.map(calculate_num_correct)
    # results_leaderboard["prior"] = np.mean(rew_bench_prior["results"])

    return results_leaderboard, rew_bench_leaderboard


def eval_lie_test(lie_dataset, tokenizer, model, batch_size=1):
    lie_dataset_results = lie_dataset.map(
        lambda ex: compute_predictions_length_dataset(
            ex,
            [
                "correct_concise",
                "correct_detailed",
                "incorrect_concise",
                "incorrect_detailed",
            ],
            tokenizer=tokenizer,
            model=model,
        ),
        remove_columns=[
            "input_ids_correct_concise",
            "attention_mask_correct_concise",
            "input_ids_correct_detailed",
            "attention_mask_correct_detailed",
            "input_ids_incorrect_concise",
            "attention_mask_incorrect_concise",
            "input_ids_incorrect_detailed",
            "attention_mask_incorrect_detailed",
        ],
        batched=True,
        batch_size=batch_size,
    )
    correct_concise_rew = np.array(
        lie_dataset_results["reward_output_correct_concise"]
    ).squeeze()
    incorrect_concise_rew = np.array(
        lie_dataset_results["reward_output_incorrect_concise"]
    ).squeeze()
    correct_det_rew = np.array(
        lie_dataset_results["reward_output_correct_detailed"]
    ).squeeze()
    incorrect_det_rew = np.array(
        lie_dataset_results["reward_output_incorrect_detailed"]
    ).squeeze()

    x = (
        [[0, 0]] * len(incorrect_concise_rew.tolist())
        + [[0, 1]] * len(correct_concise_rew.tolist())
        + [[1, 0]] * len(incorrect_det_rew.tolist())
        + [[1, 1]] * len(correct_det_rew.tolist())
    )
    y = (
        incorrect_concise_rew.tolist()
        + correct_concise_rew.tolist()
        + incorrect_det_rew.tolist()
        + correct_det_rew.tolist()
    )
    x = np.array(x)
    x = x[:, :]
    y = np.array(y)

    reg = LinearRegression().fit(x, y)
    full_length_weight, full_correctness_weight = reg.coef_
    return (
        full_length_weight,
        full_correctness_weight,
        full_correctness_weight / full_length_weight,
        lie_dataset_results,
    )


def eval_anthropic_bias(dataset, tokenizer, model, batch_size):
    dataset = dataset.map(
        lambda ex: compute_predictions_length_dataset(
            ex,
            ["chosen", "rejected"],
            tokenizer=tokenizer,
            model=model,
        ),
        remove_columns=[
            "input_ids_chosen",
            "attention_mask_chosen",
            "input_ids_rejected",
            "attention_mask_rejected",
        ],
        batched=True,
        batch_size=batch_size,
    )

    def add_reward_difference_batched(examples):
        reward_differences = []
        for i in range(len(examples["chosen_choice"])):
            if examples["chosen_choice"][i] == "A":
                reward_diff = (
                    examples["reward_output_chosen"][i][0]
                    - examples["reward_output_rejected"][i][0]
                )
            else:
                reward_diff = (
                    examples["reward_output_rejected"][i][0]
                    - examples["reward_output_chosen"][i][0]
                )

            reward_differences.append(reward_diff)

        return {"reward_difference": reward_differences}

    dataset = dataset.map(
        add_reward_difference_batched, batched=True, batch_size=batch_size
    )

    value_map = {"A": 1, "C": 0, "B": -1}

    def apply_value_mapping(examples):
        result = {}
        for col in ANTHROPIC_BIAS_CATEGORY_COLS:
            result[f"{col}_feature"] = [value_map.get(x, -1) for x in examples[col]]
        return result

    dataset = dataset.map(apply_value_mapping, batched=True, batch_size=batch_size)

    feature_cols = [f"{col}_feature" for col in ANTHROPIC_BIAS_CATEGORY_COLS]
    model = LinearRegression()
    x = np.array([dataset[col] for col in feature_cols]).T
    y = np.array(dataset["reward_difference"])
    model.fit(x, y)
    coefs = model.coef_
    return dict(zip(feature_cols, coefs)), dataset


def eval_true_model(
    val_dataset_1,
    val_dataset_2,
    cal_dataset,
    tokenizer,
    model,
    batch_size,
    score_scale=1,
):
    val_dataset_1 = val_dataset_1.map(
        lambda ex: compute_predictions_length_dataset(
            ex, ["prompt_response_group"], tokenizer=tokenizer, model=model
        ),
        remove_columns=[
            "input_ids_prompt_response_group",
            "attention_mask_prompt_response_group",
        ],
        batched=True,
        batch_size=batch_size,
    )
    val_rewards_1 = (
        torch.tensor(val_dataset_1["reward_output_prompt_response_group"]) * score_scale
    )
    val_labels_1 = torch.tensor(val_dataset_1["correct_chosen"])

    val_dataset_2 = val_dataset_2.map(
        lambda ex: compute_predictions_length_dataset(
            ex, ["prompt_response_group"], tokenizer=tokenizer, model=model
        ),
        remove_columns=[
            "input_ids_prompt_response_group",
            "attention_mask_prompt_response_group",
        ],
        batched=True,
        batch_size=batch_size,
    )
    val_rewards_2 = (
        torch.tensor(val_dataset_2["reward_output_prompt_response_group"]) * score_scale
    )
    val_labels_2 = torch.tensor(val_dataset_2["correct_chosen"])

    if cal_dataset is not None:
        cal_dataset = cal_dataset.map(
            lambda ex: compute_predictions_length_dataset(
                ex, ["prompt_response_group"], tokenizer=tokenizer, model=model
            ),
            remove_columns=[
                "input_ids_prompt_response_group",
                "attention_mask_prompt_response_group",
            ],
            batched=True,
            batch_size=batch_size,
        )
        cal_rewards = (
            torch.tensor(cal_dataset["reward_output_prompt_response_group"])
            * score_scale
        )
        cal_labels = torch.tensor(cal_dataset["correct_chosen"])
        calibrator = train_logistic_calibration(cal_rewards, cal_labels)
        calibrated_probs_1 = apply_logistic_calibration(calibrator, val_rewards_1)
        calibrated_loss_1 = log_loss(val_labels_1.numpy(), calibrated_probs_1.numpy())
        calibrated_probs_2 = apply_logistic_calibration(calibrator, val_rewards_2)
        calibrated_loss_2 = log_loss(val_labels_2.numpy(), calibrated_probs_2.numpy())
    else:
        calibrated_loss_1 = log_loss(val_labels_1.numpy(), val_rewards_1.numpy())
        calibrated_loss_2 = log_loss(val_labels_2.numpy(), val_rewards_2.numpy())

    return (
        val_dataset_1,
        val_dataset_2,
        calibrated_loss_1,
        calibrated_loss_2,
    )


def train_logistic_calibration(logits, labels):
    if logits.shape[1] == 2:
        inputs = logits[:, 1] - logits[:, 0]
        inputs = inputs.reshape(-1, 1)
    else:
        inputs = logits.detach().numpy()

    y = labels.detach().numpy()

    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(inputs, y)
    return calibrator


def apply_logistic_calibration(calibrator, logits):
    if logits.shape[1] == 2:
        inputs = (logits[:, 1] - logits[:, 0]).reshape(-1, 1).detach().numpy()
    else:
        inputs = logits.detach().numpy()

    calibrated_probs = torch.tensor(calibrator.predict_proba(inputs))
    return calibrated_probs


def eval_rm_bench(rm_bench_dataset, tokenizer, model, batch_size):
    rm_bench_dataset = rm_bench_dataset.map(
        lambda ex: compute_predictions_length_dataset(
            ex,
            ["chosen", "rejected"],
            tokenizer=tokenizer,
            model=model,
        ),
        remove_columns=[
            "input_ids_chosen",
            "attention_mask_chosen",
            "input_ids_rejected",
            "attention_mask_rejected",
        ],
        batched=True,
        batch_size=batch_size,
    )

    rm_bench_dataset = rm_bench_dataset.to_pandas()
    results = compute_accuracy_from_flattened(rm_bench_dataset)
    return results


def compute_accuracy_from_flattened(results_df):
    type_to_idx = {"concise": 0, "detailed": 1, "markdown": 2}

    matrix_size = 3
    acc_matrix = np.zeros((matrix_size, matrix_size))

    grouped = results_df.groupby("id")
    total_samples = 0

    for _, group in grouped:
        total_samples += 1

        chosen_scores = {}
        rejected_scores = {}

        for _, row in group.iterrows():
            type_idx = type_to_idx[row["type"]]
            chosen_scores[type_idx] = row["reward_output_chosen"]
            rejected_scores[type_idx] = row["reward_output_rejected"]

        for i in range(matrix_size):
            for j in range(matrix_size):
                if i in chosen_scores and j in rejected_scores:
                    if chosen_scores[i] > rejected_scores[j]:
                        acc_matrix[i][j] += 1

    acc_matrix /= total_samples

    upper_right_count = matrix_size * (matrix_size - 1) / 2
    hard_acc = np.sum(np.triu(acc_matrix, 1)) / upper_right_count

    normal_acc = np.mean(np.diag(acc_matrix))

    lower_left_count = matrix_size * (matrix_size - 1) / 2
    easy_acc = np.sum(np.tril(acc_matrix, -1)) / lower_left_count

    return {
        "hard_acc": hard_acc,
        "normal_acc": normal_acc,
        "easy_acc": easy_acc,
    }
