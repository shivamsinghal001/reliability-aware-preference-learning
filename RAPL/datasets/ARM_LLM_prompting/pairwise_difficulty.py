import argparse
import multiprocessing
import os
import random
import re
from dataclasses import dataclass
from itertools import product
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

RESPONSE_PARSE_RE = re.compile(
    r"""
        # Capture text after each .a until the next .b
        ^\s*(?P<item_number>\d)\.a\s+(?P<text>.*?)(?=\s*\1\.b)
    """,
    flags=re.DOTALL | re.MULTILINE | re.VERBOSE,
)


@dataclass
class PairwiseComparisonResponse:
    """
    The response from the pairwise comparison autograder.

    Attributes:
    picked_question: int
        The question that was picked.
    grader_model_input: str
        The input to the autograder model.
    grader_model_output: str
        The output from the autograder model.
    """

    picked_question: Optional[int]
    grader_model_input: Optional[str]
    grader_model_output: Optional[str]


def get_preference(q1, q2, dataset, model="gpt-4-turbo", ag_model="gpt-4-turbo"):
    q1_data = dataset[dataset["ordered_ids"] == q1]
    q2_data = dataset[dataset["ordered_ids"] == q2]
    question1 = f"{q1_data['questions'].values[0]} A.) {q1_data['choice1'].values[0]} B.) {q1_data['choice2'].values[0]}"
    question2 = f"{q2_data['questions'].values[0]} A.) {q2_data['choice1'].values[0]} B.) {q2_data['choice2'].values[0]}"
    question1_eval = q1_data[f"{ag_model}_autograder_reasoning_text"].values[0]
    question2_eval = q2_data[f"{ag_model}_autograder_reasoning_text"].values[0]

    with open("pairwise_difficulty_prompt.txt", "r") as f:
        autograder_prompt = f.read()

    model_input = autograder_prompt.format(
        question1=question1,
        question1_rubric=question1_eval,
        question2=question2,
        question2_rubric=question2_eval,
    )

    num_repeats = 10 if model == "gpt-3.5-turbo" else 1
    for _ in range(num_repeats):
        with openai.OpenAI() as client:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": model_input}],
                temperature=0.0,  # no randomness in the model output
            )
            model_output = completion.choices[0].message.content

        pattern = r":\s*([12])$"
        if model_output is not None:
            match = re.search(pattern, model_output)
            if match:
                return PairwiseComparisonResponse(
                    picked_question=int(match.group(1)),
                    grader_model_input=model_input,
                    grader_model_output=model_output,
                )

    return PairwiseComparisonResponse(
        picked_question=None,
        grader_model_input=model_input,
        grader_model_output=model_output,
    )


def get_preference_for_dataset(
    question1_lst, question2_lst, dataset, model="gpt-4-turbo", ag_model="gpt-4-turbo"
):
    """
    Run the pairwise comparison autograder on a dataset.

    Args:
    question1_lst: List[str]
        The list of questions to compare.
    question2_lst: List[str]
        The list of questions to compare.
    dataset: pd.DataFrame
        The dataset on which to run the autograder.
        Has to have "questions", "choice1", "choice2" columns.
    model: str
        The model to use for the autograder.
    ag_model: str
        The model that was used for evaluating the questions on the rubric

    Returns:
    pd.DataFrame
        The dataset with the scores from the autograder. The scores are stored in a column called 'score'.
    """

    model_lst = [model] * len(question1_lst)
    ag_model_lst = [ag_model] * len(question1_lst)
    dataset_lst = [dataset] * len(question1_lst)
    with multiprocessing.Pool() as pool:
        results = []
        result_objects = pool.starmap_async(
            get_preference,
            zip(question1_lst, question2_lst, dataset_lst, model_lst, ag_model_lst),
        )
        for result in tqdm(result_objects.get(), total=len(question1_lst)):
            results.append(result)

    data_dict = {
        "q1_ID": question1_lst,
        "q2_ID": question2_lst,
        "preference_ag_output": [result for result in results],
        "preference_picked_question": [result.picked_question for result in results],
        "preference_model_input": [result.grader_model_input for result in results],
        "preference_model_output": [result.grader_model_output for result in results],
    }
    data = pd.DataFrame(data_dict)
    return data


def train_model(params, question_ids):
    num_epochs, lr, lambda_reg = params
    print(
        f"Starting training with epochs: {num_epochs}, lr: {lr}, lambda_reg: {lambda_reg}"
    )

    difficulty_scores = torch.nn.Parameter(torch.randn(len(question_ids)))
    optimizer = optim.Adam([difficulty_scores], lr=lr)

    losses = []
    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        loss = preference_loss(lambda_reg, difficulty_scores)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        # if epoch % 10 == 0:
        #     print(f"Epoch {epoch}, Loss: {loss.item()}")

    difficulty_scores_np = difficulty_scores.detach().numpy()
    normalized_difficulty_scores = (
        difficulty_scores_np - np.min(difficulty_scores_np)
    ) / (np.max(difficulty_scores_np) - np.min(difficulty_scores_np))

    dataset_key = (
        f"trained_difficulty_scores_lr-{lr}_epochs-{num_epochs}_lreg-{lambda_reg}"
    )
    return dataset_key, normalized_difficulty_scores, losses


def preference_loss(lambda_reg, difficulty_scores):
    loss = torch.tensor(0)
    for _, row in data_df.iterrows():
        i = row["q1_ID"]
        j = row["q2_ID"]
        picked_question = row["preference_picked_question"]

        di = difficulty_scores[i]
        dj = difficulty_scores[j]
        if picked_question == 1:
            loss += -nn.functional.logsigmoid(di - dj)
        else:
            loss += -nn.functional.logsigmoid(dj - di)

    reg_loss = lambda_reg * torch.sum(difficulty_scores**2)
    return loss + reg_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)

    parser = argparse.ArgumentParser(
        description="Preference Learning with Bounded Cognition"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/nas/ucb/shivamsinghal/preference-learning-with-bounded-cognition/bounded_cognition/data_collection/truthful_qa/train_truthful_qa_w_survey_responses.csv",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--num_comparisons", type=int, default=4000, help="Number of comparisons"
    )
    parser.add_argument(
        "--lambda_reg", type=float, default=0.01, help="Regularization strength"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--model", type=str, default="gpt-4-turbo", help="Model to use")
    parser.add_argument(
        "--ag_model",
        type=str,
        default="gpt-4-turbo",
        help="Model that was used for autograding the questions",
    )
    parser.add_argument(
        "--preference_dataset_path",
        type=str,
        default=None,
        help="Path to the preference dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Path to save generated preferences and trained difficulty scores",
    )
    parser.add_argument(
        "--use_batching",
        type=str,
        default="True",
        help="Whether to use batching for the preference learning",
    )

    args = parser.parse_args()

    dataset_path = args.dataset_path
    num_comparisons = args.num_comparisons
    lambda_reg = args.lambda_reg
    learning_rate = args.lr
    num_epochs = args.num_epochs
    model = args.model
    ag_model = args.ag_model
    preference_dataset_path = args.preference_dataset_path
    use_batching = True if args.use_batching == "True" else False
    output_dir = args.output_dir

    dataset = pd.read_csv(dataset_path)
    question_ids = dataset["ordered_ids"].tolist()
    num_questions = len(question_ids)

    if not preference_dataset_path:
        api_key = os.getenv("OPENAI_API_KEY")
        # ask the user to set their API key interactively if it's not set
        if api_key is None:
            api_key = input("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key

        data = []
        used_q_pairs = set()
        used_qs = set()
        question1_lst = []
        question2_lst = []

        print("Getting pairwise comparisons...")
        pbar = tqdm(total=num_comparisons)

        # Ensure every question is paired at least once
        random.shuffle(question_ids)
        for i in range(0, len(question_ids) - 1, 2):
            q1, q2 = question_ids[i], question_ids[i + 1]
            used_q_pairs.add((q1, q2))
            if not use_batching:
                preference = get_preference(
                    q1, q2, dataset, model=model, ag_model=ag_model
                )
                data.append((q1, q2, preference))
            used_qs.add(q1)
            used_qs.add(q2)
            question1_lst.append(q1)
            question2_lst.append(q2)
            pbar.update(1)

        if len(question_ids) % 2 != 0:
            last_q = question_ids[-1]
            other_q = random.choice(
                [qid for qid in question_ids if qid != last_q and qid not in used_qs]
            )
            used_q_pairs.add((last_q, other_q))
            if not use_batching:
                preference = get_preference(
                    last_q, other_q, dataset, model=model, ag_model=ag_model
                )
                data.append((last_q, other_q, preference))
            used_qs.add(last_q)
            used_qs.add(other_q)
            question1_lst.append(last_q)
            question2_lst.append(other_q)
            pbar.update(1)

        # Additional random pairings till we reach the desired number of comparisons
        while len(used_q_pairs) < num_comparisons:
            q1, q2 = random.choice(question_ids), random.choice(question_ids)
            while q1 == q2 or (q1, q2) in used_q_pairs or (q2, q1) in used_q_pairs:
                q1, q2 = random.choice(question_ids), random.choice(question_ids)
            if not use_batching:
                preference = get_preference(
                    q1, q2, dataset, model=model, ag_model=ag_model
                )
                data.append((q1, q2, preference))
            used_q_pairs.add((q1, q2))
            used_qs.add(q1)
            used_qs.add(q2)
            question1_lst.append(q1)
            question2_lst.append(q2)
            pbar.update(1)

        pbar.close()

        if not use_batching:
            data_df = pd.DataFrame(
                data, columns=["q1_ID", "q2_ID", "preference_ag_output"]
            )

            def get_preference_picked_question(preference):
                return preference.picked_question

            def get_preference_model_input(preference):
                return preference.grader_model_input

            def get_preference_model_output(preference):
                return preference.grader_model_output

            data_df["preference_picked_question"] = data_df[
                "preference_ag_output"
            ].apply(get_preference_picked_question)
            data_df["preference_model_input"] = data_df["preference_ag_output"].apply(
                get_preference_model_input
            )
            data_df["preference_model_output"] = data_df["preference_ag_output"].apply(
                get_preference_model_output
            )
        else:
            data_df = get_preference_for_dataset(
                question1_lst, question2_lst, dataset, model=model, ag_model=ag_model
            )
        preference_dataset_path = (
            f"{output_dir}/{ag_model}_ag_{model}_preference_data.csv"
        )
        data_df.to_csv(
            preference_dataset_path,
            index=False,
        )

    breakpoint()
    data_df = pd.read_csv(preference_dataset_path)
    data_df = data_df.sample(frac=1)

    print("Training the model...")
    epoch_values = [100, 300, 1000, 3000]
    lr_vals = [0.01, 0.03, 0.1, 0.3]
    lambda_vals = [0.001, 0.01, 0.1]
    param_combinations = list(product(epoch_values, lr_vals, lambda_vals))

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(
            train_model, [(params, question_ids) for params in param_combinations]
        )

    print("Saving the results...")
    breakpoint()
    losses_dict = {}
    for key, scores, losses in results:
        dataset[key] = scores
        losses_dict[key] = losses
    dataset.to_csv(dataset_path, index=False)

    print("Plotting the loss...")
    for model_str, losses in losses_dict.items():
        epochs = range(1, len(losses) + 1)
        plt.plot(epochs, losses, label=model, marker="o")

    plt.title("Preference Learning Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        f"{output_dir}/difficulty_training_plots/{ag_model}_ag_{model}_preference_learning_loss.png"
    )
