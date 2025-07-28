import numpy as np
import torch
import tqdm
from transformers import TrainerCallback  # type: ignore

# import matplotlib.pyplot as plt


class EMA:
    """
    EMA -- Exponential Moving Average of model weights.
    This was implemented to enable the model to be more robust to noisy labels,
    have prediction consistency, and produce more calibrated probabilities.
    """

    def __init__(self, model, decay, path):
        self.model = model
        self.decay = decay
        self.ema_weights = {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup = {}
        self.save_path = path

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert (
                    name in self.ema_weights
                ), f"Parameter {name} not found in EMA weights."
                current_param_cpu = param.detach().cpu()
                new_average = (
                    self.decay * self.ema_weights[name]
                    + (1.0 - self.decay) * current_param_cpu
                )
                self.ema_weights[name] = new_average.clone()

    def apply_ema_weights(self):
        for name, param in tqdm.tqdm(
            self.model.named_parameters(), desc="Processing EMA weights"
        ):
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.ema_weights[name].to(param.device))

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in self.backup:
                    param.data.copy_(self.backup[name])
        self.backup = {}

    def save_weights(self):
        weight_dict = {
            name: tensor.clone() for name, tensor in self.ema_weights.items()
        }
        torch.save(weight_dict, self.save_path)

    def load_weights(self):
        loaded_weights = torch.load(self.save_path, map_location="cpu")
        for name, tensor in loaded_weights.items():
            if name in self.ema_weights:
                self.ema_weights[name] = tensor.clone()
            else:
                print(f"{name} not found in the saved model's EMA weights.")


class EMACallback(TrainerCallback):
    def __init__(self, ema_model):
        self.ema_model = ema_model

    def on_step_end(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        self.ema_model.update()

    def on_epoch_end(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        # evaluation happens at the end of every epoch, and we want to use ema weights for evaluation
        if state.is_world_process_zero:
            self.ema_model.apply_ema_weights()

    def on_evaluate(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        # restore the original model weights after evaluation
        if state.is_world_process_zero:
            self.ema_model.restore()

    def on_train_end(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        if state.is_world_process_zero:
            self.ema_model.save_weights()


def get_step_decay_lr_lambda(current_step: int, *, num_training_steps: int):
    if current_step < num_training_steps // 3:
        return 1.0
    elif current_step < (2 * num_training_steps) // 3:
        return 0.1
    else:
        return 0.01


def get_cosine_decay_lr_lambda(current_step: int, *, num_training_steps: int):
    return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * current_step / num_training_steps))


def calculate_beta(p_correctness):
    p_correctness = torch.tensor(p_correctness)
    return torch.tensor(max(0, torch.special.logit(p_correctness, eps=1e-7))).item()


def calculate_p(p_correctness):
    p_correctness = torch.tensor(p_correctness)
    return torch.tensor(max(0, 2 * p_correctness - 1)).item()


# TODO: reevaluate margin calculation? Right now, smaller margin for higher confidence and vice versa
def calculate_margin(p_correctness):
    p_correctness = torch.tensor(p_correctness)
    if p_correctness <= 0.5:
        return 1
    return torch.tensor(
        max(0.0, 1 - torch.special.logit(p_correctness, eps=1e-7))
    ).item()


class GradientLoggingCallback(TrainerCallback):
    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = param.grad.data.norm(2).item()
                grad_norms[name] = norm

        # llama_layer_norms = {name: norm for name, norm in grad_norms.items() if "layers" in name}

        # print(f"Step {state.global_step} gradient norms (sample):")
        # for name, norm in grad_norms.items():
        #     print(f"  {name}: {norm:.4f}")

        sorted_grad_norms = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)
        print(
            f"MAX NORM in {sorted_grad_norms[0][0]} layer: {sorted_grad_norms[0][1]}\n"
        )
        return control

    # def on_train_end(self, args, state, control, model=None, **kwargs):
    #     weights = model.score.weight.data.cpu().numpy()

    # plt.figure(figsize=(10, 6))
    # plt.hist(weights.flatten(), bins=50, density=True)
    # plt.title('Distribution of Classifier Weights')
    # plt.xlabel('Weight Value')
    # plt.ylabel('Density')
    # plt.grid(True)
    # plt.savefig('after_weight_distribution.png')
    # plt.close()
