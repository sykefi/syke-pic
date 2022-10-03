"""Classes and functions for neural network architecture and logic."""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


class TorchVisionNet(nn.Module):
    """Model with a (pre-trained) TorchVision network as base."""

    def __init__(
        self,
        name,
        num_classes,
        weights="DEFAULT",
        head=[256, 128],
        dropout=[],
        last_activation=None,
    ):
        """
        Parameters
        ----------
        name : str
            Name of a valid TorchVision model, e.g., resnet18, efficientnet_b0.
        num_classes : int
            Number of neurons to use in the very last layer (classification).
        weights: str, None
            Pre-trained weights name, DEFAULT = best available,
            None = no pre-training.
        head : list[int]
            List of integers, in which every number represents the
            number of neurons in a layer in the network head.
            So these are all the final layers except the last one.
        dropout : list[tuple[int, float]]
            List of tuples, where each tuple contains two values (int, float).
            The first value is the index (relative to the `head`) of where
            to place the dropout layer, the second is the probability parameter
            for that dropout.
        last_activation : str
            Activation function for the last layer, e.g., softmax, log_softmax.
            None is the default.
        """

        super().__init__()
        model = getattr(models, name)(weights=weights)
        layers = list(model.children())
        last_linear = layers[-1]
        if isinstance(last_linear, nn.Sequential):
            for layer in last_linear:
                if isinstance(layer, nn.Linear):
                    last_linear = layer
                    break
        head.insert(0, last_linear.in_features)
        head.append(num_classes)  # Last layer has num_classes neurons
        head_layers = [nn.Linear(head[i], head[i + 1]) for i in range(len(head) - 1)]
        if dropout:
            for idx, p in dropout:
                head_layers.insert(idx, nn.Dropout(p))
        self.base = nn.Sequential(*layers[:-1])
        self.head = nn.Sequential(*head_layers)
        self.last_activation = last_activation

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        if self.last_activation:
            x = getattr(F, self.last_activation)(x, dim=1)
        return x


class LRWarmup:
    """Learning rate warmup callback"""

    def __init__(
        self,
        net,
        optimizer,
        factor_1=0.1,
        factor_2=0.5,
        step_1=5,
        step_2=15,
        step_3=30,
        verbose=True,
    ):
        self.net = net
        self.optimizer = optimizer
        self.factor_1 = factor_1
        self.factor_2 = factor_2
        self.step_1 = step_1
        self.step_2 = step_2
        self.step_3 = step_3
        self.verbose = verbose

    def __call__(self, epoch):
        if epoch == self.step_1:
            # Update head lr
            self.optimizer.param_groups[0]["lr"] *= self.factor_1
            if self.verbose:
                print(f"[INFO] LRWarmup step 1 completed:\n{self.optimizer}")

        elif epoch == self.step_2:
            # Start fine tuning last sequential layer of base
            # NOTE! Index -2 might not be correct for all models
            new_part = self.net.base[-2:]
            make_trainable(new_part)
            new_params = list(filter_params(new_part))
            new_lr = self.optimizer.param_groups[0]["lr"] * self.factor_1
            self.optimizer.param_groups[1]["params"] = new_params
            self.optimizer.param_groups[1]["lr"] = new_lr
            # Update head lr
            self.optimizer.param_groups[0]["lr"] *= self.factor_2
            if self.verbose:
                print(f"[INFO] LRWarmup step 2 completed:\n{self.optimizer}")

        elif epoch == self.step_3:
            # Start fine tuning rest of base layers
            new_part = self.net.base[:-2]
            make_trainable(new_part)
            new_params = list(filter_params(new_part))
            new_lr = self.optimizer.param_groups[1]["lr"] * self.factor_1
            self.optimizer.param_groups[2]["params"] = new_params
            self.optimizer.param_groups[2]["lr"] = new_lr
            # Update head lr
            self.optimizer.param_groups[0]["lr"] *= self.factor_2
            if self.verbose:
                print(f"[INFO] LRWarmup step 3 completed:\n{self.optimizer}")


def make_trainable(module):
    """Enable gradient calculation for module parameters."""

    for param in module.parameters():
        param.requires_grad = True
    module.train()  # Necessary?


def make_untrainable(module):
    """Disable gradient calculation for module parameters."""

    for param in module.parameters():
        param.requires_grad = False
    module.eval()  # Necessary?


def freeze(module, n=None):
    """Freeze layers until layer n (not included)."""

    child_modules = list(module.children())
    max_n = int(n) if n else len(child_modules)
    for child in child_modules[:max_n]:
        recursive_freeze(child)
    for child in child_modules[max_n:]:
        make_trainable(child)


def recursive_freeze(module):
    """Freezes all child modules (layers) of parent module recursively."""

    child_modules = list(module.children())
    if child_modules:
        for child in child_modules:
            recursive_freeze(child)
    else:
        # Dont' freeze batch norm layers
        if isinstance(module, BN_TYPES):
            make_trainable(module)
        else:
            make_untrainable(module)


def filter_params(module):
    """Yields all trainable but non batch norm params."""

    child_modules = list(module.children())
    if child_modules:
        for child in child_modules:
            for param in filter_params(child):
                yield param
    else:
        if not isinstance(module, BN_TYPES):
            for param in module.parameters():
                if param.requires_grad:
                    yield param
