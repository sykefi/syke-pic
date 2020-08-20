"""Classes and functions for neural network architecture and logic."""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


class TorchVisionNet(nn.Module):
    """Network with a pretrained TorchVision model as base.

    Currently this only tested with ResNets, so other base models
    might not work as expected.
    """

    def __init__(self, model_name, num_classes, head=[256, 128],
                 dropout=[], last_activation=None):
        """
        Parameters
        ----------
        model_name : str
            resnet18, resnet50, etc.
        num_classes : int
            Number of neurons to use in the very last layer
        head : list
            List of integers, in which every number represents the
            number of neurons in a layer in the network head.
            So these are all the final layers except the last one.
        last_activation : str
            Activation function for the last layer
            e.g. softmax, log_softmax. None is the default.
        """

        super().__init__()
        base_model = getattr(models, model_name)(pretrained=True)
        base_layers = list(base_model.children())[:-1]
        head.insert(0, base_model.fc.in_features)
        head.append(num_classes)  # Last layer has num_classes neurons
        head_layers = [nn.Linear(head[i], head[i+1])
                       for i in range(len(head)-1)]
        if dropout:
            for idx, p in dropout:
                head_layers.insert(idx, nn.Dropout(p))
        self.base = nn.Sequential(*base_layers)
        self.head = nn.Sequential(*head_layers)
        self.last_activation = last_activation

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        if self.last_activation:
            x = getattr(F, self.last_activation)(x, dim=1)
        return x


class LRWarmup():
    """Learning rate warmup callback"""

    def __init__(self, net, optimizer, factor_1=0.1, factor_2=0.5,
                 step_1=5, step_2=15, step_3=30, verbose=True):
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
            self.optimizer.param_groups[0]['lr'] *= self.factor_1
            if self.verbose:
                print(f'[INFO] LRWarmup step 1 completed:\n{self.optimizer}')

        elif epoch == self.step_2:
            # Start fine tuning last layer of base
            new_part = self.net.base[-2:]
            make_trainable(new_part)
            new_params = list(filter_params(new_part))
            new_lr = self.optimizer.param_groups[0]['lr'] * self.factor_1
            self.optimizer.param_groups[1]['params'] = new_params
            self.optimizer.param_groups[1]['lr'] = new_lr
            # Update head lr
            self.optimizer.param_groups[0]['lr'] *= self.factor_2
            if self.verbose:
                print(f'[INFO] LRWarmup step 2 completed:\n{self.optimizer}')

        elif epoch == self.step_3:
            # Start fine tuning rest of base layers
            new_part = self.net.base[:-2]
            make_trainable(new_part)
            new_params = list(filter_params(new_part))
            new_lr = self.optimizer.param_groups[1]['lr'] * self.factor_1
            self.optimizer.param_groups[2]['params'] = new_params
            self.optimizer.param_groups[2]['lr'] = new_lr
            # Update head lr
            self.optimizer.param_groups[0]['lr'] *= self.factor_2
            if self.verbose:
                print(f'[INFO] LRWarmup step 3 completed:\n{self.optimizer}')


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
