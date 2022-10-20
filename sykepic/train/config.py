"""Helper functions for train config file parsing"""

from configparser import NoOptionError

from torchvision.transforms import Normalize, ToTensor

from .image import (
    ChangeBrightness,
    Compose,
    FlipHorizontal,
    FlipVertical,
    Resize,
    Rotate,
    Translate,
    Zoom,
)
from .network import TorchVisionNet


def get_img_shape(config):
    img_shape = tuple(int(i) for i in config.get("image", "shape").split(","))
    return img_shape


def get_transforms(config, img_shape):
    augmentations = [
        aug.strip() for aug in config.get("image", "augmentations").split(",")
    ]
    border = config.get("image", "border")
    # Resize applied by default
    train_transform = [Resize()]
    eval_transform = [Resize()]
    for aug in augmentations:
        if aug == "flip":
            train_transform.extend([FlipHorizontal(), FlipVertical()])
        if aug == "translate":
            train_transform.append(Translate())
        if aug == "rotate":
            max_rotation = config.getint("image", "max_rotation")
            train_transform.append(Rotate(max_rotation))
        if aug == "zoom":
            zoom_range = tuple(
                float(i) for i in config.get("image", "zoom_range").split(",")
            )
            train_transform.append(Zoom(zoom_range))
        if aug == "brightness":
            brightness_range = tuple(
                float(i) for i in config.get("image", "brightness_range").split(",")
            )
            train_transform.append(ChangeBrightness(brightness_range))
    # ToTensor applied by default
    train_transform.append(ToTensor())
    eval_transform.append(ToTensor())
    # Normalization is done after converting to tensor
    if config.getboolean("image", "imagenet_normalization"):
        train_transform.append(Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    train_transform = Compose(train_transform, img_shape[1:], border)
    eval_transform = Compose(eval_transform, img_shape[1:], border)

    return train_transform, eval_transform


def get_network(config, num_classes):
    network = config.get("model", "network")
    try:
        weights = config.get("model", "weights")
        weights = None if not weights else weights
    except NoOptionError:
        # Previous model config didn't have weights option
        weights = "DEFAULT"
    head = [int(i) for i in config.get("model", "head").split(",")]
    dropout = []
    if config.get("model", "dropout"):
        for drop in config.get("model", "dropout").split(";"):
            idx, p = drop.split(",")
            dropout.append((int(idx), float(p)))
    return TorchVisionNet(network, num_classes, weights, head, dropout)
