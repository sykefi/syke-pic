"""This module contains the main logic for model inference."""

import logging
import shutil
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sykepic.utils import ifcb
from sykepic.train.config import get_img_shape, get_transforms, get_network
from sykepic.train.data import ImageDataset

log = logging.getLogger("predict")


def main(args):
    if args.softmax_exp == "e":
        args.softmax_exp = None
    else:
        args.softmax_exp = float(args.softmax_exp)
    predict(
        args.model,
        args.raw,
        args.out,
        args.batch_size,
        args.num_workers,
        args.softmax_exp,
        None,
        args.limit,
        args.force,
    )


def predict(
    model,
    raw_dir,
    out_dir,
    batch_size,
    num_workers,
    softmax_exp=None,
    sample_filter=None,
    limit=None,
    force=False,
    progress_bar=True,
):
    # Find all samples from raw_dir, optionally filter them by name (sync)
    samples = []
    for adc in sorted(Path(raw_dir).glob("**/*.adc")):
        sample = adc.with_suffix("").name
        if sample_filter and sample not in sample_filter:
            continue
        sample_date = ifcb.sample_to_datetime(sample)
        day_path = sample_date.strftime("%Y/%m/%d")
        csv = Path(out_dir) / day_path / (adc.with_suffix(".csv").name)
        samples.append((adc, csv))

    if not samples:
        log.error("No samples to predict")

    # Prepare model
    model = Path(model)
    with open(model / "class_names.txt") as fh:
        classes = fh.read().splitlines()
    config = ConfigParser()
    config_file = model / "config.ini"
    config.read(config_file)
    img_shape = get_img_shape(config)
    _, eval_transform = get_transforms(config, img_shape)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = get_network(config, len(classes))
    net.load_state_dict(torch.load(model / "best_state.pth", map_location=device))

    # Choose a subset of samples based if limit is provided
    if limit and limit > 0 and len(samples) > limit:
        idxs = np.linspace(0, len(samples) - 1, num=limit, dtype=int)
        samples = [samples[i] for i in idxs]

    # Optionally hide tqdm progress bar
    iterator = tqdm(samples, desc="Prediction progress") if progress_bar else samples

    predicted_samples = set()
    # Start prediction process
    for adc, csv in iterator:
        sample = adc.with_suffix("").name
        log.debug(f"Predicting {sample}")
        if not force and csv.is_file():
            log.error(f"{csv.name} already exists, skipping")
            predicted_samples.add(sample)
            continue
        img_dir = f"{csv.with_suffix('')}_images"
        roi = adc.with_suffix(".roi")
        try:
            try:
                ifcb.raw_to_png(adc, roi, out_dir=img_dir)
            except FileExistsError:
                log.error(f"Images already extracted for '{sample}'")
            img_paths = sorted(Path(img_dir).glob("**/*.png"))
            dataset = ImageDataset(
                img_paths, transform=eval_transform, num_chans=img_shape[0]
            )
            dataloader = DataLoader(dataset, batch_size, num_workers=num_workers)
            predictions = make_predictions(
                net, dataloader, classes, softmax_exp, device
            )
            with open(csv, "w") as fh:
                data = "roi,"
                data += ",".join(classes)
                for roi, probs in predictions:
                    data += f"\n{roi},"
                    data += ",".join(f"{p:.5f}" for p in probs)
                fh.write(data)
        except Exception:
            log.exception("While predicting '{sample}'")
            raise
        finally:
            # Remove extracted images even in case of exception
            shutil.rmtree(img_dir)
        predicted_samples.add(sample)
    return predicted_samples


def make_predictions(net, dataloader, classes, softmax_exp=None, device="cpu"):
    """
    Returns
    -------
    list of tuples
        Each tuple consists of a roi number and
        a list of class prediction probabilities i.e. (roi, probs)
    """

    predictions = []
    net.to(device)
    net.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, paths = batch[0].to(device), batch[1]
            out = net(x)
            rois = tuple(int(Path(p).with_suffix("").name) for p in paths)
            # Change softmax exponent by multiplying by log(new_exponent)
            if softmax_exp:
                out = out * np.log(softmax_exp)
            probs = F.softmax(out, dim=1)
            predictions.extend(zip(rois, probs.tolist()))
    # Sort predictions by roi number (ascending)
    predictions = sorted(predictions)
    return predictions


def entropy_confidence(probs):
    """
    Not in use, since softmax with a smaller exponent (than e)
    seems to work equally well, if not better, in determening
    confident predictions.
    """
    num_classes = probs[0].numel()
    logs = probs * torch.log2(probs)
    numerator = 0 - torch.sum(logs, dim=1).cpu()
    denominator = np.log2(num_classes)
    result = (numerator / denominator).tolist()
    return result
