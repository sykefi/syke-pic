"""Compute class probabilities for raw IFCB data"""

import shutil
from collections import namedtuple
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sykepic.train.config import get_img_shape, get_network, get_transforms
from sykepic.train.data import ImageDataset
from sykepic.utils import files, ifcb, logger

SOFTMAX_EXP = 1.3
FILE_SUFFIX = ".prob"
log = logger.get_logger("prob")
EvalParams = namedtuple(
    "EvalParams",
    ["batch_size", "num_workers", "classes", "img_shape", "transform", "device"],
)


def call(args):
    if args.image_dir or args.images:
        samples_as_images = True
        if args.image_dir:
            img_paths = sorted(Path(args.image_dir).rglob("*.png"))
        else:
            img_paths = sorted(Path(path) for path in args.images)
        sample_paths = {}
        for sample, img_path in ((p.name.rpartition("_")[0], p) for p in img_paths):
            sample_paths.setdefault(sample, []).append(img_path)
    else:
        samples_as_images = False
        if args.raw:
            sample_paths = files.list_sample_paths(args.raw)
        else:
            sample_paths = [Path(path) for path in args.samples]
    main(
        sample_paths,
        args.model,
        args.out,
        args.batch_size,
        args.num_workers,
        args.force,
        progress_bar=True,
        samples_as_images=samples_as_images,
    )


def main(
    sample_paths,
    model_dir,
    out_dir,
    batch_size=64,
    num_workers=2,
    force=False,
    progress_bar=True,
    samples_as_images=False,
):
    # Prepare model
    net, classes, img_shape, eval_transform, device = prepare_model(model_dir)
    params = EvalParams(
        batch_size=batch_size,
        num_workers=num_workers,
        classes=classes,
        img_shape=img_shape,
        transform=eval_transform,
        device=device,
    )
    # Start probability process
    if samples_as_images:
        # Optionally hide tqdm progress bar
        iterator = (
            tqdm(sample_paths.items(), desc="Processing samples")
            if progress_bar
            else sample_paths.items()
        )
        for sample, img_paths in iterator:
            csv_path = Path(out_dir) / f"{sample}{FILE_SUFFIX}.csv"
            process_images(img_paths, net, params, csv_path, force)
    else:
        # Optionally hide tqdm progress bar
        iterator = (
            tqdm(sample_paths, desc="Processing samples")
            if progress_bar
            else sample_paths
        )
        samples_processed = set()
        for sample_path in iterator:
            try:
                samples_processed.add(
                    process_sample(sample_path, net, params, out_dir, force)
                )
            except ValueError:
                log.exception(f"Faulty raw data for {sample_path.name}")
        return samples_processed


def prepare_model(model_dir):
    model_dir = Path(model_dir)
    with open(model_dir / "class_names.txt") as fh:
        classes = fh.read().splitlines()
    config = ConfigParser()
    config_file = model_dir / "config.ini"
    config.read(config_file)
    img_shape = get_img_shape(config)
    _, eval_transform = get_transforms(config, img_shape)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_network(config, len(classes))
    model.load_state_dict(torch.load(model_dir / "best_state.pth", map_location=device))
    return model, classes, img_shape, eval_transform, device


def process_sample(sample_path, net, params, out_dir, force=False):
    sample = sample_path.name
    csv_path = files.sample_csv_path(sample_path, out_dir, suffix=FILE_SUFFIX)
    if csv_path.is_file():
        if force:
            log.warn(f"{csv_path.name} already exists, overwriting")
        else:
            log.warn(f"{csv_path.name} already exists, skipping")
            return sample
    log.debug(f"Computing probabilities for {sample}")
    img_dir = f"{sample_path}_images"
    roi = sample_path.with_suffix(".roi")
    adc = sample_path.with_suffix(".adc")
    try:
        ifcb.raw_to_png(adc, roi, out_dir=img_dir, force=True)
        img_paths = sorted(Path(img_dir).glob("**/*.png"))
        dataset = ImageDataset(
            img_paths, transform=params.transform, num_chans=params.img_shape[0]
        )
        dataloader = DataLoader(
            dataset, params.batch_size, num_workers=params.num_workers
        )
        probabilities = net_pass(net, dataloader, params.device)
    except Exception:
        raise
    finally:
        # Remove extracted images even in case of exception
        shutil.rmtree(img_dir, ignore_errors=True)
    probabilities_to_csv(probabilities, params.classes, csv_path)
    return sample


def process_images(img_paths, net, params, csv_path, force=False):
    if csv_path.is_file():
        if force:
            log.warn(f"{csv_path.name} already exists, overwriting")
        else:
            log.warn(f"{csv_path.name} already exists, skipping")
            return
    dataset = ImageDataset(
        img_paths, transform=params.transform, num_chans=params.img_shape[0]
    )
    dataloader = DataLoader(dataset, params.batch_size, num_workers=params.num_workers)
    probabilities = net_pass(net, dataloader, params.device)
    probabilities_to_csv(probabilities, params.classes, csv_path)


def net_pass(net, dataloader, device="cpu"):
    """Returns a list of tuples: [(roi, probs),...]"""

    results = []
    net.to(device)
    net.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, paths = batch[0].to(device), batch[1]
            out = net(x)
            rois = tuple(int(Path(p).stem.split("_")[-1]) for p in paths)
            # Change softmax exponent by multiplying by log(new_exponent)
            if SOFTMAX_EXP:
                out = out * np.log(SOFTMAX_EXP)
            probs = F.softmax(out, dim=1)
            results.extend(zip(rois, probs.tolist()))
    # Sort predictions by roi number (ascending)
    return sorted(results)


def probabilities_to_csv(probabilities, classes, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_content = "roi," + ",".join(classes) + "\n"
    for roi, probs in probabilities:
        csv_content += f"{roi}," + ",".join(f"{p:.5f}" for p in probs) + "\n"
    with open(csv_path, "w") as fh:
        fh.write(csv_content)
