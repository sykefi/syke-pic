"""This module contains the main logic for model inference."""

import shutil
from configparser import ConfigParser
from pathlib import Path

import boto3
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import allas
from . import ifcb
from sykepic.train.config import get_img_shape, get_transforms, get_network
from sykepic.train.data import ImageDataset


def main(args):
    if args.softmax_exp == 'e':
        args.softmax_exp = None
    else:
        args.softmax_exp = float(args.softmax_exp)
    if args.allas:
        allas_setup(args)
    else:
        predict(args.model, args.raw, args.out, args.batch_size,
                args.num_workers, args.softmax_exp, args.limit, args.force)


def allas_setup(args):
    raw_root = Path(args.raw)
    out_root = Path(args.out)
    s3 = boto3.resource('s3', endpoint_url='https://a3s.fi')
    samples = []
    for allas_adc in list(allas.ls(args.allas, '.adc', s3)):
        sample = Path(allas_adc).with_suffix('').name
        date = ifcb.sample_to_datetime(sample)
        day_path = date.strftime('%Y/%m/%d')
        csv = (out_root/day_path/sample).with_suffix('.csv')
        if csv.is_file() and not args.force:
            # print(f'[INFO] {sample} already analysed')
            continue
        sample_dir = raw_root/day_path
        adc = (sample_dir/sample).with_suffix('.adc')
        if not adc.is_file():
            if not sample_dir.is_dir():
                sample_dir.mkdir(parents=True)
            try:
                for ext in ('.adc', '.hdr', '.roi'):
                    file = (Path(args.allas)/sample).with_suffix(ext)
                    print(f'[INFO] Downloading {file}')
                    allas.download(file, sample_dir, s3)
            except Exception as e:
                print(f'[ERROR] {sample}: {e}')
                # Remove all raw files, so sample can be tried again later
                for ext in ('.adc', '.hdr', '.roi'):
                    (sample_dir/sample).with_suffix(ext).unlink(missing_ok=True)
                continue
        samples.append((adc, csv))
    if samples:
        # print(f'[INFO] Predicting {len(samples)} samples')
        predict(args.model, samples, None, args.batch_size, args.num_workers,
                args.softmax_exp, args.limit, args.force)


def predict(model, raw, out_root, batch_size, num_workers,
            softmax_exp=None, limit=None, force=False):
    if isinstance(raw, (str, Path)):
        # main called from __name__
        samples = []
        for adc in sorted(Path(raw).glob('**/*.adc')):
            # sample = adc.with_suffix('').name
            date = ifcb.sample_to_datetime(adc.name)
            day_path = date.strftime('%Y/%m/%d')
            csv = Path(out_root)/day_path/(adc.with_suffix('.csv').name)
            samples.append((adc, csv))
    else:
        # main called from allas_predict
        samples = raw

    # Prepare model
    model = Path(model)
    with open(model/'class_names.txt') as fh:
        classes = fh.read().splitlines()
    config = ConfigParser()
    config_file = model/'config.ini'
    config.read(config_file)
    img_shape = get_img_shape(config)
    _, eval_transform = get_transforms(config, img_shape)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = get_network(config, len(classes))
    net.load_state_dict(torch.load(
        model/'best_state.pth', map_location=device))

    # Choose a subset of samples based if limit is provided
    if limit and limit > 0:
        idxs = np.linspace(0, len(samples)-1, num=limit, dtype=int)
        samples = [samples[i] for i in idxs]

    for adc, csv in tqdm(samples, desc='Prediction progress'):
        if not force and csv.is_file():
            print(f'[ERROR] {csv} already exists, -f [--force] to overwrite')
            continue
        img_dir = f"{csv.with_suffix('')}_images"
        roi = adc.with_suffix('.roi')
        try:
            try:
                ifcb.raw_to_png(adc, roi, out_dir=img_dir)
            except FileExistsError:
                print('[ERROR] Images already extracted')
            img_paths = sorted(Path(img_dir).glob('**/*.png'))
            dataset = ImageDataset(img_paths, transform=eval_transform,
                                   num_chans=img_shape[0])
            dataloader = DataLoader(dataset, batch_size,
                                    num_workers=num_workers)
            predictions = make_predictions(net, dataloader, classes,
                                           softmax_exp, device)
            with open(csv, 'w') as fh:
                data = 'roi,'
                data += ','.join(classes)
                for roi, probs in predictions:
                    data += f'\n{roi},'
                    data += ','.join(f'{p:.5f}' for p in probs)
                fh.write(data)
        except Exception:
            raise
        finally:
            # Remove extracted images
            shutil.rmtree(img_dir)


def make_predictions(net, dataloader, classes, softmax_exp=None, device='cpu'):
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
            rois = tuple(int(Path(p).with_suffix('').name) for p in paths)
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
