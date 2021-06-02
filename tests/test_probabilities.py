from pathlib import Path

import numpy as np
import pytest
import torch

from sykepic.compute import probabilities as prob
from sykepic.train.network import TorchVisionNet
from sykepic.train.image import Compose


@pytest.fixture
def net_and_params():
    model_dir = "examples/models/resnet18_20201022"
    net, classes, img_shape, eval_transform, device = prob.prepare_model(model_dir)
    params = prob.EvalParams(
        batch_size=32,
        num_workers=1,
        classes=classes,
        img_shape=img_shape,
        transform=eval_transform,
        device=device,
    )
    return net, params


def test_main(tmp_path):
    sample_paths = [
        Path("tests/data/raw/valid/D20180712T065600_IFCB114"),
        Path("tests/data/raw/invalid/D20210523T053149_IFCB114"),
    ]
    model_dir = "examples/models/resnet18_20201022"
    out_dir = tmp_path / "prob"
    samples_processed = prob.main(
        sample_paths,
        model_dir,
        out_dir,
        batch_size=32,
        num_workers=1,
        progress_bar=False,
    )
    assert isinstance(samples_processed, set)
    assert len(samples_processed) == 1
    assert len(list(out_dir.glob("**/*.csv"))) == 1


def test_prepare_model():
    model_dir = "examples/models/resnet18_20201022"
    net, classes, img_shape, eval_transform, device = prob.prepare_model(model_dir)
    assert isinstance(net, TorchVisionNet)
    assert isinstance(classes, list)
    assert isinstance(classes[0], str)
    assert len(img_shape) == 3
    assert isinstance(eval_transform, Compose)
    assert isinstance(device, torch.device)


def test_process_sample(net_and_params, tmp_path):
    net, params = net_and_params
    sample_path = Path("tests/data/raw/valid/D20180712T065600_IFCB114")
    out_dir = tmp_path / "prob"
    sample = prob.process_sample(sample_path, net, params, out_dir, force=False)
    assert isinstance(sample, str)
    csv = (
        out_dir
        / "2018"
        / "07"
        / "12"
        / f"D20180712T065600_IFCB114{prob.FILE_SUFFIX}.csv"
    )
    assert csv.is_file()
    with open(csv) as fh:
        header = fh.readline().strip().split(",")
        line1 = fh.readline().strip().split(",")
        line2 = fh.readline().strip().split(",")
    assert header[0] == "roi"
    assert header[1:] == params.classes
    line1_sum = sum(map(float, line1[1:]))
    line2_sum = sum(map(float, line2[1:]))
    assert abs(line1_sum - 1.0) < 0.001
    assert abs(line2_sum - 1.0) < 0.001


# Not in use
def slow_test_process_sample(net_and_params, tmp_path):
    net, params = net_and_params
    sample_path = Path("tests/data/raw/valid/D20180816T091250_IFCB114")
    out_dir = tmp_path / "prob"
    sample = prob.process_sample(sample_path, net, params, out_dir, force=False)
    assert isinstance(sample, str)
    csv = next(out_dir.glob("**/*.csv"))
    assert csv.is_file()
    with open(csv) as fh:
        classes = fh.readline().strip().split(",")[1:]
        apha_idx = classes.index("Aphanizomenon_flosaquae")
        doli_idx = classes.index("Dolichospermum-Anabaenopsis")
        osci_idx = classes.index("Oscillatoriales")
        for i, line in enumerate(fh):
            line = line.strip().split(",")
            roi, *probs = line
            roi = int(roi)
            if roi in (125, 346, 1253, 1268, 1512):
                assert np.argmax(probs) == apha_idx
            elif roi in (1693, 1107):
                assert np.argmax(probs) == doli_idx
            elif roi in (294, 1420, 1537):
                assert np.argmax(probs) == osci_idx
            if i > 2000:
                break
