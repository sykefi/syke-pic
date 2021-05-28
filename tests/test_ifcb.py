from datetime import datetime

import pytest

from sykepic.utils import ifcb


@pytest.mark.parametrize(
    "sample",
    [
        "D20210523T053149_IFCB114.adc",
        "D20210523T053149_IFCB114",
        "D20210523T053149",
    ],
)
def test_sample_to_datetime(sample):
    expected = datetime(2021, 5, 23, 5, 31, 49)
    result = ifcb.sample_to_datetime(sample)
    assert expected == result


def test_raw_to_png_valid(tmp_path):
    adc = "tests/data/raw/valid/D20180712T065600_IFCB114.adc"
    roi = "tests/data/raw/valid/D20180712T065600_IFCB114.roi"
    out_dir = tmp_path / "images"
    ifcb.raw_to_png(adc, roi, out_dir=out_dir)
    number_of_extracted_images = len(list(out_dir.glob("*.png")))
    assert number_of_extracted_images == 2


def test_raw_to_png_invalid_1(tmp_path):
    adc = "tests/data/raw/invalid/D20210523T053149_IFCB114.adc"
    roi = "tests/data/raw/invalid/D20210523T053149_IFCB114.roi"
    out_dir = tmp_path / "images"
    with pytest.raises(ValueError):
        ifcb.raw_to_png(adc, roi, out_dir=out_dir)


def test_raw_to_png_invalid_2(tmp_path):
    adc = "tests/invalid/data/raw/D20210523T053149_IFCB114.adc"
    roi = "tests/invalid/data/raw/D20210523T053149.roi"
    out_dir = tmp_path / "images"
    with pytest.raises(FileNotFoundError):
        ifcb.raw_to_png(adc, roi, out_dir=out_dir)
