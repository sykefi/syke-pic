"""Functions for handling data from Imaging FlowCytobot (IFCB)."""

import datetime
import re
from pathlib import Path

import cv2
import numpy as np

from . import logger

log = logger.get_logger("ifcb")


def sample_to_datetime(sample):
    """Parse IFCB sample name into a datetime object

    If sample name is D20180703T093453_IFCB114, a datetime object
    is returned with the following attributes:
    year=2018, month=7, day=3, hour=9, minute=34, second=53

    Parameters
    ----------
    sample : str
        Sample name, with or without a file extension

    Returns
    -------
    datetime
        A datetime object extracted from sample name
    """

    m = re.match(r"D(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})", sample)
    timestamp = datetime.datetime(*[int(t) for t in m.groups()])
    return timestamp


def extract_sample_images(sample, raw_dir, out_dir, exist_ok=False):
    """Extract sample's raw data into images

    The images will be saved to `out_dir`. This directory will
    be created if it doesn't already exist.

    Each image has a name that corresponds to the row number
    in <sample>.adc file. This is equivalent to the roi-number.

    Parameters
    ----------
    sample : str
        Sample name without any extensions, e.g. D20180703T093453_IFCB114
    raw_dir : str, Path
        Root directory of raw IFCB data
    out_dir : str, Path
        Where to output images
    exist_ok : bool
        Whether to allow overwriting to existing out_dir
    """

    try:
        adc = next(Path(raw_dir).glob(f"**/{sample}.adc"))
    except StopIteration:
        log.error(f"Sample {sample} not found in {raw_dir}")
        raise
    roi = adc.with_suffix(".roi")
    raw_to_png(adc, roi, out_dir, force=exist_ok)


def raw_to_png(adc, roi, out_dir=None, force=False):
    """Parses .adc and .roi files into PNG images

    Parameters
    ----------
    adc : str, Path
        Path to .adc-file
    roi : str, Path
        Path to .roi-file
    out_dir : str, Path
        Defaults to adc-file's stem
    force : bool
        Overwrite existing images in out_dir
    """

    adc = Path(adc)
    roi = Path(roi)
    for f in (adc, roi):
        if not f.is_file():
            raise FileNotFoundError(f)
    sample = adc.with_suffix('').name
    out_dir = Path(adc.with_suffix("")) if not out_dir else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=force)
    # Read bytes from .roi-file into 8-bit integers
    roi_data = np.fromfile(roi, dtype="uint8")
    # Parse each line of .adc-file
    with open(adc) as adc_fh:
        for i, line in enumerate(adc_fh, start=1):
            line = line.split(",")
            roi_x = int(line[15])  # ROI width
            roi_y = int(line[16])  # ROI height
            start = int(line[17])  # start byte
            # Skip empty roi
            if roi_x < 1 or roi_y < 1:
                continue
            # roi_data is a 1-dimensional array, where
            # all roi are stacked one after another.
            end = start + (roi_x * roi_y)
            # Reshape into 2-dimensions
            img = roi_data[start:end].reshape((roi_y, roi_x))
            img_path = out_dir / f"{sample}_{i:05}.png"
            # imwrite reshapes automatically to 3-dimensions (RGB)
            cv2.imwrite(str(img_path), img)


def raw_to_numpy(adc, roi):
    adc = Path(adc)
    # Read bytes from .roi-file into 8-bit integers
    roi_data = np.fromfile(roi, dtype="uint8")
    # Parse each line of .adc-file
    with adc.open() as adc_fh:
        for i, adc_line in enumerate(adc_fh, start=1):
            np_arr = next_roi(roi_data, adc_line)
            if np_arr is not None:
                yield i, np_arr


def next_roi(roi_data, adc_line):
    adc_line = adc_line.split(",")
    roi_x = int(adc_line[15])  # ROI width
    roi_y = int(adc_line[16])  # ROI height
    # Skip empty roi
    if roi_x < 1 or roi_y < 1:
        return None
    # roi_data is a 1-dimensional array, where
    # all roi are stacked one after another.
    start = int(adc_line[17])  # start byte
    end = start + (roi_x * roi_y)
    # Reshape into 2-dimensions
    return roi_data[start:end].reshape((roi_y, roi_x))
