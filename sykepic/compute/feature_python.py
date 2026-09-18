"""Extract features for raw IFCB data"""

import os
from joblib import Parallel, delayed
from pathlib import Path
import csv

from typing import List, Tuple, Optional, Literal, Union, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm

import numpy as np
from ifcb_features import compute_features
from sykepic.utils import files, ifcb, logger
from PIL import Image

VERSION = "py-v4.1"
FILE_SUFFIX = ".feat"
log = logger.get_logger("feat")

MICRON_FACTORS = {
    'ifcb': 2.8,
    'cytosense': 3.6
}

def compute_ifcb_features(roi_array):
    """Compute WHOI IFCB features and return them as a dictionary."""
    _, features, _ = compute_features(roi_array)
    return dict(features)

# Features from here: https://aslopubs.onlinelibrary.wiley.com/doi/full/10.1002/lno.12171
EXTENDED_FEATURES = [
    "summedArea",
    "ConvexArea",
    "summedConvexArea",
    "Perimeter",
    "summedPerimeter",
    "ConvexPerimeter",
    "summedConvexPerimeter",
    "Area_over_PerimeterSquared",
    "Area_over_Perimeter",
    "summedConvexPerimeter_over_Perimeter",
    "EquivDiameter",
    "summedMajorAxisLength",
    "summedMinorAxisLength",
    "Extent",
    "Solidity",
    "Eccentricity",
    "Orientation",
    "summedBiovolume",
]

def validate_args(args):
    """Validates command line arguments for feature extraction."""
    if args.device and not args.image_dir:
        raise ValueError("--device can only be used with --image-dir")

    if args.image_dir and not args.device:
        raise ValueError("--device is required when --image-dir is used")

def call(args):
    """
    Process command line arguments and extract features from samples.
    Args:
        args: Command line arguments containing sample paths, output directory,
                parallel processing flag, and force overwrite flag.
    Returns:
        None
    """

    validate_args(args)

    if args.raw:
        sample_paths: List[Path] = files.list_sample_paths(args.raw)
        filtered_sample_paths = filter_sample_paths(sample_paths)
        process_sample_list(
            sample_paths=filtered_sample_paths,
            sample_type="ifcb",
            device="ifcb",
            out_dir=args.out,
            parallel=args.parallel,
            force=args.force,
            save_all_features=args.save_all_features,
        )
    elif args.samples:
        sample_paths: List[Path] = [Path(path) for path in args.samples]
        filtered_sample_paths = filter_sample_paths(sample_paths)
        process_sample_list(
            sample_paths=filtered_sample_paths,
            sample_type="ifcb",
            device="ifcb",
            out_dir=args.out,
            parallel=args.parallel,
            force=args.force,
            save_all_features=args.save_all_features,
        )
    else: # args.image_dir
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            raise ValueError(f"The directory {args.image_dir} does not exist.")
        img_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        sample_paths = [file for file in img_files if file.is_file()]
        process_sample_list(
            sample_paths=sample_paths,
            sample_type="img",
            device=args.device,
            out_dir=args.out,
            parallel=args.parallel,
            force=args.force,
            save_all_features=args.save_all_features,
        )

def filter_sample_paths(sample_paths: List[Path]) -> List[Path]:
    """Filter sample paths to exclude those with .roi files larger than 1GB.
    Args:
        sample_paths (List[Path]): List of sample paths to filter.
    Returns:
        List[Path]: Filtered list of sample paths.
    """
    filtered_sample_paths = []
    for sample_path in sample_paths:
        if sample_path.with_suffix(".roi").stat().st_size <= 1e9:
            filtered_sample_paths.append(sample_path)
        else:
            log.warn(f"{sample_path.name} is over 1G, skipping")
    return filtered_sample_paths

def process_sample_list(
    sample_paths: List[Path],
    sample_type: Literal["ifcb", "img"],
    device: Literal['ifcb', 'cytosense'],
    out_dir: str,
    parallel: bool = False,
    force: bool = False,
    save_all_features: bool = False,
):
    """
    Extract features from a list of sample paths and save them to CSV files.
    Handles parallelization of process_sample processing.

    Args:
        sample_paths (List[Path]): List of paths to IFCB sample files.
        sample_type (Literal["ifcb", "img"]): Type of sample being processed.
        out_dir (str): Output directory where the CSV files will be saved.
        parallel (bool): Whether to process samples in parallel.
        force (bool): Whether to overwrite existing CSV files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if sample_type == 'img':
        csv_path = make_csv_path(
            sample_type=sample_type,
            sample_path=sample_paths[0].parent.name,
            out_dir=out_dir,
            force=force
        )
        if csv_path is None:
            return

    if parallel:
        slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
        if slurm_cpus:
            available_cores = int(slurm_cpus) -1
            print(f"Extracting features in parallel with {available_cores} cores (SLURM_CPUS_PER_TASK)")
        else:
            available_cores = os.cpu_count() -1
            print(f"Extracting features in parallel with {available_cores} cores (os.cpu_count())")
        
        samples_processed = Parallel(n_jobs=available_cores)(
            delayed(process_sample)(path, sample_type, device, out_dir, save_all_features, force) for path in tqdm(sample_paths)
        )

    else:
        log.debug("Extracting features synchronously")
        samples_processed = []
        for path in tqdm(sorted(sample_paths)):
            samples_processed.append(process_sample(path, sample_type, device, out_dir, save_all_features, force))

    # Aggregate results for image samples
    if sample_type == "img":
        image_features_to_csv(
            roi_features=samples_processed,
            csv_path=csv_path
        )
    
def make_csv_path(sample_type: Literal['ifcb', 'img'], sample_path: Path, out_dir: str, force: bool = False) -> Union[Path, None]:
    """
    Check if a CSV file for the given sample already exists.
    If it exists and force is False, returs None.
    If it exists and force is True, returns the path to the CSV file, overwriting it.
    If it does not exist, returns the path to the CSV file.
    """
    if sample_type == 'ifcb':
        csv_path: Path = files.sample_csv_path(sample_path, out_dir, suffix=FILE_SUFFIX)
    else:
        csv_path = out_dir / f"{sample_path}{FILE_SUFFIX}.csv"

    pid = os.getpid()
    if csv_path.is_file():
        if force:
            print(
                f"feat [{pid}] - WARNING - {str(csv_path)} already exists, overwriting"
            )
            return csv_path
        else:
            print(f"feat [{pid}] - WARNING - {str(csv_path)} already exists, skipping")
            return None
    return csv_path

@dataclass
class ROIFeatures:
    roi_id: str
    sample_type: str
    biovol_px: int
    area: float
    major_axis_length: float
    minor_axis_length: float
    biovol_um3: Optional[float] = None
    biomass_ugl: Optional[float]= None
    volume_ml: Optional[float] = None,
    extended_features: Optional[Dict[str, Any]] = None

def process_sample(sample_path: Path,
                   sample_type: Literal["ifcb", "img"],
                   device: Literal['ifcb', 'cytosense'],
                   out_dir: str,
                   save_all_features: bool = False,
                   force: bool=False) -> Union[List[ROIFeatures], None]:
    """
    Process a single sample to extract features and save them to a CSV file.
    Handles the creation of the output directory and checks for existing files.

    For IFCB samples that contain multiple ROIs, it extracts features from each ROI
    and saves them in a CSV file.
    
    For image samples, 

    Args:
        sample_path (Path): Path to the sample file.
        sample_type (Literal["ifcb", "img"]): Type of sample being processed.
        out_dir (Path): Output directory where the CSV file will be saved.
        force (bool): Whether to overwrite existing CSV files.
    Returns:
        Union[List[ROIFeatures], None]: 
            If sample_type is 'ifcb', returns None after saving features to CSV.
            If sample_type is 'ifcb' and the CSV file already exists, returns None without processing.
            If sample_type is 'img', returns a list of ROIFeatures for the image.
    """
    pid = os.getpid()
    print(f"feat [{pid}] - INFO - Extracting features for {sample_path.name}")

    if sample_type == 'ifcb':
        csv_path = make_csv_path(sample_type, sample_path, out_dir, force)
        if csv_path is None:
            return None
        roi_features = sample_ifcb_features(sample_path, save_all_features=save_all_features)
        ifcb_features_to_csv(roi_features, csv_path)
        return None

    elif sample_type == 'img':
        roi_features = sample_image_features(sample_path, device, save_all_features=save_all_features)
        return roi_features
    else:
        raise ValueError(f"Unknown sample type: {sample_type}")


def sample_ifcb_features(sample_path: Path,
                         save_all_features: bool = False) -> List[ROIFeatures]:
    """
    Extract features from an IFCB sample file.
    Args:
        sample_path (Path): Path to the IFCB sample file (.adc).
        save_all_features (bool): Whether to save all computed features from the ifcb-features library.
    Returns:
        List[ROIFeatures]: A list of ROIFeatures dataclasses containing the extracted features.
    """
    root = Path(sample_path)
    adc = root.with_suffix(".adc")
    hdr = root.with_suffix(".hdr")
    roi = root.with_suffix(".roi")
    try:
        volume_ml = sample_volume(hdr)
        # This didn't show in logs for some reason
        # if volume_ml <= 0:
        #   log.warn(f"{root.name} volume_ml is {volume_ml}")
    except Exception:
        log.exception(f"Unable to calculate volume for {root.name}")
        return None
    roi_features = []
    for roi_id, roi_array in ifcb.raw_to_numpy(adc, roi):
        roi_features.append(calculate_roi_features(roi_id=roi_id,
                                                   sample_type='ifcb',
                                                   roi_array=roi_array,
                                                   volume_ml=volume_ml,
                                                   save_all_features=save_all_features))
    return roi_features

def sample_image_features(sample_path: Path,
                          device: Literal['ifcb', 'cytosense'],
                          save_all_features: bool = False) -> List[ROIFeatures]:
    """
    calculate features for an image sample.
    Args:
        sample_path (Path): Path to the image file.
        device (Literal['ifcb', 'cytosense']): The imaging device where the image originated from. Used to set correct micron factor.
        save_all_features (bool): Whether to save all computed features from the ifcb-features library.
    Returns:
        ROIFeatures: A dataclass containing the calculated features for the image.
    """
    roi_array = np.array(Image.open(sample_path).convert("L"))  # Convert to grayscale
    roi_id = sample_path.name

    micron_factor = MICRON_FACTORS[device]

    roi_features = calculate_roi_features(
        roi_id=roi_id,
        sample_type='img',
        roi_array=roi_array,
        micron_factor=micron_factor,
        save_all_features=save_all_features
    )
    return roi_features

def calculate_roi_features(roi_id: str,
                           sample_type: Literal['ifcb', 'img'],
                           roi_array: np.array,
                           volume_ml: Optional[float] = None,
                           micron_factor: Optional[float] = 2.8,
                           save_all_features: bool = False) -> ROIFeatures:
    """
    Calculate features for a single ROI array.
    Args:
        roi_id (str): Identifier for the ROI.
        sample_type (Literal['ifcb', 'img']): Type of sample being processed. Is saved in the features.
        roi_array (np.array): Numpy array representing the ROI.
        volume_ml (Optional[float]): Volume of the sample in milliliters. If None, biomass_ugl will be None.
        micron_factor (Optional[float]): Micron factor for converting pixels to micrometers. Default is 2.8.
    Returns:
        ROIFeatures: A dataclass containing the calculated features for the ROI.
    """
    all_roi_features = compute_ifcb_features(roi_array)

    biovol_px = all_roi_features["Biovolume"]
    area = all_roi_features["Area"]
    major_axis_length = all_roi_features["MajorAxisLength"]
    minor_axis_length = all_roi_features["MinorAxisLength"]

    if volume_ml is not None:
        biovol_um3 = pixels_to_um3(biovol_px, micron_factor=micron_factor)
        biomass_ugl = biovolume_to_biomass(biovol_um3, volume_ml)
    else:
        biovol_um3 = pixels_to_um3(biovol_px, micron_factor=micron_factor)
        biomass_ugl = None
    
    # Save extended features if requested
    if save_all_features:
        extended_features = {
            feature_name: all_roi_features.get(feature_name, None)
            for feature_name in EXTENDED_FEATURES
        }
    else:
        extended_features = None

    return ROIFeatures(
        roi_id=roi_id,
        sample_type=sample_type,
        biovol_px=biovol_px,
        biovol_um3=biovol_um3,
        biomass_ugl=biomass_ugl,
        area=area,
        major_axis_length=major_axis_length,
        minor_axis_length=minor_axis_length,
        volume_ml=volume_ml,
        extended_features=extended_features
    )


def sample_volume(hdr_file):
    ifcb_flowrate = 0.25  # ml
    run_time = None
    inhibit_time = None
    with open(hdr_file) as fh:
        for line in fh:
            if line.startswith("inhibitTime"):
                inhibit_time = float(line.split()[1])
            elif line.startswith("runTime"):
                run_time = float(line.split()[1])
    sample_vol = ifcb_flowrate * ((run_time - inhibit_time) / 60.0)
    if sample_vol <= 0:
        raise ValueError(f"Sample volume is {sample_vol}")
    return sample_vol


def pixels_to_um3(pixels, micron_factor=2.8):
    return pixels / (micron_factor**3)
    # The micron factor should be 2.8 for python features v-4, but should still be tested for instruments
    # When calculating features for image folders, micron factor is set based on device type


def biovolume_to_biomass(biovol_um3, volume_ml):
    try:
        return biovol_um3 / volume_ml / 1000
    except ZeroDivisionError:
        return 0

COMMON_FIELDS = {
    "roi_id": "roi",
    "biovol_px": "biovolume_px",
    "biovol_um3": "biovolume_um3",
    "area": "area",
    "major_axis_length": "major_axis_length",
    "minor_axis_length": "minor_axis_length",
}

IFCB_FIELD_MAPPING = {
    **COMMON_FIELDS,
    "biomass_ugl": "biomass_ugl",
}

IMAGE_FIELD_MAPPING = {
    **COMMON_FIELDS,
}

def write_features_to_csv(
    features: List[ROIFeatures],
    csv_path: Path,
    field_mapping: Dict[str, str],
    metadata: Dict[str, Any]
):
    """Write ROI features to a CSV file with metadata as comments."""
    path_obj = Path(csv_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    csv_headers = list(field_mapping.values())

    with open(path_obj, "w", newline='') as fh:
        # Write metadata as comments
        for key, value in metadata.items():
            fh.write(f"# {key}={value}\n")
        
        writer = csv.DictWriter(fh, fieldnames=csv_headers)
        writer.writeheader()

        for roi in features:
            row = {}
            for attr_name, csv_header in field_mapping.items():
                if attr_name in EXTENDED_FEATURES and roi.extended_features is not None:
                    row[csv_header] = roi.extended_features.get(attr_name, None)
                else:
                    row[csv_header] = getattr(roi, attr_name)
            writer.writerow(row)

def ifcb_features_to_csv(roi_features: List[ROIFeatures], csv_path: Path):
    """
    Save IFCB ROI features to a CSV file.
    Args:
        roi_features (List[ROIFeatures]): List of ROIFeatures dataclasses containing the extracted features.
        csv_path (Path): Path to the CSV file where the features will be saved.
    """
    if not roi_features:
        return None
    if csv_path is None:
        raise ValueError("CSV path cannot be None") 
    
    sample_types = set([f.sample_type for f in roi_features])
    if sample_types != {'ifcb'}:
        raise ValueError(f"All ROI features must be of type 'ifcb'. Found: {sample_types}")
    
    volume_mls = set([f.volume_ml for f in roi_features])
    if len(volume_mls) != 1:
        raise ValueError(
            f"All ROI features must have the same volume_ml. Found: {volume_mls}"
        )
    volume_ml = roi_features[0].volume_ml
    metadata = {
        "version": VERSION,
        "volume_ml": volume_ml,
        "feature_extractor": "WHOIGit/ifcb-features",
        "feature_extractor_version": "1.3.0",
        "micron_factor": MICRON_FACTORS["ifcb"],
    }

    if roi_features[0].extended_features is not None:
        field_mapping = {**IFCB_FIELD_MAPPING}
        for feature_name in EXTENDED_FEATURES:
            field_mapping[feature_name] = feature_name
    else:
        field_mapping = IFCB_FIELD_MAPPING

    write_features_to_csv(
        features=roi_features,
        csv_path=csv_path,
        field_mapping=field_mapping,
        metadata=metadata
    )

def image_features_to_csv(roi_features: List[ROIFeatures], csv_path: Path):
    """
    Save image ROI features to a CSV file.
    Args:
        roi_features (List[ROIFeatures]): List of ROIFeatures dataclasses containing the extracted features.
        csv_path (Path): Path to the CSV file where the features will be saved.
    """
    if not roi_features:
        return None
    if csv_path is None:
        raise ValueError("CSV path cannot be None") 
    
    sample_types = set([f.sample_type for f in roi_features])
    if sample_types != {'img'}:
        raise ValueError(f"All ROI features must be of type 'img'. Found: {sample_types}")
    
    metadata = {
        "version": VERSION,
        "volume_ml": "None"
    }

    if roi_features[0].extended_features is not None:
        field_mapping = {**IMAGE_FIELD_MAPPING}
        for feature_name in EXTENDED_FEATURES:
            field_mapping[feature_name] = feature_name
    else:
        field_mapping = IMAGE_FIELD_MAPPING

    write_features_to_csv(
        features=roi_features,
        csv_path=csv_path,
        field_mapping=field_mapping,
        metadata=metadata
    )
