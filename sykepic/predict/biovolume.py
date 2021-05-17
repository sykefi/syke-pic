import os
from multiprocessing import Pool, get_context
from pathlib import Path

from ifcb_features import compute_features
from sykepic.utils import ifcb, logger

log = logger.get_logger("feat")


def main(raw_dir, out_dir, sample_filter=None, parallel=False, force=False):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Not sorting the samples, so that each cpu gets them from random timestamps
    sample_paths = (roi.with_suffix("") for roi in raw_dir.glob("**/*.roi"))
    if sample_filter is not None:
        sample_paths = (path for path in sample_paths if path.name in sample_filter)
    if parallel:
        available_cores = os.cpu_count()
        log.debug(f"Extracting features in parallel with {available_cores} cores")
        with get_context("spawn").Pool(available_cores) as pool:
            samples_processed = pool.starmap(
                process_sample, [(path, out_dir, force) for path in sample_paths]
            )
    else:
        log.debug("Extracting features synchronously")
        samples_processed = []
        for path in sorted(sample_paths):
            samples_processed.append(process_sample(path, out_dir, force))
    return set(filter(None, samples_processed))


def process_sample(sample_path, out_dir, force=False):
    sample_path = Path(sample_path)
    sample = sample_path.name
    csv_path = (
        Path(out_dir)
        / ifcb.sample_to_datetime(sample).strftime("%Y/%m/%d")
        / f"{sample}.feat.csv"
    )
    if csv_path.is_file():
        if force:
            print(f"[WARNING] {csv_path.name} already exists, overwriting")
            # log.warn(f"{csv_path.name} already exists, overwriting")
        else:
            print(f"[WARNING] {csv_path.name} already exists, skipping")
            # log.warn(f"{csv_path.name} already exists, skipping")
            return sample_path.name
    # log.debug(f"Extracting features for {sample}")
    # print(f"Extracting features for {sample}")
    volume_ml, roi_features = sample_features(sample_path)
    features_to_csv(volume_ml, roi_features, csv_path)
    return sample_path.name


def sample_features(sample_path):
    root = Path(sample_path)
    adc = root.with_suffix(".adc")
    hdr = root.with_suffix(".hdr")
    roi = root.with_suffix(".roi")
    try:
        volume_ml = sample_volume(hdr)
    except Exception:
        log.exception(f"Unable to calculate sample volume for {root.name}")
        return
    roi_features = []
    for roi_id, roi_array in ifcb.raw_to_numpy(adc, roi):
        _, all_roi_features = compute_features(roi_array)
        all_roi_features = dict(all_roi_features)
        biovol_px = all_roi_features["Biovolume"]
        biovol_um3 = pixels_to_um3(biovol_px)
        biomass_ugl = biovolume_to_biomass(biovol_um3, volume_ml)
        roi_features.append((roi_id, biovol_px, biovol_um3, biomass_ugl))
    return (volume_ml, roi_features)


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
    sample_vol = ifcb_flowrate * ((run_time - inhibit_time) / 60)
    return sample_vol


def pixels_to_um3(pixels, micron_factor=3.5):
    return pixels / (micron_factor ** 3)


def biovolume_to_biomass(biovol_um3, volume_ml):
    return biovol_um3 / volume_ml / 1000


def features_to_csv(volume_ml, roi_features, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # csv_content = f"# {datetime.now().astimezone().isoformat()}\n"
    csv_content = "# version=3\n"
    csv_content += f"# volume_ml={volume_ml}\n"
    csv_content += "roi,biovolume_px,biovolume_um3,biomass_ugl\n"
    for roi_feat in roi_features:
        csv_content += ",".join(map(str, roi_feat)) + "\n"
    with open(csv_path, "w") as fh:
        fh.write(csv_content)
