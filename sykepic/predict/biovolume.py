import logging
import os
import shutil
import subprocess
from pathlib import Path
from multiprocessing import Pool

from ifcb_features import compute_features

from sykepic.utils import ifcb

log = logging.getLogger("biovolume")


def with_python(raw_dir, out_dir, samples=None, parallel=False):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adc_files = sorted(raw_dir.glob("**/*.adc"))
    if samples:
        adc_files = [adc for adc in adc_files if adc.stem in samples]
    if parallel:
        available_cores = os.cpu_count()
        print(f"Extracting features in parallel with {available_cores} cores")
        with Pool(available_cores) as pool:
            pool.starmap(process_sample, [(adc, out_dir) for adc in adc_files])
    else:
        print(f"Extracting features synchronously")
        for adc in sorted(adc_files):
            process_sample(adc, out_dir)


def process_sample(adc_file, out_dir):
    print(f"Extracting features for {adc_file.stem}")
    sample_csv = (
        out_dir
        / ifcb.sample_to_datetime(adc_file.stem).strftime("%Y/%m/%d")
        / f"{adc_file.stem}.csv"
    )
    if sample_csv.is_file():
        print(f"{sample_csv} already exists, skipping")
        return
    sample_csv.parent.mkdir(parents=True, exist_ok=True)
    roi_file = adc_file.with_suffix(".roi")
    csv_content = "roi,area,biovolume\n"
    for roi_id, roi_array in ifcb.raw_to_numpy(adc_file, roi_file):
        _, roi_features = compute_features(roi_array)
        roi_features = dict(roi_features)
        # Create a CSV entry for current ROI and append it to the sample's output
        csv_content += (
            ",".join(
                map(str, [roi_id, roi_features["Area"], roi_features["Biovolume"]])
            )
            + "\n"
        )
    with open(sample_csv, "w") as fh:
        fh.write(csv_content)


def with_matlab(
    matlab_bin, samples, extensions, raw_orig, raw_symb, blobs, features, biovolumes
):
    raw_orig = Path(raw_orig)
    raw_symb = Path(raw_symb)
    blobs = Path(blobs)
    features = Path(features)
    biovolumes = Path(biovolumes)
    # IFCB-analysis throws error if trying to run in parallel with just one sample
    parallel = "true" if len(samples) > 1 else ""
    try:
        for sample in samples:
            # Create a path for sample that `ifcb-analysis` understands
            symlink_sample(sample, extensions, raw_orig, raw_symb)
        blob_command = (
            f"start_blob_batch_user_training('{raw_symb.resolve()}/', "
            f"'{blobs.resolve()}/', '{parallel}')"
        )
        feat_command = (
            f"start_feature_batch_user_training('{raw_symb.resolve()}/', "
            f"'{blobs.resolve()}/', '{features.resolve()}/', '{parallel}')"
        )
        log.debug("Extracting blobs")
        call_matlab(matlab_bin, blob_command, "Blob extraction")
        log.debug("Extracting features")
        call_matlab(matlab_bin, feat_command, "Feature extraction")
        samples_extracted = extract_biovolumes(features, biovolumes)
    except Exception:
        raise
    finally:
        # Always remove directory with symbolic raw data
        shutil.rmtree(raw_symb)
    return samples_extracted


def symlink_sample(sample, extensions, raw_orig, raw_symb):
    sample_dir = raw_orig / ifcb.sample_to_datetime(sample).strftime("%Y/%m/%d")
    sample_sym_dir = raw_symb / sample[:9]
    sample_sym_dir.mkdir(parents=True, exist_ok=True)
    for ext in extensions:
        (sample_sym_dir / (sample + ext)).symlink_to(sample_dir / (sample + ext))


def call_matlab(matlab_bin, command, name="Matlab"):
    res = subprocess.run(
        [
            matlab_bin,
            "-nodisplay",
            "-nosplash",
            "-nodesktop",
            "-r",
            f"try {command}; catch me, disp(me.message), exit(1); end; exit(0)",
        ],
        # stderr=sys.stderr, stdout=sys.stdout)
        capture_output=True,
    )
    # std_output = res.stdout[375:].decode().replace("\n", " ")
    std_output = res.stdout[375:].decode()
    if res.returncode != 0:
        log.error(f"{name} failed: {std_output}")
    else:
        log.debug(std_output)


def extract_biovolumes(features, biovolumes):
    samples_extracted = set()
    for feat_csv in features.glob("*.csv"):
        sample = feat_csv.name[:24]
        sample_date = ifcb.sample_to_datetime(sample)
        day_path = sample_date.strftime("%Y/%m/%d")
        sample_dir = biovolumes / day_path
        sample_dir.mkdir(exist_ok=True, parents=True)
        with open(sample_dir / f"{sample}.csv", "w") as fh:
            res = subprocess.run(
                ["cut", feat_csv, "-d", ",", "-f", "1-3"],
                stderr=subprocess.PIPE,
                stdout=fh,
                # capture_output=True,
            )
        if res.returncode != 0:
            # std_error = res.stderr.decode().replace("\n", " ")
            std_error = res.stderr.decode()
            log.error(f"Biovolume extraction failed for {feat_csv}: {std_error}")
        else:
            samples_extracted.add(sample)
            log.debug(f"Biovolume extracted for {sample}")
    return samples_extracted
