# import sys
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from sykepic.utils import files, logger

VERSION = 2
FILE_SUFFIX = ".feat"
log = logger.get_logger("feat")


def call(args):
    if args.raw:
        sample_paths = files.list_sample_paths(args.raw)
    else:
        sample_paths = [Path(path) for path in args.samples]

    # Don't process samples with a .roi-file over 1G
    filtered_sample_paths = []
    for sample_path in sample_paths:
        if sample_path.with_suffix(".roi").stat().st_size <= 1e9:
            filtered_sample_paths.append(sample_path)
        else:
            log.warn(f"{sample_path.name} is over 1G, skipping")

    main(args.matlab, filtered_sample_paths, args.out, args.parallel)


def main(bin, sample_paths, out_dir, parallel):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    mat_blob_dir = out_dir / "matlab" / "blob"
    mat_feat_dir = out_dir / "matlab" / "feat"
    # IFCB-analysis throws error if trying to run in parallel with just one sample
    parallel = "true" if parallel and len(sample_paths) > 1 else ""
    with (TemporaryDirectory(prefix="tmp-", dir=out_dir)) as sym_dir:
        sym_dir = Path(sym_dir)
        symlink_samples(sample_paths, sym_dir)
        blob_command = (
            "start_blob_batch_user_training("
            f"'{sym_dir}/', '{mat_blob_dir.resolve()}/', '{parallel}')"
        )
        feat_command = (
            "start_feature_batch_user_training("
            f"'{sym_dir}/', '{mat_blob_dir.resolve()}/', "
            f"'{mat_feat_dir.resolve()}/', '{parallel}')"
        )
        log.debug("Extracting blobs")
        call_matlab(bin, blob_command, "Blob extraction")
        log.debug("Extracting features")
        call_matlab(bin, feat_command, "Feature extraction")

    samples_processed = set()
    for sample_path in sorted(sample_paths):
        result = sample_features(sample_path, mat_feat_dir)
        if result is not None:
            volume, feat_df = result
            out_csv = files.sample_csv_path(sample_path, out_dir, FILE_SUFFIX)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(out_csv, "w") as fh:
                fh.write(f"# version={VERSION}\n# volume_ml={volume}\n")
                feat_df.to_csv(fh, index=False)
        samples_processed.add(sample_path.stem)
    return samples_processed


def symlink_samples(sample_paths, sym_dir):
    for sample_path in sample_paths:
        for raw_file in (
            sample_path.with_suffix(ext) for ext in (".adc", ".hdr", ".roi")
        ):
            sample_sym_dir = sym_dir / sample_path.stem[:9]
            sample_sym_dir.mkdir(exist_ok=True)
            (sample_sym_dir / raw_file.name).symlink_to(raw_file.resolve())


def call_matlab(bin, command, name="Matlab"):
    res = subprocess.run(
        [
            bin,
            "-nodisplay",
            "-nosplash",
            "-nodesktop",
            "-r",
            f"try {command}; catch me, disp(me.message), exit(1); end; exit(0)",
        ],
        capture_output=True,
        #     stderr=sys.stderr,
        #     stdout=sys.stdout,
    )
    # std_output = res.stdout[375:].decode().replace("\n", " ")
    std_output = res.stdout[375:].decode()
    if res.returncode != 0:
        log.error(f"{name} failed: {std_output}")
    else:
        log.debug(std_output)


def sample_features(sample_path, mat_feat_dir):
    try:
        feat_df = pd.read_csv(mat_feat_dir / f"{sample_path.stem}_fea_v{VERSION}.csv")
        volume_ml = sample_volume(sample_path.with_suffix(".hdr"))
    except FileNotFoundError:
        log.exception(f"Matlab features missing for {sample_path.name}")
        return None
    except Exception:
        log.exception(f"Unable to calculate volume for {sample_path.name}")
        return None
    biovolume_um3 = pixels_to_um3(feat_df["Biovolume"])
    feat_df["biovolume_um3"] = biovolume_um3
    feat_df["biomass_ugl"] = biovolume_to_biomass(biovolume_um3, volume_ml)
    feat_df.rename(
        columns={
            "roi_number": "roi",
            "Area": "area",
            "Biovolume": "biovolume_px",
            "MajorAxisLength": "major_axis_length",
            "MinorAxisLength": "minor_axis_length",
        },
        inplace=True,
    )
    columns_to_keep = [
        "roi",
        "biovolume_px",
        "biovolume_um3",
        "biomass_ugl",
        "area",
        "major_axis_length",
        "minor_axis_length",
    ]
    return (
        volume_ml,
        feat_df[columns_to_keep],
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
    if sample_vol < 0:
        raise ValueError(f"Sample volume is {sample_vol}")
    return sample_vol


def pixels_to_um3(pixels, micron_factor=3.5):
    return pixels / (micron_factor**3)


def biovolume_to_biomass(biovol_um3, volume_ml):
    try:
        return biovol_um3 / volume_ml / 1000
    except ZeroDivisionError:
        return 0
