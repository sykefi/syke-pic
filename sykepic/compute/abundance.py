"""Join predictions and features to count sample statistics"""

# Note that the abundance results are not concentrations, but counts of the sample. 
# For concentrations, analyzed volume needs to be taken into account in postprocessing step

from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sykepic.utils import logger
from sykepic.utils.ifcb import sample_to_datetime, filter_out_quality_flagged_samples
from .prediction import prediction_dataframe, threshold_dictionary

log = logger.get_logger("abundance")

def main(args):
    all_probs = sorted(Path(args.probabilities).glob("**/*.csv"))

    if args.exclusion_list:
        probs = filter_out_quality_flagged_samples(all_probs, Path(args.exclusion_list))
    else:
        probs = all_probs
        
    out_file = Path(args.out)
    if out_file.suffix != ".csv":
        raise ValueError("Make sure output file ends with .csv")
    if out_file.is_file():
        if not (args.append or args.force):
            raise FileExistsError(f"{args.out} exists, --append or --force not used")
    if args.feat:
        feats = sorted(Path(args.feat).glob("**/*.csv"))
        df = class_df(
            probs,
            feats,
            thresholds_file=args.thresholds,
            summary_feature=args.value_column,
            progress_bar=True,
        )
    else:
        print("no feats?")
    df = swell_df(df)
    df_to_csv(df, out_file, args.append)

def class_df(
    probs,
    feats,
    thresholds_file,
    summary_feature="biomass_ugl",
    progress_bar=False,
):
    # Read probability thresholds
    thresholds = threshold_dictionary(thresholds_file)
    df_rows = []

    # Ensure probabilities and features match
    if len(probs) != len(feats):
        iterator = (
            (p, f)
            for f in sorted(feats)
            for p in sorted(probs)
            if p.with_suffix("").stem == f.with_suffix("").stem
        )
    else:
        iterator = zip(sorted(probs), sorted(feats))

    # Add a progress bar optionally
    if progress_bar:
        iterator = tqdm(list(iterator), desc=f"Processing {len(feats)} samples")

    for prob_csv, feat_csv in iterator:
        # Check that CSVs match
        if prob_csv.with_suffix("").stem != feat_csv.with_suffix("").stem:
            raise ValueError(f"CSV mismatch: {prob_csv.name} & {feat_csv.name}")

        sample = prob_csv.with_suffix("").stem

        # Process sample and obtain class counts, total count and volume
        try:
            counts, total_count, sample_volume = process_sample(
                prob_csv, feat_csv, thresholds
            )
        except KeyError:
            log.exception(sample)
            continue

        # Convert class counts to concentrations per mL
        row = {
            class_name: counts.get(class_name, 0) / sample_volume
            for class_name in sorted(thresholds.keys())
        }

        # Add total count and analyzed volume
        row["Total_count"] = total_count
        row["Total_per_ml"] = total_count / sample_volume
        row["ml_analyzed"] = sample_volume

        df_rows.append(pd.Series(row, name=sample))

    # Create a collective dataframe
    df = pd.DataFrame(df_rows)
    df.index.name = "sample"

    # Ensure deterministic column order
    classes = sorted(thresholds.keys())
    df = df.reindex(
        columns=classes + ["Total_count", "Total_per_ml", "ml_analyzed"],
        fill_value=0,
    )

    df.fillna(0, inplace=True)
    return df

def swell_df(df):
    # Convert sample names to ISO 8601 timestamps (without microseconds)
    df.index = df.index.map(lambda x: sample_to_datetime(x, isoformat=True))
    df.index.name = "Time"
    # Replace underscores with spaces in class names
    df.columns = df.columns.str.replace("_", " ")
    return df

def df_to_csv(df, out_file, append=False):
    append = append and Path(out_file).is_file()
    mode = "a" if append else "w"
    df.to_csv(out_file, mode=mode, header=not append)

total_counts = []

def process_sample(prob_csv, feat_csv, thresholds):

    # Read sample volume from feature-file metadata
    sample_volume = None
    with open(feat_csv, "r") as f:
        for line in f:
            if not line.startswith("#"):
                break

            key, value = line[1:].strip().split("=", 1)
            if key == "volume_ml":
                sample_volume = value
                break

    if sample_volume is None or sample_volume == "None":
        raise ValueError(
            f"Valid volume_ml metadata not found in feature file: {feat_csv}"
        )

    sample_volume = float(sample_volume)

    if sample_volume <= 0:
        raise ValueError(
            f"volume_ml must be positive in feature file: {feat_csv}"
        )

    # Join prediction and feature data by ROI number
    df = pd.concat(
        [
            prediction_dataframe(prob_csv, thresholds),
            pd.read_csv(feat_csv, index_col=0, comment="#"),
        ],
        axis=1,
    )
    df.index.name = "roi"

    # Keep only ROIs with a valid, positive biovolume
    df = df[
        df["biovolume_um3"].notna()
        & (df["biovolume_um3"] > 0)
    ]

    # Count all valid ROIs, including unclassified ones
    total_counts.append(len(df))

    # Count only classified ROIs by class
    classified_df = df[df["classified"]]
    abundances = classified_df.groupby(
        "prediction", observed=False
    ).size()

    abundances.index.name = "class"

    # Convert counts to concentrations per mL
    abundances = abundances.astype(float) / sample_volume

    return abundances, len(df), sample_volume