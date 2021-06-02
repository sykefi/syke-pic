"""Join predictions and features to make final classification results"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from sykepic.utils.ifcb import sample_to_datetime
from .prediction import prediction_dataframe, threshold_dictionary

FILE_SUFFIX = ".class"


def call(args):
    probs = sorted(Path(args.probabilities).glob("**/*.csv"))
    feats = sorted(Path(args.features).glob("**/*.csv"))
    print(f"[INFO] Processing {len(probs)} samples")
    main(probs, feats, args.out, args.thresholds, args.divisions)

    if args.summary:
        summary_csv = Path(args.out) / f"summary_{args.summary}.csv"
        classification_summary(args.out, out_file=summary_csv, column=args.summary)
        print(f"[INFO] Creating summary in {summary_csv}")
    print("[INFO] Done!")


def main(probs, feats, out_dir, prob_thresholds, feat_divisions=None):
    out_dir = Path(out_dir)
    # Read probability thresholds
    thresholds = threshold_dictionary(prob_thresholds)
    # Read feature divisions (optional)
    if feat_divisions:
        feat_divisions = read_divisions(feat_divisions)

    for prob_csv, feat_csv in tqdm(zip(probs, feats)):
        # Check that CSVs match
        if prob_csv.with_suffix("").stem != feat_csv.with_suffix("").stem:
            raise ValueError(f"CSV mismatch: {prob_csv.name} & {feat_csv.name}")
        sample = prob_csv.with_suffix("").stem

        # Join prob, feat and classifications in one df
        sample_df = process_sample(prob_csv, feat_csv, thresholds, feat_divisions)

        # Write sample results to file
        day_dir = sample_to_datetime(sample).strftime("%Y/%m/%d")
        sample_csv = out_dir / day_dir / f"{sample}{FILE_SUFFIX}.csv"
        sample_csv.parent.mkdir(exist_ok=True, parents=True)
        sample_df.to_csv(sample_csv)


def classification_summary(class_dir, out_file=None, column="biomass_ugl"):
    rows = []
    for csv in sorted(Path(class_dir).glob("**/*.csv")):
        # Extract requested column from sample
        sample_df = pd.read_csv(csv, index_col=0)
        sample_column = sample_df[column]
        sample_column.name = csv.with_suffix("").stem
        rows.append(sample_column)
    # Create a collective dataframe from all sample columns
    df = pd.DataFrame(rows)
    # Sort columns alphabetically by class name
    df = df.reindex(sorted(df.columns), axis=1)
    df.index.name = "sample"
    df.fillna(0, inplace=True)
    if out_file:
        df.to_csv(out_file)
    return df


def process_sample(prob_csv, feat_csv, prob_thresholds, feat_divisions=None):
    # Join prediction and volume data by index (roi number)
    df = pd.concat(
        [
            prediction_dataframe(prob_csv, prob_thresholds),
            pd.read_csv(feat_csv, index_col=0, comment="#"),
        ],
        axis=1,
    )
    df.index.name = "roi"
    # Discard unclassified images (below threshold)
    df = df[df["classified"]]
    # Make sure rows match (no empty biovolume values)
    assert not any(df.isna().any(axis=1))

    # Create intra-class divisions based on volume size
    if feat_divisions:
        df = df.apply(divide_row, axis=1, args=((feat_divisions,)))

    # Group rows by prediction
    group = df.groupby("prediction")
    # Join biovolumes and frequencies
    gdf = group.sum()[["classified", "biovolume_um3", "biomass_ugl"]]
    gdf.rename(columns={"classified": "frequency"}, inplace=True)
    gdf.index.name = "class"
    # Sort by highest biomass
    gdf.sort_values("biomass_ugl", ascending=False, inplace=True)
    # Drop classes without any predictions
    gdf.drop(gdf[gdf["frequency"] <= 0].index, inplace=True)

    return gdf


def read_divisions(division_file):
    divisions = {}
    with open(division_file) as fh:
        for line in fh:
            line = line.strip().split()
            key, *values = line
            divisions[key] = list(map(int, values))
    return divisions


def divide_row(row, divisions, column="biovolume_px"):
    row_name = row["prediction"]
    new_row_name = row_name
    if row_name in divisions:
        row_value = row[column]
        row_divisions = divisions[row_name]
        for i, division in enumerate(row_divisions):
            if row_value < division:
                if i == 0:
                    # prediction_under_9000
                    new_row_name = f"{row_name}_under_{division}"
                else:
                    # prediction_5000_9000
                    new_row_name = f"{row_name}_{row_divisions[i - 1]}_{division}"
            else:
                if i == len(row_divisions):
                    # prediction_9000_10000
                    new_row_name = f"{row_name}_{division}_{row_divisions[i + 1]}"
                else:
                    # prediction_over_9000
                    new_row_name = f"{row_name}_over_{division}"
    row["prediction"] = new_row_name
    return row
