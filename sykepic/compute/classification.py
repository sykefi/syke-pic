"""Join predictions and features to make final classification results"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from sykepic.utils.ifcb import sample_to_datetime
from .prediction import prediction_dataframe, threshold_dictionary


def main(args):
    probs = sorted(Path(args.probabilities).glob("**/*.csv"))
    feats = sorted(Path(args.features).glob("**/*.csv"))
    out_file = Path(args.out)
    if out_file.suffix != ".csv":
        raise ValueError("Make sure output file ends with .csv")
    if out_file.is_file():
        if not (args.append or args.force):
            raise FileExistsError(f"{args.out} exists, --append or --force not used")
    df = class_df(
        probs,
        feats,
        thresholds_file=args.thresholds,
        divisions_file=args.divisions,
        summary_feature=args.summarize,
        progress_bar=True,
    )
    df = swell_df(df)
    df_to_csv(df, out_file, args.append)


def class_df(
    probs,
    feats,
    thresholds_file,
    divisions_file=None,
    summary_feature="biomass_ugl",
    progress_bar=False,
):

    # Read probability thresholds
    thresholds = threshold_dictionary(thresholds_file)
    # Read feature divisions (optional)
    divisions = read_divisions(divisions_file) if divisions_file else None

    df_rows = []
    iterator = zip(probs, feats)
    if progress_bar:
        iterator = tqdm(list(iterator), desc=f"Processing {len(probs)} samples")
    for prob_csv, feat_csv in iterator:
        # Check that CSVs match
        if prob_csv.with_suffix("").stem != feat_csv.with_suffix("").stem:
            raise ValueError(f"CSV mismatch: {prob_csv.name} & {feat_csv.name}")
        sample = prob_csv.with_suffix("").stem
        # Join prob, feat and classifications in one df
        sample_df = process_sample(prob_csv, feat_csv, thresholds, divisions)
        # Select specific feature to summarize
        sample_column = sample_df[summary_feature]
        sample_column.name = sample
        df_rows.append(sample_column)

    # Create a collective dataframe for all samples
    # Make sure column names are deterministic
    classes = thresholds.keys()
    if divisions:
        division_names = names_of_divisions(divisions)
        classes = set(classes).union(division_names).difference(divisions.keys())
    df = pd.DataFrame(df_rows, columns=sorted(classes))
    df.index.name = "sample"
    df.fillna(0, inplace=True)
    return df


def swell_df(df):
    # Convert sample names to ISO 8601 (without microseconds)
    df.index = df.index.map(sample_to_datetime).map(
        lambda x: x.tz_localize("UTC").replace(microsecond=0).isoformat()
    )
    df.index.name = "ISO_8601"
    # Sum Dolichospermum-Anabaenopsis variants together
    df["Dolichospermum-Anabaenopsis"] = df[
        ["Dolichospermum-Anabaenopsis", "Dolichospermum-Anabaenopsis-coiled"]
    ].sum(axis=1)
    df.drop("Dolichospermum-Anabaenopsis-coiled", axis=1, inplace=True)
    # Sum all together for total biomass
    df.insert(0, "total", df.sum(axis=1))
    # df["total"] = df.sum(axis=1)
    return df


def df_to_csv(df, out_file, append=False):
    mode = "a" if append and Path(out_file).is_file() else "w"
    header = not append
    df.to_csv(out_file, mode=mode, header=header)


def process_sample(
    prob_csv, feat_csv, thresholds, divisions=None, division_column="biovolume_px"
):
    # Join prediction and volume data by index (roi number)
    df = pd.concat(
        [
            prediction_dataframe(prob_csv, thresholds),
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
    if divisions:
        df = df.apply(divide_row, axis=1, args=((divisions, division_column)))

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


def divide_row(row, divisions, column):
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


def names_of_divisions(divisions):
    new_names = []
    for key, values in divisions.items():
        values = sorted(values)
        new_names.append(f"{key}_under_{values[0]}")
        new_names.append(f"{key}_over_{values[-1]}")
        for i in range(len(values) - 1):
            new_names.append(f"{key}_{values[i]}_{values[i + 1]}")
    return new_names
