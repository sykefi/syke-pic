"""Join predictions and features to count sample statistics"""
"""Currently only works if there're data from at least 2 months!"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sykepic.utils import logger
from .prediction import prediction_dataframe, threshold_dictionary

log = logger.get_logger("features_per_prediction")

def main(args):
    probs = sorted(Path(args.probabilities).glob("**/*.csv"))
    out_file = Path(args.out)
    if out_file.suffix != ".csv":
        raise ValueError("Make sure output file ends with .csv")
    if out_file.is_file():
        if not (args.append or args.force):
            raise FileExistsError(f"{args.out} exists, --append or --force not used")
    if args.feat:
        feats = sorted(Path(args.feat).glob("**/*.csv"))
        df_list = class_df(
            probs,
            feats,
            thresholds_file=args.thresholds,
            progress_bar=True,
        )
    else:
        print("no feats?")

    identifier = 1
    for df in df_list:
        path = out_file
        out_file = path.with_name(path.stem + str(identifier) + path.suffix)
        identifier = int(identifier) + 1
        df_to_csv(df, out_file, args.append)

def class_df(
    probs,
    feats,
    thresholds_file,
    progress_bar=False,
):
    # Read probability thresholds
    thresholds = threshold_dictionary(thresholds_file)
    current_df_sample_month = ""
    df_list = list()
    
    df_rows = []
    # Ensure probs and feats match
    if len(probs) != len(feats):
        iterator = (
            (p, f)
            for f in sorted(feats)
            for p in sorted(probs)
            if p.with_suffix("").stem == f.with_suffix("").stem
        )
    else:
        iterator = zip(sorted(probs), sorted(feats))
    # Add a tqdm progress bar optionally
    if progress_bar:
        iterator = tqdm(list(iterator), desc=f"Processing {len(feats)} samples")
    
    iterator_length = len(feats)
    iteration = 0
    processing_errors = 0

    for prob_csv, feat_csv in iterator:
        # Check that CSVs match
        if prob_csv.with_suffix("").stem != feat_csv.with_suffix("").stem:
            raise ValueError(f"CSV mismatch: {prob_csv.name} & {feat_csv.name}")
        sample = prob_csv.with_suffix("").stem
        sample_month = sample[5:7]

        if current_df_sample_month == "":
            current_df_sample_month = sample_month

        # Join prob, feat and classifications in one df
        try:
            sample_df = process_sample(prob_csv, feat_csv, thresholds, sample)
        except KeyError:
            log.exception(prob_csv.with_suffix("").stem)
            processing_errors += 1
            continue  

        if sample_month == current_df_sample_month and (iteration + processing_errors) < iterator_length-1:
            df_rows.append(sample_df)
            iteration += 1
        elif sample_month == current_df_sample_month and (iteration + processing_errors) == iterator_length-1:
            df_rows.append(sample_df)
            df_sample_month = pd.concat(df_rows)
            df_list.append(df_sample_month)
            iteration += 1
        else:
            # append df_rows to df_list
            # then start new df_rows with sample data
            df_sample_month = pd.concat(df_rows)
            df_list.append(df_sample_month)
            df_rows = []
            df_rows.append(sample_df)
            current_df_sample_month = sample_month
            iteration += 1
    return df_list

def df_to_csv(df, out_file, append=False):
    append = append and Path(out_file).is_file()
    mode = "a" if append else "w"
    df.to_csv(out_file, mode=mode, header=not append)

def process_sample(
    prob_csv, feat_csv, thresholds, sample
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

    # Drop unclassified rows (below threshold)
    df = df[df["classified"]]

    df_stats = df[["prediction", "biovolume_um3", "biomass_ugl", "area", "major_axis_length", "minor_axis_length"]]

    filament_labels = ["Dolichospermum-Anabaenopsis", "Dolichospermum-Anabaenopsis_coiled", "Nodularia_spumigena", "Nodularia_spumigena-coiled", "Aphanizomenon_flosaquae"]
    filaments = df_stats[df_stats["prediction"].isin(filament_labels)]

    filaments.insert(0, "sample", sample)
    return filaments