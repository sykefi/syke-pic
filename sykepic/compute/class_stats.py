"""Join predictions and features to count sample statistics"""
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sykepic.utils import logger
from .prediction import prediction_dataframe, threshold_dictionary

log = logger.get_logger("class_stats")

def main(args):
    probs = sorted(Path(args.probabilities).glob("**/*.csv"))
    classes = args.classes
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
            classes,
            thresholds_file=args.thresholds,
            progress_bar=True,
        )
    else:
        print("no feats?")
    df_to_csv(df, out_file, args.append)

def class_df(
    probs,
    feats,
    classes,
    thresholds_file,
    progress_bar=False,
):
    # Read probability thresholds
    thresholds = threshold_dictionary(thresholds_file)
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

    for prob_csv, feat_csv in iterator:
        # Check that CSVs match
        if prob_csv.with_suffix("").stem != feat_csv.with_suffix("").stem:
            raise ValueError(f"CSV mismatch: {prob_csv.name} & {feat_csv.name}")
        sample = prob_csv.with_suffix("").stem

        # Join prob, feat and classifications in one df
        try:
            sample_df = process_sample(prob_csv, feat_csv, thresholds, sample, classes)
        except KeyError:
            log.exception(prob_csv.with_suffix("").stem)
            continue

        df_rows.append(sample_df)

    df = pd.concat(df_rows)
    return df

def df_to_csv(df, out_file, append=False):
    append = append and Path(out_file).is_file()
    mode = "a" if append else "w"
    df.to_csv(out_file, mode=mode, header=not append)

def process_sample(
    prob_csv, feat_csv, thresholds, sample, classes
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

    df_stats = df[["prediction", "classified", "biovolume_um3", "area", "major_axis_length", "minor_axis_length"]]
 
    # filament_labels = ["Dolichospermum-Anabaenopsis", "Dolichospermum-Anabaenopsis_coiled", "Nodularia_spumigena", "Nodularia_spumigena-coiled", "Aphanizomenon_flosaquae"]
    # filaments = df_stats[df_stats["prediction"].isin(filament_labels)]
    # filaments.loc[:, "prediction"] = "filamentous_cyanobacteria"
    # df_stats = pd.concat([df_stats, filaments])

    if classes:
        df_stats = df_stats[df_stats["prediction"].isin(classes)]

    stats = df_stats.groupby("prediction", observed=False).agg({"biovolume_um3": ['mean', 'median', 'min', 'max'], 
                                                               "area": ['mean', 'median', 'min', 'max'],
                                                               "major_axis_length": ['mean', 'median', 'min', 'max'],
                                                               "minor_axis_length": ['mean', 'median', 'min', 'max']})
    stats.columns = stats.columns.map('_'.join)
    stats = stats.dropna()
    stats.index.name = "class"

    stats.insert(0, "sample", sample)

    return stats