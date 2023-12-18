"""Join predictions and features to count sample statistics"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sykepic.utils import logger
from sykepic.utils.ifcb import sample_to_datetime
from .prediction import prediction_dataframe, threshold_dictionary

log = logger.get_logger("abundance")

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
            sample_df = process_sample(prob_csv, feat_csv, thresholds)
        except KeyError:
            log.exception(prob_csv.with_suffix("").stem)
            continue
        # Select specific feature to summarize
        sample_column = sample_df[summary_feature]
        sample_column.name = sample
        df_rows.append(sample_column)

    # Create a collective dataframe for all samples
    # Make sure column names are deterministic
    classes = thresholds.keys()
    classes = sorted(classes)
    classes.append("Total")
    df = pd.DataFrame(df_rows, columns=classes)
    df["Total"] = total_counts
    df.index.name = "sample"
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
    df = df.astype(int)
    append = append and Path(out_file).is_file()
    mode = "a" if append else "w"
    df.to_csv(out_file, mode=mode, header=not append)

total_counts = []
def process_sample(
    prob_csv, feat_csv, thresholds
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

    # laske kuvien kokonaismäärä
    global total_counts
    total_counts.append(len(df. index))

    # Drop unclassified rows (below threshold)
    df = df[df["classified"]]

    abundances = df.groupby("prediction", observed=False).count()
    abundances.index.name = "class"
    
    return abundances