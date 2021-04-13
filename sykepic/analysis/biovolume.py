from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from sykepic.utils.ifcb import sample_to_datetime
from sykepic.analysis.classification import read_predictions


def main(args):
    pred_dir = Path(args.predictions)
    preds = sorted(pred_dir.glob("**/*.csv"))
    vol_dir = Path(args.biovolumes)
    out_dir = Path(args.outdir)
    out_dir.mkdir(exist_ok=True, parents=True)

    files = []
    for p in preds:
        v = vol_dir / f"{p.name.split('_')[0]}_biovol.csv"
        if v.is_file():
            files.append((p, v))
        else:
            # No biovolume file, check that prediction file is not empty
            with open(p) as fh:
                num_lines = len(fh.readlines())
            if num_lines > 1:
                print(f"[ERROR] Missing biovolume data for '{p.name}'")

    # Read class names from first sample
    with open(files[0][0]) as fh:
        classes = fh.readline().strip().split(",")[1:]
    # Create a collective dataframe for all samples
    total = pd.DataFrame(columns=classes)
    total.index.name = "sample"

    # Read division instructions (optional)
    if args.division:
        divisions = read_divisions(args.division)

    print(f"[INFO] Processing {len(files)} samples")
    for p, v in tqdm(files):
        # Join prediction and volume data by index (roi number)
        df = pd.concat(
            [read_predictions(p, args.thresholds), pd.read_csv(v, index_col=0)], axis=1
        )
        df.index.name = "roi"
        # Discard unclassified images (below threshold)
        df = df[df["classified"]]
        # Make sure rows match (no empty biovolume values)
        assert not any(df.isna().any(axis=1)), f"NaNs in df for {p} and {v}"

        # Create intra-class divisions based on volume size
        if args.division:
            df = df.apply(make_division, axis=1, args=((divisions,)))

        # Group rows by prediction
        group = df.groupby("prediction")
        # Biovolumes by class
        volumes = group.sum().drop(df.columns[1:-1], axis=1)
        # Frequency by class
        counts = group.size()
        # Join biovolumes and frequencies
        gdf = pd.concat([volumes, counts], axis=1)
        gdf.columns = ["biovolume", "frequency"]
        gdf.index.name = "class"
        # Sort by highest biovolume
        gdf.sort_values("biovolume", ascending=False, inplace=True)
        # Drop classes without any predictions
        gdf.drop(gdf[gdf["frequency"] <= 0].index, inplace=True)
        # Append sample biovolumes as a row to total dataframe
        sample_vol = gdf["biovolume"]
        sample_vol.name = p.with_suffix("").name
        total = total.append(sample_vol)

        # Write sample results to file
        date = sample_to_datetime(p.name)
        day_path = date.strftime("%Y/%m/%d")
        out_file = out_dir / day_path / p.name
        out_file.parent.mkdir(exist_ok=True, parents=True)
        gdf.to_csv(out_file)

    total.to_csv(out_dir / "biovolume_summary_2018.csv", na_rep=0.0)
    print("[INFO] Done!")


def read_divisions(division_file):
    divisions = {}
    with open(division_file) as fh:
        for line in fh:
            line = line.strip().split()
            divisions[line[0]] = int(line[1])
    return divisions


def make_division(row, divisions):
    name = row["prediction"]
    if name in divisions:
        vol = row["Biovolume"]
        div = divisions[name]
        preposition = "under" if vol < div else "over"
        row["prediction"] = f"{name}_{preposition}_{div}"
    return row


if __name__ == "__main__":
    parser = ArgumentParser(description="Join predictions with biovolumes")
    parser.add_argument("predictions", help="Path to directory with predictions")
    parser.add_argument("biovolumes", help="Path to directory with biovolumes")
    parser.add_argument("thresholds", help="File with confidence thresholds")
    parser.add_argument("outdir", help="Directory where results are saved")
    parser.add_argument("-d", "--division", help="File with intra-class divisions")
    args = parser.parse_args()
    main(args)
