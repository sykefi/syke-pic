from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .feature_matlab import pixels_to_um3


def call(args):
    feats = sorted(Path(args.features).glob("**/*.csv"))
    out_file = Path(args.out)
    if out_file.suffix != ".csv":
        raise ValueError("Make sure output file ends with .csv")
    if out_file.is_file():
        if not (args.append or args.force):
            raise FileExistsError(f"{out_file} exists, --append or --force not used")
    value_column = args.value_column if args.value_column else args.size_column
    main(
        feats=feats,
        groups_file=args.groups,
        size_column=args.size_column,
        value_column=value_column,
        out_csv=args.out,
        append=args.append,
        verbose=not args.quiet,
        px_to_um3=args.pixels_to_um3,
    )


def main(
    feats, groups_file, size_column, value_column, out_csv, append, verbose=False, px_to_um3=False
):
    groups = read_size_groups(groups_file)
    df = size_df(feats, groups, size_column, value_column, verbose, px_to_um3)
    df_to_csv(df, out_csv, append)


def read_size_groups(path):
    with open(path) as fh:
        lines = (line.strip().split() for line in fh.readlines())
        groups = {name: float(size) for name, size in lines}
    groups = sorted(groups.items(), key=lambda x: x[1], reverse=True)
    return groups


def size_df(feats, groups, size_column, value_column, verbose=False, px_to_um3=False):
    rows = []
    if verbose:
        feats = tqdm(feats, desc=f"Processing {len(feats)} samples")
    for csv in feats:
        sample = csv.with_suffix("").stem
        if sample.endswith("_biovol"):
            sample = sample.split("_")[0]
        result_dict = process_sample(csv, groups, size_column, value_column, px_to_um3)
        result_dict["sample"] = sample
        rows.append(result_dict)
    df = pd.DataFrame(rows)
    df.set_index("sample", inplace=True)
    # Reverse column order, so that smallest group is first
    df = df.iloc[:, ::-1]
    # Insert total biomvolume column
    df.insert(0, "total", df.sum(axis=1))
    return df


def process_sample(csv, groups, size_column, value_column, px_to_um3=False):
    result = {name: 0.0 for name, _ in groups}
    with open(csv) as fh:
        for line in fh:
            if not line.startswith("#"):
                header = line.strip().split(",")
                break
        size_column_id = None
        value_column_id = None
        for i, name in enumerate(header):
            if name == size_column:
                size_column_id = i
            if name == value_column:
                value_column_id = i
        if size_column_id is None:
            raise ValueError(f"Column '{size_column}' not found in header")
        if value_column_id is None:
            raise ValueError(f"Column '{value_column}' not found in header")
        for line in fh:
            row = line.strip().split(",")
            size = float(row[size_column_id])
            value = float(row[value_column_id])
            if px_to_um3:
                size = pixels_to_um3(size)
            division = get_group(size, groups)
            result[division] += value
    return result


def get_group(size, groups):
    for name, lower_bound in groups:
        if size >= lower_bound:
            return name
    # Return the biggest
    return groups[-1][0]


def df_to_csv(df, out_file, append=False):
    append = append and Path(out_file).is_file()
    mode = "a" if append else "w"
    df.to_csv(out_file, mode=mode, header=not append, na_rep=0.0)
