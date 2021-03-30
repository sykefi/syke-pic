from pathlib import Path

import pandas as pd


def threshold_dictionary(thresholds, default=None):
    thres_dict = {}
    with open(thresholds) as fh:
        for line in fh:
            line = line.strip().split()
            key = line[0]
            if len(line) > 1:
                value = float(line[1])
            elif default:
                value = float(default)
            else:
                raise ValueError(f'Missing threshold for {key}, '
                                 'and no default value specified.')
            thres_dict[key] = value
    return thres_dict


def make_classification(row, thresholds):
    """Makes classification based on thresholds

    Selects the first class with the highest confidence that
    is also above the class's threshold.

    Returns:
        Tuple of (name, classified), where classified is True
        if confidence is above class threshold
    """
    if isinstance(thresholds, (int, float)):
        name = row.idxmax()
        return (name, row[name] > thresholds)
    # Generator for all classes that have condidence above
    # their specific threshold (sorted by confidence, descending)
    above_threshold = ((name, True) for name, confidence in
                       row.sort_values(ascending=False).items()
                       if confidence >= thresholds[name])
    try:
        # Return the first value from generator (highest softmax)
        return next(above_threshold)
    except StopIteration:
        return (row.idxmax(), False)


def insert_classifications(df, thresholds):
    """This function modifies `df` in place"""
    preds, status = zip(
        *df.apply(make_classification, axis=1, args=(thresholds,)))
    df.insert(0, 'prediction', preds)
    df['prediction'] = df['prediction'].astype('category')
    df.insert(1, 'classified', status)


def read_predictions(predictions, thresholds=0.0):
    if isinstance(predictions, list):
        # Need to join multiple csv-files as one df
        df_list = []
        for csv in predictions:
            df = pd.read_csv(csv)
            # Create multi-index from sample name and roi number
            sample = Path(csv).with_suffix('').name
            df.insert(0, 'sample', sample)
            df.set_index(['sample', 'roi'], inplace=True)
            df_list.append(df)
        df = pd.concat(df_list)
    elif isinstance(predictions, (str, Path)):
        df = pd.read_csv(predictions, index_col=0)
    else:
        raise ValueError('Check predictions path')
    if isinstance(thresholds, (str, Path)):
        thresholds = threshold_dictionary(thresholds)
    # Insert 'prediction' and 'classified' columns to dataframe
    if not df.empty:
        insert_classifications(df, thresholds)
    return df
