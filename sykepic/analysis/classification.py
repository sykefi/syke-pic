import datetime
from pathlib import Path

import pandas as pd


def insert_threshold(df, threshold, filter=False):
    if not isinstance(threshold, (float, str, Path)):
        raise ValueError('Threshold can be either float, list, or Path')
    if isinstance(threshold, float):
        # Add default threshold to each row
        df.insert(2, 'threshold', threshold)
    else:
        # Read threshold values from a whitespace separated file
        conf_df = pd.read_csv(threshold, header=None,
                              index_col=0, delim_whitespace=True)
        classes = df.prediction.cat.categories
        # If default value is not given,
        # make sure all classes are given a threshold value
        if (conf_df.index[0] != 'default' or
                pd.isnull(conf_df.loc['default'].item())):
            assert (classes.isin(conf_df.index).all() and
                    conf_df.notnull().values.all()), \
                'Make sure to provide default confidence threshold'
        # Add any missing classes
        conf_df = conf_df.reindex(conf_df.index.append(
            classes[~classes.isin(conf_df.index)]))
        # Fill missing values with default threshold
        conf_df = conf_df.fillna(conf_df.iloc[0])
        # Add class threshold to each row of main df
        df.insert(2, 'threshold', conf_df.loc[df['prediction'], 1].tolist())


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
    preds, status = zip(*df.apply(make_classification, axis=1,
                                  args=(thresholds,)))
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
    insert_classifications(df, thresholds)
    return df
