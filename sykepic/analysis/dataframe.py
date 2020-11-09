"""Some functions to help analyse classification predictions.

Only frequency_df() and filter_df() are intended for general use,
the rest are mainly helper functions.
"""

import datetime
from pathlib import Path

import pandas as pd

from sykepic.predict import ifcb


def frequency_df(predict_dir, conf_thres=None, start=None, end=None,
                 hour_window=None, date_format='%Y-%m-%d %H:%M'):
    """Create a pandas.DataFrame from predictions.

    Predictions are expected to be in csv-files.

    Parameters
    ----------
    predict_dir : str, Path
        Root directory of prediction csv-files.
    conf_thres : float, str, Path
        Confidence threshold, either as a single float or
        a path to a file with class names and their thresholds.
        Threholds are inclusive, meaning that a prediction is accepted
        if it's confidence value is above or equal to threshold.
    start : str
        Start date to filter csv-files by. Must follow the datetime
        format set in `date_format` e.g. '2018-07-03 00:00'.
    end : str
        End date to filter csv-files by. Must follow the datetime
        format set in `date_format` e.g. '2018-07-31 23:59'.
    hour_window : str
        Filter only those csv-files that match this hour window.
        The required string format is: `%H:%M-%H:%M`
        e.g. for samples taken around mid-day, you could specify
        hour_window='11:30-12:30'
    date_format : str
        The format value of `start` and `end`, both of which will be
        converted from string to datetime objects.

    Returns
    -------
    pandas.DataFrame
        This dataframe has class names as columns (predictions),
        and datetimes as the row indeces (samples). Each cell is thus
        the frequency of a certain class for each collected sample.
        Empty cells are filled with numpy.nan values.
    """

    csv_date_list = filter_csv_by_date(
        predict_dir, start, end, hour_window, date_format)
    if not csv_date_list:
        print('[INFO] No sample predictions match this time restraint.')
        return
    print(f'[INFO] Using predictions from {len(csv_date_list)} samples')
    df = csv_to_df(csv_date_list)
    # TODO: Fix this below. maybe: df[df['confidence'] >= df['threshold']]
    # if conf_thres:
    #     df = read_thresholds(df, conf_thres, filter=True)
    df = group_predictions(df)
    return df


def filter_df(freq_df, prediction=None, top=None):
    """Filter frequency df's columns

    Parameters
    ----------
    freq_df : pandas.DataFrame
        Only a dataframe returned by `frequency_df()` should be used.
    prediction : str, list
        Allow only the class predictions that match this string or
        list of strings.
    top : int
        Allow only the `top` most frequent classes.

    Returns
    -------
    pandas.DataFrame
        Same rows as input, but a filtered subset of columns.
    """

    if prediction:
        freq_df = freq_df.loc[:, prediction]
    if top:
        freq_df = freq_df[freq_df.sum().nlargest(top).index]
    return freq_df


def filter_csv_by_date(predict_dir, start=None, end=None, hour_window=None,
                       date_format='%Y-%m-%d %H:%M'):
    predict_dir = Path(predict_dir)
    if not predict_dir.is_dir():
        raise FileNotFoundError(f"'{predict_dir}' is not a directory")
    start = datetime.datetime.strptime(start, date_format) if start else None
    end = datetime.datetime.strptime(end, date_format) if end else None
    if hour_window:
        time_format = '%H:%M'
        hour_start, hour_end = hour_window.split('-')
        hour_start = datetime.datetime.strptime(
            hour_start.strip(), time_format)
        hour_end = datetime.datetime.strptime(hour_end.strip(), time_format)
    csv_date_list = []
    # for csv in list_files(predict_dir, '.csv'):
        # list_files() doesn't sort results, so must go through all files
    for csv in sorted(predict_dir.glob('**/*.csv')):
        date = ifcb.sample_to_datetime(csv.with_suffix('').name)
        if (start and date < start) or (end and date > end):
            continue
        # If using hour_window check that sample's time is in this range
        if (hour_window and
                not (hour_start.time() <= date.time() <= hour_end.time())):
            continue
        csv_date_list.append((csv, date))
    return csv_date_list


def csv_to_df(csv_date_list):
    df_list = []
    for csv, date in csv_date_list:
        # Read sample predictions to df, without 'roi'
        sample_df = pd.read_csv(csv).drop('roi', axis=1)
        # Insert 'prediction' and 'confidence' columns.
        insert_prediction(sample_df)
        # Drop all class probability columns, since they aren't needed
        sample_df.drop(sample_df.columns[2:], axis=1, inplace=True)
        # Insert 'timestamp' column
        sample_df.insert(0, 'timestamp', date)
        df_list.append(sample_df)
    # Join all sample predictions into one df
    df = pd.concat(df_list)
    # Convert 'prediction' column to categorical
    df['prediction'] = df['prediction'].astype('category')
    return df


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
        df['prediction'] = df['prediction'].astype('category')
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


def group_predictions(df):
    df = df.groupby('timestamp').prediction.value_counts().unstack()
    df.columns.name = ''
    df.index.name = ''
    return df


def insert_prediction(df):
    """This function modifies `df` in place"""
    # Prediction is the column with max softmax value
    df.insert(0, 'prediction', df.idxmax(axis=1))
    # Confidence is said softmax value
    df.insert(1, 'confidence', df.iloc[:, 1:].max(axis=1))


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
    # Insert 'prediction' and 'confidence' columns to dataframe
    insert_prediction(df)
    # Insert 'threshold' column to df (default threholds is 0.0)
    insert_threshold(df, thresholds)
    return df