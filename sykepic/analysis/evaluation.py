from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .dataframe import read_predictions, threshold_dictionary


def parse_evaluations(evaluations, pred_dir, thresholds=None,
                      empty='unclassifiable', threshold_search=False,
                      search_precision=0.01):
    eval_df, samples = read_evaluations(evaluations)
    predictions = []
    for sample in samples:
        try:
            predictions.append(next(Path(pred_dir).rglob(f'{sample}.csv')))
        except StopIteration:
            print(f'[ERROR] Cannot find prediction files for {sample}')
            raise
    if threshold_search:
        # Set initial threshold value
        thresholds = 0.0
    elif not thresholds:
        raise ValueError('Thresholds not provided')
    if isinstance(thresholds, (str, Path)):
        thresholds = threshold_dictionary(thresholds)
    pred_df = read_predictions(predictions, thresholds)
    result_df = results_as_df(eval_df, pred_df, empty, thresholds, threshold_search,
                              np.arange(0, 1+search_precision, search_precision))
    if threshold_search:
        # No specificity without 'all' class
        result_df.drop('specificity', axis=1, inplace=True)
    return result_df


def read_evaluations(evaluations):
    if isinstance(evaluations, (str, Path)):
        evaluations = Path(evaluations)
        if evaluations.is_dir():
            evaluations = list(evaluations.glob('*.select.csv'))
            if evaluations:
                print('[INFO] Evaluations are from these files:')
                print('\t'+'\n\t'.join([str(f) for f in evaluations]))
            else:
                raise FileNotFoundError('[ERROR] No evaluation files found')
        else:
            evaluations = [evaluations]
    df_list = []
    samples = []
    for file in evaluations:
        sample = Path(file).with_suffix('').with_suffix('').name
        samples.append(sample)
        df = pd.read_csv(file, header=None, names=['roi', 'actual'])
        df.insert(0, 'sample', sample)
        df.set_index(['sample', 'roi'], inplace=True)
        df_list.append(df)
    df = pd.concat(df_list)
    return df, samples


def results_as_df(eval_df, pred_df, empty, thres_dict,
                  threshold_search, search_range):
    result_dict = {}
    for idx, row in eval_df.iterrows():
        prediction = pred_df.loc[idx, 'prediction']
        confidence = pred_df.loc[idx, prediction]
        actual = row['actual']
        if threshold_search:
            threshold_values = search_range
        else:
            # Only one threshold per class, but iterate over it all the same
            threshold_values = [thres_dict[prediction]]
        for threshold in threshold_values:
            # Prediction is the class with largest softmax,
            # since every threshold is initially set to 0.0.
            # This means, that some these predictions differ from what
            # they would be with real thresholds, i.e., possibly more FNs.
            if confidence < threshold:
                prediction = empty
            for name, result in classification_result(prediction, actual, empty):
                result_dict.setdefault(name, {})
                result_dict[name].setdefault(threshold,
                                             {'tp': 0, 'tn': 0,
                                              'fp': 0, 'fn': 0})[result] += 1
    # Shape result_dict to have multi-index
    result_dict = {(name, thres): results
                   for name, thres_results in result_dict.items()
                   for thres, results in thres_results.items()}
    result_df = pd.DataFrame.from_dict(
        result_dict, orient='index').sort_index()

    if not threshold_search:
        # Join multiindex by class (drop threshold index)
        result_df = result_df.groupby(level=0).sum()
        # Make row 'all' for summed results
        if empty in result_df.index:
            tn = result_df.loc[empty, 'tp'].sum()
            result_df.drop(index=empty, inplace=True)
            result_df.loc['all'] = [result_df.tp.sum(), tn,
                                    result_df.fp.sum(), result_df.fn.sum()]
            # Move 'all' as the first row
            result_df = pd.concat(
                [result_df.loc[['all'], :], result_df.drop('all')])
        # Add threshold column
        result_df.insert(0, 'threshold', result_df.apply(
            lambda row: thres_dict.get(row.name, np.nan), axis=1))
    elif empty in result_df.index:
        result_df.drop(index=empty, level=0, inplace=True)
    # Calculate classifications scores for each row
    score_df = result_df.apply(lambda row: classification_scores(
                               row.tp, row.tn, row.fp, row.fn),
                               axis=1, result_type='expand')
    score_df.columns = ('precision', 'recall', 'F1', 'support', 'specificity')
    score_df['support'] = score_df['support'].astype(int)
    return pd.concat((result_df, score_df), axis=1)


def classification_result(predicted, actual, empty):
    if predicted == actual:
        # True positive (nice!)
        # Also True negatives are returned here, i.e. empty, empty
        # So, TN is the same as TP for the empty class
        return ((predicted, 'tp'),)
    elif actual == empty:
        # False positive for predicted class (boo!)
        return ((predicted, 'fp'),)
    elif predicted == empty:
        # False negative for actual class (boo!)
        return ((actual, 'fn'),)
    else:
        # Predicted wrong class (oh no, double trouble!)
        # False postive for predicted class and
        # False negative for actual class
        return ((predicted, 'fp'), (actual, 'fn'))


def classification_scores(tp, tn, fp, fn):
    if tp > 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1 = F_score(precision, recall, beta=1)
    else:
        precision = 0
        recall = 0
        F1 = 0
    # Note that support might be bigger than the actual number of
    # labeled ROIs. This is because a wrongly predicted class will produce
    # two errors (fp and fn) which are added here.
    # So the same ROI will contribute twice to the final sum.
    # I don't think this is a bad thing, since both of these values
    # affect the F1-score, and thus should be counted separately.
    support = tp + fp + fn
    if tn:
        specificity = tn / (tn + fp)
        support += tn
    else:
        specificity = np.nan
    return (precision, recall, F1, support, specificity)


def F_score(precision, recall, beta=1):
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def best_thresholds(resuls_df, criteria='F1'):
    g0 = resuls_df.groupby(level=0)
    best_idx = g0.apply(lambda name: name[criteria].idxmax())
    return resuls_df.loc[best_idx]
