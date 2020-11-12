from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .dataframe import read_predictions


def parse_evaluations(evaluations, predictions, thresholds, out_file,
                      empty='unclassifiable', threshold_search=False,
                      search_precision=0.01):
    eval_df, samples = read_evaluations(evaluations)
    try:
        predictions = [next(Path(predictions).rglob(f'{sample}.csv'))
                       for sample in samples]
    except StopIteration:
        print('[ERROR] Cannot find prediction files.')
        raise
    pred_df = read_predictions(predictions, thresholds)

    result_df = results_as_df(eval_df, pred_df, empty, threshold_search,
                              np.arange(0, 1+search_precision, search_precision))
    return result_df


def read_evaluations(evaluations):
    if isinstance(evaluations, (str, Path)):
        evaluations = Path(evaluations)
        if evaluations.is_dir():
            evaluations = list(evaluations.rglob('*.select.csv'))
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


def results_as_df(eval_df, pred_df, empty, threshold_search, search_range):
    result_dict = {}
    for idx, row in eval_df.iterrows():
        prediction, confidence, threshold = pred_df.loc[idx, [
            'prediction', 'confidence', 'threshold']]
        actual = row['actual']
        if threshold_search:
            thresholds = search_range
        else:
            # Only one threshold per class, but iterate over it all the same
            thresholds = [threshold]
        for threshold in thresholds:
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
        # Drop threshold from multiindex
        result_df.index = result_df.index.droplevel(1)
        # Make row ('all') for joined results
        if empty in result_df.index:
            tn = result_df.loc['unclassifiable', 'tp'].sum()
            result_df.drop(index=empty, inplace=True)
            result_df.loc['all'] = [result_df.tp.sum(), tn,
                                    result_df.fp.sum(), result_df.fn.sum()]
    elif empty in result_df.index:
        result_df.drop(index=empty, level=0, inplace=True)
    score_df = result_df.apply(lambda row: classification_scores(
                               row.tp, row.tn, row.fp, row.fn),
                               axis=1, result_type='expand')
    score_df.columns = ('precision', 'recall', 'F1', 'support', 'specificity')
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
        specificity = 0
    return (precision, recall, F1, support, specificity)


def F_score(precision, recall, beta=1):
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
