from collections import Counter
from pathlib import Path

import pandas as pd

from .dataframe import read_predictions


def parse_evaluations(evaluations, predictions, thresholds, out_file,
                      empty='unclassifiable'):
    eval_df, samples = read_evaluations(evaluations)
    try:
        predictions = [next(Path(predictions).rglob(f'{sample}.csv'))
                       for sample in samples]
    except StopIteration:
        print('[ERROR] Cannot find prediction files.')
        raise
    pred_df = read_predictions(predictions, thresholds)

    class_results = {}
    for idx, row in eval_df.iterrows():
        prediction, confidence, threshold = pred_df.loc[idx,
            ['prediction', 'confidence', 'threshold']]
        if confidence < threshold:
            prediction = empty
        actual = row['actual']
        for name, result in classification_result(prediction, actual, empty):
            class_results.setdefault(
                name, {'tp': 0, 'fp': 0, 'fn': 0})[result] += 1
    # Total TN = Empty TP
    if empty in class_results:
        total = Counter({'tn': class_results[empty]['tp']})
        # Empty class is not needed anymore
        del class_results[empty]
    else:
        total = Counter({'tn': 0})
    # Add class results to total
    for result in class_results.values():
        total.update(result)
    # Output scores to file
    with open(out_file, 'w') as fh:
        fh.write(f'class,precision,recall,f1-score,support,tn-rate\n')
        prec, rec, F1, sup, spec = classification_scores(
                total['tp'], total['fp'], total['fn'], tn=total['tn'])
        fh.write(f'total,{prec:.2f},{rec:.2f},{F1:.2f},{sup},{spec:.2f}\n')
        for name, values in sorted(class_results.items()):
            prec, rec, F1, sup, spec = classification_scores(
                values['tp'], values['fp'], values['fn'])
            fh.write(f'{name},{prec:.2f},{rec:.2f},{F1:.2f},{sup},\n')


def read_evaluations(evaluations):
    if isinstance(evaluations, (str, Path)):
        evaluations = Path(evaluations)
        if evaluations.is_dir():
            evaluations = list(evaluations.rglob('*.select.csv'))
        else:
            evaluations = [evaluations]
    df_list = []
    samples = []
    for file in evaluations[:1]:
        sample = Path(file).with_suffix('').with_suffix('').name
        samples.append(sample)
        df = pd.read_csv(file, header=None, names=['roi', 'actual'])
        df.insert(0, 'sample', sample)
        df.set_index(['sample', 'roi'], inplace=True)
        df_list.append(df)
    df = pd.concat(df_list)
    return df, samples


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


def classification_scores(tp, fp, fn, tn=None):
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
