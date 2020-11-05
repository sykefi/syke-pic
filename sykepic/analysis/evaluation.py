from collections import Counter
from pathlib import Path


def parse_evaluations(in_files, out_file, empty='unclassifiable'):
    if isinstance(in_files, (str, Path)):
        in_files = Path(in_files)
        if in_files.is_dir():
            in_files = list(in_files.rglob('*.eval.csv'))
        else:
            in_files = [in_files]
    class_results = {}
    for file in in_files:
        with open(file) as fh:
            fh.readline()
            for line in fh:
                line = line.strip().split(',')
                for name, result in classification_result(
                        line[1], line[2], empty=empty):
                    class_results.setdefault(
                        name, {'tp': 0, 'fp': 0, 'fn': 0})[result] += 1
    # Total TN = Empty TP
    total = Counter({'tn': class_results[empty]['tp']})
    # Empty class is not needed any more
    del class_results[empty]
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


def classification_result(predicted, actual, empty='unclassifiable'):
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
    support = tp + fp + fn
    if tn:
        specificity = tn / (tn + fp)
        support += tn
    else:
        specificity = None
    return (precision, recall, F1, support, specificity)


def F_score(precision, recall, beta=1):
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
