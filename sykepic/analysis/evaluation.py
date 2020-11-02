from pathlib import Path


def parse_evaluations(in_files, out_file):
    if isinstance(in_files, (str, Path)):
        in_files = Path(in_files)
        if in_files.is_dir():
            in_files = list(in_files.rglob('*.eval.csv'))
        else:
            in_files = [in_files]
    eval_results = {}
    for file in in_files:
        with open(file) as fh:
            fh.readline()
            for line in fh:
                line = line.strip().split(',')
                for name, res in get_classification_result(line[1], line[2]):
                    eval_results.setdefault(
                        name, {'tp': 0, 'fp': 0, 'fn': 0})[res] += 1
    with open(out_file, 'w') as fh:
        fh.write('class,tp,fp,fn\n')
        for name, res in sorted(eval_results.items()):
            fh.write(f"{name},{res['tp']},{res['fp']},{res['fn']}\n")


def get_classification_result(predicted, actual, empty='unclassifiable'):
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


def classification_error(in_file, empty='unclassifiable'):
    class_results = {'all': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}}
    with open(in_file) as fh:
        fh.readline()
        for line in fh:
            line = line.strip().split(',')
            if line[0] == empty:
                # TPs of empty class are all of the true negatives (tn)
                class_results['all']['tn'] = int(line[1])
            else:
                tp = int(line[1])
                fp = int(line[2])
                fn = int(line[3])
                class_results[line[0]] = {'tp': tp, 'fp': fp, 'fn': fn}
                class_results['all']['tp'] += tp
                class_results['all']['fp'] += fp
                class_results['all']['fn'] += fn
    print(f'--- Evaluation Results ---')
    print(f'Class\tSensitivity\tSpecificity')
    for name, res in class_results.items():
        sens = res['tp']/(res['tp']+res['fn'])
        print(f'{name}\t{sens:.2f}', end='')
        if name == 'all':
            spec = res['tn']/(res['tn']+res['fp'])
            print(f'\t\t{spec:.2f}', end='')
        print()
