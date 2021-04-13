"""Module for cleaning up a labeled dataset."""

import os
import shutil


def main(args):
    create_dataset(args.original, args.new, args.min, args.max, args.exclude)


def create_dataset(original_dataset, new_dataset, min_N, max_N, exclude):
    print(f"[INFO] Creating new dataset in {new_dataset}")
    class_names = filter_classes(original_dataset, min_N, None, exclude)
    copy_dataset(original_dataset, new_dataset, class_names, max_N)


def filter_classes(dataset, min_N=None, max_N=None, exclude=[]):
    classes = []
    start_depth = dataset.count(os.path.sep)
    for dirpath, dirnames, filenames in os.walk(dataset):
        dirname = os.path.basename(dirpath)
        # Ignore folders below dataset root
        current_depth = dirpath.count(os.path.sep) - start_depth
        if dirname in exclude or current_depth >= 1:
            continue
        if not min_N and not max_N:
            classes.append(dirname)
        elif not max_N:
            if len(filenames) >= min_N:
                classes.append(dirname)
        elif not min_N:
            if len(filenames) <= max_N:
                classes.append(dirname)
        else:
            if len(filenames) >= min_N and len(filenames) <= max_N:
                classes.append(dirname)
    return classes


def copy_dataset(original_dataset, new_dataset, classes=None, max_N=None, exclude=[]):
    for dirpath, _, filenames in os.walk(original_dataset):
        label = os.path.basename(dirpath)
        if (not classes or label in classes) and filenames:
            label = label.replace(" ", "_")
            dst_dir = os.path.join(new_dataset, label)
            os.makedirs(dst_dir)
            print(f"\tCopying files from {dirpath}")
            for i, filename in enumerate(filenames, start=1):
                if max_N and i > max_N:
                    break
                src = os.path.join(dirpath, filename)
                if src in exclude:
                    continue
                file_ext = filename.split(".")[-1]
                dst = os.path.join(dst_dir, f"{label}_{i}.{file_ext}")
                shutil.copyfile(src, dst)
