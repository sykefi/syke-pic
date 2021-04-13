"""This module contains most of the logic needed for labeled data.

Some filesystem helper functions are also present, mainly `list_files()`
"""

import os
import random
from itertools import groupby
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


class ModelData:
    """Class for handling the data required to train a CNN model."""

    def __init__(self, dataset, split, min_N, max_N, exclude=[], random_seed=24):
        self.dataset = Path(dataset)
        self.split = split
        self.min_N = min_N
        self.max_N = max_N
        self.exclude = exclude
        self.random_seed = random_seed
        self.oversampled = False
        self._init_paths()
        self._init_labels()

    def _init_paths(self):
        """Split dataset into lists of paths."""

        train_split, val_split, test_split = self.split
        self.train_x = []
        self.val_x = []
        self.test_x = []
        self.distribution = {}

        for class_dir in self.dataset.iterdir():
            paths = sorted(
                list_files(
                    class_dir,
                    ".png",
                    self.min_N,
                    self.max_N,
                    self.exclude,
                    self.random_seed,
                )
            )
            if not paths:
                continue
            random.seed(self.random_seed)
            random.shuffle(paths)
            train_stop = int(round(len(paths) * train_split))
            val_stop = train_stop + int(round(len(paths) * val_split))
            train = paths[:train_stop]
            val = paths[train_stop:val_stop]
            test = paths[val_stop:]
            assert train and val and test, (
                f"'{class_dir.name}' doesn't have enough samples ({len(paths)})."
                " Consider using another min_N or split value."
            )
            self.distribution[class_dir.name] = [
                len(paths),
                len(train),
                len(val),
                len(test),
            ]
            self.train_x.extend(train)
            self.val_x.extend(val)
            self.test_x.extend(test)
        random.seed(self.random_seed)
        random.shuffle(self.train_x)
        random.seed(self.random_seed)
        random.shuffle(self.val_x)
        random.seed(self.random_seed)
        random.shuffle(self.test_x)

    def _init_labels(self):
        """Get label from each image path."""

        train_labels = [path.parent.name for path in self.train_x]
        val_labels = [path.parent.name for path in self.val_x]
        test_labels = [path.parent.name for path in self.test_x]
        self.le = LabelEncoder()
        self.le.fit_transform(train_labels)
        self.train_y = list(self.le.transform(train_labels))
        self.val_y = list(self.le.transform(val_labels))
        self.test_y = list(self.le.transform(test_labels))

    def save(self, out_dir):
        """Save all relevant information regarding model data."""

        out_dir = Path(out_dir)
        Path.mkdir(out_dir, parents=True, exist_ok=True)
        with open(out_dir / "class_distribution.csv", "w") as fh:
            fh.write("class,total,train,validation,test")
            if self.oversampled:
                fh.write(",oversampled")
            # Order classes alphabetically
            classes = sorted(self.distribution.items())
            # Order by class size, descending
            classes = sorted(classes, key=lambda x: x[1][0], reverse=True)
            for class_ in classes:
                fh.write(f"\n{class_[0]},")
                fh.write(",".join([str(i) for i in class_[1]]))
        with open(out_dir / "class_names.txt", "w") as fh:
            fh.write("\n".join(self.le.classes_))

    def oversample(self, until, decay):
        """Reuse training samples."""

        train_zip = sorted(zip(self.train_x, self.train_y), key=lambda x: x[1])
        self.over_x = []
        self.over_y = []
        for key, group in groupby(train_zip, lambda x: x[1]):
            x, y = zip(*list(group))
            x = list(x)
            y = list(y)
            over_x, over_y = oversample(x, y, until, decay)
            name = self.le.inverse_transform([key])[0]
            self.distribution[name].append(len(over_x))
            self.distribution[name][1] += len(over_x)
            self.over_x.extend(over_x)
            self.over_y.extend(over_y)
        self.oversampled = True

    def set_data_loaders(
        self, batch_size, num_workers, train_transform, eval_transform, num_chans=3
    ):
        """Read data with PyTorch DataLoaders."""

        self.num_chans = num_chans
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        if self.oversampled:
            train_x = self.train_x + self.over_x
            train_y = self.train_y + self.over_y
            train_x, train_y = combined_shuffle(train_x, train_y, self.random_seed)
        else:
            train_x = self.train_x
            train_y = self.train_y

        train_data = ImageDataset(
            train_x, train_y, self.train_transform, num_chans, len(self.le.classes_)
        )
        val_data = ImageDataset(
            self.val_x,
            self.val_y,
            self.eval_transform,
            num_chans,
            len(self.le.classes_),
        )
        test_data = ImageDataset(
            self.test_x,
            self.test_y,
            self.eval_transform,
            num_chans,
            len(self.le.classes_),
        )
        self.train_loader = DataLoader(
            train_data, batch_size, shuffle=True, num_workers=num_workers
        )
        self.val_loader = DataLoader(val_data, batch_size, num_workers=num_workers)
        self.test_loader = DataLoader(test_data, batch_size, num_workers=num_workers)


class ImageDataset(Dataset):
    """Dataset for 1 or 3 channel images."""

    def __init__(
        self, paths, labels=None, transform=None, num_chans=3, num_classes=None
    ):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.num_chans = num_chans
        self.num_classes = num_classes

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = None

        if self.num_chans == 3:
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            # Add a dimension to grayscale images
            img = np.expand_dims(img, axis=2)

        if self.transform:
            img = self.transform(img)

        if label is not None:
            return img, label
        else:
            return img, str(path)


def list_files(root_dir, extension, min_N=None, max_N=None, exclude=[], random_seed=24):
    """Yields all files with allowed extensions.

    The absolute path is returned for each file.

    Parameters
    ----------
    root_dir : str, Path
        Top level directory
    extension : str, iterable
        One or more extensions to filter files by.
    min_N : int
        Skip sub directory if it has less files than this.
    max_N : int
        Maximum amount of files to return for each sub directory.
        If sub dir exceeds this limit, files are shuffled
        and then sliced like so [:max_N]
    exclude : iterable
        Directories to ignore

    Yields
    ------
        A new file path (Path).
    """

    if type(extension) is not list:
        extension = [extension]

    for dirpath, _, filenames in os.walk(root_dir):
        dirpath = Path(dirpath)
        if dirpath.name in exclude:
            continue
        if min_N and len(filenames) < min_N:
            continue
        if max_N and len(filenames) > max_N:
            random.seed(random_seed)
            random.shuffle(filenames)
            filenames = filenames[:max_N]
        for filename in filenames:
            filepath = dirpath / filename
            if filepath.suffix in extension:
                yield filepath.resolve()


def auto_id(directory):
    """Returns the next version number available for a sub-directory.

    previous version numbers are extracted from the number after
    the last underscore i.e. directory/version_1, directory/version_2
    In this case 3 would be the returned value.
    """

    max_id = 0
    directory = Path(directory)
    if directory.is_dir():
        for path in directory.iterdir():
            if path.is_dir() and not path.name.startswith("."):
                path_id = int(path.name.split("_")[-1])
                if path_id > max_id:
                    max_id = path_id
    return max_id + 1


def oversample(x, y, until=None, decay=None):
    """Grows lists by reusing items until a limit is reached.

    Limit can be assigned directly with `until` or with an
    exponential decay with `decay`, which is the exponent factor.
    """

    if not until and decay:
        raise ValueError("Must provide either 'until' or 'decay'")
    if not until:
        until = int((1 + 1 * decay ** len(x)) * len(x))
    over_x = []
    over_y = []
    i = 0
    while len(x) + len(over_x) < until:
        over_x.append(x[i])
        over_y.append(y[i])
        i += 1
        if i >= len(x):
            i = 0
    return over_x, over_y


def combined_shuffle(list1, list2, random_seed=24):
    """Shuffles two lists while maintaining their relative order."""

    random.seed(random_seed)
    combined = list(zip(list1, list2))
    random.shuffle(combined)
    return zip(*combined)
