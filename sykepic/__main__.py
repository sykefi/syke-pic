"""The entry point module for sykepic

All command line arguments are parsed here.
Based on which sykepic sub-command was used,
the correct module is called with said arguments.

There are two ways to run this module:

1. Executable file called `sykepic`, which is
   added to PATH with default installation.
2. Calling the python interpreter with `python -m sykepic`

Note: Make sure you are using the correct python environment.
"""

from argparse import ArgumentParser

from sykepic.train import train, dataset
from sykepic.compute import probability, feature_matlab, classification, size_group
from sykepic.utils import logger
from sykepic.sync import process


def main():
    # Set default logger settings
    logger.setup()

    parser = ArgumentParser(
        prog="sykepic",
        description="CLI tool for plankton image classification at SYKE",
    )
    subparsers = parser.add_subparsers(
        title="available sub-commands",
        required=True,
        dest="sub-command",
        help="sykepic {sub-command} -h for more information",
    )

    # Parser for 'sykepic train'
    train_parser = subparsers.add_parser(
        "train", description="Train neural network classifiers"
    )
    train_parser.set_defaults(func=train.main)
    train_parser.add_argument("config", help="Path to config file")
    train_parser.add_argument(
        "--collage",
        nargs=3,
        metavar=("HEIGHT", "WIDTH", "FILE"),
        help=("Save a HEIGHT x WIDTH collage of training images to FILE."),
    )
    train_parser.add_argument(
        "--dist", metavar="FILE", help="Save a class distribution plot to FILE"
    )

    # Parser for 'sykepic prob'
    prob_parser = subparsers.add_parser(
        "prob", description="Calculate class probabilities"
    )
    prob_parser.set_defaults(func=probability.call)
    prob_raw = prob_parser.add_mutually_exclusive_group(required=True)
    prob_raw.add_argument(
        "-r", "--raw", metavar="DIR", help="Root directory of raw IFCB data"
    )
    prob_raw.add_argument(
        "-s",
        "--samples",
        nargs="+",
        metavar="PATH",
        help="One or more sample paths (raw file without suffix)",
    )
    prob_parser.add_argument("-m", "--model", required=True, help="Model directory")
    prob_parser.add_argument("-o", "--out", required=True, help="Root output directory")
    prob_parser.add_argument(
        "-b", "--batch-size", type=int, default=64, metavar="INT", help="Default is 64"
    )
    prob_parser.add_argument(
        "-w", "--num-workers", type=int, default=2, metavar="INT", help="Default is 2"
    )
    prob_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite of previous probabilities",
    )

    # Parser for 'sykepic feat'
    feat_parser = subparsers.add_parser("feat", description="Extract features")
    feat_parser.set_defaults(func=feature_matlab.call)
    feat_parser.add_argument("-m", "--matlab", required=True, help="Matlab binary path")
    feat_raw = feat_parser.add_mutually_exclusive_group(required=True)
    feat_raw.add_argument(
        "-r", "--raw", metavar="DIR", help="Root directory of raw IFCB data"
    )
    feat_raw.add_argument(
        "-s",
        "--samples",
        nargs="+",
        metavar="PATH",
        help="One or more sample paths (raw file without suffix)",
    )
    feat_parser.add_argument(
        "-o", "--out", metavar="DIR", required=True, help="Root output directory"
    )

    # Parser for 'sykepic sync'
    sync_parser = subparsers.add_parser(
        "sync", description="Process IFCB data in an infinite loop"
    )
    sync_parser.set_defaults(func=process.call)
    sync_parser.add_argument("config", metavar="FILE", help="Configuration file")

    # Parser for 'sykepic dataset'
    dataset_parser = subparsers.add_parser(
        "dataset", description="Create a usable dataset"
    )
    dataset_parser.set_defaults(func=dataset.main)
    dataset_parser.add_argument("original", help="Original dataset path")
    dataset_parser.add_argument("new", help="New dataset path")
    dataset_parser.add_argument(
        "--min",
        type=int,
        metavar="INT",
        help="Mininmum amount of samples per class",
    )
    dataset_parser.add_argument(
        "--max",
        type=int,
        metavar="INT",
        help="Maximum amount samples per class, with random sampling.",
    )
    dataset_parser.add_argument(
        "--exclude", nargs="*", default=[], help="Sub-directories to exlude"
    )

    # Parser for `sykepic class`
    class_parser = subparsers.add_parser("class", description="Classify samples")
    class_parser.set_defaults(func=classification.main)
    class_parser.add_argument("probabilities", help="Root directory of probabilities")
    class_parser.add_argument("features", help="Root directory of features")
    class_parser.add_argument(
        "-t",
        "--thresholds",
        metavar="FILE",
        required=True,
        help="Probability thresholds file (required)",
    )
    class_parser.add_argument(
        "-d",
        "--divisions",
        metavar="FILE",
        help="Feature divisions file (optional)",
    )
    class_parser.add_argument(
        "-o",
        "--out",
        metavar="PATH",
        required=True,
        help="Output CSV-file path (required)",
    )
    class_parser.add_argument(
        "-s",
        "--summarize",
        metavar="FEATURE",
        default="biomass_ugl",
        help="Which feature to summarize, default is biomass_ugl",
    )
    class_parser.add_argument(
        "-a",
        "--append",
        action="store_true",
        help="Append to output file if it exists",
    )
    class_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite output file if it exists",
    )

    # Parser for 'sykepic size'
    size_parser = subparsers.add_parser("size", description="Extract size groups")
    size_parser.set_defaults(func=size_group.call)
    size_parser.add_argument("features", help="Root directory of features")
    size_parser.add_argument(
        "-g",
        "--groups",
        metavar="FILE",
        required=True,
        help="Size group file (required)",
    )
    size_parser.add_argument(
        "-c",
        "--column",
        metavar="FEATURE",
        required=True,
        help="Feature used to group (required)",
    )
    size_parser.add_argument(
        "-o",
        "--out",
        metavar="PATH",
        required=True,
        help="Output CSV-file path (required)",
    )
    size_parser.add_argument(
        "-a",
        "--append",
        action="store_true",
        help="Append to output file if it exists",
    )
    size_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite output file if it exists",
    )
    size_parser.add_argument(
        "--pixels-to-um3",
        action="store_true",
        help="Convert pixels to um3 before determining size group",
    )
    size_parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Don't display progress bar",
    )

    # Get arguments for the subparser specified
    args = parser.parse_args()
    # Call this subparsers default function
    args.func(args)


if __name__ == "__main__":
    main()
