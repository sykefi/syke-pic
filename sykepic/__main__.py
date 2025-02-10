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

from sykepic.compute import classification, feature, probability, size_group, abundance, class_stats, features_per_prediction

from sykepic.train import train
from sykepic.utils import logger


def main():
    # Set default logger settings
    logger.setup()

    def list_of_strings(arg):
        return arg.split(',')

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
        metavar=("ROWS", "COLUMNS", "PNG"),
        help=("Save a ROWS x COLUMNS grid of transformed images to PNG."),
    )
    train_parser.add_argument(
        "--dist", metavar="FILE", help="Save a class distribution plot to FILE"
    )
    train_parser.add_argument(
        "--save-images",
        metavar="DIR",
        help="Extract train, test, val images to this path",
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
        metavar="SAMPLE PATH",
        help="One or more sample paths (raw file without suffix)",
    )
    prob_raw.add_argument("--image-dir", metavar="DIR", help="Root directory of images")
    prob_raw.add_argument(
        "--images",
        nargs="+",
        metavar="FILE",
        help="One or more image paths",
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
    feat_parser.set_defaults(func=feature.call)
    feat_raw = feat_parser.add_mutually_exclusive_group(required=True)
    feat_raw.add_argument(
        "-r", "--raw", metavar="DIR", help="Root directory of raw IFCB data"
    )
    feat_raw.add_argument(
        "-s",
        "--samples",
        nargs="+",
        metavar="SAMPLE PATH",
        help="One or more sample paths (raw file without suffix)",
    )
    feat_parser.add_argument(
        "-o", "--out", metavar="DIR", required=True, help="Root output directory"
    )
    feat_parser.add_argument(
        "-m",
        "--matlab",
        metavar="FILE",
        help="Matlab binary path (and use it instead of Python)",
    )
    feat_parser.add_argument(
        "-p", "--parallel", action="store_true", help="Use multiple cores"
    )
    feat_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite of previous features (Python only)",
    )

    # Parser for `sykepic class`
    class_parser = subparsers.add_parser(
        "class",
        description="Use thresholds together with probabilities for classification",
    )
    class_parser.set_defaults(func=classification.main)
    class_parser.add_argument("probabilities", help="Root directory of probabilities")
    class_parser.add_argument(
        "--feat",
        metavar="DIR",
        help="Root directory of features (and use them in results)",
    )
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
        metavar="FILE",
        required=True,
        help="Output CSV-file path (required)",
    )
    class_parser.add_argument(
        "-v",
        "--value-column",
        metavar="FEATURE",
        default="biomass_ugl",
        help="Feature used to aggregate results, default is biomass_ugl",
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
    class_parser.add_argument(
        "-exc",
        "--exclusion_list",
        metavar="FILE",
        help="Text file containing a list of sample names to exclude e.g. D20180703T181501",
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
        "-s",
        "--size-column",
        metavar="FEATURE",
        required=True,
        help="Feature used to determine groups (required)",
    )
    size_parser.add_argument(
        "-v",
        "--value-column",
        metavar="FEATURE",
        required=False,
        help=(
            "Feature used to aggregate results. Can also be set to 'abundance'. "
            "Defaults to size-column."
        ),
    )
    size_parser.add_argument(
        "-o",
        "--out",
        metavar="FILE",
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
        "--volume",
        action="store_true",
        help="Include sample volume in output",
    )
    size_parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Don't display progress bar",
    )
    size_parser.add_argument(
        "-exc",
        "--exclusion_list",
        metavar="FILE",
        help="Text file containing a list of sample names to exclude e.g. D20180703T181501",
    )


    # Parser for `sykepic abundance`
    abundance_parser = subparsers.add_parser(
        "abundance",
        description="Count class abundance",
    )
    abundance_parser.set_defaults(func=abundance.main)
    abundance_parser.add_argument("probabilities", help="Root directory of probabilities")
    abundance_parser.add_argument(
        "--feat",
        metavar="DIR",
        help="Root directory of features (and use them in results)",
    )
    abundance_parser.add_argument(
        "-t",
        "--thresholds",
        metavar="FILE",
        required=True,
        help="Probability thresholds file (required)",
    )
    abundance_parser.add_argument(
        "-o",
        "--out",
        metavar="FILE",
        required=True,
        help="Output CSV-file path (required)",
    )
    abundance_parser.add_argument(
        "-v",
        "--value-column",
        metavar="FEATURE",
        default="biomass_ugl",
        help="Feature used to aggregate results, default is biomass_ugl",
    )
    abundance_parser.add_argument(
        "-a",
        "--append",
        action="store_true",
        help="Append to output file if it exists",
    )
    abundance_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite output file if it exists",
    )
    abundance_parser.add_argument(
        "-exc",
        "--exclusion_list",
        metavar="FILE",
        help="Text file containing a list of sample names to exclude e.g. D20180703T181501",
    )


        # Parser for `sykepic class_stats`
    class_stats_parser = subparsers.add_parser(
        "class_stats",
        description="Calculate class statistics",
    )
    class_stats_parser.set_defaults(func=class_stats.main)
    class_stats_parser.add_argument("probabilities", help="Root directory of probabilities")
    class_stats_parser.add_argument(
        "--feat",
        metavar="DIR",
        help="Root directory of features (and use them in results)",
    )
    class_stats_parser.add_argument(
        "-t",
        "--thresholds",
        metavar="FILE",
        required=True,
        help="Probability thresholds file (required)",
    )
    class_stats_parser.add_argument(
        "-o",
        "--out",
        metavar="FILE",
        required=True,
        help="Output CSV-file path (required)",
    )
    class_stats_parser.add_argument(
        "--classes",
        type = list_of_strings,
        metavar = "list of strings",
        help = "Comma-separated list of classes for which to calculate statistics",
    )
    class_stats_parser.add_argument(
        "-a",
        "--append",
        action="store_true",
        help="Append to output file if it exists",
    )
    class_stats_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite output file if it exists",
    )



        # Parser for `sykepic "features_per_prediction"`
    features_per_prediction_parser = subparsers.add_parser(
        "features_per_prediction",
        description="Combine particle features with prediction",
    )
    features_per_prediction_parser.set_defaults(func=features_per_prediction.main)
    features_per_prediction_parser.add_argument("probabilities", help="Root directory of probabilities")
    features_per_prediction_parser.add_argument(
        "--feat",
        metavar="DIR",
        help="Root directory of features (and use them in results)",
    )
    features_per_prediction_parser.add_argument(
        "-t",
        "--thresholds",
        metavar="FILE",
        required=True,
        help="Probability thresholds file (required)",
    )
    features_per_prediction_parser.add_argument(
        "-o",
        "--out",
        metavar="FILE",
        required=True,
        help="Output CSV-file path (required)",
    )
    features_per_prediction_parser.add_argument(
        "-a",
        "--append",
        action="store_true",
        help="Append to output file if it exists",
    )
    features_per_prediction_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite output file if it exists",
    )

    # Get arguments for the subparser specified
    args = parser.parse_args()
    # Call this subparsers default function
    args.func(args)

if __name__ == "__main__":
    main()
