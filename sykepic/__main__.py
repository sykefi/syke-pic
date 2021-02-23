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
from sykepic.predict import predict, sync


def main():
    parser = ArgumentParser(
        prog='sykepic',
        description='CLI tool for plankton image classification at SYKE',
    )
    subparsers = parser.add_subparsers(
        title='available sub-commands', required=True, dest='sub-command',
        help='sykepic {sub-command} -h for more information',
    )

    # Parser for 'sykepic train'
    train_parser = subparsers.add_parser(
        'train', description='Train neural network classifiers'
    )
    train_parser.set_defaults(func=train.main)
    train_parser.add_argument(
        'config', help='Path to config file'
    )
    train_parser.add_argument(
        '--collage', nargs=3, metavar=('HEIGHT', 'WIDTH', 'FILE'),
        help=('Save a HEIGHT x WIDTH collage of training images to FILE.')
    )
    train_parser.add_argument(
        '--dist', metavar='FILE',
        help='Save a class distribution plot to FILE'
    )

    # Parser for 'sykepic predict'
    predict_parser = subparsers.add_parser(
        'predict', description='Use a trained classifier for inference'
    )
    predict_parser.set_defaults(func=predict.main)
    predict_parser.add_argument(
        'model', help='Model directory'
    )
    predict_parser.add_argument(
        'raw', help='Root directory of raw IFCB data'
    )
    predict_parser.add_argument(
        'out', help='Root output directory'
    )
    predict_parser.add_argument(
        '-b', '--batch_size', type=int, default=64, metavar='INT',
        help='Default is 64'
    )
    predict_parser.add_argument(
        '-w', '--num_workers', type=int, default=2, metavar='INT',
        help='Default is 2'
    )
    predict_parser.add_argument(
        '-l', '--limit', type=int, metavar='INT',
        help=('Limit how many samples to process. '
              'Samples will be drawn evenly from raw directory.')
    )
    predict_parser.add_argument(
        '-e', '--softmax_exp', default=1.3, metavar='FLOAT',
        help=("Exponent to use in softmax, use 'e' for normal softmax")
    )
    predict_parser.add_argument(
        '-f', '--force', action='store_true',
        help='Force overwrite of any previous predictions'
    )
    predict_parser.add_argument(
        '--allas', metavar='PATH',
        help='Path to bucket or directory in Allas with raw files'
    )

    # Parser for 'sykepic sync'
    sync_parser = subparsers.add_parser(
        'sync', description='Synchronise local data with Allas'
    )
    sync_parser.set_defaults(func=sync.main)
    sync_parser.add_argument(
        '-l', '--local', required=True, metavar='DIR',
        help='Local directory to sync'
    )
    sync_parser.add_argument(
        '-a', '--allas', required=True, metavar='PATH',
        help='Allas bucket or bucket/directory'
    )
    sync_action = sync_parser.add_mutually_exclusive_group(required=True)
    sync_action.add_argument(
        '-d', '--download', action='store_true',
        help='Download new data from Allas to local'
    )
    sync_action.add_argument(
        '-u', '--upload', action='store_true',
        help='Archive and upload local data to Allas'
    )
    sync_action.add_argument(
        '-r', '--remove', action='store', type=str, nargs='+',
        help='Options: file, archive, allas'
    )
    sync_parser.add_argument(
        '--keep', type=int, default=0, metavar='DAYS',
        help="Don't remove files younger than this amount (in days)"
    )
    sync_parser.add_argument(
        '-f', '--force', action='store_true',
        help='Force archive creation and upload'
    )

    # Parser for 'sykepic dataset'
    dataset_parser = subparsers.add_parser(
        'dataset', description='Create a usable dataset'
    )
    dataset_parser.set_defaults(func=dataset.main)
    dataset_parser.add_argument(
        'original', help='Original dataset path'
    )
    dataset_parser.add_argument(
        'new', help='New dataset path'
    )
    dataset_parser.add_argument(
        '--min', type=int, metavar='INT',
        help='Mininmum amount of samples per class'
    )
    dataset_parser.add_argument(
        '--max', type=int, metavar='INT',
        help='Maximum amount samples per class, with random sampling.'
    )
    dataset_parser.add_argument(
        '--exclude', nargs='*', default=[],
        help='Sub-directories to exlude'
    )

    # Get arguments for the subparser specified
    args = parser.parse_args()
    # Call this subparsers default function
    args.func(args)


if __name__ == '__main__':
    main()
