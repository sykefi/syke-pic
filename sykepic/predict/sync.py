"""Module for synchronising local data with Allas"""

import datetime
import logging
import shutil
import zipfile
from argparse import ArgumentTypeError
from configparser import ConfigParser
from pathlib import Path
from time import sleep

import boto3
from botocore.exceptions import ClientError
from sykepic.utils import ifcb, logger

from . import allas
from .predict import predict

log = logging.getLogger('sync')


def main(args):
    # Parse config file and set up
    config = ConfigParser()
    config.read(args.config)
    logger.setup(config['logging']['config'])
    s3 = boto3.resource('s3', endpoint_url=allas.ENDPOINT_URL)
    record = read_record(config['local']['sample_record'])
    error_record = read_record(config['local']['error_record'])
    sample_extensions = tuple(ext.strip() for ext in config.get(
        'download', 'sample_extensions').split(','))
    download_bucket = s3.Bucket(config['download']['bucket'])
    # upload_bucket = s3.Bucket(config['upload']['bucket'])
    local_raw = Path(config['local']['raw'])
    local_raw.mkdir(parents=True, exist_ok=True)
    local_pred = Path(config['local']['predictions'])
    local_pred.mkdir(parents=True, exist_ok=True)
    model_dir = Path(config['predict']['model'])
    if not model_dir.is_dir():
        raise OSError(f"Model directory '{model_dir} not found'")
    batch_size = config.getint('predict', 'batch_size')
    num_workers = config.getint('predict', 'num_workers')
    limit = config['predict']['limit']
    limit = int(limit) if limit else None

    # Start service loop
    try:
        while True:
            samples_available = check_for_new(
                sample_extensions, record, error_record, download_bucket)
            if not samples_available:
                sleep(60)
                continue
            log.info(f'{len(samples_available)} samples available')
            samples_downloaded = download(
                samples_available, sample_extensions, local_raw, download_bucket)
            log.info(f'{len(samples_downloaded)} new samples downloaded')
            error_record.update(
                samples_available.difference(samples_downloaded))
            if not samples_downloaded:
                sleep(60)
                continue
            log.info('Calling prediction module...')
            # record.update(samples_downloaded)  # Local debug purposes
            samples_processed = predict(model_dir, local_raw, local_pred,
                                        batch_size, num_workers,
                                        sample_filter=samples_downloaded, limit=limit,
                                        progress_bar=False)
            log.info(
                f'{len(samples_processed)} new samples processed successfully')
            record.update(samples_processed)
            write_record(record, config['local']['sample_record'])
            write_record(error_record, config['local']['error_record'])
    except:
        log.critical('Unhandled exception in service loop', exc_info=True)


def check_for_new(extensions, record, error_record, bucket):
    log.info('Checking for new samples...')
    # Iterate over object keys in the given bucket, filtering out those that
    # don't end with the correct extensions. Next remove the extensions from
    # the key and add them to a set, which will keep only one name per sample.
    samples = set(
        obj.key.split('.')[0] for obj in bucket.objects.all()
        if obj.key.endswith(extensions))
    # Taking a set difference with the 'record', will return those
    # keys that are in 'samples' but not in 'record'.
    new_samples = samples.difference(record, error_record)
    return new_samples


def download(samples, extensions, local_raw, bucket):
    log.info('Downloading new samples...')
    downloaded_samples = set()
    for sample in samples:
        sample_date = ifcb.sample_to_datetime(sample)
        day_path = sample_date.strftime('%Y/%m/%d')
        to = local_raw/day_path
        to.mkdir(exist_ok=True, parents=True)
        log.debug(f'Downloading {sample}')
        try:
            for ext in extensions:
                obj = sample + ext
                if not (to/obj).is_file():
                    bucket.download_file(obj, str(to/obj))
            downloaded_samples.add(sample)
        except ClientError as e:
            status_code = e.response['ResponseMetadata']['HTTPStatusCode']
            if status_code == 404:
                log.error(f"Object '{obj}' not found in '{bucket.name}'")
            else:
                log.exception(f"While downloading object '{obj}'")
                raise
    return downloaded_samples


def read_record(file):
    Path(file).touch(exist_ok=True)
    with open(file) as fh:
        record = set(fh.read().split('\n'))
        # Remove last empty line if it exists in record
        if '' in record:
            record.remove('')
    return record


def write_record(record, file):
    log.info(f'Writing record ({len(record)} in total)')
    with open(file, 'w') as fh:
        for sample in record:
            fh.write(sample + '\n')


def upload(args):
    local = Path(args.local)
    s3 = boto3.resource('s3', endpoint_url=allas.ENDPOINT_URL)
    today = datetime.date.today()
    # Find all directories corresponding to one day
    day_dirs = []
    # find_head_dirs() modifies day_dirs in place
    find_head_dirs(local, day_dirs)
    for day_dir in day_dirs:
        try:
            year, month, day = day_dir.parts[-3:]
            date = datetime.date(int(year), int(month), int(day))
        except Exception:
            print(f'[ERROR] {day_dir} is not a valid day directory')
            continue
        # Only upload past day's data
        if date >= today:
            continue
        local_zip = day_dir.with_suffix('.zip')
        allas_zip = Path(args.allas)/year/month/local_zip.name
        if local_zip.is_file() and not args.force:
            print(f'[INFO] ZIP-file already exists for {day_dir}')
            continue
        print(f'[INFO] Creating archive {local_zip}')
        zipped_files = []
        with zipfile.ZipFile(local_zip, 'w') as zfh:
            for file in day_dir.iterdir():
                if file.is_file():
                    zfh.write(file, arcname=file.name)
                    zipped_files.append(file)
        print(f'[INFO] Uploading to {allas_zip.parent}')
        allas.upload(local_zip, allas_zip, s3)


def remove(args):
    if len(args.remove) > 3:
        raise ArgumentTypeError('--remove takes max three arguments')
    files_to_remove_from_allas = []
    today = datetime.date.today()
    # Find all directories corresponding to one day
    day_dirs = []
    # find_head_dirs() modifies day_dirs in place
    find_head_dirs(Path(args.local), day_dirs)
    for day_dir in day_dirs:
        try:
            year, month, day = day_dir.parts[-3:]
            date = datetime.date(int(year), int(month), int(day))
        except Exception:
            print(
                f'[ERROR] {day_dir} is not a valid day directory')
            continue
        if (today - date).days < args.keep:
            continue
        files_to_remove_from_allas.extend(list(day_dir.iterdir()))
        if 'file' in args.remove:
            shutil.rmtree(day_dir)
            print(f'[INFO] Removed {day_dir}')
        if 'archive' in args.remove:
            # Remove local zip file
            local_zip = day_dir.with_suffix('.zip')
            if local_zip.is_file():
                local_zip.unlink()
            # Remove month directory if it's empty
            try:
                day_dir.parent.rmdir()
            except OSError:
                pass
            print(f'[INFO] Removed {local_zip}')
    if 'allas' in args.remove:
        # Remove files from Allas
        s3 = boto3.resource('s3', endpoint_url=allas.ENDPOINT_URL)
        print(f'[INFO] Cleaning Allas bucket {args.allas}')
        allas.delete_many(args.allas, files_to_remove_from_allas, s3)
    print('[INFO] Done!')


def find_head_dirs(path, head_dirs=[]):
    sub_dirs = [p for p in sorted(path.iterdir()) if p.is_dir()]
    if sub_dirs:
        for sub_dir in sub_dirs:
            find_head_dirs(sub_dir, head_dirs)
    else:
        head_dirs.append(path)
