"""Module for synchronising local data with Allas"""

import logging
import shutil
from argparse import ArgumentTypeError
from configparser import ConfigParser
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep

import boto3
from botocore.exceptions import ClientError
from requests import HTTPError
from sykepic.utils import ifcb, logger
from sykepic.utils.files import create_archive

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
    upload_time = datetime.strptime(config['upload']['time'], '%H:%M')
    next_upload = datetime.now().replace(
        hour=upload_time.hour, minute=upload_time.minute, second=0, microsecond=0)
    upload_record = read_record(config['local']['upload_record'])
    upload_bucket = s3.Bucket(config['upload']['bucket'])
    raw_prefix = config['upload']['raw_prefix']
    pred_prefix = config['upload']['predictions_prefix']
    compression = config['upload']['compression']
    remove_raw_files = config.getboolean('remove', 'raw_files')
    remove_pred_files = config.getboolean('remove', 'prediction_files')
    remove_raw_archive = config.getboolean('remove', 'raw_archive')
    remove_pred_archive = config.getboolean('remove', 'prediction_archive')
    remove_from_bucket = config.getboolean('remove', 'from_download_bucket')
    keep = config.getint('remove', 'keep')

    # Start service loop
    log.info('Synchronization service started')
    try:
        while True:
            samples_available = check_available(
                sample_extensions, record, error_record, download_bucket)
            log.debug(f'{len(samples_available)} samples available')
            if samples_available:
                samples_downloaded = download(
                    samples_available, sample_extensions,
                    upload_record, local_raw, download_bucket)
                log.info(f'{len(samples_downloaded)} new samples downloaded')
                error_record.update(
                    samples_available.difference(samples_downloaded))
                if samples_downloaded:
                    log.debug('Running predictions')
                    samples_processed = predict(
                        model_dir, local_raw, local_pred, batch_size, num_workers,
                        sample_filter=samples_downloaded, limit=limit,
                        progress_bar=False)
                    log.info(
                        f'{len(samples_processed)} new samples processed successfully')
                    record.update(samples_processed)
                    write_record(record, config['local']['sample_record'])
                    write_record(
                        error_record, config['local']['error_record'], 'error-')
            if datetime.now() > next_upload:
                today = datetime.now().strftime('%Y/%m/%d')
                todays_upload_record = tuple(upload_record) + (today,)
                uploaded_raw = upload(todays_upload_record, compression,
                                      local_raw, raw_prefix, upload_bucket)
                uploaded_pred = upload(todays_upload_record, compression,
                                       local_pred, pred_prefix, upload_bucket)
                if uploaded_raw != uploaded_pred:
                    raise ValueError(
                        'Uploaded raw and predictions size mismatch')
                # Add new days to upload record
                if uploaded_raw:
                    upload_record.update(uploaded_raw)
                    write_record(
                        upload_record, config['local']['upload_record'], 'upload-')
                # Determine next upload time
                next_upload += timedelta(days=1)
                # Cleaning up old files
                remove(local_raw, keep, remove_raw_files, remove_raw_archive,
                       remove_from_bucket, download_bucket)
                remove(local_pred, keep, remove_pred_files, remove_pred_archive)
            else:
                # Delay next iteration a bit
                sleep(60)
    except:
        log.critical('Unhandled exception in service loop', exc_info=True)


def check_available(extensions, record, error_record, bucket):
    log.debug('Checking for new samples')
    # Iterate over object keys in the given bucket, filtering out those that
    # don't end with the correct extensions. Next remove the extensions from
    # the key and add them to a set, which will keep only one name per sample.
    samples = set(
        obj.key.split('.')[0] for obj in bucket.objects.all()
        if obj.key.endswith(extensions))
    # Taking a set difference with the 'record' and 'error_record', will return
    # those keys that are only in 'samples', i.e., they are new.
    new_samples = samples.difference(record, error_record)
    return new_samples


def download(samples, extensions, upload_record, local_raw, bucket):
    downloaded_samples = set()
    for sample in samples:
        sample_date = ifcb.sample_to_datetime(sample)
        day_path = sample_date.strftime('%Y/%m/%d')
        if day_path in upload_record:
            log.warn(
                f"Downloading '{sample}', but '{day_path}' has already been uploaded")
        to = local_raw/day_path
        to.mkdir(exist_ok=True, parents=True)
        try:
            for ext in extensions:
                obj = sample + ext
                if not (to/obj).is_file():
                    log.debug(f'Downloading {bucket.name}/{obj}')
                    bucket.download_file(obj, str(to/obj))
            downloaded_samples.add(sample)
        except ClientError as e:
            status_code = e.response['ResponseMetadata']['HTTPStatusCode']
            if status_code == 404:
                log.error(f"Object '{obj}' not found in '{bucket.name}'")
            else:
                log.exception(f'While downloading {bucket.name}/{obj}')
                raise
    return downloaded_samples


def upload(upload_record, compression, local_dir, bucket_dir, bucket):
    log.info(f'Uploading to {bucket.name}/{bucket_dir}')
    # 1. Find all day_dirs
    day_dirs = [d.relative_to(local_dir)
                for d in sorted(local_dir.glob('*/*/*')) if d.is_dir()]
    # 2. Filter out those days that are in upload record
    day_dirs = [d for d in day_dirs if str(d) not in upload_record]
    uploaded_days = []
    for day_dir in day_dirs:
        # Make sure directory trully represents a valid date
        try:
            datetime(*map(int, day_dir.parts[-3:]))
        except:
            log.error(f"'{local_dir/day_dir}' is not a valid day directory")
            continue
        try:
            # 3. Create day archive
            archive = create_archive(local_dir/day_dir, compression)
            # 4. Upload archive
            obj = f'{bucket_dir}/{archive.relative_to(local_dir)}'
            log.debug(f"Uploading '{obj}' to '{bucket.name}'")
            bucket.upload_file(str(archive), obj)
            # 5. Mark day as successfully uploaded
            uploaded_days.append(str(day_dir))
        except:
            log.exception(f"While uploading '{local_dir/day_dir}'")

    return uploaded_days


def remove(local_dir, keep, files, archive, from_bucket=False, bucket=None):
    removing = []
    if files:
        removing.append('files')
    if archive:
        removing.append('archive')
    if from_bucket:
        if not bucket:
            raise ValueError('Removal bucket not specified')
        removing.append(f'from {bucket.name}')
    log.info(f"Cleaning {local_dir.name} ({', '.join(removing)})")

    archive_suffixes = ['.zip', '.tar', '.tar.gz']
    today = datetime.today()
    day_dirs = [d for d in sorted(local_dir.glob('*/*/*')) if d.is_dir()]
    for day_dir in day_dirs:
        # breakpoint()
        try:
            date = datetime(*map(int, day_dir.parts[-3:]))
        except:
            log.error(f"'{day_dir}' is not a valid day directory")
            continue
        # Check that day is old enough to be removed
        if (today - date).days < keep:
            continue
        day_samples = [path.name for path in day_dir.iterdir()]
        if files:
            shutil.rmtree(day_dir)
        if archive:
            for suffix in archive_suffixes:
                day_archive = day_dir.with_suffix(suffix)
                if day_archive.is_file():
                    day_archive.unlink()
            # Remove month directory if it's empty
            try:
                day_dir.parent.rmdir()
            except OSError:
                pass
        if from_bucket:
            delete_many_from_bucket(day_samples, bucket)


def read_record(file):
    Path(file).touch(exist_ok=True)
    with open(file) as fh:
        record = set(fh.read().split('\n'))
        # Remove last empty line if it exists in record
        if '' in record:
            record.remove('')
    return record


def write_record(record, file, prefix=''):
    log.debug(f'Writing {prefix}record ({len(record)} items)')
    with open(file, 'w') as fh:
        for item in sorted(record):
            fh.write(item + '\n')


def delete_many_from_bucket(objects, bucket):
    response = bucket.delete_objects(
        {'Objects': [{'Key': key} for key in objects]})
    status_code = response['ResponseMetadata']['HTTPStatusCode']
    if status_code != 200:
        raise HTTPError(f's3 client returned status code {status_code}')
    # Apparently there is no way to confirm which objects were actually deleted
