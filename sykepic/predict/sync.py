"""Module for synchronising local data with Allas"""

import shutil
from configparser import ConfigParser
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep

import boto3
from botocore.exceptions import ClientError
from requests import HTTPError
from sykepic.utils import ifcb, logger
from sykepic.utils.files import create_archive

from . import biovolume
from .predict import predict

log = logger.get_logger("sync")


def main(args):
    # Parse config file and set up
    config = ConfigParser()
    config.read(args.config)
    # Logging
    logger.setup(config["logging"]["config"])
    # Local, Download
    s3 = boto3.resource("s3", endpoint_url=config["download"]["endpoint_url"])
    record = read_record(config["local"]["sample_record"])
    sample_extensions = tuple(
        ext.strip() for ext in config.get("download", "sample_extensions").split(",")
    )
    download_bucket = s3.Bucket(config["download"]["bucket"])
    local_raw = Path(config["local"]["raw"])
    local_raw.mkdir(parents=True, exist_ok=True)
    local_pred = Path(config["local"]["predictions"])
    local_pred.mkdir(parents=True, exist_ok=True)
    local_biovol = Path(config["local"]["biovolumes"])
    local_biovol.mkdir(parents=True, exist_ok=True)
    # Predict
    model_dir = Path(config["predict"]["model"])
    if not model_dir.is_dir():
        raise OSError(f"Model directory '{model_dir} not found'")
    batch_size = config.getint("predict", "batch_size")
    num_workers = config.getint("predict", "num_workers")
    limit = config["predict"]["limit"]
    limit = int(limit) if limit not in ["", None] else None
    softmax_exp = config["predict"]["softmax_exp"]
    softmax_exp = float(softmax_exp) if softmax_exp not in ["", None] else None
    # Features
    feat_parallel = config.getboolean("features", "parallel")
    feat_force = config.getboolean("features", "force")
    # Upload
    upload_time = datetime.strptime(config["upload"]["time"], "%H:%M")
    next_upload = datetime.now().replace(
        hour=upload_time.hour, minute=upload_time.minute, second=0, microsecond=0
    )
    upload_record = read_record(config["local"]["upload_record"])
    upload_bucket = s3.Bucket(config["upload"]["bucket"])
    raw_prefix = config["upload"]["raw_prefix"]
    pred_prefix = config["upload"]["predictions_prefix"]
    biovol_prefix = config["upload"]["biovolumes_prefix"]
    compression = config["upload"]["compression"]
    # Remove
    keep = config.getint("remove", "keep")
    remove_raw_files = config.getboolean("remove", "raw_files")
    remove_pred_files = config.getboolean("remove", "prediction_files")
    remove_biovol_files = config.getboolean("remove", "biovolume_files")
    remove_raw_archive = config.getboolean("remove", "raw_archive")
    remove_pred_archive = config.getboolean("remove", "prediction_archive")
    remove_biovol_archive = config.getboolean("remove", "biovolume_archive")
    remove_from_bucket = config.getboolean("remove", "from_download_bucket")

    # Start service loop
    log.info("Synchronization service started")
    try:
        while True:
            samples_available = check_available(
                sample_extensions, record, download_bucket
            )
            log.debug(f"{len(samples_available)} samples available")
            if samples_available:
                samples_downloaded, late_record = download(
                    samples_available,
                    sample_extensions,
                    upload_record,
                    local_raw,
                    download_bucket,
                )
                log.debug(f"{len(samples_downloaded)} samples downloaded")
                # Check for samples arriving after their day has already been uploaded.
                # If their day directory hasn't been removed yet,
                # these days can be re-uploaded.
                today = datetime.today()
                for day_path in late_record:
                    if (today - datetime.strptime(day_path, "%Y/%m/%d")).days < keep:
                        log.info(
                            f"{day_path} will be re-uploaded with late arriving samples"
                        )
                        upload_record.remove(day_path)
                    else:
                        log.warn(
                            f"{day_path} can't be re-uploaded, "
                            "since it's older than {keep} days"
                        )
                if samples_downloaded:
                    log.debug(
                        f"Making predictions for {len(samples_downloaded)} samples"
                    )
                    samples_predicted = predict(
                        model_dir,
                        local_raw,
                        local_pred,
                        batch_size,
                        num_workers,
                        softmax_exp=softmax_exp,
                        sample_filter=samples_downloaded,
                        limit=limit,
                        progress_bar=False,
                    )
                    log.debug(
                        f"Extracting features for {len(samples_predicted)} samples"
                    )
                    samples_processed = biovolume.main(
                        local_raw,
                        local_biovol,
                        sample_filter=samples_predicted,
                        parallel=feat_parallel,
                        force=feat_force,
                    )
                    record.update(samples_processed)
                    for sample in samples_processed:
                        log.info(f"{sample} processed")
                    write_record(record, config["local"]["sample_record"])
            if datetime.now() > next_upload:
                today = datetime.now().strftime("%Y/%m/%d")
                todays_upload_record = tuple(upload_record) + (today,)
                uploaded_raw = upload(
                    todays_upload_record,
                    compression,
                    local_raw,
                    raw_prefix,
                    upload_bucket,
                    suffix=".raw",
                )
                uploaded_pred = upload(
                    todays_upload_record,
                    compression,
                    local_pred,
                    pred_prefix,
                    upload_bucket,
                    suffix=".prob",
                )
                uploaded_biovol = upload(
                    todays_upload_record,
                    compression,
                    local_biovol,
                    biovol_prefix,
                    upload_bucket,
                    suffix=".feat",
                )
                if uploaded_raw != uploaded_pred != uploaded_biovol:
                    log.warn(
                        f"Upload mismatch: raw {len(uploaded_raw)}, "
                        f"predictions {len(uploaded_pred)}, "
                        f"biovolumes {len(uploaded_biovol)}"
                    )
                # Add new days to upload record
                if uploaded_raw:
                    upload_record.update(uploaded_raw)
                    write_record(
                        upload_record, config["local"]["upload_record"], "upload-"
                    )
                # Cleaning up old files
                remove(
                    local_raw,
                    keep,
                    remove_raw_files,
                    remove_raw_archive,
                    remove_from_bucket,
                    download_bucket,
                )
                remove(local_pred, keep, remove_pred_files, remove_pred_archive)
                remove(local_biovol, keep, remove_biovol_files, remove_biovol_archive)
                # Determine next upload time
                next_upload += timedelta(days=1)
                log.info(f"Upload and cleanup done. Next time is {next_upload}")
            else:
                # Delay next iteration a bit
                sleep(120)
    except Exception:
        log.critical("Unhandled exception in service loop", exc_info=True)


def check_available(extensions, record, bucket):
    log.debug("Checking for new samples")
    # Iterate over object keys in the given bucket, filtering out those that
    # don't end with the correct extensions. Next remove the extensions from
    # the key and add them to a set, which will keep only one name per sample.
    samples = set(
        obj.key.split(".")[0]
        for obj in bucket.objects.all()
        if obj.key.endswith(extensions)
    )
    # Taking a set difference with record, will return
    # those keys that are only in samples, i.e., they are new.
    new_samples = samples.difference(record)
    return new_samples


def download(samples, extensions, upload_record, local_raw, bucket):
    downloaded_samples = set()
    late_record = set()
    for sample in samples:
        sample_date = ifcb.sample_to_datetime(sample)
        day_path = sample_date.strftime("%Y/%m/%d")
        if day_path in upload_record:
            log.warn(f"Downloading {sample}, but {day_path} has already been uploaded")
            late_record.add(day_path)
        to = local_raw / day_path
        to.mkdir(exist_ok=True, parents=True)
        try:
            for ext in extensions:
                obj = sample + ext
                if not (to / obj).is_file():
                    log.debug(f"Downloading {bucket.name}/{obj}")
                    bucket.download_file(obj, str(to / obj))
            downloaded_samples.add(sample)
        except ClientError as e:
            status_code = e.response["ResponseMetadata"]["HTTPStatusCode"]
            if status_code == 404:
                log.error(f"Object {obj} not found in {bucket.name}")
            else:
                log.exception(f"While downloading {bucket.name}/{obj}")
                raise
    return downloaded_samples, late_record


def upload(upload_record, compression, local_dir, bucket_dir, bucket, suffix=None):
    # 1. Find all day_dirs
    day_dirs = [
        d.relative_to(local_dir) for d in sorted(local_dir.glob("*/*/*")) if d.is_dir()
    ]
    # 2. Filter out those days that are in upload record
    day_dirs = [d for d in day_dirs if str(d) not in upload_record]
    uploaded_days = []
    for day_dir in day_dirs:
        # Make sure directory trully represents a valid date
        try:
            datetime(*map(int, day_dir.parts[-3:]))
        except Exception:
            log.error(f"{local_dir/day_dir} is not a valid day directory")
            continue
        try:
            # 3. Create day archive
            # Archive path: local_dir/yyyy/mm/yyyymmdd[.suffix].compression
            archive = local_dir / day_dir.parent / f"{''.join(day_dir.parts[-3:])}"
            if suffix:
                suffix += f".{compression}"
                archive = archive.with_suffix(suffix)
            else:
                archive = archive.with_suffix(f".{compression}")
            create_archive(local_dir / day_dir, archive, compression)
            # 4. Upload archive
            obj = f"{bucket_dir}/{archive.relative_to(local_dir)}"
            log.info(f"Uploading {obj} to {bucket.name}")
            bucket.upload_file(str(archive), obj)
            # 5. Mark day as successfully uploaded
            uploaded_days.append(str(day_dir))
        except Exception:
            log.exception(f"While uploading {local_dir/day_dir}")

    return uploaded_days


def remove(local_dir, keep, files, archive, from_bucket=False, bucket=None):
    removing = []
    if files:
        removing.append("files")
    if archive:
        removing.append("archive")
    if from_bucket:
        if not bucket:
            raise ValueError("Removal bucket not specified")
        removing.append(bucket.name)

    archive_suffixes = [".zip", ".tar", ".tar.gz"]
    today = datetime.today()
    day_dirs = [d for d in sorted(local_dir.glob("*/*/*")) if d.is_dir()]
    for day_dir in day_dirs:
        try:
            date = datetime(*map(int, day_dir.parts[-3:]))
        except Exception:
            log.error(f"{day_dir} is not a valid day directory")
            continue
        # Check that day is old enough to be removed
        if (today - date).days < keep:
            continue
        log.info(
            f"Removing {day_dir.relative_to(local_dir.parent)} "
            f"({', '.join(removing)})"
        )
        day_samples = [path.name for path in day_dir.iterdir()]
        if files:
            shutil.rmtree(day_dir)
        if archive:
            for path in day_dir.parent.iterdir():
                if (
                    path.stem.starswith("".join(day_dir.parts[-3:]))
                    and path.suffix in archive_suffixes
                ):
                    path.unlink()
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
        record = set(fh.read().split("\n"))
        # Remove last empty line if it exists in record
        if "" in record:
            record.remove("")
    return record


def write_record(record, file, prefix=""):
    log.debug(f"Writing {prefix}record ({len(record)} items)")
    with open(file, "w") as fh:
        for item in sorted(record):
            fh.write(item + "\n")


def delete_many_from_bucket(objects, bucket):
    response = bucket.delete_objects(
        Delete={"Objects": [{"Key": key} for key in objects]}
    )
    status_code = response["ResponseMetadata"]["HTTPStatusCode"]
    if status_code != 200:
        raise HTTPError(f"s3 client returned status code {status_code}")
    # Apparently there is no way to confirm which objects were actually deleted
