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
from sykepic.utils.files import create_archive, list_sample_paths

from . import features, probabilities

RAW_SUFFIX = ".raw"
log = logger.get_logger("sync")


def call(args):
    main(args.config)


def main(config_file):
    # Parse config file and set up
    config = ConfigParser()
    config.read(config_file)
    # Logging
    logger.setup(config["logging"]["config"])
    # Local, Download
    s3 = boto3.resource("s3", endpoint_url=config["download"]["endpoint_url"])
    sample_record = read_record(config["local"]["sample_record"])
    sample_extensions = tuple(
        ext.strip() for ext in config.get("download", "sample_extensions").split(",")
    )
    download_bucket = s3.Bucket(config["download"]["bucket"])
    local_raw = Path(config["local"]["raw"])
    local_raw.mkdir(parents=True, exist_ok=True)
    local_prob = Path(config["local"]["probabilities"])
    local_prob.mkdir(parents=True, exist_ok=True)
    local_feat = Path(config["local"]["features"])
    local_feat.mkdir(parents=True, exist_ok=True)
    # Probabilities
    model_dir = Path(config["probabilities"]["model"])
    if not model_dir.is_dir():
        raise OSError(f"Model directory {model_dir} not found'")
    batch_size = config.getint("probabilities", "batch_size")
    num_workers = config.getint("probabilities", "num_workers")
    prob_force = config.getboolean("probabilities", "force")
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
    raw_upload_dir = config["upload"]["raw_dir"]
    prob_upload_dir = config["upload"]["probabilities_dir"]
    feat_upload_dir = config["upload"]["features_dir"]
    compression = config["upload"]["compression"]
    # Remove
    keep = config.getint("remove", "keep")
    remove_raw_files = config.getboolean("remove", "raw_files")
    remove_prob_files = config.getboolean("remove", "probabilities_files")
    remove_feat_files = config.getboolean("remove", "features_files")
    remove_raw_archive = config.getboolean("remove", "raw_archive")
    remove_prob_archive = config.getboolean("remove", "probabilities_archive")
    remove_feat_archive = config.getboolean("remove", "features_archive")
    remove_from_bucket = config.getboolean("remove", "from_download_bucket")

    # Start service loop
    log.info("Synchronization service started")
    try:
        while True:
            samples_available = check_available(
                sample_extensions, sample_record, download_bucket
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
                    sample_paths = list_sample_paths(
                        local_raw, filter=samples_downloaded
                    )
                    log.debug(
                        f"Computing probabilities for {len(sample_paths)} samples"
                    )
                    samples_prob = probabilities.main(
                        sample_paths,
                        model_dir,
                        local_prob,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        force=prob_force,
                        progress_bar=False,
                    )
                    log.debug(f"Extracting features for {len(sample_paths)} samples")
                    samples_feat = features.main(
                        sample_paths,
                        local_feat,
                        parallel=feat_parallel,
                        force=feat_force,
                    )
                    # Add to sample_record those samples that were successful
                    # in probability and feature computations
                    samples_processed = samples_prob.intersection(samples_feat)
                    sample_record.update(samples_processed)
                    for sample in samples_processed:
                        log.info(f"{sample} processed")
                    write_record(
                        sample_record, config["local"]["sample_record"], "sample-"
                    )
            if datetime.now() > next_upload:
                today = datetime.now().strftime("%Y/%m/%d")
                todays_upload_record = tuple(upload_record) + (today,)
                uploaded_raw = upload(
                    todays_upload_record,
                    compression,
                    local_raw,
                    raw_upload_dir,
                    upload_bucket,
                    suffix=RAW_SUFFIX,
                )
                uploaded_prob = upload(
                    todays_upload_record,
                    compression,
                    local_prob,
                    prob_upload_dir,
                    upload_bucket,
                    suffix=probabilities.FILE_SUFFIX,
                )
                uploaded_feat = upload(
                    todays_upload_record,
                    compression,
                    local_feat,
                    feat_upload_dir,
                    upload_bucket,
                    suffix=features.FILE_SUFFIX,
                )
                if uploaded_raw != uploaded_prob != uploaded_feat:
                    log.warn(
                        f"Upload mismatch: raw {len(uploaded_raw)}, "
                        f"probabilities {len(uploaded_prob)}, "
                        f"features {len(uploaded_feat)}"
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
                remove(local_prob, keep, remove_prob_files, remove_prob_archive)
                remove(local_feat, keep, remove_feat_files, remove_feat_archive)
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
