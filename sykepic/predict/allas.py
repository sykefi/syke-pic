"""Helper functions for accessing Allas."""

from pathlib import Path

import boto3


def delete(path, s3=None):
    if not s3:
        s3 = boto3.resource('s3', endpoint_url='https://a3s.fi')
    bucket, *target = Path(path).parts
    bucket = s3.Bucket(bucket)
    target = '/'.join(target)
    target = bucket.Object(target)
    target.delete()


def download(path, dest_dir, s3=None):
    if not s3:
        s3 = boto3.resource('s3', endpoint_url='https://a3s.fi')
    bucket, *target = Path(path).parts
    bucket = s3.Bucket(bucket)
    target = '/'.join(target)
    dest_dir = Path(dest_dir)
    dest = str(dest_dir/Path(target).name)
    bucket.download_file(target, dest)


def ls(path, extension=None, s3=None):
    if not s3:
        s3 = boto3.resource('s3', endpoint_url='https://a3s.fi')
    if isinstance(extension, str):
        extension = [extension]
    bucket, *sub_dir = Path(path).parts
    bucket = s3.Bucket(bucket)
    sub_dir = '/'.join(sub_dir)
    if sub_dir:
        objects = bucket.objects.filter(Prefix=sub_dir)
    else:
        objects = bucket.objects.all()
    for obj in objects:
        # Don't include directory objects
        if obj.key.endswith('/'):
            continue
        if extension and not Path(obj.key).suffix in extension:
            continue
        yield obj.key


def upload(src, path, s3=None):
    if not s3:
        s3 = boto3.resource('s3', endpoint_url='https://a3s.fi')
    bucket, *sub_path = Path(path).parts
    bucket = s3.Bucket(bucket)
    dest = '/'.join(sub_path)
    bucket.upload_file(str(src), dest)
