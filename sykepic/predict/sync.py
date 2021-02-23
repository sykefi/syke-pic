"""Module for synchronising local data with Allas"""

import datetime
import shutil
import zipfile
from sys import stderr
from pathlib import Path

import boto3

from . import allas
from . import ifcb


def main(args):
    if args.upload:
        upload(args)
    else:
        download(args)


def download(args):
    s3 = boto3.resource('s3', endpoint_url='https://a3s.fi')
    num_downloads = 0
    for file in list(allas.ls(args.allas, s3=s3)):
        date = ifcb.sample_to_datetime(file)
        day_path = date.strftime('%Y/%m/%d')
        save_dir = Path(args.local)/day_path
        # Check if file has already been downloaded
        if (save_dir/file).is_file() or save_dir.with_suffix('.zip').is_file():
            continue
        if not save_dir.is_dir():
            save_dir.mkdir(parents=True)
        try:
            num_downloads += 1
            allas.download(Path(args.allas)/file, save_dir, s3)
        except Exception as e:
            print(f'[ERROR] {file}: {e}', file=stderr)
            continue
    print(f'[INFO] Downloaded {num_downloads} new files')


def upload(args):
    local = Path(args.local)
    s3 = boto3.resource('s3', endpoint_url='https://a3s.fi')
    today = datetime.date.today()
    # today = datetime.date(2018, 7, 4)
    # Find all directories corresponding to one day
    day_dirs = []
    # find_head_dirs() modifies day_dirs in place
    find_head_dirs(local, day_dirs)
    for day_dir in day_dirs:
        try:
            year, month, day = day_dir.parts[-3:]
            date = datetime.date(int(year), int(month), int(day))
        except Exception:
            print(
                f'[ERROR] {day_dir} is not a valid day directory', file=stderr)
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
        # Removing local copies
        if (date - today).days() > args.keep:
            if args.remove:
                shutil.rmtree(day_dir)
                print(f'[INFO] Removed {day_dir}')
                if args.remove > 1:
                    # Remove local zip file
                    local_zip.unlink()
                    # Remove month directory if it's empty
                    try:
                        day_dir.parent.rmdir()
                    except OSError:
                        pass
                    print(f'[INFO] Removed {local_zip}')
            if args.clean_allas:
                print(f'[INFO] Cleaning {args.clean_allas} in Allas')
                for file in zipped_files:
                    allas.delete(f'{args.clean_allas}/{file.name}', s3)


def find_head_dirs(path, head_dirs=[]):
    sub_dirs = [p for p in sorted(path.iterdir()) if p.is_dir()]
    if sub_dirs:
        for sub_dir in sub_dirs:
            find_head_dirs(sub_dir, head_dirs)
    else:
        head_dirs.append(path)
