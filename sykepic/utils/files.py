import tarfile
import zipfile
from pathlib import Path

from . import ifcb, logger

log = logger.get_logger("files")


def create_archive(src, dest, compression):
    src = Path(src)
    if not src.is_dir():
        raise ValueError(f"{src} does not exist")
    if compression in ("tar", "gzip", "tar.gz", "gz"):
        mode = "w" if compression == "tar" else "w:gz"
        with tarfile.open(dest, mode) as tar:
            for src_file in src.iterdir():
                tar.add(src_file, arcname=src_file.name)
    elif compression == "zip":
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as tar:
            for src_file in src.iterdir():
                tar.write(src_file, arcname=src_file.name)
    else:
        raise ValueError(f"Unknown compression {compression}")


def sample_csv_path(sample_path, out_dir, suffix=None):
    sample_path = Path(sample_path)
    sample = sample_path.name
    if suffix:
        out_name = sample + suffix + ".csv"
    else:
        out_name = sample + ".csv"
    csv_path = (
        Path(out_dir) / ifcb.sample_to_datetime(sample).strftime("%Y/%m/%d") / out_name
    )
    return csv_path


def list_sample_paths(root_dir, filter=None):
    path_gen = (roi.with_suffix("") for roi in Path(root_dir).glob("**/*.roi"))
    if filter is not None:
        path_gen = (path for path in path_gen if path.name in filter)
    return list(path_gen)
