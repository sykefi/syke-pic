import tarfile
import zipfile
from pathlib import Path


def create_archive(directory, compression):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"'{directory}' does not exist")
    archive_name = None
    if compression in ("tar", "gzip", "tar.gz", "gz"):
        mode = "w" if compression == "tar" else "w:gz"
        suffix = ".tar" if compression == "tar" else ".tar.gz"
        archive_name = directory.with_suffix(suffix)
        with tarfile.open(archive_name, mode) as tar:
            for file in directory.iterdir():
                tar.add(file, arcname=file.name)
    elif compression == "zip":
        archive_name = directory.with_suffix(".zip")
        with zipfile.ZipFile(archive_name, "w", zipfile.ZIP_DEFLATED) as tar:
            for file in directory.iterdir():
                tar.write(file, arcname=file.name)
    else:
        raise ValueError(f"Unknown compression '{compression}")

    return archive_name
