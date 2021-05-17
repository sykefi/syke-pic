import tarfile
import zipfile
from pathlib import Path


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
