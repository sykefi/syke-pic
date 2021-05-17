import logging
import os
from logging.config import dictConfig
from pathlib import Path

import yaml

SETUP_RAN = False

logging.getLogger('s3transfer').setLevel(logging.CRITICAL)

def get_logger(name):
    global SETUP_RAN
    if not SETUP_RAN:
        setup()
        SETUP_RAN = True
    return logging.getLogger(name)


def setup(config_file=None):
    if config_file:
        with open(config_file) as fh:
            config = yaml.safe_load(fh.read())
            log_dir = Path(config["handlers"]["file"]["filename"]).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            dictConfig(config)
    else:
        logging.basicConfig(
            level=os.environ.get("LOGLEVEL", "INFO"),
            format="{asctime} - {name} - {levelname} - {message}",
            style="{",
        )
