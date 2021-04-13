import datetime
import logging
import os
from logging.config import dictConfig

import yaml


def setup(config_file=None):
    if config_file:
        with open(config_file) as fh:
            config = yaml.safe_load(fh.read())
            os.makedirs(
                os.path.dirname(config["handlers"]["file"]["filename"]), exist_ok=True
            )
            if "atTime" in config["handlers"]["file"]:
                config["handlers"]["file"]["atTime"] = datetime.time(
                    *config["handlers"]["file"]["atTime"]
                )
            dictConfig(config)
    else:
        logging.basicConfig(
            level=os.environ.get("LOGLEVEL", "INFO"),
            format="{asctime} - {name} - {levelname} - {message}",
            style="{",
        )
