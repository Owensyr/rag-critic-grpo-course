"""Compatibility shim: upstream includes only a Python 3.10 log_utils.pyc."""

import logging


def get_logger(name):
    return logging.getLogger(name)
