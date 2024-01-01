"""Collection of utility functions."""

import importlib
import logging
import os
import os.path as op
import random
import sys
import time
from contextlib import contextmanager
from uuid import uuid4

import fsspec
import joblib
import numpy as np
import pandas as pd
import yaml
import ta_pet_id

from .base_utils import silence_common_warnings

logger = logging.getLogger(__name__)
pd.options.mode.use_inf_as_na = True


def import_python_file(py_file_path):
    mod_name, ext = op.splitext(op.basename(op.abspath(py_file_path)))
    if ext != ".py":
        raise ValueError("Invalid file extension : {ext}. Expected a py file")
    spec = importlib.util.spec_from_file_location(mod_name, py_file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def save_pipeline(pipeline, loc):
    """Save an sklearn pipeline in a location.

    Parameters
    ----------
    pipeline : sklearn.Pipeline
        Pipeline object to be saved
    loc : str
        Path string of the location where the pipeline has to be saved
    """
    logger.info(f"Saving pipeline to location {loc}")

    os.makedirs(op.dirname(loc), exist_ok=True)
    joblib.dump(pipeline, loc)


def create_job_id(context):
    """Create a unique id for a job.

    Parameters
    ----------
    context : ta_pet_id.core.context.Context

    Returns
    -------
    string
        unique string identifier
    """
    return f"job-{uuid4()}"


def initialize_environment(debug=True, hide_warnings=True):
    """Initialize the OS Environ with relevant values.

    Parameters
    ----------
    debug: bool, optional
        Whether to set TA_DEBUG to True of False in the environment, default=True
    hide_warnings: bool, optional
        True will hide warnings, default True
    """
    # FIXME: support config
    if debug:
        os.environ["TA_DEBUG"] = "True"
    else:
        os.environ["TA_DEBUG"] = "False"

    # force tigerml to raise an exception on failure
    os.environ["TA_ALLOW_EXCEPTIONS"] = "True"

    if hide_warnings:
        silence_common_warnings()


def is_debug_mode():
    """Check if the current environ is in debug mode."""
    debug_mode = os.environ.get("TA_DEBUG", "True")
    return debug_mode.upper() == "TRUE"


@contextmanager
def timed_log(msg):
    """Log the provided ``msg`` with the execution time."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        logging.info(f"{msg} : {end_time-start_time} seconds")


@contextmanager
def disable_logging(highest_level=logging.CRITICAL):
    """Disable all logs below ``highest_level``."""
    # NOTE: this is the attribute that seems to be modified
    # by the call to logging.disable. so we first save this
    # and reset it when exiting the context.
    orig_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(orig_level)
        logging.info("*" * 80)


@contextmanager
def silence_stdout():
    """Silence print stmts on the console unless in debug mode."""
    # debug mode. do nothing.
    if is_debug_mode():
        try:
            yield
        finally:
            return

    # not in debug mode. silence the output by writing to the null device.
    with open(os.devnull, "w") as fp:
        old_dunder_stdout = sys.__stdout__
        old_sys_stdout = sys.stdout
        # FIXME: This doesen't help with notebooks
        # with redirect_stdout(fp):
        #    yield
        try:
            sys.__stdout__ = fp
            sys.stdout = fp
            yield

        except Exception as e:
            sys.stderr.write("Error: {}".format(str(e)))
        finally:
            sys.__stdout__ = old_dunder_stdout
            sys.stdout = old_sys_stdout


def is_relative_path(path):
    """To check if `path` is a relative path or not.

    Parameters
    ----------
    path : str
        path string to be evaluated

    Returns
    -------
    bool
        True if input path is relative else False
    """
    npath = op.normpath(path)
    return op.abspath(npath) != npath


def get_package_path():
    """Get the path of the current installed ta_pet_id package.

    Returns
    -------
    str
        path string in the current system where the ta_pet_id package is loaded from
    """
    path = ta_pet_id.__path__
    return op.dirname(op.abspath(path[0]))


def get_package_version():
    """Return the version of the package."""
    return ta_pet_id.__version__


def get_data_dir_path():
    """Fetch the data directory path."""
    return op.join(get_package_path(), "..", "data")


def get_fs_and_abs_path(path, storage_options=None):
    """Get the Filesystem and paths from a urlpath and options.

    Parameters
    ----------
    path : string or iterable
        Absolute or relative filepath, URL (may include protocols like
        ``s3://``), or globstring pointing to data.
    storage_options : dict, optional
        Additional keywords to pass to the filesystem class.

    Returns
    -------
    fsspec.FileSystem
       Filesystem Object
    list(str)
        List of paths in the input path.
    """
    fs, _, paths = fsspec.core.get_fs_token_paths(path, storage_options=storage_options)
    if len(paths) == 1:
        return fs, paths[0]
    else:
        return fs, paths


def load_yml(path, *, fs=None, **kwargs):
    """Load a yml file from the input `path`.

    Parameters
    ----------
    path : string
        Absolute or relative filepath, URL (may include protocols like
        ``s3://``).
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Returns
    -------
    dict
        dictionery of the loaded yml file
    """
    fs = fs or fsspec.filesystem("file")
    with fs.open(path, mode="r") as fp:
        return yaml.safe_load(fp, **kwargs)


def create_yml(path, config, fs=None):
    """Dump a dictionary as yaml to output `path`.

    Parameters
    ----------
    path : string
        Absolute or relative filepath, URL (may include protocols like
        ``s3://``).
    config: dict
        config dictionary to be dumped as yaml file.
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``
    """
    fs = fs or fsspec.filesystem("file")
    with fs.open(path, "w") as out_file:
        yaml.safe_dump(config, out_file, default_flow_style=False)


def initialize_random_seed(seed):
    """Initialise random seed using the input ``seed``.

    Parameters
    ----------
    seed : int

    Returns
    -------
    int
        seed integer
    """
    logger.info(f"Initialized Random Seed : {seed}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    return seed


def get_fsspec_storage_options(resource_type, credentials):
    """Get storage options from the credentials based on the resource type.

    Parameters
    ----------
    resource_type : string
        'aws' or 'azure' or 'local' etc.,
    credentials : dict
        Dictionery of the credentials

    Returns
    -------
    dict
        Dictionary of the relevant storage options

    Raises
    ------
    NotImplementedError
        Raised for all resourcetype inputs other than 'aws'
    """
    if resource_type == "aws":
        return {
            "key": credentials["aws_access_key_id"],
            "secret": credentials["aws_secret_access_key"],
        }
    else:
        raise NotImplementedError(f"resource type: {resource_type}")

