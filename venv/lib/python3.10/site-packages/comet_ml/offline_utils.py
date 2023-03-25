# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.ml
#  Copyright (C) 2015-2022 Comet ML INC
#  This file can not be copied and/or distributed without
#  the express permission of Comet ML Inc.
# *******************************************************
import logging
import os
import tempfile
from os.path import join
from zipfile import ZipFile

from comet_ml._typing import Optional, Tuple
from comet_ml.compat_utils import json_dump
from comet_ml.config import (
    DEFAULT_OFFLINE_DATA_DIRECTORY,
    OFFLINE_EXPERIMENT_JSON_FILE_NAME,
    Config,
)
from comet_ml.exceptions import InvalidOfflineDirectory
from comet_ml.logging_messages import (
    OFFLINE_DATA_DIR_DEFAULT_WARNING,
    OFFLINE_DATA_DIR_FAILED_WARNING,
)


def write_experiment_meta_file(
    tempdir,
    experiment_key,
    workspace,
    project_name,
    start_time,
    stop_time,
    tags,
    resume_strategy,
    customer_error_reported=False,
    customer_error_message=None,
):
    meta_file_path = join(tempdir, OFFLINE_EXPERIMENT_JSON_FILE_NAME)
    meta = {
        "offline_id": experiment_key,
        "project_name": project_name,
        "start_time": start_time,
        "stop_time": stop_time,
        "tags": tags,
        "workspace": workspace,
        "resume_strategy": resume_strategy,
        "customer_error_reported": customer_error_reported,
        "customer_error_message": customer_error_message,
    }
    with open(meta_file_path, "wb") as f:
        json_dump(meta, f)


def create_experiment_archive(
    offline_directory, offline_archive_file_name, data_dir, logger
):
    # type: (str, str, str, logging.Logger) -> Tuple[str, str]
    """Creates offline experiment archive by listing all files in the provided data_dir and archiving them
    into ZIP file.
    Args:
        offline_directory:
            The path to the offline directory.
        offline_archive_file_name:
            The file name for offline archive
        data_dir:
            The data directory with created experiment files.
        logger:
            The logger to be used to log messages.
    Returns:
        The path to the created ZIP file and actual path to the offline directory (it may change).
    """
    zip_file, offline_directory = create_offline_archive(
        offline_directory=offline_directory,
        offline_archive_file_name=offline_archive_file_name,
        fallback_to_temp=True,
        logger=logger,
    )

    for file in os.listdir(data_dir):
        zip_file.write(os.path.join(data_dir, file), file)

    zip_file.close()

    return zip_file.filename, offline_directory


def create_offline_archive(
    offline_directory, offline_archive_file_name, fallback_to_temp, logger
):
    # type: (str, str, bool, logging.Logger) -> Tuple[ZipFile, str]
    """Attempts to create ZIP file in the offline_directory. If attempt failed it will try to repeat in the
    temporary directory if fallback_to_temp is set. Otherwise, InvalidOfflineDirectory raised.
    Args:
        offline_directory:
            The path to the offline directory.
        offline_archive_file_name:
            The file name for offline archive
        fallback_to_temp:
            The flag to indicate whether automatic recovery should be performed in case if failed to create archive
            in the current offline_directory.
        logger:
            The logger to be used to log messages.
    Returns:
        The ZipFile for offline archive and the path to the offline directory where archive was created
    """
    # Add a random string to offline experiment archives to avoid conflict when using ExistingExperiment
    try:
        # create parent directory
        if not os.path.exists(offline_directory):
            os.mkdir(offline_directory, 0o700)

        # Try to create the archive now
        return (
            _create_zip_file(offline_directory, offline_archive_file_name),
            offline_directory,
        )
    except (OSError, IOError) as exc:
        if fallback_to_temp:
            # failed - use temporary directory instead
            offline_dir = tempfile.mkdtemp()
            logger.warning(
                OFFLINE_DATA_DIR_FAILED_WARNING,
                os.path.abspath(offline_directory),
                os.path.abspath(offline_dir),
                str(exc),
                exc_info=True,
            )
            return (
                _create_zip_file(offline_dir, offline_archive_file_name),
                offline_dir,
            )
        else:
            raise InvalidOfflineDirectory(offline_directory, str(exc))


def get_offline_data_dir_path(comet_config, logger, offline_directory=None):
    # type: (Config, logging.Logger, Optional[str]) -> Tuple[str, bool]
    """Looks for the offline data directory path. If offline_directory parameter is not None it will be returned.
    Otherwise, the experiment configuration settings will be used. Finally, if later failed the default data
    directory will be used.
    Args:
        logger:
            The logger to be used to log messages.
        comet_config:
            The Comet ML configuration options.
        offline_directory:
            The provided by user path to the offline directory or None if not set.
    Returns:
        The calculated path to the offline directory and flag to indicate if default data directory suffix was used.
    """
    if offline_directory is None:
        offline_directory = comet_config.get_string(None, "comet.offline_directory")

    if (
        offline_directory is not None
        and offline_directory != DEFAULT_OFFLINE_DATA_DIRECTORY
    ):
        return offline_directory, False

    # use DEFAULT_OFFLINE_DATA_DIRECTORY in the CWD
    offline_directory = join(os.getcwd(), DEFAULT_OFFLINE_DATA_DIRECTORY)

    # display warning to the user that offline directory was chosen with default path
    logger.info(OFFLINE_DATA_DIR_DEFAULT_WARNING, offline_directory)

    return offline_directory, True


def _create_zip_file(directory, name):
    file_path = _offline_zip_path(directory, name)
    return ZipFile(file_path, "w", allowZip64=True)


def _offline_zip_path(directory, name):
    return os.path.join(directory, name)
