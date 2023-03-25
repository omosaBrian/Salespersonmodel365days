# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.com
#  Copyright (C) 2015-2021 Comet ML INC
#  This file can not be copied and/or distributed without the express
#  permission of Comet ML Inc.
# *******************************************************

import contextlib
import logging
import os

import comet_ml
import comet_ml.api
import comet_ml.gpu_logging

import ray
from ray.air import session

LOGGER = logging.getLogger(__file__)


@contextlib.contextmanager
def comet_worker_logger(ray_config, api_key=None, **experiment_kwargs):
    """
    This context manager allows you to track resource usage from each distributed worker when
    running a distributed training job. It must be used in conjunction with
    `comet_ml.integration.ray.CometTrainLoggerCallback` callback.

    Args:
        ray_config: dict (required) ray configuration dictionary from ray driver node.
        api_key: str (optional), If not None it will be passed to ExistingExperiment.
            This argument has priority over api_key in ray_config dict and api key in environment.
        **experiment_kwargs: Other keyword arguments will be passed to the
            constructor for comet_ml.ExistingExperiment.

    Example:

    ```python
    def train_func(ray_config: Dict):
        with comet_worker_logger(ray_config) as experiment:
            # ray worker training code
    ```

    If some required information is missing (like the API Key) or something wrong happens, this will
    return a disabled Experiment, all methods calls will succeed but no data is gonna be logged.

    Returns: An Experiment object.
    """
    experiment = None
    try:
        _put_api_key_to_experiment_kwargs_if_possible(
            api_key, ray_config, experiment_kwargs
        )
        experiment_key = _get_experiment_key(ray_config)

        if experiment_key is None:
            experiment = comet_ml.OfflineExperiment(disabled=True)
            yield experiment
            return

        _setup_environment()

        try:
            experiment = comet_ml.ExistingExperiment(
                experiment_key=experiment_key,
                log_env_gpu=True,
                log_env_cpu=True,
                log_env_details=True,
                log_env_host=False,
                display_summary_level=0,
                **experiment_kwargs
            )
        except Exception:
            LOGGER.warning(
                "Internal error occured when creating experiment object."
                "\nReturning disabled experiment. Nothing will be logged.",
                exc_info=True,
            )
            experiment = comet_ml.OfflineExperiment(disabled=True)
        yield experiment

    finally:
        if experiment is not None:
            experiment.end()


def _get_api_key(config):
    if "_comet_api_key" in config:
        hidden_api_key = config["_comet_api_key"]
        return hidden_api_key.value

    return None


def _put_api_key_to_experiment_kwargs_if_possible(
    api_key, ray_config, experiment_kwargs
):
    if api_key is None:
        api_key = _get_api_key(ray_config)
        if api_key is not None:
            experiment_kwargs["api_key"] = api_key
    else:
        experiment_kwargs["api_key"] = api_key


def _get_experiment_key(config):
    if "_comet_experiment_key" in config:
        return config["_comet_experiment_key"]

    LOGGER.warning(
        "Experiment key wasn't found in RAY config."
        "Make sure you are using CometTrainLoggerCallback."
        "\nReturning disabled experiment. Nothing will be logged."
    )

    return None


def _setup_environment():
    os.environ["COMET_DISTRIBUTED_NODE_IDENTIFIER"] = str(session.get_world_rank())
    comet_ml.gpu_logging.set_devices_to_report(ray.get_gpu_ids())
