# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2021 Comet ML INC
#  This file can not be copied and/or distributed without
#  the express permission of Comet ML Inc.
# *******************************************************

import logging
from typing import Any, Dict, List

import comet_ml
from comet_ml.dataclasses import hidden_api_key

import ray.tune.experiment
import ray.tune.logger

from . import trial_result_logger, trial_save_logger

Trial = ray.tune.experiment.Trial

LOGGER = logging.getLogger(__name__)


class CometTrainLoggerCallback(ray.tune.logger.LoggerCallback):
    """
    Ray Callback for logging Train results to Comet.

    This Ray Train `LoggerCallback` sends metrics and parameters to
    Comet for tracking.

    This callback is based on the Ray native Comet callback and has been modified to allow to track
    resource usage on all distributed workers when running a distributed training job. It cannot be
    used with Ray Tune.

    Args:
        ray_config: dict (required), ray configuration dictionary to share with workers.
            It must be the same dictionary instance, not a copy.
        tags: list of string (optional), tags to add to the logged Experiment.
            Defaults to None.
        save_checkpoints: boolean (optional), if ``True``, model checkpoints will be saved to
            Comet ML as artifacts. Defaults to ``False``.
        share_api_key_to_workers: boolean (optional), if ``True``, Comet API key will be shared
            with workers via ray_config dictionary. This is an unsafe solution and we recommend you
            uses a [more secure way to set up your API Key in your
            cluster](/docs/v2/guides/tracking-ml-training/distributed-training/).
        experiment_kwargs: Other keyword arguments will be passed to the
            constructor for comet_ml.Experiment.

    Example:

    ```python
    config = {"lr": 1e-3, "batch_size": 64, "epochs": 20}

    comet_callback = CometTrainLoggerCallback(
        config,
        tags=["torch_ray_callback"],
        save_checkpoints=True,
        share_api_key_to_workers=True,
    )

    trainer = TorchTrainer(
        train_func,
        train_loop_config=config,
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        run_config=RunConfig(callbacks=[comet_callback]),
    )
    result = trainer.fit()
    ```

    Return: None
    """

    def __init__(
        self,
        ray_config: Dict[str, Any],
        tags: List[str] = None,
        save_checkpoints: bool = False,
        share_api_key_to_workers: bool = False,
        **experiment_kwargs  # fmt: skip
    ):
        self._save_checkpoints = save_checkpoints
        self._trial = None
        self._setup_shared_experiment(tags, **experiment_kwargs)
        self._push_info_into_ray_configuration(ray_config, share_api_key_to_workers)

        if share_api_key_to_workers:
            LOGGER.warning(
                "Setting CometTrainLoggerCallback(share_api_key_to_workers=True) is insecure. Use a more secure method. "
                "Check https://www.comet.com/docs/v2/guides/tracking-ml-training/distributed-training/ for more info."
            )

    @property
    def experiment_key(self):
        return self._experiment_key

    def _setup_shared_experiment(self, tags, **experiment_kwargs):
        experiment = comet_ml.Experiment(
            log_env_gpu=False,
            log_env_cpu=False,
            log_env_details=True,
            log_env_host=True,
            display_summary_level=0,
            **experiment_kwargs  # fmt: skip
        )
        if tags is not None:
            experiment.add_tags(tags)
        experiment.log_other("Created from", "Ray")

        self._experiment_key = experiment.id
        self._api_key = experiment.api_key

        experiment.end()

    def _push_info_into_ray_configuration(self, config, share_api_key_to_workers):
        config["_comet_experiment_key"] = self._experiment_key
        if share_api_key_to_workers:
            config["_comet_api_key"] = hidden_api_key.HiddenApiKey(value=self._api_key)

    def log_trial_start(self, trial: Trial):
        if self._trial is not None:
            raise Exception(
                "CometTrainLoggerCallback has been already started. Only one start is allowed "
            )

        self._trial = trial
        self._setup_existing_shared_experiment(trial)

    def _setup_existing_shared_experiment(self, trial):
        experiment = comet_ml.ExistingExperiment(
            previous_experiment=self._experiment_key,
            api_key=self._api_key,
            log_env_gpu=False,
            log_env_cpu=False,
            log_env_details=False,
            log_env_host=False,
            display_summary_level=0,
        )
        experiment.set_name(str(trial))

        config = trial.config.copy()
        config.pop("callbacks", None)
        if len(config) > 0:
            experiment.log_parameters(config)

        self._experiment = experiment

    def log_trial_result(self, iteration: int, trial: Trial, result: Dict):
        if self._trial is None:
            self.log_trial_start(trial)

        if self._trial is not trial:
            raise Exception("Only one trial is allowed for CometTrainLoggerCallback")

        result_logger = trial_result_logger.TrialResultLogger(self._experiment, result)
        result_logger.process()

    def log_trial_save(self, trial: Trial):
        if self._save_checkpoints and trial.checkpoint is not None:
            trial_save_logger.go(self._experiment, trial)

    def log_trial_end(self, trial: Trial, failed: bool = False):
        self._experiment.end()
