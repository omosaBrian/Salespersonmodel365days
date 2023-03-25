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

import box

from . import flatten_dictionary


class TrialResultLogger:
    _FIELD_CATEGORIES = {
        "not_logged": ("done", "should_checkpoint"),
        "system": ("node_ip", "hostname", "pid", "date"),
        "episode": ("hist_stats/episode_reward", "hist_stats/episode_lengths"),
        "other": ("trial_id", "experiment_id", "experiment_tag"),
    }

    def __init__(self, experiment, result):
        self._experiment = experiment
        self._result = result
        self._step = result["training_iteration"]

    def process(self):
        self._log_parameters()
        grouped = self._group_by_category()
        self._log_grouped_data(grouped)

    def _log_parameters(self):
        config_update = self._result.pop("config", {}).copy()
        config_update.pop("callbacks", None)  # Remove callbacks
        for key, value in config_update.items():
            if isinstance(value, dict):
                self._experiment.log_parameters(
                    flatten_dictionary.flatten({key: value}, "/"), step=self._step
                )
            else:
                self._experiment.log_parameter(key, value, step=self._step)

    def _group_by_category(self):
        flattened = flatten_dictionary.flatten(self._result, delimiter="/")
        result = {category: {} for category in ["metric", "other", "system", "episode"]}

        for key, value in flattened.items():
            if value is None:
                continue
            category = self._category(key)
            if category == "not_logged":
                continue
            group = result[category]
            group[key] = value

        return box.Box(result)

    def _is_valid_key_name(self, key, field):
        return key.startswith(field + "/") or key == field  # noqa: E731

    def _category(self, key):
        for category in "not_logged", "other", "system", "episode":
            if self._in_category(key, category):
                return category

        return "metric"

    def _in_category(self, key, category):
        field_names = self._FIELD_CATEGORIES[category]
        return any(self._is_valid_key_name(key, field) for field in field_names)

    def _log_grouped_data(self, grouped_data):
        experiment = self._experiment
        experiment.log_others(grouped_data.other)
        experiment.log_metrics(grouped_data.metric, step=self._step)

        for key, value in grouped_data.system.items():
            experiment.log_system_info(key, value)

        for key, value in grouped_data.episode.items():
            experiment.log_curve(key, x=range(len(value)), y=value, step=self._step)
