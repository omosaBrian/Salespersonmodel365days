# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2022 Comet ML INC
#  This file can not be copied and/or distributed without
#  the express permission of Comet ML Inc.
# *******************************************************

import sys
import typing

import comet_ml

import spacy.language
import spacy.training.loggers
import spacy.util


def comet_logger_v1(
    project_name: str = None,
    workspace: str = None,
    run_name: typing.Optional[str] = None,
    tags: typing.Optional[typing.List[str]] = None,
    remove_config_values: typing.List[str] = [],
):
    console = spacy.training.loggers.console_logger(progress_bar=False)
    return SetupLogger(
        project_name, workspace, run_name, tags, remove_config_values, console
    )


class SetupLogger:
    def __init__(
        self, project_name, workspace, run_name, tags, remove_config_values, console
    ):
        self._project_name = project_name
        self._workspace = workspace
        self._run_name = run_name
        self._tags = tags
        self._remove_config_values = remove_config_values
        self._console = console
        self._experiment = None
        self._saved_model_path = None
        self._config_dot_batches = None
        self._console_log_step = None
        self._console_finalize = None

    def __call__(
        self,
        nlp: spacy.language.Language,
        stdout: typing.IO = sys.stdout,
        stderr: typing.IO = sys.stderr,
    ):
        self._init_config_console(nlp, stdout, stderr)
        self._create_experiment()
        self._log_params()

        return self.log_step, self.finalize

    def _init_config_console(self, nlp, stdout, stderr):
        config = nlp.config.interpolate()
        config_dot = spacy.util.dict_to_dot(config)
        for field in self._remove_config_values:
            config_dot.pop(field, None)
        config = spacy.util.dot_to_dict(config_dot)
        config_dot_items = list(config_dot.items())
        self._config_dot_batches = [
            config_dot_items[i : i + 100] for i in range(0, len(config_dot_items), 100)
        ]
        self._console_log_step, self._console_finalize = self._console(
            nlp, stdout, stderr
        )

    def _create_experiment(self):
        self._experiment = comet_ml.Experiment(
            project_name=self._project_name,
            workspace=self._workspace,
        )
        self._experiment.set_name(self._run_name)
        self._experiment.add_tags(self._tags)
        self._experiment.log_other("Created from", "spaCy")

    def _log_params(self):
        for batch in self._config_dot_batches:
            self._experiment.log_parameters(
                {key.replace("@", ""): value for key, value in batch}
            )

    def log_step(self, info):
        self._console_log_step(info)
        if info is None:
            return
        self._log_score(info)
        self._log_losses(info)
        self._log_other_scores(info)
        self._saved_model_path = info.get("output_path", None)

    def _log_score(self, info):
        if info["score"] is None:
            return
        self._experiment.log_metric(
            "score", info["score"], step=info["step"], epoch=info["epoch"]
        )

    def _log_losses(self, info):
        if not info["losses"]:
            return
        self._experiment.log_metrics(
            {"loss_{}".format(key): value for key, value in info["losses"].items()}
        )

    def _log_other_scores(self, info):
        if not isinstance(info["other_scores"], dict):
            return
        other_score_dot = spacy.util.dict_to_dot(info["other_scores"])
        self._experiment.log_metrics(
            {
                key: value
                for key, value in other_score_dot.items()
                if isinstance(value, float) or isinstance(value, int)
            },
            step=info["step"],
            epoch=info["epoch"],
        )

    def finalize(self):
        self._console_finalize()
        saved_model_path = self._saved_model_path.replace("last", "best")
        self._experiment.log_model("best", saved_model_path)
        self._experiment.end()
