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

import hashlib
import json
import logging

import comet_ml
import comet_ml.integration.sanitation.metaflow as comet_metaflow
import comet_ml.messages

import metaflow

from ..._typing import Dict, List
from .. import KEY_CREATED_FROM, KEY_PIPELINE_TYPE
from . import logging_messages

LOGGER = logging.getLogger(__name__)

KEY_METAFLOW_STATUS = "metaflow_status"
KEY_METAFLOW_NAME = "metaflow_flow_name"
KEY_METAFLOW_RUN_ID = "metaflow_run_id"
KEY_METAFLOW_GRAPH_FILE = "metaflow_graph_file"
KEY_METAFLOW_STEP_NAME = "metaflow_step_name"
KEY_METAFLOW_RUN_EXPERIMENT = "metaflow_run_experiment"
KEY_METAFLOW_RUNTIME = "metaflow_runtime"
KEY_METAFLOW_ORIGIN_RUN_ID = "metaflow_origin_run_id"

KEY_COMET_RUN_ID = "comet_run_id"
KEY_COMET_STEP_ID = "comet_step_id"
KEY_COMET_TASK_ID = "comet_task_id"

VALUE_METAFLOW = "metaflow"

STATUS_RUNNING = "Running"
STATUS_COMPLETED = "Completed"
STATUS_FAILED = "Failed"


class CometDecorator(metaflow.decorators.StepDecorator):
    """
    Creates a decorator, which logs information about the Run and all the tasks to Comet ML.

    Parameters
    ----------
    workspace : str
        the name of the comet workspace.
    comet_project : str
        the name of the comet project.
    """

    name = "comet"

    def __init__(self, skip=False, workspace=None, comet_project=None, *args, **kwargs):
        kwargs["statically_defined"] = True
        super(CometDecorator, self).__init__(*args, **kwargs)
        self.comet_experiment = None
        self.run_experiment = None
        self.comet_exception = None
        self.experiment_key = None
        self.workspace = workspace
        self.comet_project = comet_project
        self.current = metaflow.current
        self._skip = skip

    @property
    def skip(self):
        return self._skip

    @skip.setter
    def skip(self, value):
        self._skip = value

    def _get_params_from_flow(self, flow, parameter_names):
        # type: (metaflow.Flow, List[str]) -> Dict

        return {p: getattr(flow, p) for p in parameter_names}

    def _log_flow_parameters(self, flow, experiment):
        # type: (metaflow.Flow, comet_ml.Experiment) -> None

        params_dict = self._get_params_from_flow(flow, self.current.parameter_names)
        if len(params_dict) > 0:
            experiment._log_parameters(
                params_dict, source=comet_ml.messages.ParameterMessage.source_autologger
            )

    def _get_run_experiment_key(self):
        # type: () -> str

        return hashlib.sha1(self.current.run_id.encode("utf-8")).hexdigest()

    def _generate_card_file_name(self, card_number):
        return "metaflow-card-{step_name}-{card_number}.html".format(
            step_name=self.current.step_name, card_number=card_number
        )

    def _log_card_html(self):
        try:
            cards = metaflow.cards.get_cards(self.current.pathspec)
            for number, card in enumerate(cards):
                self.comet_experiment._log_asset_data(
                    data=card.get(),
                    file_name=self._generate_card_file_name(number),
                    asset_type="metaflow-card",
                )
        except Exception:
            LOGGER.debug(
                "Failed to log card html asset to the experiment", exc_info=True
            )

    def _handle_project_decorator_name(self):
        if self.comet_project is not None:
            return

        self.comet_project = self.current.project_flow_name

    def _log_project_decorator_data(self, experiment):
        project_others = {
            "metaflow_project_name": self.current.project_name,
            "metaflow_branch_name": self.current.branch_name,
            "metaflow_is_user_branch": self.current.is_user_branch,
            "metaflow_is_production": self.current.is_production,
            "metaflow_project_flow_name": self.current.project_flow_name,
        }

        experiment.log_others(project_others)

    def _check_if_project_decorator_was_applied(self):
        return hasattr(self.current, "project_flow_name")

    def _create_experiment(self, **kwargs):
        if self._check_if_project_decorator_was_applied():
            self._handle_project_decorator_name()
            experiment = comet_ml.Experiment(
                workspace=self.workspace,
                project_name=self.comet_project,
                **kwargs,
            )
            self._log_project_decorator_data(experiment)
            return experiment
        return comet_ml.Experiment(
            workspace=self.workspace,
            project_name=self.comet_project,
            **kwargs,
        )

    def _create_task_experiment(self, **kwargs):
        if self.skip:
            return self._create_experiment(disabled=True, **kwargs)

        return self._create_experiment(**kwargs)

    def _log_run_experiment(self, flow, metaflow_run):
        # type: (metaflow.Flow, metaflow.Run) -> None

        env = {
            "current_flow_name": self.current.flow_name,
            "current_run_id": self.current.run_id,
        }
        run_experiment_name = "flow - {current_flow_name} - {current_run_id}".format(
            **env
        )

        self.run_experiment.set_name(run_experiment_name)
        run_graph_filename = "{current_flow_name}-{current_run_id}-graph.json".format(
            **env
        )
        run_others = {
            KEY_METAFLOW_NAME: self.current.flow_name,
            KEY_METAFLOW_RUN_ID: self.current.run_id,
            KEY_COMET_RUN_ID: self.current.run_id,
            KEY_METAFLOW_STATUS: STATUS_RUNNING,
            KEY_METAFLOW_GRAPH_FILE: run_graph_filename,
            KEY_PIPELINE_TYPE: VALUE_METAFLOW,
            KEY_CREATED_FROM: VALUE_METAFLOW,
            KEY_METAFLOW_RUNTIME: _find_runtime_name(metaflow_run.tags),
        }
        run_tags = sorted(metaflow_run.tags - metaflow_run.system_tags)
        clean_metaflow_graph = comet_metaflow.sanitize_pipeline_environment(
            self.current.graph
        )
        run_graph = json.dumps(clean_metaflow_graph)

        self.run_experiment.log_asset_data(
            data=run_graph, name=run_graph_filename, overwrite=False
        )

        self.run_experiment.log_others(run_others)
        if len(run_tags) != 0:
            self.run_experiment.add_tags(run_tags)
        self._log_flow_parameters(flow, self.run_experiment)

    def _log_task_experiment(self, step_name, flow, metaflow_task):
        # type: (str, metaflow.Flow, metaflow.Task) -> None

        env = {
            "current_task_id": self.current.task_id,
            "current_step_name": step_name,
            "current_flow_name": self.current.flow_name,
            "current_run_id": self.current.run_id,
        }
        task_exp_name = "{current_task_id}/{current_step_name} - {current_flow_name} - {current_run_id}".format(
            **env
        )
        self.comet_experiment.set_name(task_exp_name)
        task_others = {
            KEY_COMET_RUN_ID: self.current.run_id,
            KEY_COMET_STEP_ID: "{current_run_id}/{current_step_name}".format(**env),
            KEY_COMET_TASK_ID: "{current_run_id}/{current_step_name}/{current_task_id}".format(
                **env
            ),
            KEY_METAFLOW_NAME: self.current.flow_name,
            KEY_METAFLOW_RUN_ID: self.current.run_id,
            KEY_METAFLOW_STEP_NAME: step_name,
            KEY_METAFLOW_STATUS: STATUS_RUNNING,
            KEY_METAFLOW_RUN_EXPERIMENT: self._get_run_experiment_key(),
            KEY_PIPELINE_TYPE: VALUE_METAFLOW,
            KEY_CREATED_FROM: VALUE_METAFLOW,
            KEY_METAFLOW_RUNTIME: metaflow_task.runtime_name,
        }
        if self.current.origin_run_id is not None:
            task_others.update({KEY_METAFLOW_ORIGIN_RUN_ID: self.current.origin_run_id})

        task_tags = [step_name] + sorted(metaflow_task.tags - metaflow_task.system_tags)

        self.comet_experiment.log_others(task_others)
        self._log_flow_parameters(flow, self.comet_experiment)
        if len(task_tags) != 0:
            self.comet_experiment.add_tags(task_tags)

    def _end_experiment(self, experiment):
        # type: (comet_ml.Experiment) -> None
        if self.comet_exception is None:
            experiment.log_other(KEY_METAFLOW_STATUS, STATUS_COMPLETED)
        else:
            experiment.log_other(KEY_METAFLOW_STATUS, STATUS_FAILED)

        experiment.end()

    def _finish_experiments(self):
        self._end_experiment(self.comet_experiment)

        # Handle Run-level experiment at the end of the Flow
        if self.current.step_name == "end" or self.comet_exception is not None:
            run_experiment = comet_ml.ExistingExperiment(
                experiment_key=self._get_run_experiment_key(),
                workspace=self.workspace,
                project_name=self.comet_project,
                # Disable summary for the run experiment or each step will show the run experiment summary
                display_summary_level=0,
                parse_args=False,
            )

            self._end_experiment(run_experiment)

    def task_pre_step(
        self,
        step_name,
        task_datastore,
        metadata,
        run_id,
        task_id,
        flow,
        graph,
        retry_count,
        max_user_code_retries,
        ubf_context,
        inputs,
    ):
        if step_name == "start":
            self.run_experiment = self._create_experiment(
                experiment_key=self._get_run_experiment_key(),
                display_summary_level=0,
                parse_args=False,
            )
            LOGGER.info(
                logging_messages.COMET_PROJECT_LINK_ON_START,
                self.run_experiment.focus_link,
            )
            run_meta = metaflow.Run(
                "{}/{}".format(self.current.flow_name, self.current.run_id)
            )

            self._log_run_experiment(
                flow=flow,
                metaflow_run=run_meta,
            )
            self.run_experiment.end()

        self.comet_experiment = self._create_task_experiment(
            parse_args=False,
        )

        setattr(flow, "comet_experiment", self.comet_experiment)
        setattr(flow, "run_comet_experiment_key", self._get_run_experiment_key())

        task_meta = metaflow.Task(self.current.pathspec)

        self._log_task_experiment(
            step_name=step_name,
            flow=flow,
            metaflow_task=task_meta,
        )

    def _clear_task_from_comet_attributes(self, flow):
        if hasattr(flow, "comet_experiment"):
            delattr(flow, "comet_experiment")
        if hasattr(flow, "run_comet_experiment_key"):
            delattr(flow, "run_comet_experiment_key")

    def task_post_step(
        self, step_name, flow, graph, retry_count, max_user_code_retries
    ):
        self._clear_task_from_comet_attributes(flow)

    def task_exception(
        self, exception, step_name, flow, graph, retry_count, max_user_code_retries
    ):
        self._clear_task_from_comet_attributes(flow)

    def task_finished(
        self, step_name, flow, graph, is_task_ok, retry_count, max_user_code_retries
    ):
        if hasattr(self.current, "card"):
            self._log_card_html()
            self.comet_experiment.log_other("has_metaflow_cards", 1)

        try:
            self._finish_experiments()
        except Exception:
            LOGGER.debug("Error cleaning up the experiments", exc_info=True)


def _find_runtime_name(tags):
    for tag in tags:
        if tag.startswith("runtime:"):
            return tag.split(":")[1]

    return None
