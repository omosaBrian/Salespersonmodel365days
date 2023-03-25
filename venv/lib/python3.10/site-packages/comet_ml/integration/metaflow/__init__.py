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

from metaflow.plugins.cards import card_decorator

from . import comet_decorator, single_card_list

__all__ = ["comet_flow", "comet_skip"]

COMET_SKIP_DECORATOR = "comet_skip"


class CometFlow:
    def __init__(self, workspace=None, project_name=None):
        self._workspace = workspace
        self._project_name = project_name
        command_line = " ".join(sys.argv)
        self._card_on_command_line = "--with card" in command_line

    def __call__(self, flow_spec_class):
        for step in self._get_steps(flow_spec_class):
            if self._card_on_command_line:
                self._decorate_with_card(step)
            self._decorate_with_comet(step)
        return flow_spec_class

    def _decorate_with_card(self, step):
        step.decorators = single_card_list.SingleCardList(step.decorators)
        step.decorators.append(card_decorator.CardDecorator())

    def _get_steps(self, cls):
        all_methods = [
            getattr(cls, name) for name in cls.__dict__ if callable(getattr(cls, name))
        ]
        steps = [
            method
            for method in all_methods
            if hasattr(method, "is_step") and method.is_step
        ]

        return steps

    def _decorate_with_comet(self, step):
        decorator = comet_decorator.CometDecorator(
            workspace=self._workspace, comet_project=self._project_name
        )
        if COMET_SKIP_DECORATOR in step.decorators:
            decorator.skip = True
            step.decorators.remove(COMET_SKIP_DECORATOR)
        step.decorators.append(decorator)


def comet_skip(metaflow_step):
    """
    Add this decorator to prevents the step from being tracked by comet. This is useful for join
    steps or foreach step with a high number of iterations. Order is important - it must be placed
    above the @step decorator.

    [Read the documentation for more details](/docs/v2/integrations/third-party-tools/metaflow/#skip-metaflow-steps).

    Example:

    ```python
    @comet_flow
    class HelloFlow(FlowSpec):
        @comet_skip
        @step
        def start(self):
            ...
    ```
    """
    metaflow_step.decorators.append(COMET_SKIP_DECORATOR)
    return metaflow_step


def comet_flow(func=None, **kwargs):
    """
    This decorator placed before your Flow class allows you to track
    each step of your flow in a separate comet experiment.

    Args:
        project_name: str (optional), send your experiment to a specific project.
            Otherwise will be sent to `Uncategorized Experiments`.
            If project name does not already exists Comet.ml will create a new project.
        workspace: str (optional), attach an experiment to a project that belongs to this workspace

    Example:

    ```python
    @comet_flow(project_name="comet-example-metaflow-hello-world")
    class HelloFlow(FlowSpec):
        @step
        def start(self):
            ...
    ```
    """
    if func is None:
        return CometFlow(**kwargs)
    return CometFlow().__call__(func)
