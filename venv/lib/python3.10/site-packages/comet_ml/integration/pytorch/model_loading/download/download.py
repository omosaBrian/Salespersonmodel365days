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

import pathlib

import comet_ml.api
import comet_ml.exceptions

from ... import constants
from ..uri import parse


def from_registry(MODEL_URI: str, dirpath: str) -> None:
    model_request = parse.registry(MODEL_URI)
    comet_ml.api.get_instance().download_registry_model(
        workspace=model_request.workspace,
        registry_name=model_request.registry,
        output_path=dirpath,
        version=model_request.version,
    )


def _get_experiment_api_by_key(key: str, comet_api: comet_ml.api.API):
    experiment_api = comet_api.get_experiment_by_key(key)

    if experiment_api is not None:
        return experiment_api

    exception_message = (
        "Couldn't find Experiment by key '{}', double-check the "
        "Experiment key and that you have the right API Key configured."
    ).format(key)

    raise comet_ml.exceptions.ExperimentNotFound(exception_message)


def _raise_model_not_found_in_experiment_by_key(model_name, experiment_key):
    exception_message = (
        "Experiment model {} cannot be found in Experiment with "
        "key {}, double-check the model name and experiment URI"
    ).format(model_name, experiment_key)

    raise comet_ml.exceptions.ModelNotFound(exception_message)


def _get_experiment_api_by_workspace(
    workspace, project_name, experiment_name, comet_api
):
    experiment_api = comet_api.get(
        workspace=workspace,
        project_name=project_name,
        experiment=experiment_name,
    )

    if experiment_api is not None:
        return experiment_api

    exception_message = (
        "Couldn't find Experiment by URI '{}/{}/{}', double-check "
        "the Experiment URI and that you have the right API Key configured."
    ).format(
        workspace,
        project_name,
        experiment_name,
    )

    raise comet_ml.exceptions.ExperimentNotFound(exception_message)


def _raise_model_not_found_in_experiment_by_workspace(
    model_name, workspace, project_name, experiment_name
):
    exception_message = (
        "Experiment model {} cannot be found in Experiment "
        "'{}/{}/{}', double-check the model name and experiment URI"
    ).format(
        model_name,
        workspace,
        project_name,
        experiment_name,
    )

    raise comet_ml.exceptions.ModelNotFound(exception_message)


def from_experiment_by_key(MODEL_URI: str, dirpath: str) -> None:
    model_request = parse.experiment_by_key(MODEL_URI)
    experiment_key = model_request.experiment_key
    model_name = model_request.model_name

    comet_api = comet_ml.api.get_instance()
    experiment_api = _get_experiment_api_by_key(experiment_key, comet_api)
    assets = experiment_api.get_model_asset_list(model_name=model_name)
    if len(assets) == 0:
        _raise_model_not_found_in_experiment_by_key(model_name, experiment_key)

    pathlib.Path(dirpath, constants.MODEL_DATA_DIRECTORY).mkdir(
        parents=True, exist_ok=True
    )

    for asset in assets:
        comet_api.download_experiment_asset(
            experiment_key=model_request.experiment_key,
            asset_id=asset["assetId"],
            output_path=pathlib.Path(dirpath, asset["fileName"]),
        )


def from_experiment_by_workspace(MODEL_URI: str, dirpath: str) -> None:
    model_request = parse.experiment_by_workspace(MODEL_URI)
    workspace = model_request.workspace
    project_name = model_request.project_name
    experiment_name = model_request.experiment_name
    model_name = model_request.model_name

    comet_api = comet_ml.api.get_instance()
    experiment_api = _get_experiment_api_by_workspace(
        workspace, project_name, experiment_name, comet_api
    )

    assets = experiment_api.get_model_asset_list(model_name=model_name)
    if len(assets) == 0:
        _raise_model_not_found_in_experiment_by_workspace(
            workspace, project_name, experiment_name, model_name
        )

    pathlib.Path(dirpath, constants.MODEL_DATA_DIRECTORY).mkdir(
        parents=True, exist_ok=True
    )

    for asset in assets:
        comet_api.download_experiment_asset(
            experiment_key=experiment_api.key,
            asset_id=asset["assetId"],
            output_path=pathlib.Path(dirpath, asset["fileName"]),
        )
