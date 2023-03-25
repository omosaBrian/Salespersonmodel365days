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
import enum

from . import request_types, scheme


class RequestTypes(enum.Enum):
    UNDEFINED = 0
    REGISTRY = 1
    EXPERIMENT_BY_KEY = 2
    EXPERIMENT_BY_WORKSPACE = 3
    FILE = 4


def registry(uri: str) -> request_types.Registry:
    uri_components = uri.split(sep="/")
    workspace = uri_components[-2]

    if ":" in uri_components[-1]:
        model_info_components = uri_components[-1].split(sep=":")
        registry_name = model_info_components[0]
        model_version = model_info_components[1]
        return request_types.Registry(workspace, registry_name, model_version)

    registry_name = uri_components[-1]

    return request_types.Registry(workspace, registry_name, None)


def filepath(uri: str) -> str:
    return uri.replace("file://", "")


def experiment_by_key(uri: str) -> request_types.ExperimentByKey:
    uri_components = uri.split(sep="/")
    model_name = uri_components[-1]
    experiment_key = uri_components[-2]

    return request_types.ExperimentByKey(experiment_key, model_name)


def experiment_by_workspace(
    uri: str,
) -> request_types.ExperimentByWorkspace:
    uri_components = uri.split(sep="/")
    model_name = uri_components[-1]
    experiment_name = uri_components[-2]
    project_name = uri_components[-3]
    workspace = uri_components[-4]

    return request_types.ExperimentByWorkspace(
        workspace, project_name, experiment_name, model_name
    )


def request_type(uri: str) -> RequestTypes.REGISTRY:
    if scheme.is_registry(uri):
        return RequestTypes.REGISTRY
    elif scheme.is_experiment_by_key(uri):
        return RequestTypes.EXPERIMENT_BY_KEY
    elif scheme.is_experiment_by_workspace(uri):
        return RequestTypes.EXPERIMENT_BY_WORKSPACE
    elif scheme.is_file(uri):
        return RequestTypes.FILE

    return RequestTypes.UNDEFINED
