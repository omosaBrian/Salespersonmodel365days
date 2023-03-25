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


def _extract_path(uri: str, scheme: str) -> str:
    return uri.replace("{scheme}://".format(scheme=scheme), "")


def _satisfies_conditions(uri: str, scheme: str, path_components_number: int) -> bool:
    if not uri.startswith(scheme):
        return False

    uri_path = _extract_path(uri, scheme)
    uri_path_components = uri_path.split("/")

    if not len(uri_path_components) == path_components_number:
        return False

    if "" in uri_path_components:
        return False

    return True


def is_experiment_by_key(uri: str) -> bool:
    return _satisfies_conditions(uri, "experiment", 2)


def is_experiment_by_workspace(uri: str) -> bool:
    return _satisfies_conditions(uri, "experiment", 4)


def is_registry(uri: str) -> bool:
    if not _satisfies_conditions(uri, "registry", 2):
        return False

    uri_path = _extract_path(uri, "registry")
    if ":" in uri_path:
        version = uri.split(":")[-1]
        if version == "" or "/" in version:
            return False

    return True


def is_file(uri: str) -> bool:
    return uri.startswith("file://")
