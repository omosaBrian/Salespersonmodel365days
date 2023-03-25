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
#  This file can not be copied and/or distributed without the express
#  permission of Comet ML Inc.
# *******************************************************

import json
from typing import Tuple

import comet_ml

from .. import codec, secret_managers_dispatch


def get_api_key_from_secret_manager(
    project_id: str, secret_id: str, secret_version: str
) -> str:
    """
    Returns a Comet API Key Secret that can be used instead of a clear-text API Key when creating an
    Experiment or API object. The Comet API Key Secret is a string that represents the location of
    the secret in the GCP Secret Manager without containing the API Key. This means
    `get_api_key_from_secret_manager` doesn't need permission or access to GCP Secret Manager.

    Args:
        project_id: str (required) GCP project id.
        secret_id: str (required), GCP secret id.
        secret_version: str (required) GCP secret version. You can get this value
            from the
            [store_api_key_in_secret_manager](/docs/v2/api-and-sdk/python-sdk/reference/secrets.gcp/#store_api_key_in_secret_manager)
            output. You can also pass `"latest"`, in that case the function will return a Comet
            API Key Secret pointing to the latest version of the GCP Secret.

    Example:

    ```python
    api_key = get_api_key_from_secret_manager(
        GCP_PROJECT_ID, secret_id="username", secret_version=GCP_SECRET_VERSION
    )
    experiment = comet_ml.Experiment(api_key=api_key)
    ```

    Returns: (str) Comet API Key Secret.
    """
    instructions = {
        "type": "GCP",
        "details": {
            "project_id": project_id,
            "secret_id": secret_id,
            "secret_version": secret_version,
        },
        "comet_ml_version": comet_ml.get_comet_version(),
    }

    return "_SECRET_-" + codec.encode(json.dumps(instructions))


def store_api_key_in_secret_manager(
    api_key: str, project_id: str, secret_id: str
) -> Tuple[str, str]:
    """
    Stores an API key to GCP Secret Manager as a secret. After that returns an API Key Secret and
    GCP Secret version as a string.

    Args:
        api_key: str (required), Comet API to save
        project_id: str (required) GCP project id.
        secret_id: str (required), GCP secret id.

    Example:

    ```python
    api_key_secret, secret_version = store_api_key_in_secret_manager(
        COMET_API_KEY, GCP_PROJECT_ID, secret_id="username"
    )
    experiment = comet_ml.Experiment(api_key=api_key_secret)
    ```

    Returns: (Tuple[str, str]) API Key Secret, GCP Secret version
    """
    secret_version_name = secret_managers_dispatch.dispatch("GCP").store(
        api_key,
        project_id,
        secret_id,
    )
    version_number = secret_version_name.split("/")[-1]
    encrypted_key = get_api_key_from_secret_manager(
        project_id, secret_id, version_number
    )
    return encrypted_key, version_number
