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


def secret_manager_service_client_instance():
    try:
        from google.cloud import secretmanager

        return secretmanager.SecretManagerServiceClient()
    except ImportError as exception:
        raise ImportError(
            "You are likely missing the dependency 'google-cloud-secret-manager',"
            "install it with: `python -m pip install google-cloud-secret-manager`"
        ) from exception


def already_exists_exception():
    from google.api_core import exceptions

    return exceptions.AlreadyExists
