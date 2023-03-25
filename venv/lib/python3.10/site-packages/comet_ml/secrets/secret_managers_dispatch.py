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

import types

from .gcp import secret_manager as gcp_secret_manager

_SECRET_MANAGERS = types.MappingProxyType(
    {
        "GCP": gcp_secret_manager.SecretManager(),
    }
)


def dispatch(key: str):
    if key not in _SECRET_MANAGERS:
        return None

    return _SECRET_MANAGERS[key]
