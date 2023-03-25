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
import logging

from comet_ml import logging_messages

from . import codec, secret_managers_dispatch

LOGGER = logging.getLogger(__name__)

_SECRET_INSTRUCTION_PREFIX = "_SECRET_-"


def interpret(api_key):
    if not isinstance(api_key, str):
        return api_key

    if not api_key.startswith(_SECRET_INSTRUCTION_PREFIX):
        return api_key

    instruction = _decode_instruction(api_key)
    if instruction is None:
        return api_key

    fetched_api_key = _fetch_api_key(instruction)
    if fetched_api_key is None:
        return api_key

    return fetched_api_key


def _decode_instruction(encoded):
    try:
        encoded = encoded[len(_SECRET_INSTRUCTION_PREFIX) :]
        decoded = codec.decode(encoded)
        return json.loads(decoded)
    except Exception:
        LOGGER.error(
            logging_messages.BAD_ENCODED_SECRET_API_KEY,
            exc_info=True,
            extra={"show_traceback": True},
        )
        return None


def _fetch_api_key(instruction):
    try:
        secret_manager_name = instruction["type"]
        details = instruction["details"]

        fetcher = secret_managers_dispatch.dispatch(secret_manager_name)
        if fetcher is None:
            version = instruction["comet_ml_version"]
            LOGGER.error(
                logging_messages.UNSUPPORTED_SECRET_MANAGER
                % (secret_manager_name, version)
            )
            return None

        return fetcher.fetch(details)
    except Exception:
        LOGGER.error(
            logging_messages.FAILED_TO_GET_API_KEY_FROM_SECRET_MANAGER,
            exc_info=True,
            extra={"show_traceback": True},
        )
        return None
