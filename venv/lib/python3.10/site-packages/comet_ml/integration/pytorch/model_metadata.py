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

import logging
import pathlib
from typing import Any, Dict

import comet_ml
import comet_ml.inspect

import torch

from . import constants
from .types import Module

LOGGER = logging.getLogger(__name__)


def collect(model_name, pickle_module):
    model_metadata = {}

    metadata = {
        "format": "pytorch",
        "comet_sdk_version": comet_ml.__version__,
        "model_metadata": model_metadata,
    }

    model_metadata["pytorch_version"] = torch.__version__
    model_metadata["pickle_module"] = pickle_module.__name__
    model_metadata["model_path"] = str(
        pathlib.Path(model_name, constants.MODEL_FILENAME)
    )

    return metadata


def warn_if_has_mismatches_with_environment(
    comet_model_metadata: Dict[str, Any], pickle_module: Module
) -> None:
    model_metadata = comet_model_metadata["model_metadata"]
    pickle_name = pickle_module.__name__
    if model_metadata["pickle_module"] != pickle_name:
        LOGGER.warn(
            constants.PICKLE_PACKAGE_MISMATH_WARN_MESSAGE.format(
                prev_pickle=model_metadata["pickle_module"], new_pickle=pickle_name
            )
        )
    if model_metadata["pytorch_version"] != torch.__version__:
        LOGGER.warn(
            constants.TORCH_VERSION_MISMATCH_WARN_MESSAGE.format(
                prev_torch=model_metadata["pytorch_version"],
                new_torch=torch.__version__,
            )
        )


def get_torch_pickle_module():
    torch_save_default_args = comet_ml.inspect.default_arguments(torch.save)
    return torch_save_default_args["pickle_module"]
