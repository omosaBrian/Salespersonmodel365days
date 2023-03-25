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

import io
import json
import logging
import pathlib
import tempfile

import torch

from . import constants, model_metadata

LOGGER = logging.getLogger(__name__)


def get_state_dict(model):
    if isinstance(model, dict):
        return model
    elif isinstance(model, torch.nn.Module):
        return model.state_dict()
    raise TypeError("Model must be torch.nn.Module or state_dict")


def log_comet_model_metadata(experiment, model_name, pickle_module):
    metadata = model_metadata.collect(model_name, pickle_module=pickle_module)
    experiment._log_model(
        model_name,
        io.StringIO(json.dumps(metadata)),
        file_name=constants.COMET_MODEL_METADATA_FILENAME,
        critical=True,
    )


def log_state_dict(
    experiment, model_name, state_dict, metadata, pickle_module, **torch_save_args
):
    model_temp_file = tempfile.NamedTemporaryFile()

    try:
        torch.save(
            state_dict, model_temp_file, pickle_module=pickle_module, **torch_save_args
        )
    except:  # noqa: E722
        LOGGER.debug("Failed to save model's state dict.", exc_info=True)
        model_temp_file.close()
        raise

    model_temp_file.seek(0)

    experiment._log_model(
        model_name,
        model_temp_file,
        file_name=pathlib.Path(
            constants.MODEL_DATA_DIRECTORY, constants.MODEL_FILENAME
        ),
        metadata=metadata,
        critical=True,
        on_model_upload=lambda response: model_temp_file.close(),
        on_failed_model_upload=lambda response: model_temp_file.close(),
    )
