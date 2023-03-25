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
from typing import Any, Optional

from .. import model_metadata
from ..types import ModelStateDict, Module
from . import load
from .uri import parse, scheme

LOGGER = logging.getLogger(__name__)


def load_model(
    MODEL_URI: str,
    map_location: Any = None,
    pickle_module: Optional[Module] = None,
    **torch_load_args
) -> ModelStateDict:
    """
    Load model's state_dict from experiment, registry or from disk by uri. This will returns a
    Pytorch state_dict that you will need to load into your model. This will load the model using
    [torch.load](https://pytorch.org/docs/stable/generated/torch.load.html).

    Here is an example of loading a model from the Model Registry for inference:

    ```python
    from comet_ml.integration.pytorch import load_model

    class TheModelClass(nn.Module):
        def __init__(self):
            super(TheModelClass, self).__init__()
            ...

        def forward(self, x):
            ...
            return x

    # Initialize model
    model = TheModelClass()

    # Load the model state dict from Comet Registry
    model.load_state_dict(load_model("registry://WORKSPACE/TheModel:1.2.4"))

    model.eval()

    prediction = model(...)
    ```

    Here is an example of loading a model from an Experiment for Resume Training:

    ```python
    from comet_ml.integration.pytorch import load_model

    # Initialize model
    model = TheModelClass()

    # Load the model state dict from a Comet Experiment
    checkpoint = load_model("experiment://e1098c4e1e764ff89881b868e4c70f5/TheModel")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.train()
    ```

    Args:
        uri: string (required), a uri string defining model location. Possible options are:
            - file://data/my-model
            - file:///path/to/my-model
            - registry://workspace/registry_name (takes the last version)
            - registry://workspace/registry_name:version
            - experiment://experiment_key/model_name
            - experiment://workspace/project_name/experiment_name/model_name
        map_location: (optional) passed to torch.load (see [torch.load](https://pytorch.org/docs/stable/generated/torch.load.html))
        pickle_module: (optional) passed to torch.load (see [torch.load](https://pytorch.org/docs/stable/generated/torch.load.html))
        torch_load_args: (optional) passed to torch.load (see [torch.load](https://pytorch.org/docs/stable/generated/torch.load.html))

    Returns: model's state dict
    """
    if pickle_module is None:
        pickle_module = model_metadata.get_torch_pickle_module()

    if parse.request_type(MODEL_URI) == parse.RequestTypes.UNDEFINED:
        raise ValueError("Invalid MODEL_URI: '{}'".format(MODEL_URI))

    if scheme.is_file(MODEL_URI):
        model = load.from_disk(
            MODEL_URI,
            map_location=map_location,
            pickle_module=pickle_module,
            **torch_load_args
        )
    else:
        model = load.from_remote(
            MODEL_URI,
            map_location=map_location,
            pickle_module=pickle_module,
            **torch_load_args
        )

    return model
