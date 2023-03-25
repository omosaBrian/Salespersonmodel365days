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

import comet_ml._reporting
import comet_ml.event_tracker

from . import model_logging, model_metadata
from .model_loading.entrypoint import load_model


def log_model(
    experiment, model, model_name, metadata=None, pickle_module=None, **torch_save_args
):
    """
    Logs a Pytorch model to an experiment. This will save the model using
    [torch.save](https://pytorch.org/docs/stable/generated/torch.save.html) and save it as an
    Experiment Model.

    The model parameter can either be an instance of `torch.nn.Module` or any input supported by
    torch.save, see the [tutorial about saving and loading Pytorch
    models](https://pytorch.org/tutorials/beginner/saving_loading_models.html) for more details.

    Here is an example of logging a model for inference:

    ```python
    class TheModelClass(nn.Module):
        def __init__(self):
            super(TheModelClass, self).__init__()
            ...

        def forward(self, x):
            ...
            return x

    # Initialize model
    model = TheModelClass()

    # Train model
    train(model)

    # Save the model for inference
    log_model(experiment, model, model_name="TheModel")
    ```

    Here is an example of logging a checkpoint for resume training:

    ```python
    model_checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        ...
    }
    log_model(experiment, model_checkpoint, model_name="TheModel")
    ```

    Args:
        experiment: Experiment (required), instance of experiment to log model
        model: model's state dict or torch.nn.Module (required), model to log
        model_name: string (required), the name of the model
        metadata: dict (optional), some additional data to attach to the the data. Must be a JSON-encodable dict
        pickle_module: (optional) passed to torch.save (see [torch.save](https://pytorch.org/docs/stable/generated/torch.save.html) documentation)
        torch_save_args: (optional) passed to torch.save (see [torch.save](https://pytorch.org/docs/stable/generated/torch.save.html) documentation)

    Returns: None
    """
    _track_usage(experiment, model)

    state_dict = model_logging.get_state_dict(model)

    if pickle_module is None:
        pickle_module = model_metadata.get_torch_pickle_module()

    model_logging.log_comet_model_metadata(experiment, model_name, pickle_module)
    model_logging.log_state_dict(
        experiment, model_name, state_dict, metadata, pickle_module, **torch_save_args
    )


def _track_usage(experiment, model):
    comet_ml.event_tracker.register(
        "comet_ml.integration.pytorch.log_model-called", experiment_key=experiment.id
    )
    experiment._report(
        event_name=comet_ml._reporting.PYTORCH_MODEL_SAVING_EXPLICIT_CALL,
        err_msg=str(type(model)),
    )
