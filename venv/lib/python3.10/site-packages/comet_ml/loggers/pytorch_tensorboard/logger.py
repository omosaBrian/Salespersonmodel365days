# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.ml
#  Copyright (C) 2015-2021 Comet ML INC
#  This file can not be copied and/or distributed without the express
#  permission of Comet ML Inc.
# *******************************************************

import logging

from ... import _logging, monkey_patching
from ...messages import ParameterMessage
from . import summary_writer_arg_parsers as arg_parsers

LOGGER = logging.getLogger(__name__)

FRAMEWORK = "pytorch-tensorboard"


def _load_actual_value(summary_writer, item):
    """
    This code is taken from SummaryWriter implementation
    It repeats in almost every add_* method
    """
    if summary_writer._check_caffe2_blob(item):
        from caffe2.python import workspace

        return workspace.FetchBlob(item)

    return item


def _log_scalar(experiment, summary_writer, tag, scalar, global_step):
    scalar = _load_actual_value(summary_writer, scalar)
    experiment._log_metric(tag, scalar, step=global_step, framework=FRAMEWORK)


def _log_scalars(experiment, summary_writer, main_tag, tag_scalar_dict, global_step):
    tag_scalar_dict = {
        tag: _load_actual_value(summary_writer, scalar)
        for (tag, scalar) in sorted(tag_scalar_dict.items())
    }
    experiment._log_metrics(
        tag_scalar_dict,
        step=global_step,
        prefix=main_tag,
        framework=FRAMEWORK,
    )


def _get_channels_position(dataformats):
    return "first" if "C" in dataformats[:2] else "last"


def _log_image(experiment, summary_writer, tag, image, global_step, dataformats):
    image = _load_actual_value(summary_writer, image)
    image_channels = "first" if dataformats.startswith("C") else "last"
    experiment.log_image(
        image, name=tag, step=global_step, image_channels=image_channels
    )


def _log_images(experiment, summary_writer, tag, image_batch, global_step, dataformats):
    image_batch = _load_actual_value(summary_writer, image_batch)
    image_channels = _get_channels_position(dataformats)
    if dataformats[0] == "N":
        for image in image_batch:
            experiment.log_image(
                image, name=tag, step=global_step, image_channels=image_channels
            )
    else:
        experiment.log_image(
            image_batch,
            name=tag,
            step=global_step,
            image_channels=image_channels,
        )


def _log_hparams(experiment, tag_hparam_dict, prefix):
    experiment._log_parameters(
        tag_hparam_dict, prefix=prefix, source=ParameterMessage.source_autologger
    )


def _log_others(experiment, tag_value_dict):
    experiment.log_others(tag_value_dict)


@_logging.convert_exception_to_log_message(
    "Failed to extract scalar from SummaryWriter.add_scalar()", logger=LOGGER
)
def _summary_writer_add_scalar(experiment, original, return_value, *args, **kwargs):
    if not experiment.auto_metric_logging:
        return

    summary_writer, tag, scalar, global_step = arg_parsers.add_scalar(
        original, *args, **kwargs
    )
    _log_scalar(experiment, summary_writer, tag, scalar, global_step)


@_logging.convert_exception_to_log_message(
    "Failed to extract scalars from SummaryWriter.add_scalars()", logger=LOGGER
)
def _summary_writer_add_scalars(experiment, original, return_value, *args, **kwargs):
    if not experiment.auto_metric_logging:
        return

    (
        summary_writer,
        main_tag,
        tag_scalar_dict,
        global_step,
    ) = arg_parsers.add_scalars(original, *args, **kwargs)
    _log_scalars(experiment, summary_writer, main_tag, tag_scalar_dict, global_step)


@_logging.convert_exception_to_log_message(
    "Failed to extract scalar from SummaryWriter.add_image()", logger=LOGGER
)
def _summary_writer_add_image(experiment, original, return_value, *args, **kwargs):
    if not experiment.auto_metric_logging:
        return

    summary_writer, tag, image, global_step, dataformats = arg_parsers.add_image(
        original, *args, **kwargs
    )
    _log_image(experiment, summary_writer, tag, image, global_step, dataformats)


@_logging.convert_exception_to_log_message(
    "Failed to extract scalar from SummaryWriter.add_images()", logger=LOGGER
)
def _summary_writer_add_images(experiment, original, return_value, *args, **kwargs):
    if not experiment.auto_metric_logging:
        return

    (
        summary_writer,
        tag,
        image_batch,
        global_step,
        dataformats,
    ) = arg_parsers.add_images(original, *args, **kwargs)
    _log_images(experiment, summary_writer, tag, image_batch, global_step, dataformats)


@_logging.convert_exception_to_log_message(
    "Failed to extract scalar from SummaryWriter.add_hparams()", logger=LOGGER
)
def _summary_writer_add_hparams(experiment, original, return_value, *args, **kwargs):
    if not experiment.auto_metric_logging:
        return

    (
        summary_writer,
        hparams_dict,
        metrics_dict,
        hparam_domain_discrete,
        run_name,
    ) = arg_parsers.add_hparams(original, *args, **kwargs)
    _log_hparams(experiment, hparams_dict, run_name)
    _log_scalars(experiment, summary_writer, run_name, metrics_dict, None)
    if hparam_domain_discrete is not None:
        _log_others(experiment, hparam_domain_discrete)


def patch(module_finder):
    monkey_patching.check_module("torch")
    module_finder.register_after(
        "torch.utils.tensorboard",
        "SummaryWriter.add_scalar",
        _summary_writer_add_scalar,
    )
    module_finder.register_after(
        "torch.utils.tensorboard",
        "SummaryWriter.add_scalars",
        _summary_writer_add_scalars,
    )
    module_finder.register_after(
        "torch.utils.tensorboard", "SummaryWriter.add_image", _summary_writer_add_image
    )
    module_finder.register_after(
        "torch.utils.tensorboard",
        "SummaryWriter.add_images",
        _summary_writer_add_images,
    )
    module_finder.register_after(
        "torch.utils.tensorboard",
        "SummaryWriter.add_hparams",
        _summary_writer_add_hparams,
    )


monkey_patching.check_module("torch")
