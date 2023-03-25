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
import comet_ml.inspect


class UndefinedValueMock:
    pass


def add_scalar(
    original_function,
    summary_writer,
    tag,
    data,
    global_step=UndefinedValueMock,
    walltime=UndefinedValueMock,
    **kwargs
):
    original_default_arguments = comet_ml.inspect.default_arguments(original_function)

    if global_step is UndefinedValueMock:
        global_step = original_default_arguments["global_step"]

    return summary_writer, tag, data, global_step


def add_scalars(
    original_function,
    summary_writer,
    main_tag,
    tag_scalar_dict,
    global_step=UndefinedValueMock,
    walltime=UndefinedValueMock,
    **kwargs
):
    original_default_arguments = comet_ml.inspect.default_arguments(original_function)

    if global_step is UndefinedValueMock:
        global_step = original_default_arguments["global_step"]

    return summary_writer, main_tag, tag_scalar_dict, global_step


def add_image(
    original_function,
    summary_writer,
    tag,
    img_tensor,
    global_step=UndefinedValueMock,
    walltime=UndefinedValueMock,
    dataformats=UndefinedValueMock,
    **kwargs
):
    original_default_arguments = comet_ml.inspect.default_arguments(original_function)

    if global_step is UndefinedValueMock:
        global_step = original_default_arguments["global_step"]
    if dataformats is UndefinedValueMock:
        dataformats = original_default_arguments["dataformats"]

    return summary_writer, tag, img_tensor, global_step, dataformats


def add_images(
    original_function,
    summary_writer,
    tag,
    img_tensor,
    global_step=UndefinedValueMock,
    walltime=UndefinedValueMock,
    dataformats=UndefinedValueMock,
    **kwargs
):
    original_default_arguments = comet_ml.inspect.default_arguments(original_function)

    if global_step is UndefinedValueMock:
        global_step = original_default_arguments["global_step"]
    if dataformats is UndefinedValueMock:
        dataformats = original_default_arguments["dataformats"]

    return summary_writer, tag, img_tensor, global_step, dataformats


def add_hparams(
    original_function,
    summary_writer,
    hparams_dict,
    metrics_dict,
    hparam_domain_discrete=UndefinedValueMock,
    run_name=UndefinedValueMock,
    **kwargs
):
    original_default_arguments = comet_ml.inspect.default_arguments(original_function)

    if hparam_domain_discrete is UndefinedValueMock:
        hparam_domain_discrete = original_default_arguments["hparam_domain_discrete"]
    if run_name is UndefinedValueMock:
        run_name = original_default_arguments["run_name"]

    return (
        summary_writer,
        hparams_dict,
        metrics_dict,
        hparam_domain_discrete,
        run_name,
    )
