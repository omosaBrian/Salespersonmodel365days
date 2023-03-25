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

import inspect


def default_arguments(function):
    signature = inspect.signature(function)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def identify_caller(callstack_frame, known_callers, return_if_unknown=None):
    CALLER_IS_UNKNOWN = return_if_unknown

    module = inspect.getmodule(callstack_frame[0])

    if module is None:
        return CALLER_IS_UNKNOWN

    module_name = module.__name__

    for known_caller in known_callers:
        if module_name.split(".")[0] == known_caller:
            return known_caller

    return CALLER_IS_UNKNOWN
