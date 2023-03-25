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
import math
import traceback
from inspect import istraceback

convert_functions = []

try:
    import numpy

    def convert_numpy_array_pre_1_16(value):
        try:
            return numpy.asscalar(value)
        except (ValueError, IndexError, AttributeError, TypeError):
            return

    def convert_numpy_array_post_1_16(value):
        try:
            return value.item()
        except (ValueError, IndexError, AttributeError, TypeError):
            return

    convert_functions.append(convert_numpy_array_post_1_16)
    convert_functions.append(convert_numpy_array_pre_1_16)
except ImportError:
    pass


def nan2None(obj):
    if isinstance(obj, dict):
        return {k: nan2None(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [nan2None(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


class NestedEncoder(json.JSONEncoder):
    """
    A JSON Encoder that converts floats/decimals to strings and allows nested objects
    """

    def encode(self, obj, *args, **kwargs):
        obj = nan2None(obj)
        return super().encode(obj, *args, **kwargs)

    def iterencode(self, obj, *args, **kwargs):
        obj = nan2None(obj)
        return super().iterencode(obj, *args, **kwargs)

    def default(self, obj):

        # raise TypeError("test")

        # First convert the object
        obj = self.convert(obj)

        # Check if the object is convertible
        try:
            json.JSONEncoder().encode(obj)
            return obj

        except TypeError:
            pass

        # Custom conversion
        if type(obj) == Exception or isinstance(obj, Exception) or type(obj) == type:
            return str(obj)

        elif istraceback(obj):
            return "".join(traceback.format_tb(obj)).strip()

        elif hasattr(obj, "repr_json"):
            return obj.repr_json()

        elif isinstance(obj, complex):
            return str(obj)

        else:
            try:
                return json.JSONEncoder.default(self, obj)

            except TypeError:
                return "%s not JSON serializable" % obj.__class__.__name__

    def convert(self, obj):
        """
        Try converting the obj to something json-encodable
        """
        for converter in convert_functions:
            converted = converter(obj)

            if converted is not None:
                obj = converted

        return obj
