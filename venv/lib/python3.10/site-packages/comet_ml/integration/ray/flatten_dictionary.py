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
import copy
from typing import Dict


def flatten(
    dictionary: Dict,
    delimiter: str = "/",
    prevent_delimiter: bool = False,
    flatten_list: bool = False,
):  # pragma: no cover
    """
    THIS FUNCTION IS COPIED FROM DEPRECATED ray.tune.utils.flatten_dict

    Flatten dict.


    Output and input are of the same dict type.
    Input dict remains the same after the operation.
    """

    def _raise_delimiter_exception():
        raise ValueError(
            "Found delimiter `{}` in key when trying to flatten "
            "dict. Please avoid using the delimiter in your specification.".format(
                delimiter
            )
        )

    dictionary = copy.copy(dictionary)
    if prevent_delimiter and any(delimiter in key for key in dictionary):
        # Raise if delimiter is any of the keys
        _raise_delimiter_exception()

    while_check = (dict, list) if flatten_list else dict

    while any(isinstance(value, while_check) for value in dictionary.values()):
        remove = []
        add = {}
        for key, value in dictionary.items():
            if isinstance(value, dict):
                for subkey, value in value.items():
                    if prevent_delimiter and delimiter in subkey:
                        # Raise if delimiter is in any of the subkeys
                        _raise_delimiter_exception()

                    add[delimiter.join([key, str(subkey)])] = value
                remove.append(key)
            elif flatten_list and isinstance(value, list):
                for i, value in enumerate(value):
                    if prevent_delimiter and delimiter in subkey:
                        # Raise if delimiter is in any of the subkeys
                        _raise_delimiter_exception()

                    add[delimiter.join([key, str(i)])] = value
                remove.append(key)

        dictionary.update(add)
        for key in remove:
            del dictionary[key]
    return dictionary
