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

from typing import List

from comet_ml.vendor.nvidia_ml import pynvml


def all() -> List[str]:
    device_count = pynvml.nvmlDeviceGetCount()
    uuids = [get_by_index(id) for id in range(device_count)]

    return uuids


def get_by_index(id: int) -> str:
    device_handler = pynvml.nvmlDeviceGetHandleByIndex(id)

    return pynvml.nvmlDeviceGetUUID(device_handler)
