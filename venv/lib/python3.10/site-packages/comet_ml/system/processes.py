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

import os
from typing import Dict, List

from comet_ml.system.gpu import identifiers
from comet_ml.vendor.nvidia_ml import pynvml

try:
    import psutil
except Exception:
    psutil = None


def _get_gpu_consumers_by_uuid(uuid: str) -> List[int]:
    handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
    processes = pynvml.nvmlDeviceGetComputeRunningProcesses_v3(handle)

    return [process.pid for process in processes]


def gpus_consumers() -> Dict[str, List[int]]:
    pynvml.nvmlInit()
    all_uuids = identifiers.all()
    consumers = {uuid: _get_gpu_consumers_by_uuid(uuid) for uuid in all_uuids}

    return consumers


def family() -> List[int]:
    main_process = psutil.Process(os.getpid())
    children = main_process.children(recursive=True)

    return [main_process.pid] + [child.pid for child in children]


def is_available():
    return psutil is not None
