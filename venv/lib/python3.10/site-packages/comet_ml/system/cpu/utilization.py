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

import time

try:
    import psutil
except Exception:
    psutil = None

from .. import processes


def process_tree():
    processes_family = [psutil.Process(pid=pid) for pid in processes.family()]
    for process in processes_family:
        process.cpu_percent()

    UTILIZATION_MEASURE_INTERVAL = 0.3
    time.sleep(UTILIZATION_MEASURE_INTERVAL)

    result = sum([process.cpu_percent() for process in processes_family])

    return result
