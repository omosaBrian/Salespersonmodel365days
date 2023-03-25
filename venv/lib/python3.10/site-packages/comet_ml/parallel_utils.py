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
#  This file can not be copied and/or distributed
#  without the express permission of Comet ML Inc.
# *******************************************************
import os
import threading
from functools import wraps

from ._typing import Callable


def synchronised(func):
    # type: (Callable) -> Callable
    """The decorator to make particular function synchronized"""
    func.__lock__ = threading.Lock()  # type: ignore

    @wraps(func)
    def synced_func(*args, **kws):
        with func.__lock__:  # type: ignore
            return func(*args, **kws)

    return synced_func


@synchronised
def makedirs_synchronized(name, exist_ok=False):
    """
    Replacement for Python2's version lacking exist_ok
    """
    if not os.path.exists(name) or not exist_ok:
        os.makedirs(name)
