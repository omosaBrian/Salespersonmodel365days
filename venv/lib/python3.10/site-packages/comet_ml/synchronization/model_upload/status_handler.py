# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.com
#  Copyright (C) 2015-2021 Comet ML INC
#  This file can not be copied and/or distributed without
#  the express permission of Comet ML Inc.
# *******************************************************
import collections
import threading
from typing import Callable, Tuple

from . import callbacks


class StatusHandler:
    def __init__(
        self,
    ):
        self._lock = threading.Lock()
        self._uploaded_events = collections.defaultdict(list)
        self._failed_events = collections.defaultdict(list)

    def start_upload(self, model_name: str) -> Tuple[Callable, Callable]:
        with self._lock:
            uploaded_event = threading.Event()
            failed_event = threading.Event()
            self._uploaded_events[model_name].append(uploaded_event)
            self._failed_events[model_name].append(failed_event)

            return uploaded_event.set, failed_event.set

    def observer(self, model_name: str) -> callbacks.StatusObserver:
        with self._lock:
            if model_name not in self._uploaded_events:
                raise KeyError(
                    "Upload status for model {} can't be observed because it wasn't logged".format(
                        model_name
                    )
                )
            observer_callback = callbacks.StatusObserver(
                self._uploaded_events[model_name],
                self._failed_events[model_name],
                self._lock,
            )

            return observer_callback
