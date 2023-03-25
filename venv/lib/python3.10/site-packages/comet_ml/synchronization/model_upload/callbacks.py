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
import threading
from typing import List

Status = str


class StatusObserver:
    def __init__(
        self,
        uploaded_events: List[threading.Event],
        failed_events: List[threading.Event],
        lock: threading.Lock,
    ):
        self._uploaded_events = uploaded_events
        self._failed_events = failed_events
        self._lock = lock

    def __call__(self) -> Status:
        with self._lock:
            all_failed = all(
                failed_event.is_set() for failed_event in self._failed_events
            )
            if all_failed:
                return "FAILED"

            any_uploaded = any(
                uploaded_event.is_set() for uploaded_event in self._uploaded_events
            )
            if any_uploaded:
                return "COMPLETED"

            return "IN_PROGRESS"
