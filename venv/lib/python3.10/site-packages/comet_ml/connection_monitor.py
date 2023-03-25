# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2022 Comet ML INC
#  This file can not be copied and/or distributed without
#  the express permission of Comet ML Inc.
# *******************************************************
import time
from typing import Callable


class ServerConnectionMonitor(object):
    """
    This class provides methods to monitor server connection status using particular failure detection scheme.
    """

    def __init__(self, max_failed_ping_attempts: int, ping_interval: float):
        self.max_failed_ping_attempts = max_failed_ping_attempts
        self.ping_interval = ping_interval
        self._failed_ping_count = 0
        self._has_server_connection = True
        self._last_beat = float("-inf")

    @property
    def has_server_connection(self) -> bool:
        return self._has_server_connection

    @has_server_connection.setter
    def has_server_connection(self, new_value: bool) -> None:
        self._has_server_connection = new_value

    def tick(
        self,
        connection_probe: Callable[..., bool],
        connection_failure_report: Callable[[int], None],
    ) -> None:
        """Invoked at each appropriate execution tick. If appropriate, this method will attempt to check
        connectivity using provided connection probe callable."""
        if not self._has_server_connection:
            # already in connection failed mode
            return

        next_beat = self._last_beat + self.ping_interval
        now = time.time()
        if next_beat <= now:
            self._last_beat = now
            result = connection_probe()
            self._on_ping_result(result, connection_failure_report)

    def _on_ping_result(
        self, success: bool, connection_failure_report: Callable[[int], None]
    ):
        """Invoked to signal ping result"""
        if success:
            if self._failed_ping_count > 0:
                connection_failure_report(self._failed_ping_count)

            self._failed_ping_count = 0  # reset failed attempts counter
        else:
            self._failed_ping_count += 1

        if self._failed_ping_count >= self.max_failed_ping_attempts:
            self._has_server_connection = False
