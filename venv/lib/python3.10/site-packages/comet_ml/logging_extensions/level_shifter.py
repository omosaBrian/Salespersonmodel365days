# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2023 Comet ML INC
#  This file can not be copied and/or distributed without the express
#  permission of Comet ML Inc.
# *******************************************************

from logging import Logger
from typing import Any


class LevelShiftingLogger(object):
    """The standard logger wrapper allowing to shift log level after specific number of logging attempts."""

    def __init__(
        self,
        logger: Logger,
        initial_level: int,
        level: int,
        shift_after: int = 1,
    ) -> None:
        self._logger = logger
        self._initial_level = initial_level
        self._level = level
        self._shift_after = shift_after
        self._log_attempts_count = 0

    def log(self, msg: Any, *args: Any, **kwargs: Any):
        self._log_attempts_count += 1
        log_level = (
            self._level
            if self._log_attempts_count > self._shift_after
            else self._initial_level
        )

        self._logger.log(log_level, msg, *args, **kwargs)
