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

import comet_ml
import comet_ml.logging_messages

from . import checkers


def _announcements_disabled():
    config = comet_ml.get_config()
    return config["comet.disable_announcement"]


def announce(logger, experiment_key):
    if _announcements_disabled():
        return

    if checkers.check_pytorch_integration_log_model(experiment_key):
        logger.info(
            comet_ml.logging_messages.PYTORCH_INTEGRATION_LOG_MODEL_ANNOUNCEMENT
        )
