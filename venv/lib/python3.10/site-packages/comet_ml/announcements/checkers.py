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

from comet_ml import event_tracker


def check_pytorch_integration_log_model(experiment_key):
    if not event_tracker.is_registered("torch.save-called-by-unknown", experiment_key):
        return False

    comet_log_model_called = any(
        event_tracker.is_registered(function_name, experiment_key)
        for function_name in [
            "experiment.log_model-called",
            "comet_ml.integration.pytorch.log_model-called",
        ]
    )
    if comet_log_model_called:
        return False

    return True
