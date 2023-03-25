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
#  This file can not be copied and/or distributed without
#  the express permission of Comet ML Inc.
# *******************************************************

MODEL_FILENAME = "comet-torch-model.pth"

COMET_MODEL_METADATA_FILENAME = "CometModel"

MODEL_DATA_DIRECTORY = "model-data"

TORCH_VERSION_MISMATCH_WARN_MESSAGE = "Torch versions mismatch during model loading. Model was uploaded with torch=={prev_torch}. Current version is {new_torch}"

PICKLE_PACKAGE_MISMATH_WARN_MESSAGE = "Pickle packages mismatch during model loading. Model was uploaded with {prev_pickle}. You provided {new_pickle}"
