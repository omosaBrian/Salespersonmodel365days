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

import os
import pathlib

import comet_ml


def go(experiment: comet_ml.Experiment, trial):
    artifact = _setup_artifact(name=str(trial), directory=trial.checkpoint.dir_or_data)

    experiment.log_artifact(artifact)


def _setup_artifact(name: str, directory: str):
    artifact = comet_ml.Artifact(
        name="checkpoint_{}".format(name), artifact_type="model"
    )
    checkpoint_root = pathlib.Path(directory)
    for root, _, files in os.walk(checkpoint_root):
        rel_root = pathlib.Path(root).relative_to(checkpoint_root)
        for file in files:
            local_file = checkpoint_root / rel_root / file
            logical_path = rel_root / file

            artifact.add(str(local_file), logical_path=str(logical_path))

    return artifact
