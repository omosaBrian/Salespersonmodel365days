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


import calendar
import collections

import comet_ml._jupyter
import comet_ml.env_logging
from comet_ml.api import APIExperiment

import boto3
from sagemaker.analytics import TrainingJobAnalytics


def _get_boto_client():
    return boto3.client("sagemaker")


def _flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def log_last_sagemaker_training_job_v1(api_key=None, workspace=None, project_name=None):
    """
    This function retrieves the last training job and logs its data as a Comet Experiment. The training job must be in completed status.

    This API is in BETA.

    Args:
        api_key: string (optional), the Comet API Key. If not provided must be [configured in another way](/docs/v2/guides/tracking-ml-training/configuring-comet/)
        workspace: string (optional), attach the experiment to a project that belongs to this workspace. If not provided must be [configured in another way](/docs/v2/guides/tracking-ml-training/configuring-comet/)
        project_name: string (optional), send the experiment to a specific project. If not provided must be [configured in another way](/docs/v2/guides/tracking-ml-training/configuring-comet/)

    Returns: an instance of [APIExperiment](/docs/v2/api-and-sdk/python-sdk/reference/APIExperiment/) for the created Experiment
    """
    client = _get_boto_client()
    last_name = client.list_training_jobs()["TrainingJobSummaries"][0][
        "TrainingJobName"
    ]
    return log_sagemaker_training_job_by_name_v1(
        last_name, api_key=api_key, workspace=workspace, project_name=project_name
    )


def log_sagemaker_training_job_v1(
    estimator, api_key=None, workspace=None, project_name=None
):
    """
    This function retrieves the last training job from an
    [`sagemaker.estimator.Estimator`](https://sagemaker.readthedocs.io/en/v2.16.0/api/training/estimators.html#sagemaker.estimator.Estimator)
    object and log its data as a Comet Experiment. The training job must be in completed status.

    This API is in BETA.

    Here is an example of using this function:

    ```python
    import sagemaker

    from comet_ml.integration.sagemaker import log_sagemaker_training_job_v1

    estimator = sagemaker.estimator.Estimator(
        training_image,
        role,
        instance_count=instance_count,
        instance_type=instance_type,
        output_path=s3_output_location,
    )

    estimator.fit(s3_input_location)

    api_experiment = log_sagemaker_training_job_v1(
        estimator, api_key=API_KEY, workspace=WORKSPACE, project_name=PROJECT_NAME
    )
    ```

    Args:
        estimator: sagemaker.estimator.Estimator (required), the estimator object that was used to start the training job.
        api_key: string (optional), the Comet API Key. If not provided must be [configured in another way](/docs/v2/guides/tracking-ml-training/configuring-comet/).
        workspace: string (optional), attach the experiment to a project that belongs to this workspace. If not provided must be [configured in another way](/docs/v2/guides/tracking-ml-training/configuring-comet/).
        project_name: string (optional), send the experiment to a specific project. If not provided must be [configured in another way](/docs/v2/guides/tracking-ml-training/configuring-comet/).

    Returns: an instance of [APIExperiment](/docs/v2/api-and-sdk/python-sdk/reference/APIExperiment/) for the created Experiment
    """
    # Retrieve the training job name from the estimator
    if not hasattr(estimator, "latest_training_job"):
        raise ValueError("log_sagemaker_job expect a sagemaker Estimator object")

    if estimator.latest_training_job is None:
        raise ValueError(
            "The given Estimator object doesn't seems to have trained a model, call log_sagemaker_job after calling the fit method"
        )

    return log_sagemaker_training_job_by_name_v1(
        estimator.latest_training_job.job_name,
        api_key=api_key,
        workspace=workspace,
        project_name=project_name,
    )


def log_sagemaker_training_job_by_name_v1(
    sagemaker_job_name, api_key=None, workspace=None, project_name=None
):
    """
    This function logs the training job identified by the `sagemaker_job_name` as a Comet Experiment.
    The training job must be in completed status.

    This API is in BETA.

    Args:
        sagemaker_job_name: string (required), the name of the Sagemaker Training Job.
        api_key: string (optional), the Comet API Key. If not provided must be [configured in another way](/docs/v2/guides/tracking-ml-training/configuring-comet/)
        workspace: string (optional), attach the experiment to a project that belongs to this workspace. If not provided must be [configured in another way](/docs/v2/guides/tracking-ml-training/configuring-comet/)
        project_name: string (optional), send the experiment to a specific project. If not provided must be [configured in another way](/docs/v2/guides/tracking-ml-training/configuring-comet/)

    Returns: an instance of [APIExperiment](/docs/v2/api-and-sdk/python-sdk/reference/APIExperiment/) for the created Experiment
    """
    # Metadata
    client = _get_boto_client()
    metadata = client.describe_training_job(TrainingJobName=sagemaker_job_name)

    if metadata["TrainingJobStatus"] != "Completed":
        raise ValueError(
            "Not importing %r as it's not completed, status %r"
            % (sagemaker_job_name, metadata["TrainingJobStatus"])
        )

    experiment = APIExperiment(
        api_key=api_key,
        workspace=workspace,
        project_name=project_name,
        experiment_name=sagemaker_job_name,
    )
    start_time = metadata["TrainingStartTime"]
    start_time_timestamp = calendar.timegm(start_time.utctimetuple())
    experiment.set_start_time(start_time_timestamp * 1000)
    end_time = metadata.get("TrainingEndTime")
    if end_time:
        experiment.set_end_time(calendar.timegm(end_time.utctimetuple()) * 1000)

    for param_name, param_value in metadata["HyperParameters"].items():
        experiment.log_parameter(param_name, param_value)

    other_list = [
        "BillableTimeInSeconds",
        "EnableInterContainerTrafficEncryption",
        "EnableManagedSpotTraining",
        "EnableNetworkIsolation",
        "RoleArn",
        "TrainingJobArn",
        "TrainingJobName",
        "TrainingJobStatus",
        "TrainingTimeInSeconds",
    ]
    for other_name in other_list:
        other_value = metadata.get(other_name)
        if other_value:
            experiment.log_other(other_name, other_value)

    experiment.log_other(
        "TrainingImage", metadata["AlgorithmSpecification"]["TrainingImage"]
    )
    experiment.log_other(
        "TrainingInputMode", metadata["AlgorithmSpecification"]["TrainingInputMode"]
    )

    for other_key, other_value in _flatten(
        metadata.get("ModelArtifacts", {}), "ModelArtifacts"
    ).items():
        experiment.log_other(other_key, other_value)

    for other_key, other_value in _flatten(
        metadata["OutputDataConfig"], "OutputDataConfig"
    ).items():
        experiment.log_other(other_key, other_value)

    for other_key, other_value in _flatten(
        metadata["ResourceConfig"], "ResourceConfig"
    ).items():
        experiment.log_other(other_key, other_value)

    for i, _input in enumerate(metadata["InputDataConfig"]):
        for other_key, other_value in _flatten(
            _input, "InputDataConfig.%d" % i
        ).items():
            experiment.log_other(other_key, other_value)

    response = client.list_tags(ResourceArn=metadata["TrainingJobArn"])
    for tag_name, tag_value in response["Tags"]:
        experiment.add_tags(["%s:%s" % (tag_name, tag_value)])

    # Metrics
    metrics_dataframe = TrainingJobAnalytics(
        training_job_name=sagemaker_job_name
    ).dataframe()

    for iloc, (timestamp, metric_name, value) in metrics_dataframe.iterrows():
        experiment.log_metric(
            metric=metric_name, value=value, timestamp=start_time_timestamp + timestamp
        )

    # Log pip packages from the current environment
    experiment.set_installed_packages(comet_ml.env_logging.get_pip_packages())

    # Log notebook code if ran from a notebook only
    if comet_ml._jupyter._in_ipython_environment():
        source_code = comet_ml.env_logging.get_ipython_source_code()

        if source_code != "":
            experiment.set_code(source_code)

    return experiment
