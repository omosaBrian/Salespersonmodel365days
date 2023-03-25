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
#  This file can not be copied and/or distributed
#  without the express permission of Comet ML Inc.
# *******************************************************
from comet_ml._typing import Optional

from six.moves.urllib.parse import urlparse

META_SYNCED = "synced"
META_S3_PATH = "s3_path"
META_CHECKSUM = "checksum"
META_FILE_SIZE = "file_size"
META_VERSION_ID = "version_id"
META_ERROR_MESSAGE = "error_message"
META_OBJECT_S3_PLATFORM = "s3_platform"
META_VALUE_S3_PLATFORM_AWS = "aws"

META_GS_PATH = "gs_path"


def _parse_cloud_storage_uri(storage_uri):
    # type: (str) -> (str, str)
    """Parses cloud storage URI and extracts bucket name and object prefix (parent folders path)."""
    o = urlparse(storage_uri)
    bucket_prefix = o.path
    bucket_name = o.hostname
    if bucket_prefix == "":
        return bucket_name, None

    # remove first slash
    if bucket_prefix.startswith("/"):
        bucket_prefix = bucket_prefix.replace("/", "", 1)

    return bucket_name, bucket_prefix


def _remove_leading_slash(uri_path):
    if uri_path.startswith("/"):
        uri_path = uri_path[1:]
    return uri_path


def _create_cloud_storage_obj_logical_path(
    uri_path, uri_path_folder=None, path_prefix=None
):
    # type: (str, Optional[str], Optional[str]) -> str
    """Creates logical path for the assets associated with remote artifact
    which is stored in the cloud storage bucket."""
    if uri_path_folder is not None:
        if uri_path == uri_path_folder:
            asset_basename = uri_path.split("/")[-1]
            uri_path = asset_basename
        else:
            uri_path = uri_path.replace(uri_path_folder, "", 1)

    uri_path = _remove_leading_slash(uri_path)

    if path_prefix is not None:
        uri_path = _remove_leading_slash(path_prefix) + "/" + uri_path

    return uri_path
