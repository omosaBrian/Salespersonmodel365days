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
#  This file can not be copied and/or distributed without the express
#  permission of Comet ML Inc.
# *******************************************************

from . import google_api_importer


class SecretManager:
    def __init__(self):
        self._client = None

    def _initialize_client_if_needed(self):
        if self._client is not None:
            return

        self._client = google_api_importer.secret_manager_service_client_instance()

    def _secret_version_name(self, project_id, secret_id, version_number):
        version_name = "projects/{project_id}/secrets/{secret_id}/versions/{version_number}".format(
            project_id=project_id,
            secret_id=secret_id,
            version_number=version_number,
        )
        return version_name

    def fetch(self, details):
        self._initialize_client_if_needed()

        project_id = details["project_id"]
        secret_id = details["secret_id"]
        version_number = details["secret_version"]

        version_name = self._secret_version_name(project_id, secret_id, version_number)
        response = self._client.access_secret_version(request={"name": version_name})
        api_key = response.payload.data.decode("UTF-8")

        return api_key

    def _create_secret_if_not_exist(self, parent_name, secret_id):
        AlreadyExists = google_api_importer.already_exists_exception()
        try:
            self._client.create_secret(
                request={
                    "parent": parent_name,
                    "secret_id": secret_id,
                    "secret": {"replication": {"automatic": {}}},
                }
            )
        except AlreadyExists:
            pass

    def store(self, secret_value, project_id, secret_id):
        self._initialize_client_if_needed()

        secret_parent_name = "projects/{}".format(project_id)
        self._create_secret_if_not_exist(secret_parent_name, secret_id)

        secret_name = "{parent_name}/secrets/{secret_id}".format(
            parent_name=secret_parent_name, secret_id=secret_id
        )

        version = self._client.add_secret_version(
            request={
                "parent": secret_name,
                "payload": {"data": str.encode(secret_value)},
            }
        )

        return version.name
