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

import base64


def decode(data: str) -> str:
    encoded_bytes = data.encode("ascii")
    decoded_bytes = base64.b64decode(encoded_bytes)
    decoded_string = decoded_bytes.decode("ascii")
    return decoded_string


def encode(data: str) -> str:
    bytes = data.encode("ascii")
    encoded_bytes = base64.b64encode(bytes)
    encoded_string = encoded_bytes.decode("ascii")
    return encoded_string
