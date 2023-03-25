# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.com
#  Copyright (C) 2015-2021 Comet ML INC
#  This file can not be copied and/or distributed without the express
#  permission of Comet ML Inc.
# *******************************************************


class HiddenApiKey:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "*** Comet API key ***"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, HiddenApiKey):
            return False
        return self.value == other.value
