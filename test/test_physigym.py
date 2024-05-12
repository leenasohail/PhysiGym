####
# title: test/test_physigym.py
#
# language: python3
# author: Elmar Bucher
# date: 2024-05-07
# license: BSD 3-Clause
#
# run:
#     python3 test/test_physigym.py
#
# description:
#     unit test code for the physigym project
#     note: pytest and physigym enviroment are incompatible.
#####


# modules
import os
import pcdl
import shutil
import subprocess

################
# install test #
################
subprocess.run(['python3', 'test/test_install.py'], check=True)

#############
# run tests #
#############
subprocess.run(['python3', 'test/test_epoch.py'], check=True)

################
# install test #
################
subprocess.run(['python3', 'test/test_uninstall.py'], check=True)

