####
# title: test/all_physigym.py
#
# language: python3
# author: Elmar Bucher
# date: 2024-05-07
# license: BSD 3-Clause
#
# run:
#     python3 test/all_physigym.py
#
# description:
#     run complete unit test code for the physigym project
#     note: pytest and physigym enviroment are incompatible.
#####


# modules
import subprocess

# install test
subprocess.run(['python3', 'test/install_unittest.py'], check=True)

# run tests
subprocess.run(['python3', 'test/test_epoch.py'], check=True)

# install test
subprocess.run(['python3', 'test/uninstall_unittest.py'], check=True)
