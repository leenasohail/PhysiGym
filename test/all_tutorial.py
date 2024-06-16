####
# title: test/all_tutorial.py
#
# language: python3
# author: Elmar Bucher
# date: 2024-05-07
# license: BSD 3-Clause
#
# run:
#     python3 test/all_tutorial.py
#
# description:
#     run complete tutorial code, and unittest on the tutorial,
#     for the physigym project.
#####


# modules
import subprocess

# install test
subprocess.run(['python3', 'test/install_tutorial.py'], check=True)

# run tests
subprocess.run(['python3', 'test/test_tutorial.py'], check=True)

# install test
subprocess.run(['python3', 'test/uninstall.py'], check=True)
