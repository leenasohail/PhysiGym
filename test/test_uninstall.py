####
# title: test/test_uninstall.py
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


###########################
# restore backuped model #
###########################

print('\nUNITTEST restore backuped model ...')
os.chdir('../PhysiCell')
os.system('make load PROJ=backup')
os.chdir('../PhysiGym')
print('UNITTEST: ok!')

