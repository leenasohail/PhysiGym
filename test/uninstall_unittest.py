####
# title: test/uninstall_physigym.py
#
# language: python3
# author: Elmar Bucher
# date: 2024-05-07
# license: BSD 3-Clause
#
# run:
#     python3 test/uninstall_physigym.py
#
# description:
#     uninstall unit test code for the physigym project
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

