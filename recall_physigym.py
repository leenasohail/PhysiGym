######
# title: recall_physicell_userproject.py
#
# language: python3
# license: BSD-3-Clause
# date: 2024-03-29
# author: Elmar Bucher
# source code: https://github.com/elmbeech/physicellembedding
#
# run:
#   python3 recall_physicell_userproject.py
#
# description:
#   script to copy the entire PhysiCell user_projects into
#   this source code folder, which is under git version control.
#####


# modules
import argparse
import os
import shutil
import sys


# const
s_prj = 'physigym'


# function
def recall_pcuserproj(s_root='../PhysiCell/'):
    """
    input:
        check out argparse in __main__ .

    output:
        <project>/ folder fetched from ../PhysiCell/user_projects/<project>/

    description:
        check out argparse in __main__ .
    """
    # check for PhysiCell root folder
    s_root = s_root.replace('\\','/')
    if not s_root.endswith('/'):
        s_root = s_root + '/'

    if not (os.path.isdir(s_root)):
        sys.exit(f"Error @ recall_pcuserproj : no PhysiCell root directory found at '{s_root}'.")

    # recall project
    print(f'recall {s_prj} ...')
    s_path_prj = f'{s_root}user_projects/{s_prj}/'

    # overwrite the existing source code repository version
    if os.path.exists(s_prj):
        shutil.rmtree(s_prj)
    shutil.copytree(s_path_prj, s_prj, dirs_exist_ok=True)

    # going home
    print("ok!")


# run
if __name__ == "__main__":
    print(f'run {s_prj} script ...')

    # argv
    parser = argparse.ArgumentParser(
        prog = f'recall PhysiCell/user_projects/{s_prj}',
        description = 'script to copy the current  PhysiCell/user_projects/{s_prj} into this source code repository here.',
        epilog = 'afterwards {s_prj} can be git status, git diff, git add, git commit git push, git pull, git log, git as usual.',
    )
    # s_root
    parser.add_argument(
        'root',
        nargs = '?',
        default = '../PhysiCell/',
        help = 'path to the PhysiCell root directory.'
    )

    # parse arguments
    args = parser.parse_args()
    #print(args)

    # processing
    recall_pcuserproj(
        s_root = args.root,
    )

