######
# title: install_physicell_userproject.py
#
# language: python3
# license: BSD-3-Clause
# date: 2024-02-03
# author: Elmar Bucher
# source code: https://github.com/elmbeech/physicellembedding
#
# run:
#   python3 install_physicell_userproject.py
#   python3 install_physicell_userproject.py -f
#   python3 install_physicell_userproject.py <path/to/PhysiCell/>
#   python3 install_physicell_userproject.py <path/to/PhysiCell/> -f
#
# description:
#   script to install the this PhysiCell user project
#   into the PhysiCell/user_projects/ folder.
#####


# modules
import argparse
import os
import shutil
import sys


# const
s_prj = 'physigym'
s_module = 'lyceum'


# function
def install_pcuserproj(s_root='../PhysiCell/', b_force=False):
    """
    input:
        check out argparse in __main__ .

    output:
        PhysiCell/user_projects/<project>/ folder.

    description:
        check out argparse in __main__ .
    """

    # check for physicell root folder
    s_root = s_root.replace('\\','/')
    if not s_root.endswith('/'):
        s_root = s_root + '/'
    if not (os.path.isdir(s_root)):
        sys.exit(f"Error @ install_pcuserproj : no PhysiCell root directory found at '{s_root}'.")

    # check for physicell user_projects project folder
    s_path_prj = f'{s_root}user_projects/{s_prj}/'
    if ((not b_force) and os.path.exists(s_path_prj)):
        sys.exit(f"Error @ install_pcuserproj : {s_path_prj} already exists!\nUse the command line -f or --force argument to overwrite the existing project with this {s_prj} template project.")

    # install project
    print(f'install {s_path_prj} ...')
    if os.path.exists(s_path_prj):
        shutil.rmtree(s_path_prj)

    # copy the entire folder into PhysiCell user_projects
    shutil.copytree(s_prj, s_path_prj)

    # copy files to the user_project's custom_modules folder
    s_path_cmodules = f'{s_path_prj}custom_modules/'
    s_path_cmodules_module = f'{s_path_prj}custom_modules/{s_module}/'
    os.makedirs(s_path_cmodules, exist_ok=True)
    shutil.copy('LICENSE', s_path_cmodules)
    shutil.copy('PHYSICELL', s_path_cmodules)
    shutil.copy('VERSION.txt', s_path_cmodules_module)

    # going home
    print("ok!")


# run
if __name__ == "__main__":
    print(f'run {s_prj} installation script ...')

    # argv
    parser = argparse.ArgumentParser(
        prog = f'install {s_prj}',
        description = f'script to copy the {s_prj} PhysiCell user project into the correct folder structure.',
        epilog = f'afterwards {s_prj} can be built, run, and further developed within Physicell as usual.',
    )
    # s_root
    parser.add_argument(
        'root',
        nargs = '?',
        default = '../PhysiCell/',
        help = 'path to the PhysiCell root directory.'
    )
    # b_force
    parser.add_argument(
        '-f', '--force',
        #type = bool,
        #nargs = 0,
        action=argparse.BooleanOptionalAction,
        #default = False,
        help = ''
    )

    # parse arguments
    args = parser.parse_args()
    #print(args)

    # processing
    install_pcuserproj(
        s_root = args.root,
        b_force = args.force,
    )

