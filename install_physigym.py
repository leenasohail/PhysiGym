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
#   python3 install_physicell_userproject.py <model>
#   python3 install_physicell_userproject.py <model> -f
#   python3 install_physicell_userproject.py <model model>
#   python3 install_physicell_userproject.py <model model> -f
#   python3 install_physicell_userproject.py <all>
#   python3 install_physicell_userproject.py <all> -f
#
# description:
#   script to install the physigym PhysiCell user projects
#   into the PhysiCell/user_projects/ folder.
#####


# modules
import argparse
import os
import shutil
import sys

# const
s_root = '../PhysiCell/'  # absolute or relative path to physicell root folder

# function
def install_pcuserproj(ls_model=[], b_force=False, s_root=s_root):
    """
    input:
        check out argparse in __main__ .

    output:
        PhysiCell/user_projects/<project>/ folder.

    description:
        check out argparse in __main__ .
    """

    # check for PhysiCell root folder
    s_root = s_root.replace('\\','/')
    if not s_root.endswith('/'):
        s_root = s_root + '/'
    if not (os.path.isdir(s_root)):
        sys.exit(f"Error @ install_pcuserproj : no PhysiCell root directory found at '{s_root}'.")

    # check if model specified
    if (ls_model == ['all']):
        ls_model = sorted([s_file for s_file in os.listdir('model/') if os.path.isdir(f'model/{s_file}')])
    if (len(ls_model) == 0):
        sys.exit(f"Error @ install_pcuserproj : no model specified for installation '{ls_model}'.")

    # install models
    for s_model in ls_model:
        s_prj = f'physigym_{s_model}'

        # check for physigym model folder
        s_path_model = f'model/{s_model}/'
        if not (os.path.isdir(s_path_model)):
            sys.exit(f"Error @ install_pcuserproj : {s_path_model} does not exists!")

        # check for PhysiCell user_projects project folder
        s_path_prj = f'{s_root}user_projects/{s_prj}/'
        if ((not b_force) and os.path.exists(s_path_prj)):
            sys.exit(f"Error @ install_pcuserproj : {s_path_prj} already exists!\nUse the command line -f or --force argument to overwrite the existing project with this {s_prj} template project.")

        # erase possibly old model
        print(f'erase {s_path_prj} ...')
        if os.path.exists(s_path_prj):
            shutil.rmtree(s_path_prj)

        # copy the entire base folder into PhysiCell user_projects
        print(f'install: {s_path_prj} physigym basics ...')
        shutil.copytree(src='physigym', dst=s_path_prj)

        # copy files to the user_project's root folder
        print('copy from: PHYSICELL ...')
        shutil.copy('PHYSICELL', s_path_prj)
        for s_file in sorted(os.listdir(s_path_model)):
            if os.path.isfile(f'{s_path_model}{s_file}'):
                print(f'copy from: {s_path_model}{s_file} ...')
                shutil.copy(src=f'{s_path_model}{s_file}', dst=s_path_prj)

        # copy the user_project's config folder
        print(f'copy from: {s_path_model}config/ ...')
        shutil.copytree(src=f'{s_path_model}config/', dst=f'{s_path_prj}config/')

        # copy files to the user_project's custom_modules folder
        for s_file in sorted(os.listdir(f'{s_path_model}custom_modules/')):
            if os.path.isfile(f'{s_path_model}custom_modules/{s_file}'):
                print(f'copy from: {s_path_model}custom_modules/{s_file} ...')
                shutil.copy(
                    src=f'{s_path_model}custom_modules/{s_file}',
                    dst=f'{s_path_prj}custom_modules/',
                )
            elif os.path.isdir(f'{s_path_model}custom_modules/{s_file}') and not (s_file in {'extending','physigym'}):
                shutil.copytree(src=f'{s_path_model}custom_modules/{s_file}/', dst=f'{s_path_prj}custom_modules/{s_file}/')

        # copy files to the user_project's custom_modules extending folder
        print(f'copy from: {s_path_model}custom_modules/extending/physicellmodule.cpp ...')
        shutil.copy(
            src=f'{s_path_model}custom_modules/extending/physicellmodule.cpp',
            dst=f'{s_path_prj}custom_modules/extending/',
        )

        # copy files to the user_project's custom_modules physigym folder
        for s_file in sorted(os.listdir(f'{s_path_model}custom_modules/physigym/')):
            print(f'copy from: {s_path_model}custom_modules/physigym/{s_file} ...')
            shutil.copy(
                src=f'{s_path_model}custom_modules/physigym/{s_file}',
                dst=f'{s_path_prj}custom_modules/physigym/physigym/envs/',
            )

    # going home
    print("ok!")


# run
if __name__ == "__main__":
    print(f'run installation script ...')

    # argv
    parser = argparse.ArgumentParser(
        prog = f'install physigym models',
        description = f'script to copy the physigym PhysiCell user_projects into the correct folder structure.',
        epilog = f'afterwards the project can be built, run, and further developed within Physicell as usual.',
    )
    # s_model
    parser.add_argument(
        'model',
        nargs = '+',
        #default = ['template'],
        help = 'model to be installed. have to match one or more folder names under ./model. all will install all models under ./model.'
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
        ls_model = args.model,
        b_force = args.force,
    )

