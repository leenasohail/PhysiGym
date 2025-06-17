######
# title: capture_physicell_userproject.py
#
# language: python3
# license: BSD-3-Clause
# date: 2024-03-29
# author: Elmar Bucher
# source code: https://github.com/elmbeech/physicellembedding
#
# run:
#   python3 capture_physicell_userproject.py <model>
#   python3 capture_physicell_userproject.py <model> -f
#   python3 capture_physicell_userproject.py <model model>
#   python3 capture_physicell_userproject.py <model model> -f
#
# description:
#   script to seize the physigym PhysiCell user_projects into
#   this source code folder, which is under git version control.
#####


# modules
import argparse
import os
import shutil
import sys

# const
s_root = '../PhysiCell/'  # absolute or relative path to physicell root folder

# function
def capture_pcuserproj(ls_model=[], b_force=False, s_root=s_root):
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
        sys.exit(f"Error @ capture_pcuserproj : no PhysiCell root directory found at '{s_root}'.")

    # check if model specified
    if (len(ls_model) == 0):
        sys.exit(f"Error @ capture_pcuserproj : no model specified to capture '{ls_model}'.")

    # capture models
    for s_model in ls_model:
        print(f'capture {s_model} ...')
        s_prj = f'physigym_{s_model}'

        # check for PhysiCell user_projects project folder
        s_path_prj = f'{s_root}user_projects/{s_prj}/'
        if not os.path.exists(s_path_prj):
            sys.exit(f"Error @ capture_pcuserproj : {s_path_prj} does not exists!")

        # check for physigym model folder
        s_path_model = f'model/{s_model}/'
        if ((not b_force) and (not os.path.isdir(s_path_model))):
            sys.exit(f"Error @ capture_pcuserproj : {s_path_model} does not exists!\nUse the command line -f or --force argument to save {s_prj} as a new model under {s_path_model}.")

        # erase possibly old model
        print(f'erase {s_path_model} ...')
        if os.path.exists(s_path_model):
            shutil.rmtree(s_path_model)

        # copy files to the model's root folder
        os.makedirs(s_path_model, exist_ok=True)
        for s_file in sorted(os.listdir(s_path_prj)):
            if os.path.isfile(f'{s_path_prj}{s_file}') and not (s_file in {'LICENSE', 'main.cpp','Makefile','PHYSICELL','studio_debug.log','VERSION.txt'}):
                print(f'copy to: {s_path_model}{s_file} ...')
                shutil.copy(src=f'{s_path_prj}{s_file}', dst=s_path_model)

        # copy the model's config folder
        print(f'copy to: {s_path_model}config/ ...')
        shutil.copytree(src=f'{s_path_prj}config/', dst=f'{s_path_model}config/')

        # copy files to the model's custom_modules folder
        os.makedirs(f'{s_path_model}custom_modules/', exist_ok=True)
        for s_file in sorted(os.listdir(f'{s_path_prj}custom_modules/')):
            if os.path.isfile(f'{s_path_prj}custom_modules/{s_file}') and not (s_file in {'empty.txt','LICENSE','studio_debug.log'}):
                print(f'copy to: {s_path_model}custom_modules/{s_file} ...')
                shutil.copy(
                    src=f'{s_path_prj}custom_modules/{s_file}',
                    dst=f'{s_path_model}custom_modules/',
                )
            elif os.path.isdir(f'{s_path_prj}custom_modules/{s_file}') and not (s_file in {'extending','physigym'}):
                shutil.copytree(src=f'{s_path_prj}custom_modules/{s_file}/', dst=f'{s_path_model}custom_modules/{s_file}/')

        # copy files to the model's custom_modules extending folder
        os.makedirs(f'{s_path_model}custom_modules/extending/', exist_ok=True)
        print(f'copy to: {s_path_model}custom_modules/extending/physicellmodule.cpp ...')
        shutil.copy(
            src=f'{s_path_prj}custom_modules/extending/physicellmodule.cpp',
            dst=f'{s_path_model}custom_modules/extending/',
        )

        # copy files to the model's custom_modules physigym folder
        os.makedirs(f'{s_path_model}custom_modules/physigym/', exist_ok=True)
        for s_file in sorted(os.listdir(f'{s_path_prj}custom_modules/physigym/physigym/envs/')):
            if not (s_file.endswith('__init__.py') or s_file.endswith('physicell_core.py')):
                print(f'copy to: {s_path_model}custom_modules/physigym/{s_file} ...')
                shutil.copy(
                    src=f'{s_path_prj}custom_modules/physigym/physigym/envs/{s_file}',
                    dst=f'{s_path_model}custom_modules/physigym/',
                )

        # copy the user_project's img folder
        print(f'copy to: {s_path_model}img/ ...')
        shutil.copytree(src=f'{s_path_prj}img/', dst=f'{s_path_model}img/')


    # going home
    print("ok!")


# run
if __name__ == "__main__":
    print(f'run capture script ...')

    # argv
    parser = argparse.ArgumentParser(
        prog = f'capture physigym models',
        description = 'script to copy the current physigym PhysiCell/user_projects/ model into this source code repository here.',
        epilog = 'afterwards model can be git status, git diff, git add, git commit git push, git pull, git log, git as usual.',
    )
    # s_model
    parser.add_argument(
        'model',
        nargs = '+',
        #default = ['template'],
        help = 'model to be seized. have to match one or more folder names under ./model.'
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
    capture_pcuserproj(
        ls_model = args.model,
        b_force = args.force,
    )

