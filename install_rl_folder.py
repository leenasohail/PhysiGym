#####
# please, add header!
#####


# modules
import argparse
import os
import shutil
import sys

# const
s_root = '../PhysiCell/'  # relative path to PhysiCell root folder


# function
def install_rl_project(b_force=False, s_root=s_root):
    """
    Install the 'rl' folder into PhysiCell/.

    Parameters:
        b_force (bool): If True, overwrite existing folder.
        s_root (str): Path to the PhysiCell root folder.
    """
    # Ensure s_root path formatting
    s_root = s_root.replace('\\', '/')
    if not s_root.endswith('/'):
        s_root = s_root + '/'
    if not os.path.isdir(s_root):
        sys.exit(f"Error: No PhysiCell root directory found at '{s_root}'.")

    # Define source and destination paths
    s_src = 'rl/'
    s_dst = f'{s_root}/rl/'

    # Check if the 'rl' source folder exists
    if not os.path.isdir(s_src):
        sys.exit(f"Error: Source folder '{s_src}' does not exist!")

    # Handle existing destination folder
    if os.path.exists(s_dst):
        if not b_force:
            sys.exit(f"Error: Destination folder '{s_dst}' already exists! Use --force to overwrite.")
        else:
            print(f"Removing existing folder: {s_dst} ...")
            shutil.rmtree(s_dst)


    # Copy the 'rl' folder to the destination
    print(f"Copying '{s_src}' to '{s_dst}' ...")
    shutil.copytree(s_src, s_dst)
    print("Installation completed successfully!")


# run
if __name__ == "__main__":
    print("Running RL installation script ...")

    # Argument parser
    parser = argparse.ArgumentParser(
        prog='install_rl_project',
        description='Script to install the RL folder into PhysiCell/user_projects.',
    )
    parser.add_argument(
        '-f', '--force',
        action=argparse.BooleanOptionalAction,
        help='Force overwrite if the RL folder already exists.',
    )

    # Parse arguments
    args = parser.parse_args()

    # Install the RL project
    install_rl_project(b_force=args.force)
