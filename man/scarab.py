####
# title: scarab.py
#
# language: python3
# date: 2024-03-07
# license: BSD-3-Clause
# author: Elmar Bucher
#
# rootdir: PhysiGym/
# run:
#     python3 man/scarab.py
#
# description:
#     inspired by sphinx, scarabaeus is a super lightweight script,
#     that turns input: output: description: docstrings
#     and argparse command line help into markdown files,
#     for source code reference api documentation.
####


# library
from embedding import physicell  # import the Python/PhysiCell API module
import gymnasium
import physigym  # import the Gymnasium PhysiCell bridge module
import os


# function
def help_md(s_command, s_opath='../PhysiGym/man/docstring/'):
    """
    input:
        s_command: string
            command line command name.

        s_opath: string, default ../PhysiGym/man/docstring/
            output path.

    output:
        opath/command.md file.

    description:
        function to generate an markdown file from the
        argparse help information.
    """
    print(f'processing: {s_command} ...')
    s_opathfile = f'{s_opath}{s_command}.md'
    f = open(s_opathfile, 'w')
    f.write('```\n')
    f.close()
    os.system(f'{s_command} -h >> {s_opathfile}')
    f = open(s_opathfile, 'a')
    f.write('```\n')
    f.close()


def docstring_md(s_function, ls_doc, s_header=None, s_opath='../PhysiGym/man/docstring/'):
    """
    input:
        s_function: string
            function name.

        ls_doc: list of string.
            module.function.__doc__.split('\n')

        s_header: string, default None
            default markdown text title is function().
            with this s_header argument you can set another title
            than the default one.

        s_opath: string, default ../PhysiGym/man/docstring/
            output path.

    output:
        opath/command.md file.

    description:
        function to generate a markdown file from
        docstring information.
    """
    print(f'processing: {s_function} ...')
    os.makedirs(s_opath, exist_ok=True)
    s_opathfile = s_opath + f'{s_function}.md'
    f = open(s_opathfile, 'w')
    if (s_header is None):
        s_header = f'{s_function}()'
    f.write(f'# {s_header}\n\n')
    for s_doc in ls_doc:
        if (s_doc.find('input:') > -1):
            f.write(f'## {s_doc.strip()}\n```\n')
        elif (s_doc.find('output:') > -1):
            f.write(f'```\n\n## {s_doc.strip()}\n```\n')
        elif (s_doc.find('run:') > -1):
            f.write(f'```\n\n## {s_doc.strip()}\n```\n')
        elif (s_doc.find('description:') > -1):
            f.write(f'```\n\n## {s_doc.strip()}\n```\n')
        else:
            f.write(f'{s_doc}\n')
    f.write('```')
    f.close()


# load classes
os.chdir('../PhysiCell')
#coreenv = physigym.envs.physicell_core.CorePhysiCellEnv
#env = physigym.envs.physicell_core.ModelPhysiCellEnv
env = gymnasium.make('physigym/ModelPhysiCellEnv-v0')


# write physigym CorePhysiCellEnv function makdown files
# about the ModelPhysiCellEnv class
docstring_md(
    s_function = 'physigym.envs.ModelPhysiCellEnv',
    ls_doc = physigym.envs.ModelPhysiCellEnv.__doc__.split('\n'),
)

# to run epochs
docstring_md(
    s_function = 'physigym.envs.ModelPhysiCellEnv.__init__',
    ls_doc = physigym.envs.CorePhysiCellEnv.__init__.__doc__.split('\n'),
)
docstring_md(
    s_function = 'physigym.envs.ModelPhysiCellEnv.reset',
    ls_doc = physigym.envs.ModelPhysiCellEnv.reset.__doc__.split('\n'),
)
docstring_md(
    s_function = 'physigym.envs.ModelPhysiCellEnv.render',
    ls_doc = physigym.envs.ModelPhysiCellEnv.render.__doc__.split('\n'),
)
docstring_md(
    s_function = 'physigym.envs.ModelPhysiCellEnv.step',
    ls_doc = physigym.envs.ModelPhysiCellEnv.step.__doc__.split('\n'),
)
docstring_md(
    s_function = 'physigym.envs.ModelPhysiCellEnv.close',
    ls_doc = physigym.envs.ModelPhysiCellEnv.close.__doc__.split('\n'),
)
docstring_md(
    s_function = 'physigym.envs.ModelPhysiCellEnv.verbose_true',
    ls_doc = physigym.envs.ModelPhysiCellEnv.verbose_true.__doc__.split('\n'),
)
docstring_md(
    s_function = 'physigym.envs.ModelPhysiCellEnv.verbose_false',
    ls_doc = physigym.envs.ModelPhysiCellEnv.verbose_false.__doc__.split('\n'),
)

# to specify models
docstring_md(
    s_function = 'physigym.envs.ModelPhysiCellEnv.get_action_space',
    ls_doc = physigym.envs.ModelPhysiCellEnv.get_action_space.__doc__.split('\n'),
)
docstring_md(
    s_function = 'physigym.envs.ModelPhysiCellEnv.get_observation_space',
    ls_doc = physigym.envs.ModelPhysiCellEnv.get_observation_space.__doc__.split('\n'),
)
docstring_md(
    s_function = 'physigym.envs.ModelPhysiCellEnv.get_img',
    ls_doc = physigym.envs.ModelPhysiCellEnv.get_img.__doc__.split('\n'),
)
docstring_md(
    s_function = 'physigym.envs.ModelPhysiCellEnv.get_observation',
    ls_doc = physigym.envs.ModelPhysiCellEnv.get_observation.__doc__.split('\n'),
)
docstring_md(
    s_function = 'physigym.envs.ModelPhysiCellEnv.get_info',
    ls_doc = physigym.envs.ModelPhysiCellEnv.get_info.__doc__.split('\n'),
)
docstring_md(
    s_function = 'physigym.envs.ModelPhysiCellEnv.get_terminated',
    ls_doc = physigym.envs.ModelPhysiCellEnv.get_terminated.__doc__.split('\n'),
)
docstring_md(
    s_function = 'physigym.envs.ModelPhysiCellEnv.get_reward',
    ls_doc = physigym.envs.ModelPhysiCellEnv.get_reward.__doc__.split('\n'),
)

# pure internal functions
docstring_md(
    s_function = 'physigym.envs.ModelPhysiCellEnv.get_truncated',
    ls_doc = physigym.envs.ModelPhysiCellEnv.get_truncated.__doc__.split('\n'),
)

# python physicell api functions
docstring_md(
    s_function = 'physicell.start',
    ls_doc = physicell.start.__doc__.split('\n'),
)
docstring_md(
    s_function = 'physicell.step',
    ls_doc = physicell.step.__doc__.split('\n'),
)
docstring_md(
    s_function = 'physicell.stop',
    ls_doc = physicell.stop.__doc__.split('\n'),
)
docstring_md(
    s_function = 'physicell.reset',
    ls_doc = physicell.reset.__doc__.split('\n'),
)
docstring_md(
    s_function = 'physicell.set_parameter',
    ls_doc = physicell.set_parameter.__doc__.split('\n'),
)
docstring_md(
    s_function = 'physicell.get_parameter',
    ls_doc = physicell.get_parameter.__doc__.split('\n'),
)
docstring_md(
    s_function = 'physicell.set_variable',
    ls_doc = physicell.get_variable.__doc__.split('\n'),
)
docstring_md(
    s_function = 'physicell.get_variable',
    ls_doc = physicell.get_variable.__doc__.split('\n'),
)
docstring_md(
    s_function = 'physicell.set_vector',
    ls_doc = physicell.set_vector.__doc__.split('\n'),
)
docstring_md(
    s_function = 'physicell.get_vector',
    ls_doc = physicell.get_vector.__doc__.split('\n'),
)
docstring_md(
    s_function = 'physicell.get_cell',
    ls_doc = physicell.get_cell.__doc__.split('\n'),
)
docstring_md(
    s_function = 'physicell.get_microenv',
    ls_doc = physicell.get_microenv.__doc__.split('\n'),
)

os.chdir('../PhysiGym')
