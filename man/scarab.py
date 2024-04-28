####
# title: scarab.py
#
# language: python3
# date: 2024-03-07
# license: BSD-3-Clause
# author: Elmar Bucher
#
# run: 
#     python3 ../PhysiGym/man/scarab.py
#
# description:
#     inspired by sphinx, scarabaeus is a super lightweight script,
#     that turns input: output: description: docstrings
#     and argparse command line help into markdown files,
#     for source code reference api documentation.
####


# library
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
        elif (s_doc.find('description:') > -1):
            f.write(f'```\n\n## {s_doc.strip()}\n```\n')
        else:
            f.write(f'{s_doc}\n')
    f.write('```')
    f.close()


# load classes
#coreenv = physigym.envs.physicell_core.CorePhysiCellEnv
#env = physigym.envs.physicell_core.ModelPhysiCellEnv
env = gymnasium.make('physigym/ModelPhysiCellEnv-v0')


# write physigym CorePhysiCellEnv function makdown files
# to run epochs 
docstring_md(
    s_function = 'env.__init__',
    ls_doc = env.__init__.__doc__.split('\n'),
    s_header = "env = gymnasium.make('physigym/ModelPhysiCellEnv')"
)
docstring_md(
    s_function = 'env.reset',
    ls_doc = env.reset.__doc__.split('\n'),
)
docstring_md(
    s_function = 'env.render',
    ls_doc = env.render.__doc__.split('\n'),
)
docstring_md(
    s_function = 'env.step',
    ls_doc = env.step.__doc__.split('\n'),
)
docstring_md(
    s_function = 'env.close',
    ls_doc = env.close.__doc__.split('\n'),
)
docstring_md(
    s_function = 'env.verbose_true',
    ls_doc = env.verbose_true.__doc__.split('\n'),
)
docstring_md(
    s_function = 'env.verbose_false',
    ls_doc = env.verbose_false.__doc__.split('\n'),
)

# to specify models
docstring_md(
    s_function = 'env.get_action_space',
    ls_doc = env.get_action_space.__doc__.split('\n'),
)
docstring_md(
    s_function = 'env.get_observation_space',
    ls_doc = env.get_observation_space.__doc__.split('\n'),
)
docstring_md(
    s_function = 'env.get_img',
    ls_doc = env.get_img.__doc__.split('\n'),
)
docstring_md(
    s_function = 'env.get_observation',
    ls_doc = env.get_observation.__doc__.split('\n'),
)
docstring_md(
    s_function = 'env.get_info',
    ls_doc = env.get_info.__doc__.split('\n'),
)
docstring_md(
    s_function = 'env.get_terminated',
    ls_doc = env.get_terminated.__doc__.split('\n'),
)
docstring_md(
    s_function = 'env.get_reward',
    ls_doc = env.get_reward.__doc__.split('\n'),
)

# pure internal functions
docstring_md(
    s_function = 'env.get_truncated',
    ls_doc = env.get_truncated.__doc__.split('\n'),
)

