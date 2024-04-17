#####
# title: setup.py
#
# languag: setuptools
# date: 2024-spring
# license: BSD-3-Clause
# author: Alexandre Bertin
#
# description:
#   Building a setuptools based python3 lyceum python library.
# https://setuptools.pypa.io/en/latest/userguide/ext_modules.html
#####


from setuptools import setup

# extract the version number from the VERSION.txt file
exec(open('./VERSION.txt').read())

setup(
    name="lyceum",
    version=__version__,
    install_requires=[
        "gymnasium"
    ],
)

