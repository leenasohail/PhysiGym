#####
# title: setup.py
#
# languag: setuptools
# date: 2024-spring
# license: BSD-3-Clause
# author: Alexandre Bertin
#
# description:
#   Building a setuptools based python3 lycee python library.
# https://setuptools.pypa.io/en/latest/userguide/ext_modules.html
#####


from setuptools import setup

# extract the version number from the VERSION.txt file
exec(open('./VERSION.txt').read())

setup(
    name="lycee",
    version=__version__,
    install_requires=[
        "gymnasium"
    ],
)

