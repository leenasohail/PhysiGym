![physicellembedding logo & title](man/img/physicellembedding_title_v0.0.0.png)


# Header:

+ Language: C++11 and Python [>= 3.8](https://devguide.python.org/versions/)
+ Library dependencies: regular PhysiCell and Python3 installation. no other dependencies.
+ Author: Elmar Bucher, Alexandre Bertin
+ Date: 2024-01-08
+ Doi:
+ License: [BSD-3-Clause](https://en.wikipedia.org/wiki/BSD_licenses)
+ User manual: this README.md file
+ Source code: [https://github.com/elmbeech/physicellembedding](https://github.com/elmbeech/physicellembedding)


# Abstract:

**Introduction:**

PhysiCell is a physics-based cell simulator for 3D multicellular systems.
More precisely, [PhysiCell](https://github.com/MathCancer/PhysiCell) is an agent-based model and diffusion transport solver, off-lattice, center-based, multiscale in space and time, written in [C++](https://en.wikipedia.org/wiki/C%2B%2B).

This physicellembeding project aims to offer a PhysiCell [Python3](https://en.wikipedia.org/wiki/Python_(programming_language)) application interface.
Such an interface makes sense in two ways.
1. Embedding PhysiCell into as Python3 module, so that PhysiCell can be controlled from within Python3 (**py3pc_embed**).
1. Embedding a Python3 interpreter into PhysiCell, so that PhysiCell can execute Python3 code within its main loop (**pcpy3_embed**).

**Method:**

The implementation is based on the chapter "[Extending and Embedding the Python Interpreter](https://docs.python.org/3/extending/index.html)" from the official Python3 documentation.
The implementation is super lightweight, as it is only based on the CPython C/C++ application interface.
CPython is the Python language reference implementation, which is written in C.
To achieve the same, third-party modules like Cython, SWIG, or Pybind11 may have offered more sophisticated approaches and potentially faster execution, but the lightweight and bare-bones nature of our code would have to be scarified.

**Result:**

This physicellembedding project provides two PhysiCell template user projects:
1. **py3pc_embed**, to extend Python3, to embed PhysiCell into a Python3 module.
1. **pcpy3_embed**, to extend PhysiCell, to embed a Python3 interpreter into the PhysiCell main loop.

This PhysiCell template user projects can be easily installed and further developed to address the users' needs, in the same way PhysiCell models are usually developed.

**Conclusion:**

The core of the PhysiCell model will always be C++.
Nonetheless, the possibility of interacting with the model in both C++ and Python3 will permit the development of models in ways previously unimaginable.

May the force be with you!


# HowTo Guide:

+ [install py3pc_embed](https://github.com/elmbeech/physicellembedding/tree/master/man/HOWTO_py3pc.md)
+ [install pcpy3_embed](https://github.com/elmbeech/physicellembedding/tree/master/man/HOWTO_pcpy3.md)
+ [uninstall](https://github.com/elmbeech/physicellembedding/tree/master/man/HOWTO_purge.md)


# Tutorial:

+ [py3pc_embed tutorial](https://github.com/elmbeech/physicellembedding/tree/master/man/TUTORIAL_py3pc.md)
+ [pcpy3_embed tutorial](https://github.com/elmbeech/physicellembedding/tree/master/man/TUTORIAL_pcpy3.md)


# Reference Manual:

+ [reference manual](https://github.com/elmbeech/physicellembedding/tree/master/man/REFERENCE.md)


# Discussion:

To be developed.


# About Documentation:

Within the physicellembedding library, we tried to stick to the documentation policy laid out by Daniele Procida in his "[what nobody tells you about documentation](https://www.youtube.com/watch?v=azf6yzuJt54)" talk at PyCon 2017 in Portland, Oregon.


# Contributions:

+ Concept and implementation: Elmar Bucher, Alexandre Bertin
+ Testing: Marcelo Hurtado, Aneequa Sundus, Furkan Kurtoglu, John Metzcar, Randy Heiland
+ Supervision: Vincent François, Emmanuel Rachelson, Vera Pancaldi, Heber Lima da Rocha, Paul Macklin


# Cite:

To be BibTeX.


# Road Map:

+ write unit test code.
+ test on windows.
+ test on mac os intel.
+ test on mac os m.
+ continuous integration on github.
+ py3pc_embed gym torchrl backend smoketest.


# Release Notes:

+ 0.0.4 (2024-nn-nn): unit tests and continuous integration are implemented and stable.
+ 0.0.3 (2024-04-13): make the template more generic, add recall.py scripts,
                      add physicell.get_cell and physicell.get_microenv function,
                      update py3pc_embed tutorial.
+ 0.0.2 (2024-02-19): documentation is stable.
+ 0.0.1 (2024-02-09): Makefile, py3pc/pyproject.toml and py3pc/setup.py are stable.
+ 0.0.0 (2024-02-02): the basic pcpy3 and py3pc embedding works.

