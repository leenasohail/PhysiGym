<!--
# find files with traling spaces:
find . -type f -exec egrep -l " +$" {} \;

# tag commit
git tag -a v0.0.0 -m'version 0.0.0'
git push --tag
-->

![physigym logo & title](man/img/logo/physigym_title_v0.0.0.png)

PhysiGym is still BETA!

# Header:

+ Language: C++11 and Python [>= 3.9](https://devguide.python.org/versions/)
+ Software dependencies: PhysiCell >= v1.14.2
+ Python library dependencies: gymnasium, lxml, matplotlib, numpy, pandas, (ipython, PyQt6)
+ Operating system dependencies: compiles on Linux, Windows Subsystem for Linux, and Mac OS X.
+ Author: Alexandre Bertin, Elmar Bucher
+ Date: 2024-spring
+ Doi:
+ License: [BSD-3-Clause](https://en.wikipedia.org/wiki/BSD_licenses)
+ User manual: this README.md file
+ Source code: [https://github.com/Dante-Berth/PhysiGym](https://github.com/Dante-Berth/PhysiGym)


# Abstract:

PhysiCell is a physics-based cell simulator for 3D multicellular systems.
More precisely, [PhysiCell](https://github.com/MathCancer/PhysiCell) is an agent-based model and diffusion transport solver that is off-lattice, center-based, multiscale in space and time, and written in [C++](https://en.wikipedia.org/wiki/C%2B%2B).

[Gymnasium](https://gymnasium.farama.org/main/) is the API standard for reinforcement learning, written in [Python](https://en.wikipedia.org/wiki/Python_(programming_language)).

The Python-based physigym module presented here was written on top of the python\_with\_physicell extending module from the [physicellembedding](https://github.com/elmbeech/physicellembedding) project, which makes it possible to extend the python interpreter to interact with PhysiCell models in the Python language.

Both, physigym and extending, are PhysiCell custom\_modules.

Walking through the tutorial, you will gain the understanding needed to tackle more complex PhysiCell-based reinforcement learning projects.
First, you will set up a very basic template model that can figure as a starting point for your own project.
Then you will learn on a somewhat more complex tumor\_immune\_base model how to deep reinforcement learn a policy using the sac (soft actor critic) algorithm.

**Limitations:** Because of the way how PhysiCell is implemented and run, it is not possible to generate more than one PhysiCell Gymnasium environment per runtime. A runtime warning will be thrown if you try to do so.

May the force be with you!


# &#x1F9E9; HowTo Guide:

+ [install PhysiCell](https://github.com/Dante-Berth/PhysiGym/blob/main/man/HOWTO_physicell.md)
+ [install and troubleshoot the physigym user_project](https://github.com/Dante-Berth/PhysiGym/blob/main/man/HOWTO_physigym.md)
+ [uninstall the physigym user_project](https://github.com/Dante-Berth/PhysiGym/blob/main/man/HOWTO_purge.md)


# &#x1F9E9; Tutorial:

+ [physigym modelling tutorial](https://github.com/Dante-Berth/PhysiGym/blob/main/man/TUTORIAL_physigym_model.md)
+ [physigym reinforcement learning tutorial](https://github.com/Dante-Berth/PhysiGym/blob/main/man/TUTORIAL_physigym_rl.md)


# &#x1F9E9; Reference Manual:

+ [reference manual](https://github.com/Dante-Berth/PhysiGym/blob/main/man/REFERENCE.md)
+ [class ModelPhysiCellEnv gymnasium environment structure](https://github.com/Dante-Berth/PhysiGym/blob/main/man/ModelPhysiCellEnv_struct.md)


# Discussion:

Check out our preprint at bioRxiv.
+ https://doi.org/10.1101/2025.09.18.677030


# About Documentation:

Within the PhysiGym library, we tried to stick to the documentation policy laid out by Daniele Procida in his "[what nobody tells you about documentation](https://www.youtube.com/watch?v=azf6yzuJt54)" talk at PyCon 2017 in Portland, Oregon.


# Contributions:

+ Concept and implementation: Alexandre Bertin, Elmar Bucher
+ Involved: Emmanuel Rachelson, Heber Lima da Rocha, Marcelo Hurtado, Owen Griere, Paul Macklin, Randy Heiland, Vera Pancaldi, Vincent François

If you like to contribute to the software with models or rl algorithms, please make a pull request to the [development branch](https://github.com/Dante-Berth/PhysiGym/tree/development).


# Funding:

This material is based upon research supported by the [Chateaubriand Fellowship](https://chateaubriand-fellowship.org/) from the Office for Science & Technology from the Embassy of France in the United States, the [Occitanie Region Toulouse](https://www.laregion.fr/), France, and [Inserm](https://www.inserm.fr/en/home/), France. This research was also supported in part by Lilly Endowment, Inc., through its support for the [Indiana University Pervasive Technology Institute](https://pti.iu.edu/).


# Cite:

To be BibTeX.


# Road Map:

+ Add more models and rl algorithm.
+ Vectorize the physigym environment with the python multiprocessing library, taking inspiration from the stable-baseline3 SubprocVecEnv. 


# Release Notes:

+ 1.0.0 (2025-07-14): public beta release.
+ 0.4.1 (2025-06-21): bue ok release.
+ 0.4.0 (2025-02-28): models, rl, os portability, unit tests, and documentation complete.
+ 0.3.1 (2025-02-05): custom\_modules/embedding to custom\_module/extending change.
+ 0.3.0 (2025-01-16): complete physigym PhysiCell >= v1.14.2 compatibility.
+ 0.2.1 (2024-11-29): physicell gymnasium environment limitation handling.
+ 0.2.0 (2024-10-27): physigym PhysiCell v1.13.1 to v1.14.n adaption.
+ 0.1.0 (2024-06-20): the basic physigym implementation works.
+ 0.0.0 (2024-04-15): physigym rises from the ashes.
