# physigym tutorial project

+ language: c, c++, python
+ software dependencies: physicell >= v1.14.2
+ python library dependencies: gymnasium, lxml, matplotlib, numpy, pandas, (ipython, PyQt6)
+ date: 2024-spring
+ license: <has to be comatiple with bsb-3-clause>
+ authors: \
    original work 2015-2025, Paul Mackli, the BioFVM Project and the PhysiCell Project. \
    modified work 2024-2025, Elmar Bucher, physicell embedding. \
    modified work 2024-2025, Alexandre Bertin, Elmar Bucher, physigym. \
    modified work YYYY-YYYY, <your name goes here>.

+ description: \
    physigym physicell user_project based on the physigym tutotial. \
    + https://github.com/Dante-Berth/PhysiGym/blob/main/man/TUTORIAL_physigym.md

+ source: https://github.com/Dante-Berth/PhysiGym/tree/main/model

+ install:
    1. cd path/to/PhysiGym
    1. python3 install_physigym.py tutorial
    1. cd ../PhysiCell
    1. make data-cleanup clean reset
    1. make load PROJ=physigym_tutorial
    1. make

+ run classic c++ episodes
```bash
make classic
./project
```

+ run python controlled one episode:
```bash
python3 custom_modules/physigym/physigym/envs/run_physigym_tutorial.py
```

+ run python controlled episodes:
```bash
python3 custom_modules/physigym/physigym/envs/run_physigym_tutorial_eposodes.py
```

+ run python controlled reinforcement learning:
```bash
not implemented.
```
