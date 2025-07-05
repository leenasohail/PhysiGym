# physigym template project

+ language: c, c++, python
+ software dependencies: physicell >= v1.14.2
+ python library dependencies: gymnasium, lxml, matplotlib, numpy, pandas, (ipython, PyQt6)
+ date:
+ license: <has to be comatiple with bsb-3-clause>
+ authors: \
    original work 2015-2025, Paul Mackli, the BioFVM Project and the PhysiCell Project. \
    modified work 2024-2025, Elmar Bucher, physicell embedding. \
    modified work 2024-2025, Alexandre Bertin, Elmar Bucher, physigym. \
    modified work YYYY-YYYY, <your name goes here>.

+ description: \
    most minimalistic physigym physicell user_project.

+ source: <https:// your url goes here>

+ install:
    1. cd path/to/PhysiGym
    1. python3 install_physigym.py template
    1. cd ../PhysiCell
    1. make data-cleanup clean reset
    1. make load PROJ=physigym_template
    1. make

+ run classic c++ episodes
```bash
make classic
./project
```

+ run python controlled one episode:
```bash
python3 custom_modules/physigym/physigym/envs/run_physigym_template.py
```

+ run python controlled episodes:
```bash
python3 custom_modules/physigym/physigym/envs/run_physigym_template_eposodes.py
```

+ run python controlled reinforcemnt learning:
```bash
not implemented.
```


