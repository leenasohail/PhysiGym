# physigym tibbue project

+ language: c, c++, python

+ software dependencies: physicell >= v1.14.2

+ python library dependencies:
    gymnasium, lxml, matplotlib, numpy, pandas, (ipython, PyQt6) \
    torch installed as described here: https://pytorch.org/get-started/locally/ \
    pip3 install -r model/tibbue/custom_modules/physigym/requirements.txt

+ date: 2025-spring

+ license: bsb-3-clause

+ authors:
    original work 2015-2025, Paul Mackli, the BioFVM Project and the PhysiCell Project. \
    modified work 2024-2025, Elmar Bucher, physicell extending. \
    modified work 2024-2025, Alexandre Bertin, Elmar Bucher, physigym. \
    modified work 2024-2025, Alexandre Bertin, Owen Griere, tumor_immune_base model. \
    modified work 2024-2025, Alexandre Bertin, sac_tib.py. \
    modified work 2025-2025, Elmar Bucher, code refactoring.

+ description: \
    simple but complex enough model for non-trivial reinforcement learning.

+ source: https://github.com/Dante-Berth/PhysiGym/tree/main/model

+ install:
```bash
cd path/to/PhysiGym
```
```bash
python3 install_physigym.py tibbue
```
```bash
cd ../PhysiCell
```
```bash
make data-cleanup clean reset
make load PROJ=physigym_tibbue
make
```

+ run classic c++ episodes
```bash
make classic
./project
```

+ run python controlled one episode:
```bash
python3 custom_modules/physigym/physigym/envs/run_physigym_tibbue.py
```

+ run python controlled episodes:
```bash
python3 custom_modules/physigym/physigym/envs/run_physigym_tibbue_eposodes.py
```

+ run python controlled reinforcement learning:
```bash
python3 custom_modules/physigym/physigym/envs/run_physigym_tibbue_sac.py
```
