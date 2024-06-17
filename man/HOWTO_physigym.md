## How to install the physigym PhysiCell user project

1. Download this repository in the same folder where your PhysiCell is installed, right next to the PhysiCell folder, not into it!
```bash
git clone https://github.com/Dante-Berth/PhysiGym
```

2. cd into the physigym folder and run the install_physigym.py script.
```bash
cd path/to/PhysiGym
python3 install_physigym.py
```

3. If you are using environments, this is the time to activate the python3 environment in which you would like to run physigym.

4. cd into the PhysiCell folder, reset PhysiCell, load, and compile the physigym project.
This will install two python3 modules, the first one named `embedding`, the second one named `physigym`.

Notice: the `sed` command below is needed in PhysiCell <= 1.13\* because the load command in these Makefiles does not allow custom modules packed in a folder structure, like the python modules are.
This step will no longer be needed with the coming PhysiCell release.
```bash
cd ../PhysiCell
make clean data-cleanup reset
make list-user-projects
sed -i 's/cp .\/user_projects\/$(PROJ)\/custom_modules\//cp -r .\/user_projects\/$(PROJ)\/custom_modules\//' ./Makefile
make load PROJ=physigym
make
```

5. Now you're good to go! Open a python3 shell and type the following:
```python
import gymnasium
import physigym

env = gymnasium.make('physigym/ModelPhysiCellEnv')
env.reset()
env.step(action={})
env.close()

exit()
```

6. Check out the [tutorial](https://github.com/Dante-Berth/PhysiGym/blob/main/man/TUTORIAL_physigym.md) to understand what you just ran.


## How to fetch the latest version from this PhysiCell user project into this source code repository

1. Save the project in the PhysiCell foler:
```bash
cd path/to/PhysiCell
make save PROJ=physigym
```

2. Fetch the project in to the PhysiGym folder and git:
```bash
cd ../PhysiGym
python3 recall_physigym.py
git status
git diff
```
