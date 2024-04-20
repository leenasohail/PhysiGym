## How to install the py3pc_embed PhysiCell user project

1. Download this repository in the same folder where your PhysiCell is installed, right next to the PhysiCell folder, not into it!
```bash
git clone https://github.com/Dante-Berth/PhysiGym
```

2. cd into the physigym folder and run the install_physigym.py script.
```bash
cd PhysiGym
python3 install_physigym.py
```

3. If you are using environments, this is the time to activate the python3 environment in which you would like to run physigym.

4. cd into the PhysiCell folder, reset PhysiCell, load, and compile the physigym project.
This will install two python3 modules the first one named `embedding`, the second one named `physigym`.

Notice: the `sed` command below is needed in PhysiCell <= 1.13\* because the load command in these Makefiles does not allow custom modules packed in a folder structure, like the python modules are.
```bash
cd ../PhysiCell
make clean data-cleanup reset
make list-user-projects
sed -i 's/cp .\/user_projects\/$(PROJ)\/custom_modules\//cp -r .\/user_projects\/$(PROJ)\/custom_modules\//' ./Makefile
make load PROJ=physigym
make
```

5. Now you're good to go! open a python3 shell and type the following:
```python
from embedding import physicell

physicell.start()
physicell.step()
physicell.get_parameter('my_bool')
physicell.set_parameter('my_bool', True)
physicell.get_parameter('my_int')
physicell.set_parameter('my_int', 9)
physicell.get_parameter('my_float')
physicell.set_parameter('my_float', 9)
physicell.get_parameter('my_str')
physicell.set_parameter('my_str', 'never daunted')
physicell.get_variable('my_variable')
physicell.set_variable('my_variable', 9)
physicell.get_vector('my_vector')
physicell.set_vector('my_vector', [0,1,2,3])
physicell.step()
physicell.stop()
exit()
```

6. Take a look at the `output` folder and marvel over the output from the two mcds time steps we run.


## How to fetch the latest version from this PhysiCell user project into this source code repository

7. fetch and git
```bash
python3 recall_physigym.py
git status
git diff
```
