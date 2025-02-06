## How to apply Reinforcement Learning on your Controllable biological model made by PhysiCell & PhysiGym

A quick definition about Reinforcement Learning (RL) is a machine learning field where the aim is to control a dynamic of a system. In RL, an important concept called Markovian decion process should be taken in order to build your controllable bliological model and to build your PhysiCell Gymnasium level child environment such as get_action, get_observation, get_reward. In fact a Markovian decision process (MDP) is composed of a state noted $S$ what the reinforcement learning algorithm will get as input, an action space $A$ how we control the process and also the reward $R$ function which is the cumulative of this reward which has to be maximized than can be seen as the goal. But a MDP is also composed of a Transition Kernel which is related to the evolution of the environment it is noted $T$.

We take as example tme, in this model the state space is the number of cancer cells, the action space is described by two drugs, the transition matrix refers to the tme model made by PhysiCell. The reward is linked to the aim, in our case the aim is to maintain the number of cancer cells thanks two drugs, the reward is the opposite absolute value of a soustraction between the current number of cancer cells and the cancer cells target.

We applied a Deep Reinforcement Learning Algorithm called SAC is adapted for continous action space. To Launch SAC on tme model you just need to install libraries, but before in your PhysiGym launch:
```
python3 install_rl_folder.py
```
The `rl`folder from PhysiGym will be installed into PhysiCell, but you can use your own Reinforcement Learning Algorithms or Folders, you have to keep it mind than the python script related to Reinforcemetn Learning should be launched since PhysiCell because PhysiCell is the main tool to compute the microenvironment.

If you have selected to use the current work you can keep follow the steps, we only focus on launching the SAC reinforcement learning algorithm on TME model.

In the sac folder there are three files, the `sac_requirements.txt`in order to install the libraries nedded, `launch_sac.sh`a script to launch the script `sac.py` multiple times with multiple different seeds. `sac.py`contains almost all the code needed to launch a SAC algorithm the other important files are in the folder `rl/utils`. In this folder, there is the replay buffer mainly used in Deep Reinforcement Learning Algorithms but also a wrapper file called `wrapper_physicell_tme.py`this wrapper is a easiest way to wrap  the environment from `model/custom_modules/physigym/physicell_model.py` from tme model ()



1. Download this repository *in the same folder where your PhysiCell is installed, right next to the PhysiCell folder, not into it*!
```bash
git clone https://github.com/Dante-Berth/PhysiGym
```

2. cd into the physigym folder and run the install_physigym.py script.
```bash
cd path/to/PhysiGym
python3 install_physigym.py template
```

3. If you are using environments, this is the time to activate the Python environment in which you would like to run physigym.

4. Check that you are hooked up to the internet because pip must be able to check for build dependencies.

5. cd into the PhysiCell folder, reset PhysiCell, load, and compile the physigym template project.
This will install two Python modules, the first one named `extending`, the second one named `physigym`.
```bash
cd ../PhysiCell
make clean data-cleanup reset
make list-user-projects
make load PROJ=physigym_template
pip3 install --force-reinstall custom_modules/physigym   # optional to install and update dependencies
make
```

6. Now you're good to go! Open a Python shell and type the following:
```python
import gymnasium
import physigym

env = gymnasium.make('physigym/ModelPhysiCellEnv')
env.reset()
env.step(action={})
env.close()

exit()
```

7. Check out the [tutorial](https://github.com/Dante-Berth/PhysiGym/blob/main/man/TUTORIAL_physigym.md) to understand what you just ran.


## How to fetch the latest version from this PhysiCell user project into this source code repository

1. Save the project in the PhysiCell folder:
```bash
cd path/to/PhysiCell
make save PROJ=physigym_myproject
```

2. Fetch the project in to the PhysiGym folder and git:
```bash
cd ../PhysiGym
python3 capture_physigym.py myproject
git status
git diff
```
