# Physigym and Reinforcement Learning with Gymnasium

How to find the best treatment regime of any controllalble biological model ?

## How to apply Reinforcement Learning on your Controllable biological model made by PhysiCell & PhysiGym

A quick definition about Reinforcement Learning (RL) is a machine learning field where the aim is to control a dynamic of a system. In RL, an important concept called Markovian decion process should be taken in order to build your controllable bliological model and to build your PhysiCell Gymnasium level child environment such as get_action, get_observation, get_reward. In fact a Markovian decision process (MDP) is composed of a state noted $S$ what the reinforcement learning algorithm will get as input, an action space $A$ how we control the process and also the reward $R$ function which is the cumulative of this reward which has to be maximized than can be seen as the goal. But a MDP is also composed of a Transition Kernel which is related to the evolution of the environment it is noted $T$.

## The [tme](https://github.com/Dante-Berth/PhysiGym/tree/main/model/tme) model.

We take as example tme, in this model the state space is the number of cancer cells, the action space is described by two drugs, the transition matrix refers to the tme model made by PhysiCell. The reward is linked to the aim, in our case the aim is to maintain the number of cancer cells thanks two drugs, the reward is the opposite absolute value of a soustraction between the current number of cancer cells and the cancer cells target.
## Tme model description
### Cell Types
The model consists of two types of cells:

- **Cancer Cells**: These cells can divide and die through apoptosis. They can produce a chemical that attracts nurse cells.
- **Nurse Cells**: These cells produce an anti-apoptosis chemical that reduces the apoptosis rate of cancer cells and increases their division rate. This chemical essentially protects and feeds the cancer cells.

### Cancer Cells
- **Division**: Cancer cells can proliferate within the system.
- **Apoptosis**: They can undergo programmed cell death.
- **Chemical Production**: Cancer cells produce a stress chemical (`stress_chemical_cc`) that attracts nurse cells.

### Nurse Cells
- `antiapoptosis_chemical_nc`: Nurse cells produce an anti-apoptosis chemical  that:
  - Reduces the apoptosis rate of cancer cells.
  - Increases the division rate of cancer cells.

Two types of drugs can be applied to the system:

- `drug_apoptosis`: Increases the apoptosis rate of cancer cells.
- `drug_reducing_antiapoptosis`: Deactivates the protein produced by nurse cells, rendering them ineffective. This makes nurse cells unable to protect and feed cancer cells.

A quick view of tme model:
![Same TME model with three different representations, cancer cells are in grey hwile nurse cells are in red. THe first presentation is the raw representation , while the second one is displaying the production of stress chemicall cc and the last image represents the production of antiapoptosis chemical nc produced by nurse cells](https://github.com/Dante-Berth/PhysiGym/blob/main/man/img/tme_model.png)
The presentation presented can be modified given different intial state, the initial state is given by  `cells.csv`.
man/img/tme_model.png

## Installation/Running tme model

0. Install and load the model (Bash).

Install tme model:

```bash
cd path/to/PhysiGym
python3 install_physigym.py tme -f
```

Load the model:

```bash
cd ../PhysiCell
make clean data-cleanup reset
make list-user-projects
make load PROJ=physigym_episode
```

1. Compile and run the model the classic way (Bash).


For model development, it is sometimes useful to be able to compile and run the model the old-fashioned way.
In fact, this is the only reason why we kept the original [main.cpp](https://github.com/Dante-Berth/PhysiGym/blob/main/physigym/main.cpp) in the physigym code base.
Physigym as such is written on top of the [physicell embedding](https://github.com/elmbeech/physicellembedding) python_with_physicell module,
for which the main.cpp file had to be ported to [custom/extending/physicellmodule.cpp](https://github.com/Dante-Berth/PhysiGym/blob/main/model/template/custom_modules/extending/physicellmodule.cpp) that you can find in the physigym code base too.



```bash
make classic -j8
./project
```

2. Compile the embedded way (Bash).

```bash
make
```

3. Run the model the embeded way (Python).

Open a ipython shell. If ipython not installed you can download it thanks to pip.

```python
# library
from extending import physicell
import gymnasium
import numpy as np
import physigym
from random import randrange

# load PhysiCell Gymnasium environment
%matplotlib
env = gymnasium.make(
    'physigym/ModelPhysiCellEnv-v0',
    settingxml='config/PhysiCell_settings.xml',
    figsize=(8,6),
    render_mode='human',
    render_fps=10
)

# reset the environment
r_reward = 0.0
o_observation, d_info = env.reset()

# time step loop
b_episode_over = False
while not b_episode_over:

    # policy according to o_observation
    d_observation = o_observation
    d_action = {
        "drug_apoptosis": np.array([randrange(30)]),
        "drug_reducing_antiapoptosis": np.array([randrange(30)]),
    }

    # action
    o_observation, r_reward, b_terminated, b_truncated, d_info = env.step(d_action)
    b_episode_over = b_terminated or b_truncated

# drop the environment
env.close()
```
But before taking control, let's dive deep into the [`physicell_model.py`](https://github.com/Dante-Berth/PhysiGym/blob/main/model/tme/custom_modules/physigym/physicell_model.py), which is a child class of [`physicell_core.py`](https://github.com/Dante-Berth/PhysiGym/blob/main/physigym/custom_modules/physigym/physigym/envs/physicell_core.py).  

The code contains important functions such as `get_action_space`, `get_observation_space`, `get_observation`, `get_info`, `get_terminated`, and `get_reward`. Each function should be modified for each new model, but in our case, we are focused on the TME model.  

In this problem, where we aim to control the environment using two drugs, we understand that the function `get_action_space` consists of two drugs:
```python
def get_action_space(self):
    d_action_space = spaces.Dict(
            {
            "drug_apoptosis": spaces.Box(
                    low=0.0, high=30.0, shape=(1,), dtype=np.float64
            ),
            "drug_reducing_antiapoptosis": spaces.Box(
                    low=0.0, high=30.0, shape=(1,), dtype=np.float64
            ),
            }
        )
    return d_action_space
```
The lowest value is zero, and the maximum value is 30. In Gymnasium, you need to note both the minimum and maximum values. The maximum value might seem unusual, but it is merely a parameter representing toxicity. If the value is too low, it means the maximum dosage of the drug would be sufficient to regulate the environment.

Be careful, if you modify certain rules of your environment, you may need to increase the maximum value, as the drug’s efficiency will be affected. However, this value will be normalized in our reinforcement learning algorithm. Mathematically, the action space is noted $A$ which refers to $[0,30]^{2}$ and $a=(drug apoptosis, drug reducing antiapoptosis) \ in A$ and $a'$ is noted as the next action (next time step).

The reward is defined as $r(t)=-|c_{t}-c|$ where $c_{t}$ represents the number of cancer cells while $c$ represents the number of target of cancer cells.
```python
def get_action_space(self):
    return -np.abs(self.nb_cancer_cells - self.cell_count_cancer_cell_target)
```

\( c \), the target number of cancer cells, is determined by an element from the child class. This element is computed during initialization using:  

```python
self.cell_count_cancer_cell_target = int(self.x_root.xpath("//cell_count_cancer_cell_target")[0].text)
```
It is important to note that in this reward function, the maximum reward the algorithm can achieve occurs when $c_{t}=c$.

The observation in our case is an image of cells. For more details, you can go to the code ```get_observation```but the output of it is an image height and width of parameters in the ```PhysiCell.xml``` file. The number of channels can be RGB or can be gray.



4. Reinforcement learn a policy for the model.

4.1 Introduction
In Reinforcement Learning (RL), the aim is to maxmize the expected cumulative reward with $\gamma$ (discount factor), $r_t$ the reward function, $\pi$ the policy which can be seen as the strategy, $s_0$ the initial state given by ``cells.csv``.
```math
\underset{<constraints>}{\operatorname{<argmax>}}_{\pi}\mathbb{E}\left[\sum_{t=0}^{T} \gamma^t r_t \mid s_0 = s, \pi \right].
```
The aim of an agent is to maximizes it, in the next chapter, we would use a deep reinforcement learning algorithm to solve our problem.

4.2 Description folder rl
We applied a Deep Reinforcement Learning Algorithm called [SAC](https://arxiv.org/pdf/1812.05905) is adapted for continous action space. To Launch SAC on tme model you just need to install libraries, but before in your PhysiGym launch:
```
python3 install_rl_folder.py
```
The `rl`folder from PhysiGym will be installed into PhysiCell, but you can use your own Reinforcement Learning Algorithms, you have to keep it mind than the python script related to Reinforcemetn Learning should be launched since PhysiCell because PhysiCell is the main tool to compute the microenvironment.

If you have selected to use the current work you can keep follow the steps, we only focus on launching the SAC reinforcement learning algorithm on TME model.

In the sac folder there are three files, the [`sac_requirements.txt`](https://github.com/Dante-Berth/PhysiGym/blob/main/rl/sac/sac_requirements.txt) in order to install the libraries needed, [`launch_sac.sh`](https://github.com/Dante-Berth/PhysiGym/blob/main/rl/sac/launch_sac.sh) a script to launch [`sac.py`](https://github.com/Dante-Berth/PhysiGym/blob/main/rl/sac/sac.py) multiple times with multiple different seeds. [`sac.py`](https://github.com/Dante-Berth/PhysiGym/blob/main/rl/sac/sac.py) contains almost all the code needed to launch a SAC algorithm the other important files are in the folder [`rl/utils`](https://github.com/Dante-Berth/PhysiGym/tree/main/rl/utils), in this folder, there is a replay buffer mainly used in Deep Reinforcement Learning Algorithms but also a wrapper file called [`wrapper_physicell_tme.py`](https://github.com/Dante-Berth/PhysiGym/blob/main/rl/utils/wrappers/wrapper_physicell_tme.py) contains wrapper for [`physicell_model.py`](https://github.com/Dante-Berth/PhysiGym/blob/main/model/tme/custom_modules/physigym/physicell_model.py).

4.3 Launch SAC

First install the libraries needed by:
```bash
pip install -r rl/sac/sac_requirements.txt
```
In the [`sac_requirements.txt`](https://github.com/Dante-Berth/PhysiGym/blob/main/rl/sac/sac_requirements.txt), you may figure out a library called [wandb](https://wandb.ai/site). This popular library allows you to save results in the cloud. To use it, you need to create an account before launching SAC and saving results.

Depending on your available resources, you can run multiple seeds in parallel. In reinforcement learning, it is recommended to run multiple seeds to ensure consistency regardless of the seed used.

Finally, navigate to the root of your PhysiCell folder.
```bash
./rl/sac/launch_sac.sh
```