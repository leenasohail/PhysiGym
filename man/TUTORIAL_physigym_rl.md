# physigym and reinforcement learning with Gymnasium

How to find the best treatment regime of any controllalble biological model ?

## How to apply Reinforcement Learning on your Controllable biological model made by PhysiCell & PhysiGym

A quick definition about Reinforcement Learning (RL) is a machine learning field where the aim is to control a dynamic of a system. In RL, an important concept called Markovian decion process should be taken in order to build your controllable bliological model and to build your PhysiCell Gymnasium level child environment such as get_action, get_observation, get_reward. In fact a Markovian decision process (MDP) is composed of a state noted $S$ what the reinforcement learning algorithm will get as input, an action space $A$ how we control the process and also the reward $R$ function which is the cumulative of this reward which has to be maximized than can be seen as the goal. But a MDP is also composed of a Transition Kernel which is related to the evolution of the environment it is noted $T$.

## The [tme](https://github.com/Dante-Berth/PhysiGym/tree/main/model/tme) model.

We take as example tme, in this model the state space is the number of cancer cells, the action space is described by two drugs, the transition matrix refers to the tme model made by PhysiCell. The reward is linked to the aim, in our case the aim is to maintain the number of cancer cells thanks two drugs, the reward is the opposite absolute value of a soustraction between the current number of cancer cells and the cancer cells target.
## tme model description
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
        "drug_apoptosis": np.array([randrange(30*45)]),
        "drug_reducing_antiapoptosis": np.array([randrange(30*45)]),
    }

    # action
    o_observation, r_reward, b_terminated, b_truncated, d_info = env.step(d_action)
    b_episode_over = b_terminated or b_truncated

# drop the environment
env.close()
```
Congratulations :huggingface:, you can control the tme model. But before controlling, let's deep dive into the [physicell_model.py](https://github.com/Dante-Berth/PhysiGym/blob/main/model/tme/custom_modules/physigym/physicell_model.py) which is the child class from 

4. Reinforcement learn a policy for the model.

< ALEX, HERE COMES THE BIG CHUNK >
