# physigym and reinforcement learning with Gymnasium

## The [tme](https://github.com/Dante-Berth/PhysiGym/tree/main/model/tme) model.

< ALEX cloud you please describe the model here, like done for the episode model in the TUTORIAL_physigym.py? >

0. Install and load the model (Bash).

```bash
cd path/to/PhysiGym
python3 install_physigym.py episode
```
```bash
cd ../PhysiCell
make clean data-cleanup reset
make list-user-projects
make load PROJ=physigym_episode
```

1. Compile and run the model the classic way (Bash).

< ALEX here please some blah about the insights and output runing the model the classic way >

```bask
make classic -j8
./project
```

2. Compile the embedded way (Bash).

```bash
make
```

3. Run the model the embeded way (Python).

< ALEX here please some blah about the insights and output from running the model the embedded way >

Open a ipython shell.

```python
# library
from embedding import physicell
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

4. Reinforcement learn a policy for the model.

< ALEX, HERE COMES THE BIG CHUNK >
