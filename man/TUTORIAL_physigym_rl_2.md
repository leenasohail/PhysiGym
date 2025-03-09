# Physigym and Reinforcement Learning with Gymnasium
In this tutorial, you will learn how to use reinforcement learning for your controllable biological model.
We will take tme model as example [tme](https://github.com/Dante-Berth/PhysiGym/tree/main/model/tme) model.
First, we have to recall somme important elements in Reinforcement Learning.

## Reinforcement Learning theory and example
In reinforcement learning, our aim is to maximize the expected cumulative reward. In the name expected cumulative reward, there is reward which is a function which helps the learning agent to have a feed back. Because in reinforcement learning, it is mostly trial and erros, besides, the learning agent see the env that can be scalars but in our case of having an agent based model , we can feed to our agent images for instance images are fed to our learning agent. Given the image, the data recieverd coming from tumor environment or the environment in general. Given it, the learning agent would output an action and given this action a reward would be recieved by the agent to know if its action given the state was great or not.
The action in the case of our tme is a set of two drugs.

## Installation tme

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

2. Compile the embedded way (Bash).

```bash
make
```
## Applying Deep Reinforcement Learning on tme
The state space is an image, thus we would use Deep Reinforcement Learning in order to 

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