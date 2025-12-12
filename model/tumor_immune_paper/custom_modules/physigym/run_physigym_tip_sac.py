#####
# title: model/tumor_immune_paper/custom_modules/physigym/run_physigym_tip_sac.py
#
# language: python3
# main libraries: gymnasium, physigym, torch
#
# date: 2024-spring
# license: BSD-3-Clause
# author: Alexandre Bertin, Elmar Bucher
# original source code: https://github.com/Dante-Berth/PhysiGym
#
# description:
#     vectorized sac implementation for tumor immune paper
#####


#### IMPORT LIBRARIES ####
# Standard Python Libraries
import argparse
import os
import random
import shutil
import time
from lxml import etree

# Non-standard Python Libraries
import matplotlib

matplotlib.use("agg")  # set the plotting backend e.g. agg qtagg

import numpy as np

import gymnasium as gym

# Load Gymnasium PhysiCell bridge module namespace physigym
import physigym

# Torch ecosystem
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import tensorboard
from torch_geometric.data import Data, Batch


# Utils code related to the project
from vectorized_tip import vec_envs
from nn_tip import Actor, QNetwork
from rb_tip import ReplayBuffer
from wrapper_tip import PhysiCellModelWrapper

# Tracking
import wandb
from tqdm import tqdm
import psutil


def flatten_dict(d, parent_key=""):
    """Flatten a nested dictionary, joining keys with dots."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def obs_to_pyg(obs, o_device):
    # obs is a batch dict:
    # node_features: (B, MAX_NODES, node_dim)
    # edge_index:    (B, 2, MAX_EDGES)
    # edge_attr:     (B, MAX_EDGES, edge_dim)
    # node_mask:     (B, MAX_NODES)
    # edge_mask:     (B, MAX_EDGES)

    graphs = []
    B = obs["node_features"].shape[0]

    for i in range(B):
        node_mask = obs["node_mask"][i] > 0.5
        edge_mask = obs["edge_mask"][i] > 0.5

        # Filter valid nodes
        x = obs["node_features"][i][node_mask]

        # Filter valid edges
        edge_index = obs["edge_index"][i][:, edge_mask]
        edge_attr = obs["edge_attr"][i][edge_mask]

        # Build a normal PyG graph (create tensors on CPU; we'll move the whole batch later)
        g = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        )
        # Do NOT set device here — Batch.to(device) is simpler/safer
        g.batch = torch.full((x.shape[0],), i, dtype=torch.long)

        graphs.append(g)

    batch = Batch.from_data_list(graphs)

    # Move entire batch to the model device — this ensures data.x, edge_index, edge_attr, batch, etc. are on o_device
    return batch.to(o_device)


###################
# Algorithm Logic #
###################
# description:
#   The code is mainly inspired from:
#   https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py


def run(
    s_settingxml="config/PhysiCell_settings.xml",  # xpath
    s_settingcells="cells.csv",  # cells csv path
    i_seed=int(42),  # int or none: seed of the experiment
    s_observation_mode="scalars_cells_substrates",  # str: observation mode
    s_render_mode=None,  # render is none or rgb_array or human
    r_max_time_episode=12900.0,  #  8[d]=12900[min] = 8 * 3 = 24[steps]
    i_total_step_learn=int(1e5),  # int: the total number of steps
    b_gpu=False,  # bool: if using GPU
    s_name="vec_sac",  # str: the name of this experiment
    b_wandb=False,  # bool: track with wandb, if false local tensorboard
    s_entity="corporate-manu-sureli",  # name of your project in wandb
    i_tumor=512,
    i_cell_1=128,
    r_cell_2_fraction=None,  # fraction of cell_1 into cell_2
    i_num_envs=6,
    s_frequency_save_data=64,
    neural_architecture_image="impala",
    rl_threads=2,
):
    d_arg_simulation = {
        # basics
        "name": s_name,  # str: the name of this experiment
        # hardware
        "cuda": b_gpu,  # bool: should torch check for gpu (nvidia cuda, amd mroc) accelerator?
        # tracking
        "wandb_track": b_wandb,  # bool: track with wandb, if false local tensorboard
        # random seed
        "seed": i_seed,  # int or none: seed of the experiment
        # steps
        "max_time": r_max_time_episode,
    }
    # wandb
    d_arg_wandb = {
        "entity": s_entity,  # str: the wandb s entity name
        "project": "SAC_VEC_TIP",  # str: the wandb s project name
        "sync_tensorboard": True,
        "monitor_gym": True,
        "save_code": True,
    }

    # physigym
    d_arg_physigym_model = {
        "id": "physigym/ModelPhysiCellEnv-v0",  # str: the id of the gymnasium environmenit
        "settingxml": s_settingxml,
        "settingcells": s_settingcells,
        "output_dir": None,
        "cell_type_cmap": {
            "tumor": "yellow",
            "cell_1": "green",
            "cell_2": "navy",
            "other_tissue": "red",
        },  # viridis
        "figsize": (6, 6),
        "observation_mode": s_observation_mode,  # str: scalars , img_rgb , img_mc, graph_neighbor, graph_delaunay
        "render_mode": s_render_mode,  # human, rgb_array
        "verbose": False,
        "img_rgb_grid_size_x": 64,  # pixel size
        "img_rgb_grid_size_y": 64,  # pixel size
        "img_mc_grid_size_x": 64,  # pixel size
        "img_mc_grid_size_y": 64,  # pixel size
        "normalization_factor": i_tumor,  # normalization factor
    }
    d_arg_physigym_wrapper = {
        "list_variable_name": ["drug_1"],  # list of str: of action varaible names
        "weight": 0.8,  # float: weight for the reduction of tumor
        "frequency_save_data": s_frequency_save_data,
    }

    # rl algorithm
    d_arg_rl = {
        "total_timesteps": i_total_step_learn,  # int: the total number of steps
        # algoritm neural network I
        "buffer_size": int(1e5),  # int: the replay memory buffer size
        "batch_size": int(
            64 * i_num_envs
        ),  # int: the batch size of sample from the replay memory
        "learning_starts": 1000,  # 20[years] float: timestep to start learning (25e3)
        "policy_frequency": 2,  # int: the frequency of training policy (delayed)
        "target_network_frequency": 1,  # int: the frequency of updates for the target nerworks (Denis Yarats" implementation delays this by 2.)
        # algorithm neural network II
        "autotune": True,  # bool: automatic tuning the the entropy coefficient.
        "alpha": 0.05,  # float: set manual entropy regularization coefficient.
        "tau": 0.005,  # float: target smoothing coefficient (default" : 0.005)
        "q_lr": 3e-4,  # float: the learning rate of the Q network network optimizer
        "policy_lr": 3e-4,  # float: the learning rate of the policy network optimizer
        # algorithm neural network III
        "gamma": 0.99,  # float: the discount factor gamma (how much learning)
    }
    d_arg_vect = {
        "num_envs": i_num_envs,
        "rl_threads": rl_threads,
    }

    # all in one
    d_arg = {}
    d_arg["simulation"] = d_arg_simulation
    d_arg["vectorization"] = d_arg_vect
    d_arg["wandb"] = d_arg_wandb
    d_arg["rl"] = d_arg_rl
    d_arg["wrapper"] = d_arg_physigym_wrapper
    d_arg["model"] = d_arg_physigym_model
    d_arg["neural_architecture_image"] = neural_architecture_image
    num_envs = d_arg["vectorization"]["num_envs"]

    model_cfg_ghost = d_arg["model"].copy()
    del model_cfg_ghost["settingcells"]
    del model_cfg_ghost["output_dir"]
    ghost_env = gym.make(**model_cfg_ghost)
    ghost_env = PhysiCellModelWrapper(ghost_env, **d_arg_physigym_wrapper)
    # gpu cpu
    if (d_arg["simulation"]["cuda"] and not torch.cuda.is_available()) or (
        not d_arg["simulation"]["cuda"] and torch.cuda.is_available()
    ):
        raise ValueError(
            f"argument cuda set {d_arg['simulation']['cuda']} but torch GPU detection {torch.cuda.is_available()}."
        )

    # initialize tracking
    s_run = f"{d_arg['simulation']['name']}_seed_{d_arg['simulation']['seed']}_observation_mode_{d_arg['model']['observation_mode']}_weight_{d_arg['wrapper']['weight']}_time_{int(time.time())}"
    if d_arg["simulation"]["wandb_track"]:
        print("tracking: wandb ...")
        run = wandb.init(name=s_run, config=d_arg["simulation"], **d_arg["wandb"])
        s_dir_run = os.path.join(run.dir, s_run)
    else:
        print("tracking tensorboard ...")
        s_dir_run = os.path.join("tensorboard", s_run)

    # initialize tensorbord recording
    writer = tensorboard.SummaryWriter(s_dir_run)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % (
            "\n".join(
                [
                    f"|{s_key}|{s_value}|"
                    for s_key, s_value in sorted(flatten_dict(d_arg).items())
                ]
            )
        ),
    )

    # set random seed
    random.seed(d_arg_simulation["seed"])
    np.random.seed(d_arg_simulation["seed"])
    if d_arg_simulation["seed"] is None:
        torch.seed()
        torch.backends.cudnn.deterministic = False
    else:
        torch.manual_seed(d_arg_simulation["seed"])
        torch.backends.cudnn.deterministic = True
    if r_cell_2_fraction is None:
        r_cell_2_fraction = [0.0, 0.25, 0.5, 0.75, 1.0]
    # d_arg_generation control the generation of initial states, you should not modify it, at your own risk
    # but you may change the number of tumor cells (n_tumor) and you may also change (n_cell_1)
    params = {
        "tumor": {"correlation_length": 45, "threshold": 0.55, "number_cells": i_tumor},
        "cell_1": {
            "correlation_length": 45,
            "threshold": 0.55,
            "number_cells": i_cell_1,
        },
    }
    d_arg_generation = {
        "x_min": ghost_env.unwrapped.x_min,
        "x_max": ghost_env.unwrapped.x_max,
        "y_min": ghost_env.unwrapped.y_min,
        "y_max": ghost_env.unwrapped.y_max,
        "params": params,
        "cell_2_fraction": r_cell_2_fraction,
        "seed": d_arg_simulation["seed"],
    }

    d_arg["generation"] = d_arg_generation

    envs = vec_envs(d_arg)
    d_arg_env = {
        "action_space_shape": ghost_env.action_space.shape,
        "observation_space_shape": ghost_env.observation_space.shape,
        "observation_mode": ghost_env.unwrapped.kwargs["observation_mode"],
        "node_feature_dim": getattr(
            ghost_env.observation_space, "node_feature_dim", None
        ),
        "x_min": ghost_env.unwrapped.x_min,
        "x_max": ghost_env.unwrapped.x_max,
        "y_min": ghost_env.unwrapped.y_min,
        "y_max": ghost_env.unwrapped.y_max,
        "action_space_high": ghost_env.action_space.high,
        "action_space_low": ghost_env.action_space.low,
        "observation_space_dtype": ghost_env.observation_space.dtype,
        "is_graph": True if "graph" in d_arg["model"]["observation_mode"] else False,
    }
    is_graph = d_arg_env["is_graph"]

    # Initialize neural networks
    o_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Networks
    actor = Actor(d_arg_env, d_arg["neural_architecture_image"]).to(o_device)
    qf1 = QNetwork(d_arg_env, d_arg["neural_architecture_image"]).to(o_device)
    qf2 = QNetwork(d_arg_env, d_arg["neural_architecture_image"]).to(o_device)
    qf1_target = QNetwork(d_arg_env, d_arg["neural_architecture_image"]).to(o_device)
    qf2_target = QNetwork(d_arg_env, d_arg["neural_architecture_image"]).to(o_device)
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=d_arg["rl"]["q_lr"]
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=d_arg["rl"]["policy_lr"])

    # Set neural network entropy alpha by automatic tuning or manual
    if d_arg["rl"]["autotune"]:
        target_entropy = -torch.prod(
            torch.Tensor(ghost_env.action_space.shape).to(o_device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=o_device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=d_arg["rl"]["q_lr"])
    else:
        alpha = d_arg["rl"]["alpha"]

    is_graph = False
    if hasattr(ghost_env.unwrapped, "kwargs"):
        obs_mode = ghost_env.unwrapped.kwargs.get("observation_mode", "")
        is_graph = "graph" in str(obs_mode)

    # Initialize the reply buffer
    rb = ReplayBuffer(
        state_dim=ghost_env.observation_space.shape,
        action_dim=ghost_env.action_space.shape,
        device=o_device,
        buffer_size=d_arg["rl"]["buffer_size"],
        batch_size=d_arg["rl"]["batch_size"],
        state_type=ghost_env.observation_space.dtype,
        is_graph=is_graph,
    )
    del ghost_env
    total_discounted_cumulative_returns = np.zeros((num_envs))
    total_cumulative_returns = np.zeros((num_envs))
    discounted_cumulative_returns = np.zeros((num_envs))
    cumulative_returns = np.zeros((num_envs))
    o_observations = envs.reset()
    total = d_arg["rl"]["total_timesteps"]
    pbar = tqdm(total=total)
    for global_step in range(d_arg["rl"]["total_timesteps"]):
        # sample the action space or learn
        if global_step <= d_arg["rl"]["learning_starts"]:
            a_actions = np.array(
                [envs.action_space.sample() for _ in range(num_envs)],
                dtype=np.float32,
            )

        else:
            if is_graph:
                x = obs_to_pyg(o_observations, o_device)
            else:
                x = torch.Tensor(o_observations).to(o_device)
            actions, _, _ = actor.get_action(x)
            a_actions = actions.detach().cpu().numpy()

        # physigym step
        o_observations_next, r_rewards, b_dones, infos = envs.step(a_actions)
        for i in range(num_envs):
            obs_i = (
                {k: v[i] for k, v in o_observations.items()}
                if is_graph
                else o_observations[i]
            )
            next_obs_i = (
                {k: v[i] for k, v in o_observations_next.items()}
                if is_graph
                else o_observations_next[i]
            )
            rb.add(
                state=obs_i,
                action=a_actions[i],
                next_state=next_obs_i,
                reward=r_rewards[i],
                done=b_dones[i],
            )
            discounted_cumulative_returns[i] += (
                r_rewards[i] * d_arg["rl"]["gamma"] ** (infos[i]["step_episode"])
            )
            cumulative_returns[i] += r_rewards[i]

        pbar.update(global_step - pbar.n)
        pbar.set_postfix(
            {"rb": len(rb), "memory": f"{psutil.virtual_memory().percent}"}
        )
        # handle observation
        o_observations = o_observations_next

        # learning
        if global_step > d_arg["rl"]["learning_starts"]:
            data = rb.sample()
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    data["next_state"]
                )
                qf1_next_target = qf1_target(data["next_state"], next_state_actions)
                qf2_next_target = qf2_target(data["next_state"], next_state_actions)
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - alpha * next_state_log_pi
                )
                next_q_value = data["reward"].flatten() + (
                    1 - data["done"].flatten()
                ) * d_arg["rl"]["gamma"] * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data["state"], data["action"]).view(-1)
            qf2_a_values = qf2(data["state"], data["action"]).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # update the target networks
            if global_step % d_arg["rl"]["target_network_frequency"] == 0:
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        d_arg["rl"]["tau"] * param.data
                        + (1 - d_arg["rl"]["tau"]) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        d_arg["rl"]["tau"] * param.data
                        + (1 - d_arg["rl"]["tau"]) * target_param.data
                    )

            # update the policy
            if global_step % d_arg["rl"]["policy_frequency"] == 0:
                for _ in range(d_arg["rl"]["policy_frequency"]):
                    pi, log_pi, _ = actor.get_action(data["state"])

                    qf1_pi = qf1(data["state"], pi)
                    qf2_pi = qf2(data["state"], pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # entropy autotune
                    if d_arg["rl"]["autotune"]:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data["state"])

                        alpha_loss = (
                            -log_alpha.exp() * (log_pi + target_entropy)
                        ).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()

                        alpha = log_alpha.exp().item()

                # record policy update to tensoboard
                """
                losses = {
                    "losses/min_qf_next_target": min_qf_next_target.mean().item(),
                    "losses/qf_loss": qf_loss.item() / 2.0,
                    "losses/actor_loss": actor_loss.item(),
                }

                if d_arg["simulation"]["wandb_track"]:
                    run.log(losses)
                else:
                    for tag, value in losses.items():
                        writer.add_scalar(
                            tag=tag, scalar_value=value, global_step=global_step
                        )
                """

        # recording episode to tensorboard
        scalars = {}
        for i in range(num_envs):
            if b_dones[i]:
                # scalars[f"charts/env_{i}_discounted_cumulative_return"] = discounted_cumulative_returns[i]
                # scalars[f"charts/env_{i}_cumulative_return"] = cumulative_returns[i]
                total_discounted_cumulative_returns[i] = discounted_cumulative_returns[
                    i
                ]
                total_cumulative_returns = cumulative_returns[i]

        if global_step > 200:
            scalars["charts/mean_discounted_cumulative_return"] = np.mean(
                total_discounted_cumulative_returns
            )
            scalars["charts/mean_cumulative_return"] = np.mean(total_cumulative_returns)
            scalars["charts/steps"] = global_step * num_envs

        if d_arg["simulation"]["wandb_track"]:
            run.log(scalars)
        else:
            for tag, value in scalars.items():
                writer.add_scalar(tag, value, global_step)

        discounted_cumulative_returns *= 1 - b_dones
        cumulative_returns *= 1 - b_dones

    # finish
    envs.close()
    writer.close()


########
# Main #
########

if __name__ == "__main__":
    print("run physigym learing ...")

    # argv
    parser = argparse.ArgumentParser(
        prog="run physigym episodes",
        description="script to run physigym episodes.",
    )

    # settingxml file
    parser.add_argument(
        "--settingxml",
        # type = str,
        nargs="?",
        default="config/PhysiCell_settings.xml",
        help="path/to/settings.xml file.",
    )
    parser.add_argument(
        "--settingcells", nargs="?", default="config/cells.csv", help="name cells.csv ."
    )
    # seed
    parser.add_argument(
        "--seed",
        # type = str,
        nargs="?",
        default="5",
        help="set options random_seed in the settings.xml file and python.",
    )
    # observation_mode
    parser.add_argument(
        "--observation_mode",
        # type = str,
        nargs="?",
        default="scalars_cells_substrates",
        help="different observation modes possible",
    )
    # render_mode
    parser.add_argument(
        "--render_mode",
        # type = str,
        nargs="?",
        default="None",
        help="render mode None, rgb_array, or human. observation mode scalars needs either render mode rgb_array or human.",
    )
    # max_time
    parser.add_argument(
        "--max_time_episode",
        type=float,
        nargs="?",
        default=12900.0,
        help="set overall max_time in min in the settings.xml file.",
    )
    # total timesteps
    parser.add_argument(
        "--total_step_learn",
        type=int,
        nargs="?",
        default=int(5e4),
        help="set total time steps for the learing process to take.",
    )
    # gpu
    parser.add_argument(
        "--gpu",
        # type=bool,
        nargs="?",
        default="true",
        help="gpu for pytorch available?",
    )
    # name
    parser.add_argument(
        "--name",
        # type = str,
        nargs="?",
        default="vec_sac",
        help="experiment name.",
    )
    # wandb tracking
    parser.add_argument(
        "--wandb",
        # type=bool,
        nargs="?",
        default="true",
        help="tracking online with wandb? false with track locally with tensorboard.",
    )
    # entity
    parser.add_argument(
        "--entity",
        # type = str,
        nargs="?",
        default="corporate-manu-sureli",
        help="weight and biases team.",
    )

    parser.add_argument(
        "--tumor",
        type=int,
        nargs="?",
        default=512,
        help="number of tumor cells",
    )
    parser.add_argument(
        "--cell_1",
        type=int,
        nargs="?",
        default=128,
        help="number of tumor cell_1",
    )
    parser.add_argument(
        "--cell_2_fraction",
        type=float,
        nargs="?",
        default=None,
        help="fraction of cell_1 into cell_2 ie 0.5 means 50%",
    )

    parser.add_argument(
        "--num_envs",
        type=int,
        nargs="?",
        default=7,
        help="number of parallelized environments",
    )

    parser.add_argument(
        "--s_frequency_save_data",
        type=int,
        nargs="?",
        default=None,
        help="each number of episode data is saved",
    )

    parser.add_argument(
        "--neural_architecture_image",
        type=str,
        nargs="?",
        default="impala",
        help="neural architecture for image it is else impala or hadamax",
    )

    parser.add_argument(
        "--rl_threads",
        type=int,
        nargs="?",
        default=4,
        help="number of threads dedicated to the reinforcement learning algorithm",
    )

    # parse arguments
    args = parser.parse_args()
    print(args)

    # processing
    run(
        s_settingxml=args.settingxml,
        s_settingcells=args.settingcells,
        i_seed=None if args.seed.lower() == "none" else int(args.seed),
        s_observation_mode=args.observation_mode,
        s_render_mode=None if args.render_mode.lower() == "none" else args.render_mode,
        r_max_time_episode=args.max_time_episode,
        i_total_step_learn=args.total_step_learn,
        i_num_envs=args.num_envs,
        b_gpu=True if args.gpu.lower().startswith("t") else False,
        s_name=args.name,
        b_wandb=True if args.wandb.lower().startswith("t") else False,
        s_entity=args.entity,
        i_tumor=args.tumor,
        i_cell_1=args.cell_1,
        r_cell_2_fraction=args.cell_2_fraction,
        s_frequency_save_data=args.s_frequency_save_data,
        neural_architecture_image=args.neural_architecture_image,
        rl_threads=args.rl_threads,
    )
