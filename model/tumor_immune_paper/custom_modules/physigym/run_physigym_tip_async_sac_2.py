# run_physigym_tip_sac_async.py
import argparse
import os
import random
import time
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
import wandb
from torch.utils.tensorboard import SummaryWriter

# Your project imports
from vectorized_tip import vec_envs
from nn_tip import Actor, QNetwork
from rb_tip import ReplayBuffer
from wrapper_tip import PhysiCellModelWrapper
from torch_geometric.data import Data, Batch

# --------------------------------------------------------------
# Helper: convert dict-of-arrays → PyG Batch (same as your original)
# --------------------------------------------------------------
def obs_to_pyg(obs_dict, device):
    graphs = []
    B = obs_dict["node_features"].shape[0]
    for i in range(B):
        node_mask = obs_dict["node_mask"][i] > 0.5
        edge_mask = obs_dict["edge_mask"][i] > 0.5

        x = obs_dict["node_features"][i][node_mask]
        edge_index = obs_dict["edge_index"][i][:, edge_mask]
        edge_attr = obs_dict["edge_attr"][i][edge_mask]

        g = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        )
        g.batch = torch.full((x.shape[0],), i, dtype=torch.long)
        graphs.append(g)

    batch = Batch.from_data_list(graphs)
    return batch.to(device)


# --------------------------------------------------------------
# Actor process – runs PhysiCell envs and pushes transitions
# --------------------------------------------------------------
def actor_process(policy_queue, sample_queue, d_arg, ghost_env):
    # One actor → one process → runs ALL vectorized envs
    seed = d_arg["simulation"]["seed"] or 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    envs = vec_envs(d_arg)  # ← This already creates N parallel PhysiCell instances
    is_graph = "graph" in d_arg["model"]["observation_mode"]

    actor = Actor(ghost_env, d_arg.get("neural_architecture_image", "impala")).cpu()
    actor.eval()

    obs = envs.reset()

    while True:
        # Update policy (non-blocking)
        try:
            while not policy_queue.empty():
                new_params = policy_queue.get_nowait()
                actor.load_state_dict(new_params)
        except:
            pass

        with torch.no_grad():
            if is_graph:
                pyg_batch = obs_to_pyg(obs, "cpu")
                actions, _, _ = actor.get_action(pyg_batch)
            else:
                x = torch.from_numpy(obs).cpu()
                actions, _, _ = actor.get_action(x)
            actions = actions.cpu().numpy()

        next_obs, rewards, dones, infos = envs.step(actions)

        # Push ALL transitions from the vectorized batch
        for i in range(envs.num_envs):
            if is_graph:
                o = {k: v[i] for k, v in obs.items()}
                no = {k: v[i] for k, v in next_obs.items()}
            else:
                o = obs[i]
                no = next_obs[i]

            sample_queue.put((
                o,
                actions[i],
                float(rewards[i]),
                no,
                bool(dones[i])
            ))

        obs = next_obs
        time.sleep(0.0001)  # prevent 100% CPU


# --------------------------------------------------------------
# Main learner + async runner
# --------------------------------------------------------------
def run_async_sac(d_arg):
    mp.set_start_method("spawn", force=True)

    device = torch.device("cuda" if d_arg["simulation"]["cuda"] and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ghost env to get shapes / spaces
    model_cfg = d_arg["model"].copy()
    del model_cfg["settingcells"], model_cfg["output_dir"]
    ghost_env = gym.make(**model_cfg)
    ghost_env = PhysiCellModelWrapper(ghost_env, **d_arg["wrapper"])

    is_graph = "graph" in d_arg["model"]["observation_mode"]
    d_arg["is_graph"] = is_graph

    # Networks
    actor = Actor(ghost_env, d_arg.get("neural_architecture_image", "impala")).to(device)
    qf1 = QNetwork(ghost_env, d_arg.get("neural_architecture_image", "impala")).to(device)
    qf2 = QNetwork(ghost_env, d_arg.get("neural_architecture_image", "impala")).to(device)
    qf1_target = deepcopy(qf1).to(device)
    qf2_target = deepcopy(qf2).to(device)

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=d_arg["rl"]["q_lr"])
    actor_optimizer = optim.Adam(actor.parameters(), lr=d_arg["rl"]["policy_lr"])

    # Alpha (entropy)
    if d_arg["rl"]["autotune"]:
        target_entropy = -np.prod(ghost_env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = optim.Adam([log_alpha], lr=d_arg["rl"]["q_lr"])
        alpha = log_alpha.exp().item()
    else:
        alpha = d_arg["rl"]["alpha"]

    # Replay buffer
    rb = ReplayBuffer(
        buffer_size=d_arg["rl"]["buffer_size"],
        batch_size=d_arg["rl"]["batch_size"],
        device=device,
        is_graph=is_graph,
    )

        # === ONLY ONE ACTOR PROCESS ===
    policy_queue = mp.Queue(maxsize=10)
    sample_queue = mp.Queue(maxsize=30000)

    # Start exactly ONE actor process
    actor_proc = mp.Process(
        target=actor_process,
        args=(policy_queue, sample_queue, d_arg, ghost_env),
        daemon=True
    )
    actor_proc.start()


    # Send initial policy
    policy_queue.put(actor.state_dict())

    # Logging
    run_name = f"{d_arg['simulation']['name']}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    if d_arg["simulation"]["wandb_track"]:
        wandb.init(project="SAC_ASYNC_TIP", name=run_name, config=d_arg)

    global_step = 0
    start_time = time.time()

    print("Starting training loop...")
    while global_step < d_arg["rl"]["total_timesteps"]:
        # Collect samples
        samples_collected = 0
        while samples_collected < 2048 and not sample_queue.empty():
            try:
                obs, act, rew, next_obs, done = sample_queue.get_nowait()
                rb.add(obs, act, next_obs, rew, done)
                global_step += 1
                samples_collected += 1
            except:
                break

        if len_rb = len(rb)
        if len_rb < max(5000, d_arg["rl"]["batch_size"]):
            time.sleep(0.1)
            continue

        # Sample batch and train
        batch = rb.sample()

        # SAC update
        with torch.no_grad():
            next_actions, next_log_pi, _ = actor.get_action(batch.next_state)
            q1_next = qf1_target(batch.next_state, next_actions)
            q2_next = qf2_target(batch.next_state, next_actions)
            min_q_next = torch.min(q1_next, q2_next) - alpha * next_log_pi
            next_q = batch.reward.flatten() + (1 - batch.done.flatten()) * d_arg["rl"]["gamma"] * min_q_next.squeeze()

        q1 = qf1(batch.state, batch.action).view(-1)
        q2 = qf2(batch.state, batch.action).view(-1)
        qf1_loss = F.mse_loss(q1, next_q)
        qf2_loss = F.mse_loss(q2, next_q)
        qf_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()

        # Policy & alpha update
        if global_step % d_arg["rl"]["policy_frequency"] == 0:
            for _ in range(d_arg["rl"]["policy_frequency"]):
                actions, log_pi, _ = actor.get_action(batch.state)
                q1_pi = qf1(batch.state, actions)
                q2_pi = qf2(batch.state, actions)
                min_q_pi = torch.min(q1_pi, q2_pi)

                actor_loss = (alpha * log_pi - min_q_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if d_arg["rl"]["autotune"]:
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy).detach()).mean()
                    alpha_optim.zero_grad()
                    alpha_loss.backward()
                    alpha_optim.step()
                    alpha = log_alpha.exp().item()

        # Target network soft update
        if global_step % d_arg["rl"]["target_network_frequency"] == 0:
            tau = d_arg["rl"]["tau"]
            for p, p_t in zip(qf1.parameters(), qf1_target.parameters()):
                p_t.data.copy_(tau * p.data + (1 - tau) * p_t.data)
            for p, p_t in zip(qf2.parameters(), qf2_target.parameters()):
                p_t.data.copy_(tau * p.data + (1 - tau) * p_t.data)

        # Send new policy to actors
        if global_step % 1000 == 0:
            try:
                policy_queue.put_nowait(actor.state_dict())
            except:
                pass

        # Logging
        if global_step % 2000 == 0:
            fps = global_step / (time.time() - start_time)
            print(f"Step {global_step//1000}k | Buffer {len_rb//1000}k | FPS {fps:.1f}")

            log_dict = {
                "train/step": global_step,
                "train/fps": fps,
                "train/buffer_size": len_rb,
                "losses/qf_loss": qf_loss.item(),
                "losses/actor_loss": actor_loss.item() if 'actor_loss' in locals() else 0,
                "losses/alpha": alpha,
            }
            writer.add_scalar("charts/fps", fps, global_step)
            if d_arg["simulation"]["wandb_track"]:
                wandb.log(log_dict)
            else:
                for k, v in log_dict.items():
                    writer.add_scalar(k, v, global_step)

    print("Training finished!")
    for p in processes:
        p.terminate()
    writer.close()
    if d_arg["simulation"]["wandb_track"]:
        wandb.finish()


# --------------------------------------------------------------
# Entry point
# --------------------------------------------------------------
if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    print("Starting asynchronous SAC for PhysiGym...")

    parser = argparse.ArgumentParser(
        prog="run physigym episodes",
        description="Asynchronous SAC with PhysiCell + PyG graph support",
    )

    # === All your original arguments ===
    parser.add_argument("--settingxml", nargs="?", default="config/PhysiCell_settings.xml")
    parser.add_argument("--settingcells", nargs="?", default="cells.csv")
    parser.add_argument("--seed", nargs="?", default="5")
    parser.add_argument("--observation_mode", nargs="?", default="graph_neighbor")  # change default if you want
    parser.add_argument("--render_mode", nargs="?", default="None")
    parser.add_argument("--max_time_episode", type=float, nargs="?", default=12900.0)
    parser.add_argument("--total_step_learn", type=int, nargs="?", default=int(2e6))
    parser.add_argument("--gpu", nargs="?", default="true")
    parser.add_argument("--name", nargs="?", default="async_sac_tip")
    parser.add_argument("--wandb", nargs="?", default="true")
    parser.add_argument("--entity", nargs="?", default="corporate-manu-sureli")
    parser.add_argument("--init_mode", nargs='+', default=['circular_mode', 'asymmetric_mode', 'connected_mst_mode'])
    parser.add_argument("--tumor", type=int, nargs="?", default=512)
    parser.add_argument("--cell_1", type=int, nargs="?", default=128)
    parser.add_argument("--cell_2_fraction", type=float, nargs="?", default=None)
    parser.add_argument("--num_envs", type=int, nargs="?", default=7)           # total parallel PhysiCell instances
    parser.add_argument("--s_frequency_save_data", type=int, nargs="?", default=64)
    parser.add_argument("--neural_architecture_image", type=str, nargs="?", default="impala")
    parser.add_argument("--rl_threads", type=int, nargs="?", default=4)
    parser.add_argument("--num_actors", type=int, nargs="?", default=None)       # optional: override auto-detect

    args = parser.add_argument("--buffer_size", type=int, nargs="?", default=int(1e6))
    parser.add_argument("--batch_size_multiplier", type=int, nargs="?", default=64)

    args = parser.parse_args()
    print("Arguments:", args)

    # === Build d_arg exactly like your original run() function ===
    i_seed = None if str(args.seed).lower() == "none" else int(args.seed)
    b_gpu = args.gpu.lower().startswith("t")
    b_wandb = args.wandb.lower().startswith("t")
    s_render_mode = None if args.render_mode.lower() == "none" else args.render_mode

    if args.cell_2_fraction is None:
        r_cell_2_fraction = [0.0, 0.25, 0.5, 0.75, 1.0]
    else:
        r_cell_2_fraction = [args.cell_2_fraction]

    d_arg_simulation = {
        "name": args.name,
        "cuda": b_gpu,
        "wandb_track": b_wandb,
        "seed": i_seed,
        "max_time": args.max_time_episode,
    }

    d_arg_wandb = {
        "entity": args.entity,
        "project": "SAC_ASYNC_TIP",
        "sync_tensorboard": True,
        "monitor_gym": True,
        "save_code": True,
    }

    d_arg_physigym_model = {
        "id": "physigym/ModelPhysiCellEnv-v0",
        "settingxml": args.settingxml,
        "settingcells": args.settingcells,
        "output_dir": None,
        "cell_type_cmap": {"tumor": "yellow", "cell_1": "green", "cell_2": "navy", "other_tissue": "red"},
        "figsize": (6, 6),
        "observation_mode": args.observation_mode,
        "render_mode": s_render_mode,
        "verbose": False,
        "img_rgb_grid_size_x": 64,
        "img_rgb_grid_size_y": 64,
        "img_mc_grid_size_x": 64,
        "img_mc_grid_size_y": 64,
        "normalization_factor": args.tumor,
    }

    d_arg_physigym_wrapper = {
        "list_variable_name": ["drug_1"],
        "weight": 0.8,
        "frequency_save_data": args.s_frequency_save_data or 64,
    }

    d_arg_rl = {
        "total_timesteps": args.total_step_learn,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size_multiplier * args.num_envs,   # e.g. 64 × num_envs
        "learning_starts": 5000,
        "policy_frequency": 2,
        "target_network_frequency": 1,
        "autotune": True,
        "alpha": 0.05,
        "tau": 0.005,
        "q_lr": 3e-4,
        "policy_lr": 3e-4,
        "gamma": 0.99,
    }

    d_arg_vect = {
        "num_envs": args.num_envs,
        "rl_threads": args.rl_threads,
    }

    # === Final d_arg ===
    d_arg = {
        "simulation": d_arg_simulation,
        "vectorization": d_arg_vect,
        "wandb": d_arg_wandb,
        "rl": d_arg_rl,
        "wrapper": d_arg_physigym_wrapper,
        "model": d_arg_physigym_model,
        "neural_architecture_image": args.neural_architecture_image,  # passed to Actor/QNetwork
    }

    # === Add generation config (requires ghost_env) ===
    model_cfg_ghost = d_arg["model"].copy()
    del model_cfg_ghost["settingcells"], model_cfg_ghost["output_dir"]
    ghost_env = gym.make(**model_cfg_ghost)
    ghost_env = PhysiCellModelWrapper(ghost_env, **d_arg["wrapper"])

    d_arg_generation = {
        "x_min": ghost_env.unwrapped.x_min,
        "x_max": ghost_env.unwrapped.x_max,
        "y_min": ghost_env.unwrapped.y_min,
        "y_max": ghost_env.unwrapped.y_max,
        "n_tumor": args.tumor,
        "n_cell_1": args.cell_1,
        "range_jitter_tumor": (5, 15),
        "range_cell_1": (5, 10),
        "range_r2_frac_tumor": (0.1, 0.4),
        "range_frac_cell_1": (0.1, 0.4),
        "range_r1": (0.1, 0.4),
        "range_cell_dist": (1.5, 2.0),
        "init_mode": args.init_mode,
        "cell_2_fraction": r_cell_2_fraction,
    }
    d_arg["generation"] = d_arg_generation

    # Optional: override number of actor processes
    if args.num_actors:
        d_arg["num_actors"] = args.num_actors

    # === LAUNCH! ===
    run_async_sac(d_arg)