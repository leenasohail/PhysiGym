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
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch
import wandb

from tqdm import tqdm

# Your project imports
from vectorized_tip import vec_envs
from nn_tip import Actor, QNetwork
from rb_tip import ReplayBuffer
from wrapper_tip import PhysiCellModelWrapper


from multiprocessing import Event, Queue
import queue


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
def actor_process(
    actor_queue, sample_queue, stats_queue, d_arg, stop_event: Event, dummy_state
):
    # One actor → one process → runs ALL vectorized envs
    seed = d_arg["simulation"]["seed"] or 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    envs = vec_envs(d_arg)  # ← This already creates N parallel PhysiCell instances
    d_arg_env = d_arg["env"]
    actor_local = Actor(
        d_arg["env"], d_arg.get("neural_architecture_image", "impala")
    ).cpu()
    with torch.no_grad():
        _, _, _ = actor_local(dummy_state)
    actor_local.eval()
    episode_returns = np.zeros(envs.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(envs.num_envs, dtype=np.int32)
    local_step = 0
    obs = envs.reset()

    while not stop_event.is_set():
        # Try to fetch a new policy (non-blocking)
        try:
            while True:
                new_params = actor_queue.get_nowait()
                # load params safely
                try:
                    actor_local.load_state_dict(new_params)
                except Exception:
                    # if state_dict was saved on CUDA, map_location might be required
                    actor_local.load_state_dict(
                        {k: v.cpu() for k, v in new_params.items()}
                    )
        except queue.Empty:
            pass

        # Inference
        with torch.no_grad():
            if d_arg_env["is_graph"]:
                pyg_batch = obs_to_pyg(obs, "cpu")
                actions_tensor, _, _ = actor_local.get_action(pyg_batch)
            else:
                x = torch.from_numpy(obs).cpu()
                actions_tensor, _, _ = actor_local.get_action(x)
            actions = actions_tensor.cpu().numpy()

        # Step envs
        next_obs, rewards, dones, infos = envs.step(actions)

        # Bookkeeping per-env
        episode_returns += rewards.astype(np.float64)
        episode_lengths += 1
        local_step += 1

        # Push transitions (vectorized batch -> individual transitions)
        for i in range(envs.num_envs):
            if d_arg_env["is_graph"]:
                o = {k: v[i] for k, v in obs.items()}
                no = {k: v[i] for k, v in next_obs.items()}
            else:
                o = obs[i].copy() if isinstance(obs[i], np.ndarray) else obs[i]
                no = (
                    next_obs[i].copy()
                    if isinstance(next_obs[i], np.ndarray)
                    else next_obs[i]
                )

            # send stats if episode ended
            if dones[i]:
                stats = {
                    "episode_return": float(episode_returns[i]),
                    "episode_length": int(episode_lengths[i]),
                    "step": int(local_step),
                    "timestamp": time.time(),
                }
                try:
                    stats_queue.put_nowait(stats)
                except queue.Full:
                    # drop a stat if main can't keep up
                    pass

                # reset counters for that env (envs.reset() should also have reset it internally)
                episode_returns[i] = 0.0
                episode_lengths[i] = 0

            # send sample; use non-blocking to avoid actor stall
            try:
                sample_queue.put_nowait(
                    (o, actions[i], float(rewards[i]), no, bool(dones[i]))
                )
            except queue.Full:
                # if sample queue is full, drop sample (or implement backpressure)
                # dropping occasionally is safer than blocking the actor indefinitely
                pass

        obs = next_obs
        # small sleep to reduce busy loop (tunable)
        time.sleep(0.0001)

    # Clean up envs before process exit
    try:
        envs.close()
    except Exception:
        pass


# --------------------------------------------------------------
# Main learner + async runner
# --------------------------------------------------------------
def run_async_sac(d_arg, init_obs):
    mp.set_start_method("spawn", force=True)

    device = torch.device(
        "cuda" if d_arg["simulation"]["cuda"] and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    d_arg_env = d_arg["env"]
    is_graph = d_arg_env["is_graph"]

    # Networks
    actor = Actor(d_arg_env, d_arg.get("neural_architecture_image", "impala")).to(
        device
    )
    qf1 = QNetwork(d_arg_env, d_arg.get("neural_architecture_image", "impala")).to(
        device
    )
    qf2 = QNetwork(d_arg_env, d_arg.get("neural_architecture_image", "impala")).to(
        device
    )
    if is_graph:
        dummy_state = obs_to_pyg([init_obs])
    else:
        dummy_state = torch.Tensor(init_obs).to(device).unsqueeze(0)

    with torch.no_grad():
        if is_graph:
            actions_tensor, _, _ = actor.get_action(dummy_state)
        else:
            actions_tensor, _, _ = actor.get_action(dummy_state)
        _ = qf1(dummy_state, actions_tensor)
        _ = qf2(dummy_state, actions_tensor)

    qf1_target = deepcopy(qf1).to(device)
    qf2_target = deepcopy(qf2).to(device)

    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=d_arg["rl"]["q_lr"]
    )
    actor_optimizer = optim.Adam(actor.parameters(), lr=d_arg["rl"]["policy_lr"])

    # Alpha (entropy)
    if d_arg["rl"]["autotune"]:
        target_entropy = -float(np.prod(d_arg_env["action_space_shape"]))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = optim.Adam([log_alpha], lr=d_arg["rl"]["q_lr"])
        alpha = log_alpha.exp().item()
    else:
        alpha = float(d_arg["rl"]["alpha"])

    # Replay buffer
    rb = ReplayBuffer(
        state_dim=d_arg_env["observation_space_shape"],
        action_dim=d_arg_env["action_space_shape"],
        device=device,
        buffer_size=d_arg["rl"]["buffer_size"],
        batch_size=d_arg["rl"]["batch_size"],
        state_type=d_arg_env["observation_space_dtype"],
        is_graph=is_graph,
    )

    # Process communication
    actor_queue = mp.Queue(maxsize=10)
    sample_queue = mp.Queue(maxsize=30000)
    stats_queue = mp.Queue(maxsize=1000)
    stop_event = mp.Event()

    # Start actor process (single actor)
    actor_proc = mp.Process(
        target=actor_process,
        args=(
            actor_queue,
            sample_queue,
            stats_queue,
            d_arg,
            stop_event,
            dummy_state.cpu(),
        ),
        daemon=False,  # MUST be False so this process can spawn SubprocVecEnv children
    )

    actor_proc.start()

    # send initial policy
    try:
        actor_queue.put_nowait(
            {k: v.detach().cpu() for k, v in actor.state_dict().items()}
        )
    except queue.Full:
        actor_queue.put({k: v.detach().cpu() for k, v in actor.state_dict().items()})

    # Logging
    run_name = f"{d_arg['simulation']['name']}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    if d_arg["simulation"]["wandb_track"]:
        wandb.init(
            project=d_arg["wandb"]["project"] if "wandb" in d_arg else "SAC_ASYNC_TIP",
            name=run_name,
            config=d_arg,
        )

    tau = d_arg["rl"]["tau"]

    print("Starting training loop...")
    try:
        for step in tqdm(range(d_arg["rl"]["total_timesteps"])):
            # 1) Drain sample_queue into replay buffer until we've reached learning_starts
            drained = 0
            while not sample_queue.empty():
                try:
                    state, action, reward, next_state, done = sample_queue.get_nowait()
                except queue.Empty:
                    break
                rb.add(state, action, reward, next_state, done)
                drained += 1

            # 2) Log any stats reported by actors
            while not stats_queue.empty():
                try:
                    stat = stats_queue.get_nowait()
                except queue.Empty:
                    break
                return_val = stat["episode_return"]
                length = stat["episode_length"]
                log_dict = {
                    "episode/return": return_val,
                    "episode/length": length,
                    "train/step": step,
                }
                writer.add_scalar("episode/return", return_val, step)
                writer.add_scalar("episode/length", length, step)
                if d_arg["simulation"]["wandb_track"]:
                    wandb.log(log_dict, step=step)

            # If not enough samples yet, wait a little and continue
            if len(rb) < max(d_arg["rl"]["learning_starts"], d_arg["rl"]["batch_size"]):
                time.sleep(0.1)
                continue
            K = 3
            for _ in range(K):
                # 3) Sample batch and do SAC updates
                batch = rb.sample()

                # compute targets
                with torch.no_grad():
                    next_actions, next_log_pi, _ = actor.get_action(batch.next_state)
                    q1_next = qf1_target(batch.next_state, next_actions)
                    q2_next = qf2_target(batch.next_state, next_actions)
                    min_q_next = torch.min(q1_next, q2_next) - alpha * next_log_pi
                    next_q = (
                        batch.reward.flatten()
                        + (1 - batch.done.flatten())
                        * d_arg["rl"]["gamma"]
                        * min_q_next.squeeze()
                    )

                q1 = qf1(batch.state, batch.action).view(-1)
                q2 = qf2(batch.state, batch.action).view(-1)
                qf1_loss = F.mse_loss(q1, next_q)
                qf2_loss = F.mse_loss(q2, next_q)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # Policy & alpha update
                if step % d_arg["rl"]["policy_frequency"] == 0:
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
                            alpha_loss = (
                                -log_alpha.exp() * (log_pi + target_entropy).detach()
                            ).mean()
                            alpha_optim.zero_grad()
                            alpha_loss.backward()
                            alpha_optim.step()
                            alpha = log_alpha.exp().item()

                # Soft-update targets periodically (frequency param controls how often)
                if step % d_arg["rl"]["target_network_frequency"] == 0:
                    for param, target_param in zip(
                        qf1.parameters(), qf1_target.parameters()
                    ):
                        target_param.data.copy_(
                            tau * param.data + (1.0 - tau) * target_param.data
                        )
                    for param, target_param in zip(
                        qf2.parameters(), qf2_target.parameters()
                    ):
                        target_param.data.copy_(
                            tau * param.data + (1.0 - tau) * target_param.data
                        )

            # Periodically send new policy to actors
            if step % 2000 == 0:
                try:
                    actor_queue.put_nowait(
                        {k: v.detach().cpu() for k, v in actor.state_dict().items()}
                    )

                except queue.Full:
                    # if actor queue full, skip this update (actor will pick up later)
                    pass

    except KeyboardInterrupt:
        print("Interrupted by user — shutting down.")
    finally:
        # Ask actor process to stop, wait and terminate if necessary
        stop_event.set()
        actor_proc.join(timeout=5.0)
        if actor_proc.is_alive():
            actor_proc.terminate()
            actor_proc.join(timeout=1.0)

        # Close writer / wandb
        writer.close()
        if d_arg["simulation"]["wandb_track"]:
            wandb.finish()

        print("Training finished and cleaned up.")


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
    parser.add_argument(
        "--settingxml", nargs="?", default="config/PhysiCell_settings.xml"
    )
    parser.add_argument("--settingcells", nargs="?", default="cells.csv")
    parser.add_argument("--seed", nargs="?", default="5")
    parser.add_argument(
        "--observation_mode", nargs="?", default="graph_knn"
    )  # change default if you want
    parser.add_argument("--render_mode", nargs="?", default="None")
    parser.add_argument("--max_time_episode", type=float, nargs="?", default=12900.0)
    parser.add_argument("--total_step_learn", type=int, nargs="?", default=int(2e6))
    parser.add_argument("--gpu", nargs="?", default="true")
    parser.add_argument("--name", nargs="?", default="async_sac_tip")
    parser.add_argument("--wandb", nargs="?", default="true")
    parser.add_argument("--entity", nargs="?", default="corporate-manu-sureli")
    parser.add_argument(
        "--init_mode",
        nargs="+",
        default=["circular_mode", "asymmetric_mode", "connected_mst_mode"],
    )
    parser.add_argument("--tumor", type=int, nargs="?", default=512)
    parser.add_argument("--cell_1", type=int, nargs="?", default=128)
    parser.add_argument("--cell_2_fraction", type=float, nargs="?", default=None)
    parser.add_argument(
        "--num_envs", type=int, nargs="?", default=7
    )  # total parallel PhysiCell instances
    parser.add_argument("--s_frequency_save_data", type=int, nargs="?", default=None)
    parser.add_argument(
        "--neural_architecture_image", type=str, nargs="?", default="impala"
    )
    parser.add_argument("--rl_threads", type=int, nargs="?", default=4)
    parser.add_argument(
        "--num_actors", type=int, nargs="?", default=None
    )  # optional: override auto-detect

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
        "cell_type_cmap": {
            "tumor": "yellow",
            "cell_1": "green",
            "cell_2": "navy",
            "other_tissue": "red",
        },
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
        "frequency_save_data": args.s_frequency_save_data or None,
    }

    d_arg_rl = {
        "total_timesteps": args.total_step_learn,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size_multiplier * args.num_envs,  # e.g. 64 × num_envs
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
    d_arg["env"] = d_arg_env
    # Optional: override number of actor processes
    if args.num_actors:
        d_arg["num_actors"] = args.num_actors
    d_arg_generation["seed"] = d_arg_simulation["seed"]
    init_obs, _ = ghost_env.reset(seed=d_arg_simulation["seed"])
    ghost_env.close()
    del ghost_env
    # === LAUNCH! ===
    run_async_sac(d_arg=d_arg, init_obs=init_obs)
