#!/usr/bin/env python3
"""
title: run_physigym_vectorized_combined.py

description:
    Run multiple PhysiCell simulations in parallel with Gymnasium + SubprocVecEnv.
    Each environment:
      - uses a unique XML config and output folder
      - runs on its own CPU core range (affinity pinned)
      - is wrapped in PhysiCellModelWrapper for simplified Box actions

author: Alexandre Bertin (with ChatGPT)
date: 2025
"""

import os
import shutil
import argparse
import psutil
import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from lxml import etree
import time
import physigym
from extending import physicell
from wrapper_tip import PhysiCellModelWrapper


# ============================================================
# Global Thread Splitting
# ============================================================
def configure_thread_splitting(rl_threads: int):
    """
    Configure threads globally for the RL side (PyTorch) and
    leave remaining threads for the PhysiCell simulations.
    """
    import torch

    total = psutil.cpu_count(logical=True)
    rl_threads = max(1, rl_threads)
    rl_threads = min(rl_threads, total - 1)

    # RL / PyTorch threading
    torch.set_num_threads(rl_threads)
    os.environ["OMP_NUM_THREADS"] = str(rl_threads)
    os.environ["MKL_NUM_THREADS"] = str(rl_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(rl_threads)

    print(f"[ThreadSplit] RL threads = {rl_threads} / {total} total cores")
    print(f"[ThreadSplit] Remaining = {total - rl_threads} for PhysiCell envs")

    return rl_threads, total - rl_threads


# ============================================================
# Helper: CPU affinity per environment
# ============================================================
def assign_cpu_affinity(env_id: int, threads_per_env: int, offset_threads: int):
    os.environ["OMP_NUM_THREADS"] = str(threads_per_env)
    total_cores = psutil.cpu_count(logical=True)
    start = env_id * threads_per_env + offset_threads
    end = min(start + threads_per_env, total_cores)
    core_list = list(range(start, end))
    try:
        psutil.Process().cpu_affinity(core_list)
        print(f"[Affinity] Env {env_id}: pinned to cores {core_list}")
    except Exception as e:
        print(f"[Affinity] Warning: failed to pin Env {env_id}: {e}")


# ============================================================
# Helper: Environment factory
# ============================================================
def make_physigym_env(env_id: int, cfg: dict):
    """
    cfg structure:
    {
        "simulation": {...},
        "vectorization": {...},
        "model": {...},
        "wrapper": {...}
    }
    """

    sim_cfg = cfg["simulation"]
    vect_cfg = cfg["vectorization"]
    model_cfg = cfg["model"]
    wrapper_cfg = cfg["wrapper"]
    generation_cfg = cfg["generation"]

    base_xml = model_cfg["settingxml"]
    base_cells = model_cfg["settingcells"]
    model_cfg_copy = model_cfg.copy()
    threads_per_env = vect_cfg["threads_per_env"]
    seed = sim_cfg.get("seed")
    env_xml = f"config/PhysiCell_settings_env{env_id}.xml"
    env_cells = f"config/cells_{env_id}.csv"
    if not os.path.exists(env_xml):
        shutil.copy(base_xml, env_xml)
    if not os.path.exists(env_cells):
        shutil.copy(base_cells, env_cells)
    if model_cfg_copy["output_dir"] is None:
        model_cfg_copy["output_dir"] = "output"
    del model_cfg_copy["settingcells"]
    rl_threads = vect_cfg["rl_threads"]

    def _init():
        assign_cpu_affinity(env_id, threads_per_env, offset_threads=rl_threads)

        # Modify XML for this env
        tree = etree.parse(env_xml)
        root = tree.getroot()
        root.xpath("//overall/max_time")[0].text = str(sim_cfg["max_time"])
        root.xpath("//parallel/omp_num_threads")[0].text = str(threads_per_env)
        root.xpath("//save/folder")[0].text = os.path.join(
            model_cfg_copy["output_dir"], f"env{env_id}"
        )
        root.xpath("//save/full_data/enable")[0].text = "false"
        root.xpath("//save/SVG/enable")[0].text = "false"
        root.xpath("//initial_conditions/cell_positions/filename")[
            0
        ].text = f"cells_{env_id}.csv"
        tree.write(env_xml, pretty_print=True)
        model_cfg_copy["settingxml"] = env_xml
        del model_cfg_copy["output_dir"]
        if env_id != 0:
            wrapper_cfg["frequency_save_data"] = None
        # Create the base PhysiCell environment
        env = gym.make(**model_cfg_copy)
        # Wrap it for simplified action and custom reward
        env = PhysiCellModelWrapper(env, **wrapper_cfg)

        if seed is not None:
            env.reset(seed=seed + env_id, generation_cfg=generation_cfg)

        return env

    return _init


def vec_envs(cfg: dict):
    vect_cfg = cfg["vectorization"]
    sim_cfg = cfg["simulation"]
    num_envs = vect_cfg["num_envs"]
    rl_threads, remaining_threads = configure_thread_splitting(vect_cfg["rl_threads"])
    total_cores = psutil.cpu_count(logical=True)
    threads_per_env = (total_cores - rl_threads) // num_envs
    cfg["vectorization"]["threads_per_env"] = threads_per_env
    print(f"[INFO] Detected {total_cores} cores")
    print(f"[INFO] Launching {num_envs} envs × {threads_per_env} threads each")

    env_fns = [make_physigym_env(i, cfg) for i in range(num_envs)]

    return SubprocVecEnv(env_fns)


# ============================================================
# Runner
# ============================================================
def run_vectorized(cfg: dict):
    vect_cfg = cfg["vectorization"]
    sim_cfg = cfg["simulation"]
    num_envs = vect_cfg["num_envs"]

    envs = vec_envs(cfg)

    obs = envs.reset()
    time_1 = time.time()
    for t in range(50000):
        actions = np.array(
            [envs.action_space.sample() for _ in range(num_envs)],
            dtype=np.float32,
        )
        # actions = np.random.uniform(low=0, high=1, size=(num_envs, 1))
        obs, rewards, dones, infos = envs.step(actions)
        print(f"[Step {t}] rewards = {rewards}")

    envs.close()
    print("[INFO] Simulation complete.")
    print(f"[INFO] Total time = {time.time() - time_1:.2f} s")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vectorized PhysiCell runner with CPU pinning."
    )
    parser.add_argument(
        "settingxml", nargs="?", default="config/PhysiCell_settings.xml"
    )
    parser.add_argument("settingcells", nargs="?", default="config/cells.csv")
    parser.add_argument("-m", "--max_time", type=float, default=1440.0)
    parser.add_argument("-n", "--num_envs", type=int, default=8)
    parser.add_argument("-t", "--rl_threads", type=int, default=4)
    parser.add_argument("-s", "--seed", type=int, default=3)
    args = parser.parse_args()

    # ---- Unified nested configuration ----
    cfg = {
        "simulation": {
            "max_time": args.max_time,
            "seed": args.seed,
        },
        "vectorization": {
            "num_envs": args.num_envs,
            "rl_threads": args.rl_threads,
        },
        "model": {
            "id": "physigym/ModelPhysiCellEnv-v0",
            "settingxml": args.settingxml,
            "settingcells": args.settingcells,
            "output_dir": "test",
            "cell_type_cmap": {
                "tumor": "yellow",
                "cell_1": "green",
                "cell_2": "navy",
                "other_tissue": "red",
            },
            "figsize": (6, 6),
            "observation_mode": "img_mc_cells_substrates",
            "render_mode": None,
            "verbose": False,
            "img_rgb_grid_size_x": 64,
            "img_rgb_grid_size_y": 64,
            "img_mc_grid_size_x": 64,
            "img_mc_grid_size_y": 64,
            "normalization_factor": 512,
        },
        "wrapper": {
            "list_variable_name": ["drug_1"],
            "weight": 0.8,
            "frequency_save_data": 64,
        },
        "generation": {
            "x_min": -256,
            "x_max": 256,
            "y_min": -256,
            "y_max": 256,
            "n_tumor": 512,  # number of tumor cells for the initial state
            "n_cell_1": 128,  # number of cell 1 for the initial state
            "range_jitter_tumor": (
                5,
                15,
            ),  # range of std for the Gaussian noise jitter applied to tumor cells' positions inside ellipse
            "range_cell_1": (
                5,
                10,
            ),  # range  of std for the Gaussian noise jitter applied to surrounding cell_1 positions
            "range_r2_frac_tumor": (
                0.1,
                0.4,
            ),  # range for the fractional size of the semi-minor axis (y-axis radius) of the tumor ellipse relative to bounding box
            "range_frac_cell_1": (
                0.1,
                0.4,
            ),  # range for fractional size of semi-minor axis of the surrounding cells' ellipse (cell_1)
            "range_r1": (
                0.1,
                0.4,
            ),  # range for fractional size of the semi-major axis (x-axis radius) of the tumor ellipse
            "range_cell_dist": (
                1.5,
                2.0,
            ),  # multiplier that modifies the r2 fractional size of the surrounding cell_1 ellipse
            "init_mode": ["circular_mode", "asymmetric_mode", "connected_mst_mode"],
            "cell_2_fraction": 0.3,
            "seed": 2,
        },
    }

    run_vectorized(cfg)
