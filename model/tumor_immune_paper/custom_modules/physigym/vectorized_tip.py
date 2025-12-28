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
from lxml import etree
import time
from tqdm import tqdm

from stable_baselines3.common.vec_env import SubprocVecEnv
from resilient_sub_vec_env import ResilientSubprocVecEnv
from wrapper_tip import PhysiCellModelWrapper
import sys
import faulthandler

faulthandler.enable(file=sys.stderr, all_threads=True)
import physigym
from extending import physicell


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
    seed = sim_cfg["seed"]
    master_seed = seed if seed is not None else 42
    rng = np.random.default_rng(master_seed)
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

        generation_cfg["seed"] = int(rng.integers(0, 2**12 - 1)) + env_id
        env.reset(generation_cfg=generation_cfg)

        return env

    return _init


def vec_envs(cfg: dict):
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    vect_cfg = cfg["vectorization"]
    num_envs = vect_cfg["num_envs"]
    rl_threads, _ = configure_thread_splitting(vect_cfg["rl_threads"])
    total_cores = psutil.cpu_count(logical=True)
    threads_per_env = (total_cores - rl_threads) // num_envs
    cfg["vectorization"]["threads_per_env"] = threads_per_env
    print(f"[INFO] Detected {total_cores} cores")
    print(f"[INFO] Launching {num_envs} envs × {threads_per_env} threads each")

    env_fns = [make_physigym_env(i, cfg) for i in range(num_envs)]

    return ResilientSubprocVecEnv(
        env_fns=env_fns, start_method="spawn"
    )  # SubprocVecEnv(env_fns=env_fns, start_method="spawn")


def test_vec_env(cfg: dict):
    vect_cfg = cfg["vectorization"]
    num_envs = vect_cfg["num_envs"]
    rl_threads, _ = configure_thread_splitting(vect_cfg["rl_threads"])
    total_cores = psutil.cpu_count(logical=True)
    threads_per_env = (total_cores - rl_threads) // num_envs
    cfg["vectorization"]["threads_per_env"] = threads_per_env

    return make_physigym_env(0, cfg)


# ============================================================
# Runner
# ============================================================
def run_vectorized(cfg: dict):
    vect_cfg = cfg["vectorization"]
    num_envs = vect_cfg["num_envs"]
    envs = vec_envs(cfg)
    _ = envs.reset()
    time_1 = time.time()
    for t in tqdm(range(2500)):
        actions = np.array(
            [envs.action_space.sample() for _ in range(num_envs)],
            dtype=np.float32,
        )
        # actions = np.random.uniform(low=0, high=1, size=(num_envs, 1))
        _, _, _, _ = envs.step(actions)

    envs.close()
    print("[INFO] Simulation complete.")
    print(f"[INFO] Total time = {time.time() - time_1:.2f} s")


def test_run_vectorized(cfg: dict):
    vect_cfg = cfg["vectorization"]
    sim_cfg = cfg["simulation"]
    vect_cfg = cfg["vectorization"]
    model_cfg = cfg["model"]
    wrapper_cfg = cfg["wrapper"]
    generation_cfg = cfg["generation"]

    base_xml = model_cfg["settingxml"]
    base_cells = model_cfg["settingcells"]
    model_cfg_copy = model_cfg.copy()
    vect_cfg["threads_per_env"] = 28
    seed = sim_cfg["seed"]
    env_id = 0
    master_seed = seed if seed is not None else 42
    rng = np.random.default_rng(master_seed)
    env_xml = f"config/PhysiCell_settings_env{env_id}.xml"
    env_cells = f"config/cells_{env_id}.csv"
    if not os.path.exists(env_xml):
        shutil.copy(base_xml, env_xml)
    if not os.path.exists(env_cells):
        shutil.copy(base_cells, env_cells)
    if model_cfg_copy["output_dir"] is None:
        model_cfg_copy["output_dir"] = "output"
    del model_cfg_copy["settingcells"]

    # Modify XML for this env
    tree = etree.parse(env_xml)
    root = tree.getroot()
    root.xpath("//overall/max_time")[0].text = str(sim_cfg["max_time"])
    root.xpath("//parallel/omp_num_threads")[0].text = str(28)
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
    generation_cfg["seed"] = int(rng.integers(0, 2**32 - 1)) + env_id
    _, _ = env.reset(seed=generation_cfg["seed"], generation_cfg=generation_cfg)
    time_1 = time.time()
    for t in tqdm(range(2500)):
        actions = np.random.uniform(low=0, high=1, size=(1, 1))
        obs, reward, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            env.reset()

    env.close()
    print("[INFO] Simulation complete.")
    print(f"[INFO] Total time = {time.time() - time_1:.2f} s")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import faulthandler

    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(
        description="Vectorized PhysiCell runner with CPU pinning."
    )
    parser.add_argument(
        "settingxml", nargs="?", default="config/PhysiCell_settings.xml"
    )
    parser.add_argument("settingcells", nargs="?", default="config/cells.csv")
    parser.add_argument("-m", "--max_time", type=float, default=180.0)
    parser.add_argument("-n", "--num_envs", type=int, default=7)
    parser.add_argument("-t", "--rl_threads", type=int, default=4)
    parser.add_argument("-s", "--seed", type=int, default=3)
    args = parser.parse_args()
    params = {
        "tumor": {"correlation_length": 45, "threshold": 0.55, "number_cells": 512},
        "cell_1": {"correlation_length": 45, "threshold": 0.55, "number_cells": 128},
    }
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
            "output_dir": None,
            "cell_type_cmap": {
                "tumor": "yellow",
                "cell_1": "green",
                "cell_2": "navy",
                "other_tissue": "red",
            },
            "figsize": (6, 6),
            "observation_mode": "scalars_cells_substrates",  # "img_mc_cells_substrates",
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
            "frequency_save_data": None,
        },
        "generation": {
            "x_min": -256,
            "x_max": 256,
            "y_min": -256,
            "y_max": 256,
            "params": params,  # number of tumor cells for the initial state
            "seed": 128,  # seed
            "cell_2_fraction": args.seed,
        },
    }
    # test_run_vectorized(cfg)
    run_vectorized(cfg)
