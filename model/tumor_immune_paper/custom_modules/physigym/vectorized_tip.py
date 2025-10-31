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
# Helper: CPU affinity per environment
# ============================================================
def assign_cpu_affinity(env_id: int, threads_per_env: int):
    total_cores = psutil.cpu_count(logical=True)
    start = env_id * threads_per_env
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

    base_xml = model_cfg["settingxml"]
    threads_per_env = vect_cfg["threads_per_env"]
    seed = sim_cfg.get("seed")

    env_xml = f"config/PhysiCell_settings_env{env_id}.xml"
    if not os.path.exists(env_xml):
        shutil.copy(base_xml, env_xml)

    output_dir = f"output/env{env_id}"
    os.makedirs(output_dir, exist_ok=True)

    def _init():
        assign_cpu_affinity(env_id, threads_per_env)

        # Modify XML for this env
        tree = etree.parse(env_xml)
        root = tree.getroot()
        root.xpath("//overall/max_time")[0].text = str(sim_cfg["max_time"])
        root.xpath("//parallel/omp_num_threads")[0].text = str(threads_per_env)
        root.xpath("//save/folder")[0].text = output_dir
        tree.write(env_xml, pretty_print=True)
        model_cfg["settingxml"] = env_xml
        # Create the base PhysiCell environment
        env = gym.make(**model_cfg)
        # Wrap it for simplified action and custom reward
        env = PhysiCellModelWrapper(env, **wrapper_cfg)

        if seed is not None:
            env.reset(seed=seed + env_id)

        return env

    return _init


def vec_envs(cfg: dict):
    vect_cfg = cfg["vectorization"]
    sim_cfg = cfg["simulation"]
    num_envs = vect_cfg["num_envs"]
    threads_per_env = vect_cfg.get("threads_per_env")
    total_cores = psutil.cpu_count(logical=True)

    if threads_per_env is None:
        threads_per_env = max(1, total_cores // num_envs)
        vect_cfg["threads_per_env"] = threads_per_env

    print(f"[INFO] Detected {total_cores} cores")
    print(f"[INFO] Launching {num_envs} envs × {threads_per_env} threads each")

    env_fns = [make_physigym_env(i, cfg) for i in range(num_envs)]
    return SubprocVecEnv(env_fns)


# ============================================================
# Runner
# ============================================================
def run_vectorized(cfg: dict):
    num_envs = cfg["vectorization"]["num_envs"]
    envs = vec_envs(cfg)

    obs = envs.reset()
    print(f"[INFO] Observation shape: {np.shape(obs[0])}")

    time_1 = time.time()
    for t in range(50):
        actions = np.random.uniform(low=0, high=1, size=(num_envs, 1))
        obs, rewards, dones, infos = envs.step(actions)

        print(f"[Step {t}] rewards = {rewards}")
        if np.any(dones):
            print(f"[INFO] Envs done: {np.where(dones)[0]}")

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
    parser.add_argument("-m", "--max_time", type=float, default=1440.0)
    parser.add_argument("-n", "--num_envs", type=int, default=4)
    parser.add_argument("-t", "--threads", type=int, default=None)
    parser.add_argument("-s", "--seed", type=int, default=3)
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"

    # ---- Unified nested configuration ----
    cfg = {
        "simulation": {
            "max_time": args.max_time,
            "seed": args.seed,
        },
        "vectorization": {
            "num_envs": args.num_envs,
            "threads_per_env": args.threads,
        },
        "model": {
            "id": "physigym/ModelPhysiCellEnv-v0",
            "settingxml": args.settingxml,
            "cell_type_cmap": {"tumor": "yellow", "cell_1": "green", "cell_2": "navy"},
            "figsize": (6, 6),
            "observation_mode": "scalars",
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
        },
    }

    run_vectorized(cfg)
