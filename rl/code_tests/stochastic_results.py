import gymnasium as gym
import physigym
import random
import numpy as np
from rl.utils.wrappers.wrapper_physicell_complex_tme import PhysiCellModelWrapper
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import tyro
from dataclasses import dataclass, field

liste = [round(x * 0.1, 1) for x in range(11)] + [-1]
list_pairs_drugs = [[a, b] for a, b in zip(liste, liste)]


@dataclass
class Args:
    list_actions_value: list[list[float]] = field(
        default_factory=lambda: list_pairs_drugs
    )
    """Actions applied on the TME"""
    seed: int = 1
    """seed of the experiment"""
    maximum_episode: int = 50
    """maximum number of trajectories"""
    observation_type: str = "image"
    """the type of observation"""
    path_save: str = "code_tests"
    """path save"""
    name_file: str = "stochastic_results"


def main(args):
    os.makedirs(args.path_save, exist_ok=True)
    env = gym.make("physigym/ModelPhysiCellEnv", observation_type="simple")
    env = PhysiCellModelWrapper(env, list_variable_name=["anti_M2", "anti_pd1"])
    _, info = env.reset(seed=args.seed)
    liste = []
    list_actions_value = args.list_actions_value
    for actions_value in list_actions_value:
        episode = 1
        step = 1
        while episode < args.maximum_episode:
            begin_time = time.time()
            random_actions = np.array(env.action_space.sample())
            actions = np.array(
                [
                    random_actions[0] if actions_value[0] == -1 else actions_value[0],
                    random_actions[1] if actions_value[1] == -1 else actions_value[1],
                ]
            )
            o, r, t, ter, info = env.step(actions)
            nb_cancer_cells = info["number_cancer_cells"]
            end_time = time.time()
            time_step = end_time - begin_time
            liste.append(
                [
                    episode,
                    step,
                    actions[0],
                    actions[1],
                    nb_cancer_cells,
                    r[0],
                    time_step,
                ]
            )
            step += 1
            if t or ter:
                episode += 1
                step = 0
                o, info = env.reset()

    df = pd.DataFrame(
        liste,
        columns=[
            "episode",
            "step",
            "anti_M2",
            "anti_pd1",
            "number_cancer_cells",
            "reward",
            "time_step_seconds",
        ],
    )
    df_sorted = df.sort_values(by=["episode", "step"])
    df_sorted.to_csv(
        os.path.join(args.path_save, "stochastic_results_1.csv"), index=False
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
