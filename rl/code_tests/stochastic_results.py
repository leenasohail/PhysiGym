import gymnasium as gym
import physigym
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import tyro
from dataclasses import dataclass, field
import plotly.express as px

absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("PhysiCell") + len("PhysiCell")
]
sys.path.append(absolute_path)
from rl.utils.wrappers.wrapper_physicell_complex_tme import PhysiCellModelWrapper

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


def main(args):
    os.makedirs(args.path_save, exist_ok=True)
    env = gym.make("physigym/ModelPhysiCellEnv", observation_type="simple")
    env = PhysiCellModelWrapper(env, list_variable_name=["anti_M2", "anti_pd1"])
    _, info = env.reset(seed=args.seed)
    list_actions_value = args.list_actions_value
    for actions_value in list_actions_value:
        episode = 1
        step = 1
        liste = []
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
        save_path = os.path.join(
            args.path_save,
            f"stochastic_results_actions_{actions_value[0]}_{actions_value[1]}",
        )
        os.makedirs(
            save_path,
            exist_ok=True,
        )
        df_sorted.to_csv(
            os.path.join(
                save_path
                + f"/stochastic_results_actions_{actions_value[0]}_{actions_value[1]}.csv",
            ),
            index=False,
        )
        fig = px.line(
            df_sorted,
            x="step",
            y="number_cancer_cells",
            color="episode",
            labels={"step": "Step", "number_cancer_cells": "Number of Cancer Cells"},
            title="Step vs Number of Cancer Cells for Each Episode",
        )

        fig.write_image(
            save_path + "/Step vs Number of Cancer Cells for Each Episode.png"
        )
        del fig
        # Calculate the quantiles for 'number_cancer_cells' at each 'step'
        quantiles = (
            df.groupby("step")["number_cancer_cells"]
            .quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            .unstack()
            .reset_index()
        )

        # Rename columns for clarity
        quantiles.columns = ["step", "q0.1", "q0.25", "median", "q0.75", "q0.9"]

        # Create the plot
        fig = px.line(
            quantiles,
            x="step",
            y=["q0.1", "q0.25", "median", "q0.75", "q0.9"],
            labels={"step": "Step", "value": "Cancer Cells (Quantiles)"},
            title="Quantiles of Number of Cancer Cells at Each Step",
        )
        fig.write_image(
            save_path + "/Step vs Number of Cancer Cells for Each Episode.png"
        )
        df["cumulative_reward"] = df.groupby("episode")["reward"].cumsum()

        # Calculate the quantiles for 'cumulative_reward' at each 'step'
        quantiles = (
            df.groupby("step")["cumulative_reward"]
            .quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            .unstack()
            .reset_index()
        )

        # Rename columns for clarity
        quantiles.columns = ["step", "q0.1", "q0.25", "median", "q0.75", "q0.9"]

        # Create the plot
        fig = px.line(
            quantiles,
            x="step",
            y=["q0.1", "q0.25", "median", "q0.75", "q0.9"],
            labels={"step": "Step", "value": "Cumulative Reward (Quantiles)"},
            title="Quantiles of Cumulative Reward at Each Step",
        )

        fig.write_image(save_path + "/Quantiles of Cumulative Reward at Each Step.png")

        # Compute the cumulative time for each episode
        df["cumulative_time"] = df.groupby("episode")["time_step_seconds"].cumsum()
        # Calculate the quantiles for 'cumulative_reward' at each 'step'
        quantiles = (
            df.groupby("step")["cumulative_time"]
            .quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            .unstack()
            .reset_index()
        )

        # Rename columns for clarity
        quantiles.columns = ["step", "q0.1", "q0.25", "median", "q0.75", "q0.9"]

        # Create the plot
        fig = px.line(
            quantiles,
            x="step",
            y=["q0.1", "q0.25", "median", "q0.75", "q0.9"],
            labels={
                "step": "Step",
                "value": "Cumulative Time (in seconds) (Quantiles)",
            },
            title="Quantiles of Cumulative Time (in seconds) at Each Step",
        )

        fig.write_image(
            save_path + "/Quantiles of Cumulative Time (in seconds) at Each Step.png"
        )

        del df, liste


if __name__ == "__main__":
    main(tyro.cli(Args))
