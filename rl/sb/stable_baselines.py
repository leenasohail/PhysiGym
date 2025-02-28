import gymnasium as gym
import numpy as np
import os
import wandb
import physigym
import stable_baselines3
import sb3_contrib
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import tyro
import time
from gymnasium.spaces import Box
from dataclasses import dataclass
from embedding import physicell  # from extending import physicell
import matplotlib.pyplot as plt
import pandas as pd


# ----------------------
# 🌟 Dataclass
# ----------------------
# Explication: This dataclass stores minimum elements than you can change with CLI without modifying the script
# It is done thanks to the library tyro
@dataclass
class Args:
    algo_name: str = "SAC"
    """the name of the algo"""
    wandb_project_name: str = "IMAGE_TME_PHYSIGYM"
    """the wandb's project name"""
    wandb_entity: str = "corporate-manu-sureli"
    env_id: str = "physigym/ModelPhysiCellEnv-v0"
    """the id of the environment"""
    observation_type: str = "image"
    """the observation type depends on the environment"""
    seed: int = 1
    """seed"""


args = tyro.cli(Args)
config = vars(args)
# ----------------------
# 🏆 Initialize WandB
# ----------------------

##### ----------------------
##### 📍 Choose Algorithm (SB3 or SB3-Contrib)
##### ----------------------
algo_name = args.algo_name
if algo_name in sb3_contrib.__all__:
    algorithm = getattr(sb3_contrib, algo_name)
elif algo_name in stable_baselines3.__all__:
    algorithm = getattr(stable_baselines3, algo_name)
else:
    raise f"Algorith name does not exist: {algo_name}"

##### ----------------------
##### 📍 Choose Algorithm (SB3 or SB3-Contrib)
##### ----------------------
run_name = f"{args.env_id}__{args.algo_name}_{args.wandb_entity}_{int(time.time())}"
run_dir = f"runs/{run_name}"
os.makedirs(run_dir, exist_ok=True)

wandb.init(
    project=args.wandb_project_name,
    entity=args.wandb_entity,
    name=f"{args.algo_name}: observation {args.observation_type}, seed {args.seed}",
    sync_tensorboard=True,  # Sync TensorBoard logs
    config=config,
    monitor_gym=True,  # Monitor Gym environment
    save_code=True,  # Save the training script
)


class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional values to TensorBoard.
    """

    def __init__(self, verbose=0, video_frequency=50000):
        super().__init__(verbose)
        self.global_step = 0  # Track global training steps

    def _on_step(self) -> bool:
        # Get information from the environment
        if "reward" in self.locals:
            self.logger.record("env/reward_value", self.locals["rewards"][0])

        if "number_cancer_cells" in self.locals["infos"][0]:
            self.logger.record(
                "env/cancer_cell_count", self.locals["infos"][0]["number_cancer_cells"]
            )

        if "actions" in self.locals:
            actions = self.locals["actions"][0]
            self.logger.record("env/drug_apoptosis", actions[0])
            self.logger.record("env/drug_reducing_antiapoptosis", actions[1])

        self.global_step += 1  # Increment step counter
        self.logger.dump(step=self.global_step)
        return True


class PhysiCellModelWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        list_variable_name: list[str] = [
            "drug_apoptosis",
            "drug_reducing_antiapoptosis",
        ],
    ):
        """
        Args:
            env (gym.Env): The environment to wrap.
            list_variable_name (list[str]): List of variable names corresponding to actions in the original env.
        """
        super().__init__(env)

        # Check that all variable names are strings
        for variable_name in list_variable_name:
            if not isinstance(variable_name, str):
                raise ValueError(
                    f"Expected variable_name to be of type str, but got {type(variable_name).__name__}"
                )

        self.list_variable_name = list_variable_name
        self.unwrapped_env = env.unwrapped
        self.x_min = self.unwrapped_env.x_min
        self.x_max = self.unwrapped_env.x_max
        self.y_min = self.unwrapped_env.y_min
        self.y_max = self.unwrapped_env.y_max
        # Flatten the action space to match the expected two values
        low = np.array(
            [
                env.action_space[variable_name].low[0]
                for variable_name in list_variable_name
            ]
        )
        high = np.array(
            [
                env.action_space[variable_name].high[0]
                for variable_name in list_variable_name
            ]
        )

        self._action_space = Box(low=low, high=high, dtype=np.float64)

    @property
    def action_space(self):
        """Returns the flattened action space for the wrapper."""
        return self._action_space

    def reset(self, seed=None, options={}):
        """
        Resets the environment and preprocesses the observation.
        """
        o_observation, info = self.env.reset(seed=seed, options=options)

        # Preprocess observation (if needed)
        o_observation = np.array(o_observation, dtype=float)

        return o_observation, info

    def step(self, action: np.ndarray):
        """
        Steps through the environment using the flattened action.

        Args:
            action (np.ndarray): The flattened action array.

        Returns:
            Tuple: Observation, reward, terminated, truncated, info.
        """
        # Convert the flat action array to the dictionary expected by the env
        d_action = {
            variable_name: np.array([value])
            for variable_name, value in zip(self.list_variable_name, action)
        }
        # Take a step in the environment
        o_observation, r_reward, b_terminated, b_truncated, info = self.env.step(
            d_action
        )
        # Preprocess observation (if needed)
        o_observation = np.array(o_observation, dtype=float)
        info["action"] = d_action
        self.info = info
        return o_observation, r_reward, b_terminated, b_truncated, info

    def render(
        self,
        path="./output/image",
        saving_title: str = "output_simulation_image_episode",
    ):
        os.makedirs(path, exist_ok=True)
        df_cell = pd.DataFrame(
            physicell.get_cell(), columns=["ID", "x", "y", "z", "dead", "cell_type"]
        )
        fig, ax = plt.subplots(
            1, 3, figsize=(10, 6), gridspec_kw={"width_ratios": [1, 0.2, 0.2]}
        )
        count_cancer_cell = physicell.get_parameter("count_cancer_cell")

        for s_celltype, s_color in sorted(
            {"cancer_cell": "gray", "nurse_cell": "red"}.items()
        ):
            df_celltype = df_cell.loc[
                (df_cell.z == 0.0) & (df_cell.cell_type == s_celltype), :
            ]
            df_celltype.plot(
                kind="scatter",
                x="x",
                y="y",
                c=s_color,
                xlim=[
                    self.x_min,
                    self.x_max,
                ],
                ylim=[
                    self.y_min,
                    self.y_max,
                ],
                grid=True,
                label=s_celltype,
                s=100,
                title=f"episode step {str(self.unwrapped_env.step_episode).zfill(3)}, cancer cell: {count_cancer_cell}",
                ax=ax[0],
            ).legend(loc="lower left")

        # Create a colormap for the color bars (from -1 to 1)
        list_colors = ["royalblue", "darkorange"]

        # Function to create fluid-like color bars
        def create_fluid_bar(ax_bar, drug_amount, title, max_amount=30, color="cyan"):
            ax_bar.set_xlim(0, 1)
            ax_bar.set_ylim(
                0, 1
            )  # Set y-axis from 0 to 1 for percentage representation
            ax_bar.set_title(title, fontsize=10)
            ax_bar.set_xticks([])
            ax_bar.set_yticks(np.linspace(0, 1, 5))  # 0% to 100% scale

            # Normalize drug amount (convert to percentage of max)
            fill_level = drug_amount / max_amount  # Converts to a range of [0,1]

            # Fill up to the corresponding level
            ax_bar.fill_betweenx(np.linspace(0, fill_level, 100), 0, 1, color=color)

            # Draw container border
            ax_bar.spines["left"].set_visible(False)
            ax_bar.spines["right"].set_visible(False)
            ax_bar.spines["top"].set_visible(True)
            ax_bar.spines["bottom"].set_visible(True)

        action = self.info["action"]
        for i, (key, value) in enumerate(action.items(), start=1):  # Start index from 1
            create_fluid_bar(ax[i], value[0], f"drug_{i}", color=list_colors[i - 1])

        plt.savefig(
            path
            + f"/{saving_title} step {str(self.unwrapped_env.step_episode).zfill(3)}"
        )
        plt.close(fig)


import subprocess


def png_to_video_ffmpeg(image_folder, output_video, fps=10):
    command = [
        "ffmpeg",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        f"{image_folder}/*.png",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output_video,
    ]
    subprocess.run(command, check=True)
    print(f"✅ Video saved as {output_video}")


import os
import glob
import imageio
import imageio.v3 as iio  # Newer version of imageio
import imageio_ffmpeg  # Ensure ffmpeg support


def png_to_video_imageio(image_folder, output_video, fps=10):
    images = sorted(glob.glob(os.path.join(image_folder, "*.png")))

    if not images:
        print("❌ No images found in the directory:", image_folder)
        return

    print(f"🖼️ Found {len(images)} images. First image: {images[0]}")

    # Read first image to get size
    frame = iio.imread(images[0])
    height, width, _ = frame.shape
    print(f"📏 Image size: {width}x{height}")

    writer = imageio.get_writer(
        output_video, fps=fps, codec="libx264", format="FFMPEG", pixelformat="yuv420p"
    )

    for img in images:
        frame = iio.imread(img)
        writer.append_data(frame)

    writer.close()
    print(f"✅ Video saved as {output_video}")


def _video_save(
    env,
    seed,
    step,
    image_folder="./output/image",
    deterministic=False,
    wandb_path="test/simulation_video",
    wandb=wandb,
):
    output_video = f"seed_{seed}_step_{step}.mp4"
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            png_to_video_imageio(image_folder, output_video, fps=10)
            wandb.log({wandb_path: wandb.Video(output_video, fps=10, format="mp4")})
            obs, info = env.reset(seed=args.seed)


# ----------------------
# 🏗️  Environment Setup
# ----------------------
env = gym.make(args.env_id, observation_type=args.observation_type)
env = PhysiCellModelWrapper(env)
env = gym.wrappers.RescaleAction(env, min_action=-1, max_action=1)
env = gym.wrappers.GrayscaleObservation(env)
env = gym.wrappers.FrameStackObservation(env, stack_size=1)
obs, info = env.reset(seed=args.seed)

# ----------------------
# 📂 Logging Setup
# ----------------------
log_dir = f"./tensorboard_logs/{algo_name}"
os.makedirs(log_dir, exist_ok=True)

# ----------------------
# 🏃 Train the Model (with WandB Callback)
# ----------------------
model = algorithm("CnnPolicy", env, verbose=1, tensorboard_log=log_dir, seed=args.seed)
new_logger = configure(log_dir, ["tensorboard"])
model.set_logger(new_logger)
# ✅ Finish WandB run
# del model # remove to demonstrate saving and loading
# ----------------------
# 🎮 Run the Trained Agent
# ----------------------
# model = algorithm.load(path_saving_model) # load model
for i in range(20):
    model.learn(
        total_timesteps=int(25000),
        log_interval=1,
        progress_bar=False,
        callback=TensorboardCallback(),
    )
    _video_save(env=env, seed=args.seed, step=(i) * 25000, wandb=wandb)
    _video_save(
        env=env, seed=args.seed, step=(i) * 25000 + 1, wandb=wandb, deterministic=True
    )

path_saving_model = run_name + "/model"
model.save(path_saving_model)

print("Finished")
wandb.finish()  # ✅ Finish WandB run
