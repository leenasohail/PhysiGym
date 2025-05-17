import gymnasium as gym
import numpy as np
import os
import physigym  # import the Gymnasium PhysiCell bridge module
import random
import shutil
import os
import random
import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import wandb
import tyro
import sys, os
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("PhysiCell") + len("PhysiCell")
]
sys.path.append(absolute_path)
from rl.utils.wrappers.wrapper_physicell_complex_tme import PhysiCellModelWrapper
from rl.utils.replay_buffer.set_replay_buffer import (
    MinimalImgReplayBuffer,
    ReplayBuffer,
)
import matplotlib.pyplot as plt
import os
import imageio.v3 as iio  # Newer version of imageio


class PixelPreprocess(nn.Module):
    """
    Normalizes pixel observations to [-0.5, 0.5].
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div(255.0).sub(0.5)


class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, simnorm_dim: int = 8):
        super().__init__()
        self.dim = simnorm_dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


class FeatureExtractor(nn.Module):
    """Handles both image-based and vector-based state inputs dynamically."""

    def __init__(self, env, cfg):
        super().__init__()
        self.cfg = cfg

        obs_shape = env.observation_space.shape
        self.is_image = len(obs_shape) == 3  # Check if input is an image (C, H, W)

        if self.is_image:
            # CNN feature extractor
            num_channels = 8
            layers = [
                PixelPreprocess(),
                nn.Conv2d(obs_shape[0], num_channels, 7 * 8, stride=5),
                nn.Mish(inplace=False),
                nn.Conv2d(num_channels, num_channels, 8, stride=5),
                nn.Mish(inplace=False),
                nn.Conv2d(num_channels, num_channels, 3, stride=3),
                nn.Mish(inplace=False),
                nn.Conv2d(num_channels, num_channels, 3, stride=1),
                nn.Flatten(),
            ]
            self.feature_extractor = nn.Sequential(*layers)
            self.feature_size = self._get_feature_size(obs_shape)
        else:
            # Directly flatten vector input
            self.feature_extractor = nn.Identity()
            self.feature_size = np.prod(obs_shape)

    def _get_feature_size(self, obs_shape):
        """Pass a dummy tensor through CNN to compute feature size dynamically."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            out = self.feature_extractor(dummy_input)
            return int(np.prod(out.shape[1:]))

    def forward(self, x):
        if self.is_image:
            x = self.feature_extractor(x)  # Apply CNN
            x = x.view(x.size(0), -1)  # Flatten
        return x


class QNetwork(nn.Module):
    def __init__(self, env, n_atoms=101, v_min=-100, v_max=100):
        super().__init__()
        self.env = env
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.n = env.single_action_space.n
        obs_shape = env.observation_space.shape
        self.network = nn.Sequential(
            nn.Linear(np.array(obs_shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.n * n_atoms),
        )

    def get_action(self, x, action=None):
        logits = self.network(x)
        # probability mass function for each action
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(x)), action]


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


# Wrap the environment
list_variable_name = ["anti_M2", "anti_pd1"]


@dataclass
class Args:
    name: str = "c51"
    """The name of this experiment."""

    # Environment and experiment settings
    weight: float = 0.8
    """Weight for the reduction of tumor in reward shaping."""
    reward_type: str = "simple"
    """Type of the reward function used."""
    seed: int = 1
    """Seed for reproducibility."""
    torch_deterministic: bool = True
    """If True, makes torch backend deterministic."""
    cuda: bool = True
    """If True, enables CUDA (GPU support)."""
    track: bool = False
    """Enable tracking (e.g., for wandb)."""
    wandb_project_name: str = "C51_IMAGE_COMPLEX_TME"
    wandb_entity: str = "corporate-manu-sureli"
    wandb_track: bool = True
    """If True, logs metrics to Weights & Biases."""

    # Environment and observation
    env_id: str = "physigym/ModelPhysiCellEnv-v0"
    observation_type: str = "simple"

    # Training control
    total_timesteps: int = int(1e6)
    buffer_size: int = int(1e6)
    batch_size: int = 256
    learning_starts: int = 10_000
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    autotune: bool = True

    # C51-specific settings
    n_atoms: int = 201  # increased from 101 → improves value resolution
    """Number of atoms in the value distribution (higher for better granularity)."""
    v_min: float = 0.0
    """Minimum value of value distribution — matches min cumulative reward (since reward is [0,1])."""
    v_max: float = 256.0
    """Maximum value of value distribution — 1 (max reward/step) × 256 steps/episode."""
    
    # Exploration schedule
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5

    # Target network update (soft)
    tau: float = 0.005
    """Soft update coefficient for the target network."""
    target_network_frequency: int = 2
    """Deprecated if using soft update, retained for compatibility."""

def saving_img(
    image_folder: str,
    info: dict,
    step_episode: int,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    saving_title: str = "output_simulation_image_episode",
    color_mapping: dict = {},
):
    def rgb_to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255) if rgb[0] <= 1 else int(rgb[0]),
            int(rgb[1] * 255) if rgb[1] <= 1 else int(rgb[1]),
            int(rgb[2] * 255) if rgb[2] <= 1 else int(rgb[2]),
        )

    os.makedirs(image_folder, exist_ok=True)
    df_cell = info["df_cell"]
    fig, ax = plt.subplots(
        1, 3, figsize=(10, 6), gridspec_kw={"width_ratios": [1, 0.2, 0.2]}
    )
    count_cancer_cell = info["number_cancer_cells"]
    unique_cell_types = df_cell["type"].unique().tolist()
    for cell_type in unique_cell_types:
        tuple_color = color_mapping[cell_type]
        df_celltype = df_cell.loc[
            (df_cell.dead == 0.0) & (df_cell.type == cell_type), :
        ]
        df_celltype.plot(
            kind="scatter",
            x="x",
            y="y",
            c=rgb_to_hex(tuple_color),
            xlim=[
                x_min,
                x_max,
            ],
            ylim=[
                y_min,
                y_max,
            ],
            grid=True,
            label=cell_type,
            s=100,
            title=f"episode step {str(step_episode).zfill(3)}, cancer cell: {count_cancer_cell}",
            ax=ax[0],
        ).legend(loc="lower left")

    # Create a colormap for the color bars (from -1 to 1)
    list_colors = ["royalblue", "darkorange"]

    # Function to create fluid-like color bars
    def create_fluid_bar(ax_bar, drug_amount, title, max_amount=1, color="cyan"):
        ax_bar.set_xlim(0, 1)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_title(title, fontsize=10)
        ax_bar.set_xticks([])
        ax_bar.set_yticks(np.linspace(0, 1, 5))  # 0% to 100% scale

        # Normalize drug amount (convert to percentage of max)
        fill_level = drug_amount / max_amount

        # Fill up to the corresponding level
        ax_bar.fill_betweenx(np.linspace(0, fill_level, 100), 0, 1, color=color)

        # Draw container border
        ax_bar.spines["left"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.spines["top"].set_visible(True)
        ax_bar.spines["bottom"].set_visible(True)

    action = info["action"]
    for i, (key, value) in enumerate(action.items(), start=1):  # Start index from 1
        create_fluid_bar(ax[i], value[0], f"drug_{i}", color=list_colors[i - 1])

    plt.savefig(image_folder + f"/{saving_title} step {str(step_episode).zfill(3)}")
    plt.close(fig)


def main():
    args = tyro.cli(Args)
    config = vars(args)
    run_name = f"{args.env_id}__{args.name}_{args.wandb_entity}_{int(time.time())}"
    run_dir = f"runs/{run_name}"
    if args.wandb_track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=f"{args.name}: seed_{args.seed}_observationtype_{args.observation_type}_weight_{args.weight}_rewardtype_{args.reward_type}",
            sync_tensorboard=True,
            config=config,
            monitor_gym=True,
            save_code=True,
        )
        print("Wandb selected")
    else:
        print("Tensorboard selected")

    os.makedirs(run_dir, exist_ok=True)
    image_folder = run_dir + "/image"
    os.makedirs(image_folder, exist_ok=True)
    model_dir = run_dir + "/models"
    os.makedirs(model_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    env = gym.make(
        args.env_id,
        observation_type=args.observation_type,
        reward_type=args.reward_type,
    )
    height = env.unwrapped.height
    width = env.unwrapped.width
    x_min = env.unwrapped.x_min
    y_min = env.unwrapped.y_min
    x_max = env.unwrapped.x_max
    y_max = env.unwrapped.y_max
    color_mapping = env.unwrapped.color_mapping
    cumulative_return = 0
    env = PhysiCellModelWrapper(env=env, discrete=True)
    is_gray = True if args.observation_type == "image_gray" else False
    cfg = {"cfg_FeatureExtractor": {}}
    q_network = QNetwork(
        env, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max
    ).to(device)
    optimizer = optim.Adam(
        q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size
    )
    target_network = QNetwork(
        env, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max
    ).to(device)
    target_network.load_state_dict(q_network.state_dict())
    type_to_int = {
        name: idx for idx, name in enumerate(sorted(env.unwrapped.unique_cell_types))
    }
    rb = (
        ReplayBuffer(
            state_dim=np.array(env.observation_space.shape).prod(),
            action_dim=np.array(env.action_space.shape).prod(),
            device=device,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            state_type=env.observation_space.dtype,
        )
        if args.observation_type == "simple"
        else MinimalImgReplayBuffer(
            action_dim=np.array(env.action_space.shape).prod(),
            device=device,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            height=height,
            width=width,
            x_min=x_min,
            y_min=y_min,
            type_to_color={v: color_mapping[k] for k, v in type_to_int.items()},
            image_gray=is_gray,
        )
    )

    # TRY NOT TO MODIFY: start the game
    obs, info = env.reset(seed=args.seed)
    episode = 1
    step_episode = 0
    df_cell_obs = info["df_cell"] if "image" in args.observation_type else None
    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )
        if random.random() < epsilon:
            actions = np.array(env.action_space.sample())
        else:
            x = obs
            x = torch.Tensor(x).to(device).unsqueeze(0)
            actions, pmf = q_network.get_action(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, info = env.step(actions)
        step_episode += 1
        saving_img(
            image_folder=image_folder + f"/{episode}",
            info=info,
            step_episode=step_episode,
            x_max=x_max,
            y_max=y_max,
            x_min=x_min,
            y_min=y_min,
            color_mapping=color_mapping,
        )
        next_df_cell_obs = info["df_cell"] if "image" in args.observation_type else None
        done = terminations or truncations
        cumulative_return += rewards
        if args.observation_type == "simple":
            rb.add(obs, actions, rewards, next_obs, done)
        else:
            rb.add(df_cell_obs, actions, rewards, next_df_cell_obs, done, type_to_int)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs.copy()
        df_cell_obs = (
            next_df_cell_obs.copy() if "image" in args.observation_type else None
        )
        if global_step == args.batch_size:
            data = rb.sample()
            data_state = data["state"]

            with torch.no_grad():
                actions, pmf = q_network.get_action(torch.Tensor(data_state).to(device))
            del actions, data_state, data

        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample()
                data_next_state = data["next_state"]
                data_state = data["state"]
                with torch.no_grad():
                    _, next_pmfs = target_network.get_action(data_next_state)
                    next_atoms = data.rewards + args.gamma * target_network.atoms * (
                        1 - data.dones
                    )
                    # projection
                    delta_z = target_network.atoms[1] - target_network.atoms[0]
                    tz = next_atoms.clamp(args.v_min, args.v_max)

                    b = (tz - args.v_min) / delta_z
                    l = b.floor().clamp(0, args.n_atoms - 1)
                    u = b.ceil().clamp(0, args.n_atoms - 1)
                    # (l == u).float() handles the case where bj is exactly an integer
                    # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
                    d_m_l = (u + (l == u).float() - b) * next_pmfs
                    d_m_u = (b - l) * next_pmfs
                    target_pmfs = torch.zeros_like(next_pmfs)
                    for i in range(target_pmfs.size(0)):
                        target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                        target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

                _, old_pmfs = q_network.get_action(data_state, data.actions.flatten())
                loss = (
                    -(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(
                        -1
                    )
                ).mean()

                if global_step % 100 == 0:
                    writer.add_scalar("losses/loss", loss.item(), global_step)
                    old_val = (old_pmfs * q_network.atoms).sum(1)
                    writer.add_scalar(
                        "losses/q_values", old_val.mean().item(), global_step
                    )

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(
                    q_network.parameters(), target_network.parameters()
                ):
                    target_network.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )


        writer.add_scalar("env/anti_M2", actions[0], global_step)
        writer.add_scalar("env/anti_pd1", actions[1], global_step)

        writer.add_scalar("env/reward_value", rewards, global_step)

        writer.add_scalar(
            "env/number_cancer_cells",
            info["number_cancer_cells"],
            global_step,
        )
        writer.add_scalar(
            "env/number_m2",
            info["number_m2"],
            global_step,
        )

        writer.add_scalar(
            "env/number_cd8",
            info["number_cd8"],
            global_step,
        )

        writer.add_scalar(
            "env/number_cd8exhausted",
            info["number_cd8exhausted"],
            global_step,
        )
        if done:
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            print(
                f"global_step={global_step}, episodic_return={cumulative_return / step_episode}"
            )
            writer.add_scalar(
                "charts/episodic_return", cumulative_return / step_episode, global_step
            )
            writer.add_scalar("charts/episodic_length", step_episode, global_step)
            episode += 1
            step_episode = 0
            total_steps = 0
            cumulative_return = 0
            obs, info = env.reset(seed=None)
            done = False
            if episode % 50 == 0:
                cumumative_return_episode = 0
                step_episode = 0
                checkpoint = {
                    "q_network_state_dict": q_network.state_dict(),
                    "episode": episode,  # if defined
                }

                torch.save(checkpoint, model_dir + f"/sac_checkpoint_{episode}.pth")
                for k in range(1, 6):
                    while not done:
                        x = obs
                        x = torch.Tensor(x).to(device).unsqueeze(0)
                        with torch.no_grad():
                            actions, pmf = q_network.get_action(
                                torch.Tensor(data_state).to(device)
                            )
                        actions = actions.cpu().numpy()
                        obs, reward, terminated, truncated, info = env.step(actions)
                        saving_img(
                            image_folder=image_folder + f"/{episode}_test_{k}",
                            info=info,
                            step_episode=step_episode,
                            x_max=x_max,
                            y_max=y_max,
                            x_min=x_min,
                            y_min=y_min,
                            color_mapping=color_mapping,
                        )
                        step_episode += 1
                        total_steps += 1
                        cumumative_return_episode += reward
                        done = terminated or truncated
                    if done:
                        done = False
                        cumulative_return += cumumative_return_episode / step_episode
                        writer.add_scalar(
                            "charts/episodic_return_test",
                            cumumative_return_episode / step_episode,
                            global_step,
                        )
                        step_episode = 0
                        cumumative_return_episode = 0
                        obs, info = env.reset(seed=None)
                writer.add_scalar(
                    "charts/episodic_return_mean_test",
                    cumulative_return / k,
                    global_step,
                )

    env.close()
    writer.close()


if __name__ == "__main__":
    main()
