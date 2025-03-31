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

absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("PhysiCell") + len("PhysiCell")
]
sys.path.append(absolute_path)
from rl.utils.wrappers.wrapper_physicell_complex_tme import PhysiCellModelWrapper
from rl.utils.replay_buffer.simple_replay_buffer import ReplayBuffer
from rl.utils.replay_buffer.smart_image_replay_buffer import ImgReplayBuffer
import matplotlib.pyplot as plt
import os
import glob
import imageio
import imageio.v3 as iio  # Newer version of imageio

LOG_STD_MAX = 2
LOG_STD_MIN = -5


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
                nn.Conv2d(obs_shape[0], num_channels, 7, stride=2),
                nn.Mish(inplace=False),
                nn.Conv2d(num_channels, num_channels, 5, stride=2),
                nn.Mish(inplace=False),
                nn.Conv2d(num_channels, num_channels, 3, stride=2),
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
    """Critic network (Q-function)"""

    def __init__(self, env, cfg):
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = FeatureExtractor(env, cfg["cfg_FeatureExtractor"])

        # Fully connected layers
        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.LazyLinear(256)
        self.fc3 = nn.LazyLinear(1)
        self.mish = nn.Mish()

    def forward(self, x, a):
        x = self.feature_extractor(x)  # Extract features
        x = torch.cat([x, a], dim=1)  # Concatenate state and action

        x = self.mish(self.fc1(x))
        x = self.mish(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    """Policy network (Actor)"""

    def __init__(self, env, cfg):
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = FeatureExtractor(env, cfg["cfg_FeatureExtractor"])
        action_dim = np.prod(env.action_space.shape)

        # Fully connected layers
        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.LazyLinear(256)
        self.fc_mean = nn.LazyLinear(action_dim)
        self.fc_logstd = nn.LazyLinear(action_dim)
        self.relu = nn.ReLU()
        # Action scaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = self.feature_extractor(x)  # Extract features

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # Stable variance scaling

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


# Wrap the environment
list_variable_name = ["anti_M2", "anti_pd1"]


@dataclass
class Args:
    name: str = "sac"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    wandb_project_name: str = "SAC_IMAGE_COMPLEX_TME"
    """the wandb's project name"""
    wandb_entity: str = "corporate-manu-sureli"

    # Algorithm specific arguments
    env_id: str = "physigym/ModelPhysiCellEnv-v0"
    """the id of the environment"""
    observation_type: str = "simple"
    """the type of observation"""
    total_timesteps: int = int(1e6)
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: float = 5e3
    """timestep to start learning"""
    policy_lr: float = 1e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 5e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    wandb_track: bool = True
    """track with wandb"""
    video: bool = False
    """save video"""


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


def png_to_video_imageio(
    output_video: str, image_folder: str = "./output/image", fps: int = 10
):
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


def main():
    args = tyro.cli(Args)
    config = vars(args)
    run_name = f"{args.env_id}__{args.name}_{args.wandb_entity}_{int(time.time())}"
    run_dir = f"runs/{run_name}"
    if args.wandb_track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=f"{args.name}: seed_{args.seed}_observationtype_{args.observation_type}",
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
    env = gym.make(args.env_id, observation_type=args.observation_type)
    height = env.unwrapped.height
    width = env.unwrapped.width
    x_min = env.unwrapped.x_min
    y_min = env.unwrapped.y_min
    x_max = env.unwrapped.x_max
    y_max = env.unwrapped.y_max
    color_mapping = env.unwrapped.color_mapping

    env = PhysiCellModelWrapper(env=env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    shape_observation_space_env = env.observation_space.shape
    is_gray = True if args.observation_type == "image_gray" else False

    cfg = {"cfg_FeatureExtractor": {}}
    actor = Actor(env, cfg).to(device)
    qf1 = QNetwork(env, cfg).to(device)
    qf2 = QNetwork(env, cfg).to(device)
    qf1_target = QNetwork(env, cfg).to(device)
    qf2_target = QNetwork(env, cfg).to(device)
    target_actor = Actor(env, cfg).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(env.action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha
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
        else ImgReplayBuffer(
            action_dim=np.array(env.action_space.shape).prod(),
            device=device,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            height=height,
            width=width,
            x_min=x_min,
            y_min=y_min,
            color_mapping=color_mapping,
            image_gray=is_gray,
        )
    )

    # TRY NOT TO MODIFY: start the game
    obs, info = env.reset()
    episode = 1
    df_cell_obs = info["df_cell"] if "image" in args.observation_type else None
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step <= args.learning_starts:
            actions = np.array(env.action_space.sample())
        else:
            x = obs
            x = torch.Tensor(x).to(device).unsqueeze(0)
            actions, _, _ = actor.get_action(x)
            actions = actions.detach().squeeze(0).cpu().numpy()
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, info = env.step(actions)
        next_df_cell_obs = info["df_cell"] if "image" in args.observation_type else None
        done = terminations or truncations

        if args.observation_type == "simple":
            rb.add(obs, actions, rewards, next_obs, done)
        else:
            rb.add(df_cell_obs, actions, rewards, next_df_cell_obs, done)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs.copy()
        df_cell_obs = (
            next_df_cell_obs.copy() if "image" in args.observation_type else None
        )
        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample()
            with torch.no_grad():
                data_next_state = data["next_state"]

                data_state = data["state"]

                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    data_next_state
                )
                qf1_next_target = qf1_target(data_next_state, next_state_actions)
                qf2_next_target = qf2_target(data_next_state, next_state_actions)
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - alpha * next_state_log_pi
                )
                next_q_value = data["reward"].flatten() + (
                    1 - data["done"].flatten()
                ) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data_state, data["action"]).view(-1)
            qf2_a_values = qf2(data_state, data["action"]).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data_state)
                    qf1_pi = qf1(data_state, pi)
                    qf2_pi = qf2(data_state, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data_state)
                        alpha_loss = (
                            -log_alpha.exp() * (log_pi + target_entropy)
                        ).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

        writer.add_scalar("env/reward_value", rewards, global_step)
        writer.add_scalar(
            "env/number_cancer_cells",
            info["number_cancer_cells"],
            global_step,
        )
        print(f"Number of cancer cells {info['number_cancer_cells']}")
        writer.add_scalar(
            "env/number_m2",
            info["number_m2"],
            global_step,
        )
        writer.add_scalar("env/drug_apoptosis", actions[0], global_step)
        writer.add_scalar("env/drug_reducing_antiapoptosis", actions[1], global_step)
        if done:
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            writer.add_scalar(
                "charts/episodic_return", info["episode"]["r"], global_step
            )
            writer.add_scalar(
                "charts/episodic_length", info["episode"]["l"], global_step
            )
            episode += 1
            if episode % 128 == 0:
                output_video = f"name_{args.name}_seed_{args.seed}_step_{episode}.mp4"
                obs, info = env.reset()
                done = False
                step_episode = 0
                while not done:
                    x = obs
                    x = torch.Tensor(x).to(device).unsqueeze(0)
                    with torch.no_grad():  # Disable gradients for inference
                        actions, _, _ = actor.get_action(x)
                    actions = actions.detach().squeeze(0).cpu().numpy()
                    obs, _, terminated, truncated, info = env.step(actions)
                    step_episode += 1
                    if args.video:
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
                    done = terminated or truncated
                    if done:
                        if args.video:
                            png_to_video_imageio(
                                image_folder + f"/{episode}/" + output_video,
                                image_folder + f"/{episode}",
                                fps=10,
                            )
                            if args.wandb_track:
                                wandb.log(
                                    {
                                        "test/simulation_video": wandb.Video(
                                            image_folder
                                            + f"/{episode}/"
                                            + output_video,
                                            fps=10,
                                            format="mp4",
                                        )
                                    }
                                )
    env.close()
    writer.close()


if __name__ == "__main__":
    main()
