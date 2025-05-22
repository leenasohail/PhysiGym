import gymnasium as gym
import numpy as np
import physigym  # import the Gymnasium PhysiCell bridge module
import random
import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import tyro
import sys, os
from torch.utils.tensorboard import SummaryWriter

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
from rl.utils.img_vid.save_img import saving_img
from rl.utils.utils_layers.layers import QNetwork, ActorContinuous


def l2_project_weights(model):
    """Project weights of all linear layers to unit L2 norm (per row)."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.LazyLinear):
                w = module.weight.data
                w.div_(w.norm(p=2, dim=1, keepdim=True).clamp(min=1e-6))




# Wrap the environment
list_variable_name = ["anti_M2", "anti_pd1"]


@dataclass
class Args:
    name: str = "sac_normlayer_l2"
    """the name of this experiment"""
    weight: float = 0.8
    """weight for the reduction of tumor"""
    reward_type: str = "log"
    """type of the reward"""
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
    observation_type: str = "image_gray"
    """the type of observation"""
    total_timesteps: int = int(1e6)
    """the learning rate of the optimizer"""
    buffer_size: int = int(2e5)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: float = 10e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
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
    env = PhysiCellModelWrapper(env=env)
    is_gray = True if args.observation_type == "image_gray" else False
    cfg = {"cfg_FeatureExtractor": {}}
    actor = ActorContinuous(env, cfg).to(device)
    qf1 = QNetwork(env, cfg).to(device)
    qf2 = QNetwork(env, cfg).to(device)
    qf1_target = QNetwork(env, cfg).to(device)
    qf2_target = QNetwork(env, cfg).to(device)
    target_actor = ActorContinuous(env, cfg).to(device)
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
            data_next_state = data["next_state"]
            data_state = data["state"]

            with torch.no_grad():
                next_state_actions, _, _ = actor.get_action(data_next_state)
                _, _ = (
                    qf1(data_next_state, next_state_actions),
                    qf2(data_next_state, next_state_actions),
                )
                _, _ = (
                    qf1_target(data_next_state, next_state_actions),
                    qf2_target(data_next_state, next_state_actions),
                )
            del next_state_actions, data_next_state, data_state, data

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample()
            data_next_state = data["next_state"]
            data_state = data["state"]

            with torch.no_grad():
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
            l2_project_weights(qf1_target)
            l2_project_weights(qf2_target)

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
                    l2_project_weights(actor)

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data_state)

                        alpha_loss = (
                            -log_alpha.exp() * (log_pi + target_entropy)
                        ).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        l2_project_weights(actor)
                        alpha = log_alpha.exp().item()
                    entropy = -log_pi.mean().item()

                losses = {
                    "losses/min_qf_next_target": min_qf_next_target.mean().item(),
                    "losses/qf1_values": qf1_a_values.mean().item(),
                    "losses/qf2_values": qf2_a_values.mean().item(),
                    "losses/qf1_loss": qf1_loss.item(),
                    "losses/qf2_loss": qf2_loss.item(),
                    "losses/qf_loss": qf_loss.item() / 2.0,
                    "losses/actor_loss": actor_loss.item(),
                    "lossed/entropy": entropy,
                }

                for tag, value in losses.items():
                    writer.add_scalar(tag, value, global_step)

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
        scalars = {
            "env/anti_M2": actions[0],
            "env/anti_pd1": actions[1],
            "env/reward_value": rewards,
            "env/number_cancer_cells": info["number_cancer_cells"],
            "env/number_m2": info["number_m2"],
            "env/number_cd8": info["number_cd8"],
            "env/number_cd8exhausted": info["number_cd8exhausted"],
            "env/reward_cancer_cells": info["reward_cancer_cells"],
            "env/reward_drugs": info["reward_drugs"],
        }
        for tag, value in scalars.items():
            writer.add_scalar(tag, value, global_step)
        if done:
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
                    "actor_state_dict": actor.state_dict(),
                    "qf1_state_dict": qf1.state_dict(),
                    "qf2_state_dict": qf2.state_dict(),
                    "episode": episode,  # if defined
                }

                torch.save(
                    checkpoint, model_dir + f"/{args.name}_checkpoint_{episode}.pth"
                )
                for k in range(1, 4):
                    while not done:
                        x = obs
                        x = torch.Tensor(x).to(device).unsqueeze(0)
                        with torch.no_grad():  # Disable gradients for inference
                            actions, _, _ = actor.get_action(x)
                        actions = actions.detach().squeeze(0).cpu().numpy()
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
