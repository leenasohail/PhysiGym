import os
from collections import defaultdict
from functools import partial

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import sys
# Tensorboard: Prevent tf from allocating full GPU memory
import tensorflow as tf
import tqdm
from flax.metrics import tensorboard
from flax.training.train_state import TrainState
absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("PhysiCell") + len("PhysiCell")
]
sys.path.append(absolute_path)
from rl.tdmpc2.tdmpc2_jax.tdmpc2 import TDMPC2, WorldModel
from rl.tdmpc2.tdmpc2_jax.common.activations import mish, simnorm
from rl.tdmpc2.tdmpc2_jax.data import SequentialReplayBuffer
from rl.tdmpc2.tdmpc2_jax.envs.dmcontrol import make_dmc_env
from rl.tdmpc2.tdmpc2_jax.networks import NormedLinear
from rl.utils.wrappers.wrapper_physicell_tme import PhysiCellModelWrapper, wrap_env_with_rescale_stats_autoreset

import wandb
import time
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from torch.utils.tensorboard import SummaryWriter
list_variable_name = ["drug_apoptosis", "drug_reducing_antiapoptosis"]

from dataclasses import dataclass, field, asdict
import physigym

@dataclass
class EnvConfig:
    backend: str = "gymnasium"
    env_id: str = "physigym/ModelPhysiCellEnv-v0"
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "TDMPC2_ModelTmePhysiCellEnv_PhysiGym"
    """the wandb's project name"""
    wandb_entity: str = "corporate-manu-sureli"
    num_envs: int = 1
    utd_ratio: float = 0.5
    asynchronous: bool = True
    dmc_obs_type: str = "state"

@dataclass
class EncoderConfig:
    encoder_dim: int = 256
    num_encoder_layers: int = 2
    learning_rate: float = 1e-4
    tabulate: bool = False

@dataclass
class WorldModelConfig:
    mlp_dim: int = 512
    latent_dim: int = 512
    value_dropout: float = 0.01
    num_value_nets: int = 5
    num_bins: int = 101
    symlog_min: int = -10
    symlog_max: int = 10
    symlog_obs: bool = False
    simnorm_dim: int = 8
    learning_rate: float = 3e-4
    predict_continues: bool = False
    dtype: str = "float32"
    max_grad_norm: int = 20
    tabulate: bool = False

@dataclass
class TDMPCC2Config:
    # Planning
    mpc: bool = True
    horizon: int = 3
    mppi_iterations: int = 6
    population_size: int = 512
    policy_prior_samples: int = 24
    num_elites: int = 64
    min_plan_std: float = 0.05
    max_plan_std: float = 2.0
    temperature: float = 0.5
    # Optimization
    batch_size: int = 256
    discount: float = 0.99
    rho: float = 0.5
    consistency_coef: int = 20
    reward_coef: float = 0.1
    continue_coef: float = 1.0
    value_coef: float = 0.1
    entropy_coef: float = 1e-4
    tau: float = 0.01

@dataclass
class Config:
    seed: int = 0
    max_steps: int = 500_000
    save_interval_steps: int = 10_000
    log_interval_steps: int = 1_000
    env: EnvConfig = field(default_factory=EnvConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    tdmpc2: TDMPCC2Config = field(default_factory=TDMPCC2Config)


def train(cfg: dict):
    env_config = cfg.env
    encoder_config = cfg.encoder
    model_config = cfg.world_model
    tdmpc_config = cfg.tdmpc2
    #args = tyro.cli(cfg)
    run_name = f"{env_config.env_id}__{env_config.wandb_entity}_{int(time.time())}"
    wandb.init(
        project=env_config.wandb_project_name,
        sync_tensorboard=True,
        config=cfg,
        monitor_gym=True,
        save_code=True,
        allow_val_change=True,
    )
    config = wandb.config
    run_dir = f"runs/{run_name}"
    os.makedirs(run_dir, exist_ok=True)
    output_dir = run_dir
    for key in vars(config):
        if key in config:
            setattr(config, key, config[key])

    ##############################
    # Logger setup
    ##############################
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
    )

    ##############################
    # Environment setup
    ##############################
    def make_gym_env(env_id):
        env = gym.make(env_id)
        env = PhysiCellModelWrapper(env)
        env = wrap_env_with_rescale_stats_autoreset(env)
        return env
        

    
    env = gym.vector.SyncVectorEnv([partial(make_gym_env,env_config.env_id) 
                            for seed in range(cfg.seed, cfg.seed+env_config.num_envs)])
    
    np.random.seed(cfg.seed)
    rng = jax.random.PRNGKey(cfg.seed)

    ##############################
    # Agent setup
    ##############################
    dtype = jnp.dtype(model_config.dtype)
    rng, model_key, encoder_key = jax.random.split(rng, 3)
    encoder_module = nn.Sequential(
        [
            NormedLinear(encoder_config.encoder_dim, activation=mish, dtype=dtype)
            for _ in range(encoder_config.num_encoder_layers - 1)
        ]
        + [
            NormedLinear(
                model_config.latent_dim,
                activation=partial(simnorm, simplex_dim=model_config.simnorm_dim),
                dtype=dtype,
            )
        ]
    )

    if encoder_config.tabulate:
        print("Encoder")
        print("--------------")
        print(
            encoder_module.tabulate(
                jax.random.key(0), env.observation_space.sample(), compute_flops=True
            )
        )

    ##############################
    # Replay buffer setup
    ##############################
    dummy_obs, _ = env.reset()
    dummy_action = env.action_space.sample()
    dummy_next_obs, dummy_reward, dummy_term, dummy_trunc, _ = env.step(dummy_action)
    replay_buffer = SequentialReplayBuffer(
        capacity=cfg.max_steps // env_config.num_envs,
        num_envs=env_config.num_envs,
        seed=cfg.seed,
        dummy_input=dict(
            observation=dummy_obs,
            action=dummy_action,
            reward=dummy_reward,
            next_observation=dummy_next_obs,
            terminated=dummy_term,
            truncated=dummy_trunc,
        ),
    )

    encoder = TrainState.create(
        apply_fn=encoder_module.apply,
        params=encoder_module.init(encoder_key, dummy_obs)["params"],
        tx=optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(model_config.max_grad_norm),
            optax.adam(encoder_config.learning_rate),
        ),
    )

    model = WorldModel.create(
        action_dim=np.prod(env.action_space.shape),
        encoder=encoder,
        **asdict(model_config),
        key=model_key,
    )
    if model.action_dim >= 20:
        tdmpc_config.mppi_iterations += 2

    agent = TDMPC2.create(world_model=model, **asdict(tdmpc_config))
    global_step = 0

    options = ocp.CheckpointManagerOptions(
        max_to_keep=1, save_interval_steps=cfg.save_interval_steps
    )
    checkpoint_path = os.path.abspath(os.path.join(output_dir, "checkpoint"))
    with ocp.CheckpointManager(
        checkpoint_path,
        options=options,
        item_names=("agent", "global_step", "buffer_state"),
    ) as mngr:
        if mngr.latest_step() is not None:
            print("Checkpoint folder found, restoring from", mngr.latest_step())
            abstract_buffer_state = jax.tree.map(
                ocp.utils.to_shape_dtype_struct, replay_buffer.get_state()
            )
            restored = mngr.restore(
                mngr.latest_step(),
                args=ocp.args.Composite(
                    agent=ocp.args.StandardRestore(agent),
                    global_step=ocp.args.JsonRestore(),
                    buffer_state=ocp.args.StandardRestore(abstract_buffer_state),
                ),
            )
            agent, global_step = restored.agent, restored.global_step
            replay_buffer.restore(restored.buffer_state)
        else:
            print("No checkpoint folder found, starting from scratch")
            mngr.save(
                global_step,
                args=ocp.args.Composite(
                    agent=ocp.args.StandardSave(agent),
                    global_step=ocp.args.JsonSave(global_step),
                    buffer_state=ocp.args.StandardSave(replay_buffer.get_state()),
                ),
            )
            mngr.wait_until_finished()
        ##############################
        # Training loop
        ##############################
        ep_info = {}
        ep_count = np.zeros(env_config.num_envs, dtype=int)
        prev_logged_step = global_step
        prev_plan = (
            jnp.zeros((env_config.num_envs, agent.horizon, agent.model.action_dim)),
            jnp.full(
                (env_config.num_envs, agent.horizon, agent.model.action_dim),
                agent.max_plan_std,
            ),
        )
        observation, _ = env.reset(seed=cfg.seed)

        T = 500
        seed_steps = int(max(5 * T, 1000) * env_config.num_envs * env_config.utd_ratio)
        pbar = tqdm.tqdm(initial=global_step, total=cfg.max_steps)
        done = np.zeros(env_config.num_envs)
        for global_step in range(global_step, cfg.max_steps, env_config.num_envs):
            if global_step <= seed_steps:
                action = env.action_space.sample()
            else:
                rng, action_key = jax.random.split(rng)
                prev_plan = (
                    prev_plan[0],
                    jnp.full_like(prev_plan[1], agent.max_plan_std),
                )
                action, prev_plan = agent.act(
                    observation, prev_plan=prev_plan, train=True, key=action_key
                )

            next_observation, reward, terminated, truncated, info = env.step(action)

            writer.add_scalar(
            "env/number_cancer_cells",
            info["number_cancer_cells"],
            global_step,
            )
            writer.add_scalar("env/drug_apoptosis", action[0][0], global_step)
            writer.add_scalar("env/drug_reducing_antiapoptosis", action[0][1], global_step)

            # Get real final observation and store transition
            real_next_observation = next_observation.copy()
            # https://gymnasium.farama.org/gymnasium_release_notes/index.html#release-v1-0-0
            if np.any(done):
                for ienv in range(env_config.num_envs):
                    if not done[ienv]:
                        replay_buffer.insert(
                            dict(
                                observation=observation[ienv],
                                action=action[ienv],
                                reward=reward[ienv],
                                next_observation=real_next_observation[ienv],
                                terminated=terminated[ienv],
                                truncated=truncated[ienv],
                            )
                        )
            else:
                replay_buffer.insert(
                    dict(
                        observation=observation,
                        action=action,
                        reward=reward,
                        next_observation=real_next_observation,
                        terminated=terminated,
                        truncated=truncated,
                    )
                )

            observation = next_observation
            # Handle terminations/truncations
            done = np.logical_or(terminated, truncated)
            if np.any(done):
                prev_plan = (
                    prev_plan[0].at[done].set(0),
                    prev_plan[1].at[done].set(agent.max_plan_std),
                )
                for ienv in range(env_config.num_envs):
                    if done[ienv]:
                        print(
                            f"Episode {ep_count[ienv]}: {info['episode']['r'][ienv]:.2f}, {info['episode']['l'][ienv]}"
                        )
                        writer.add_scalar(
                            f"charts/episodic_return",
                            info["episode"]["r"][ienv],
                            global_step + ienv,
                        )
                        writer.add_scalar(
                            f"charts/episodic_length",
                            info["episode"]["l"][ienv],
                            global_step + ienv,
                        )
                        ep_count[ienv] += 1

            if global_step >= seed_steps:
                if global_step == seed_steps:
                    print("Pre-training on seed data...")
                    num_updates = seed_steps
                else:
                    num_updates = max(
                        1, int(env_config.num_envs * env_config.utd_ratio)
                    )

                rng, *update_keys = jax.random.split(rng, num_updates + 1)
                log_this_step = (
                    global_step >= prev_logged_step + cfg.log_interval_steps
                )
                if log_this_step:
                    all_train_info = defaultdict(list)
                    prev_logged_step = global_step

                for iupdate in range(num_updates):
                    batch = replay_buffer.sample(agent.batch_size, agent.horizon)
                    agent, train_info = agent.update(
                        observations=batch["observation"],
                        actions=batch["action"],
                        rewards=batch["reward"],
                        next_observations=batch["next_observation"],
                        terminated=batch["terminated"],
                        truncated=batch["truncated"],
                        key=update_keys[iupdate],
                    )

                    if log_this_step:
                        for k, v in train_info.items():
                            all_train_info[k].append(np.array(v))

                if log_this_step:
                    for k, v in all_train_info.items():
                        writer.add_scalar(f"train/{k}_mean", np.mean(v), global_step)
                        writer.add_scalar(f"train/{k}_std", np.std(v), global_step)

                mngr.save(
                    global_step,
                    args=ocp.args.Composite(
                        agent=ocp.args.StandardSave(agent),
                        global_step=ocp.args.JsonSave(global_step),
                        buffer_state=ocp.args.StandardSave(replay_buffer.get_state()),
                    ),
                )

            pbar.update(env_config.num_envs)
        pbar.close()


if __name__ == "__main__":
    cfg = Config()

    train(cfg)
