from gymnasium.envs.registration import register

register(
     id="lycee/CorePhysiCellEnv-v0",
     entry_point="lycee.envs:CorePhysiCellEnv",
     max_episode_steps=300,
)
