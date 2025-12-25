import multiprocessing as mp
import numpy as np

from stable_baselines3.common.vec_env.subproc_vec_env import (
    SubprocVecEnv,
    _worker,
    _stack_obs,
)
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper


class ResilientSubprocVecEnv(SubprocVecEnv):
    def __init__(self, env_fns, start_method="spawn"):
        assert start_method == "spawn", "PhysiCell requires spawn"

        self.env_fns = env_fns
        super().__init__(env_fns, start_method=start_method)

        # 🔧 Make remotes/processes mutable
        self.remotes = list(self.remotes)
        self.processes = list(self.processes)

    def _restart_env(self, i):
        print(f"[ResilientVecEnv] Restarting env {i}")

        try:
            if self.processes[i].is_alive():
                self.processes[i].terminate()
        except Exception:
            pass

        try:
            self.remotes[i].close()
        except Exception:
            pass

        ctx = mp.get_context("spawn")
        remote, work_remote = ctx.Pipe()

        proc = ctx.Process(
            target=_worker,
            args=(work_remote, remote, CloudpickleWrapper(self.env_fns[i])),
            daemon=True,
        )
        proc.start()
        work_remote.close()

        self.remotes[i] = remote
        self.processes[i] = proc

        # Sync spaces (SB3 expects this)
        remote.send(("get_spaces", None))
        remote.recv()

    def step_wait(self):
        results = []

        for i, remote in enumerate(self.remotes):
            try:
                results.append(remote.recv())
            except (EOFError, BrokenPipeError):
                self._restart_env(i)

                obs = self.observation_space.sample()
                reward = 0.0
                done = True
                info = {"not_crashed": False}
                results.append((obs, reward, done, info, reset_info))

        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results)

        return (
            _stack_obs(obs, self.observation_space),
            np.stack(rews),
            np.stack(dones),
            infos,
        )

    def reset(self):
        for env_idx, remote in enumerate(self.remotes):
            try:
                remote.send(("reset", (self._seeds[env_idx], self._options[env_idx])))
            except Exception:
                self._restart_env(env_idx)
                remote = self.remotes[env_idx]
                remote.send(("reset", (None, None)))

        results = []
        for i, remote in enumerate(self.remotes):
            try:
                results.append(remote.recv())
            except Exception:
                self._restart_env(i)
                obs = self.observation_space.sample()
                reset_info = {}
                results.append((obs, reset_info))

        obs, self.reset_infos = zip(*results)
        self._reset_seeds()
        self._reset_options()

        return _stack_obs(obs, self.observation_space)
