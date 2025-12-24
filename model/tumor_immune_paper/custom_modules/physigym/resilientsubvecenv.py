import multiprocessing as mp
import numpy as np
from stable_baselines.common.vec_env.base_vec_env import CloudpickleWrapper
from stable_baselines3.common.vec_env import _worker, _flatten_obs, SubprocVecEnv


class ResilientSubprocVecEnv(SubprocVecEnv):
    def __init__(self, env_fns, start_method="spawn"):
        assert start_method == "spawn", "Use spawn for PhysiCell"
        self.env_fns = env_fns
        super().__init__(env_fns, start_method=start_method)

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
        remote, work_remote = ctx.Pipe(duplex=True)

        proc = ctx.Process(
            target=_worker,
            args=(work_remote, remote, CloudpickleWrapper(self.env_fns[i])),
            daemon=True,
        )
        proc.start()
        work_remote.close()

        self.remotes[i] = remote
        self.processes[i] = proc

        # sync spaces
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
                info = {"crashed": True}
                results.append((obs, reward, done, info))

        obs, rews, dones, infos = zip(*results)
        return (
            _flatten_obs(obs, self.observation_space),
            np.array(rews),
            np.array(dones),
            infos,
        )

    def reset(self):
        for i, remote in enumerate(self.remotes):
            try:
                remote.send(("reset", None))
            except Exception:
                self._restart_env(i)

        obs = []
        for i, remote in enumerate(self.remotes):
            try:
                obs.append(remote.recv())
            except Exception:
                self._restart_env(i)
                obs.append(self.observation_space.sample())

        return _flatten_obs(obs, self.observation_space)
