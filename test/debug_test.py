import gymnasium
import physigym
env = gymnasium.make('physigym/ModelPhysiCellEnv-v0')

for _ in range(5):
    env.reset()
    env.step(action={})
    env.close()