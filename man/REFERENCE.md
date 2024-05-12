# PhysiGym Reference Man Page

This is the technical description of the machinery and how to operate it.


## physigym module

References are maintained in each module's [docstring](https://en.wikipedia.org/wiki/Docstring).\
You can access them through the [source code](https://github.com/Dante-Berth/PhysiGym/tree/main/physigym/custom_modules/physigym/physigym/envs) 
or by first loading the physigym module and environment,

```bash
cd path/to/PhysiCell
```

```python3
import gymnasium
import physigym

env = gymnasium.make('physigym/ModelPhysiCellEnv')
```

then, for each physicell module, getting on-the-fly reference information with the [help](https://en.wikipedia.org/wiki/Help!) command.

### About the ModulePhysiCell class
+ [help(physigym.envs.ModelPhysiCellEnv)](https://github.com/Dante-Berth/PhysiGym/blob/main/man/docstring/physigym.envs.ModelPhysiCellEnv.md)

### Class functions to run epochs:
+ [help(physigym.envs.ModelPhysiCellEnv.__init__)](https://github.com/Dante-Berth/PhysiGym/blob/main/man/docstring/physigym.envs.ModelPhysiCellEnv.__init__.md)  # initialize environment
+ [help(physigym.envs.ModelPhysiCellEnv.reset)](https://github.com/Dante-Berth/PhysiGym/blob/main/man/docstring/physigym.envs.ModelPhysiCellEnv.reset.md)  # reset environment
+ [help(physigym.envs.ModelPhysiCellEnv.render)](https://github.com/Dante-Berth/PhysiGym/blob/main/man/docstring/physigym.envs.ModelPhysiCellEnv.render.md)  # render environment output
+ [help(physigym.envs.ModelPhysiCellEnv.step)](https://github.com/Dante-Berth/PhysiGym/blob/main/man/docstring/physigym.envs.ModelPhysiCellEnv.step.md)  # step through environment
+ [help(physigym.envs.ModelPhysiCellEnv.close)](https://github.com/Dante-Berth/PhysiGym/blob/main/man/docstring/physigym.envs.ModelPhysiCellEnv.close.md)  # close environment
+ [help(physigym.envs.ModelPhysiCellEnv.verbose_true)](https://github.com/Dante-Berth/PhysiGym/blob/main/man/docstring/physigym.envs.ModelPhysiCellEnv.verbose_true.md)  # physigym standard stream output on
+ [help(physigym.envs.ModelPhysiCellEnv.verbose_false)](https://github.com/Dante-Berth/PhysiGym/blob/main/man/docstring/physigym.envs.ModelPhysiCellEnv.verbose_false.md)  # physigym standard stream output off

### Edit this class functions from this [template](https://github.com/Dante-Berth/PhysiGym/blob/main/physigym/custom_modules/physigym/physigym/envs/physicell_model.py) to specify the model:
+ [help(physigym.envs.ModelPhysiCellEnv.get_action_space)](https://github.com/Dante-Berth/PhysiGym/blob/main/man/docstring/physigym.envs.ModelPhysiCellEnv.get_action_space.md)
+ [help(physigym.envs.ModelPhysiCellEnv.get_observation_space)](https://github.com/Dante-Berth/PhysiGym/blob/main/man/docstring/physigym.envs.ModelPhysiCellEnv.get_observation_space.md)
+ [help(physigym.envs.ModelPhysiCellEnv.get_img)](https://github.com/Dante-Berth/PhysiGym/blob/main/man/docstring/physigym.envs.ModelPhysiCellEnv.get_img.md)
+ [help(physigym.envs.ModelPhysiCellEnv.get_observation)](https://github.com/Dante-Berth/PhysiGym/blob/main/man/docstring/physigym.envs.ModelPhysiCellEnv.get_observation.md)
+ [help(physigym.envs.ModelPhysiCellEnv.get_info)](https://github.com/Dante-Berth/PhysiGym/blob/main/man/docstring/physigym.envs.ModelPhysiCellEnv.get_info.md)
+ [help(physigym.envs.ModelPhysiCellEnv.get_terminated)](https://github.com/Dante-Berth/PhysiGym/blob/main/man/docstring/physigym.envs.ModelPhysiCellEnv.get_terminated.md)
+ [help(physigym.envs.ModelPhysiCellEnv.get_reward)](https://github.com/Dante-Berth/PhysiGym/blob/main/man/docstring/physigym.envs.ModelPhysiCellEnv.get_reward.md)

### Pure internal class functions:
help(physigym.envs.CorePhysiCellEnv.get_truncated)
