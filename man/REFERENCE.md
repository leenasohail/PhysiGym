# PhysiGym Reference Man Page

This is the technical description of the machinery and how to operate it.


# physigym module

References are maintained in each module's [docstring](https://en.wikipedia.org/wiki/Docstring).\
You can access them through the [source code](https://github.com/elmbeech/physicellembedding/blob/main/py3pc_embedding/custom_modules/physicellmodule.cpp#L449), or by first loading the physicell module.

```python3
import gymnasium
import physigym
```
Then, for each physicell module, get on the fly reference information with the [help](https://en.wikipedia.org/wiki/Help!) command.

```python3
# all module functions
help(physigym)

# to run epochs
help(env.__init__)
help(env.reset)
help(env.render)
help(env.step)
help(env.close)
help(env.verbose_false)
help(env.verbose_true)

# to specify models 
help(env.get_action_space)
help(env.get_observation_space)
help(env.get_img)
help(env.get_observation)
help(env.get_info)
help(env.get_terminated)
help(env.get_reward)

# pure internal functions
help(env.get_truncated)
