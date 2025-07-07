# ModelPhysiCellEnv Gymnasium Environment Structure

## Main Functions

# env
```python
env.__init__
    +- settingxml="config/PhysiCell_settings.xml"
    +- cell_type_cmap="turbo"
    +- figsize=(8, 6)
    +- render_mode=None
    +- render_fps=10  # frames per second
    +- verbose=True
    +- **kwargs
```

```python
env.reset  # initiate a new episode and return the episode's first observation and info object.
    +- seed=None
    +- options={}
    +- **kwargs
```

```python
env.step  # apply action, do a dt_gym episode time step, and return an observation, reward, terminated, truncated, and info object.
    +- action
    +- **kwargs
```

```python
env.close  # close the gym environment.
    +- **kwargs
```

```python
env.render  # if render_mode is human or rgb_array, then render the image into an 8-bit numpy array.
    +- **kwargs
```

```python
env.verbose_true  # enable standard output verbosity.
```

```python
env.verbose_false  # disable standard output verbosity.
```

## Internal Functions (model specific)
```python
env.get_action_space  # specify type and range of each action parameter.
```

```python
env.get_observation_space  # specify type and range of each observation parameter.
```

```python
env.get_observation  # retrive the current state of the environment as an observation object.
```

```python
env.get_info  # retrive additional info about the current state of the environment as an info dictionary.
```

```python
env.get_terminated  # determine if the target was reached and the episode was terminated.
```

```python
env.get_reset_values  # reset model specific self.variables.
```

```python
env.get_reward  # evaulate the cost function and return a float value.
```

```python
env.get_img  # specify the render image as a matplotlib plot.
```

## Internal Function (generic)
```python
env.get_truncated  # determine if the episode reached the max_time specified in the PhysiCell settings.xml.
```

## Internal Variables

### Time
```python
env.episode  # integer: episode number.
env.step_episode  # integer: step within the episode.
env.step_env  # integer: step within the learning process.
env.time_simulation  # integer: current simulation time in minutes.
```

### Gymnasium rendering
```python
env.render_mode  # None or string (rgb_array or human).
env.metadata  # render_modes  render_fps  (currently all metadata has to do with rendering).
env.figsize  # tuple of x and y in inch.
env.fig   # matplotlib figure object.
```

### Gymnasium spaces
```python
env.action_space  # gymnasium space object.
env.observation_space  # gymnasium space object.
```

### PhysiCell settings xml
```python
env.settingxml  # setting.xml file name xml etree object from the setting xml file.
env.x_tree  # lxml etree ElementTree object form the parsed setting xml file.
env.x_root  # lxml etree Element object for xpath operations on the parsed xml file.
```

### PhysiCell domain
```python
env.x_min  # float
env.x_max  # float
env.y_min  # float
env.y_max  # float
env.z_min  # float
env.z_max  # float
env.dx  # float
env.dy  # float
env.dz  # float
env.width  # float
env.height  # float
env.depth  # float
```

### PhysiCell substrate
```python
env.substrate_to_id  # dictionaty with name string to integer id mapping.
env.substrate_unique  # list of name strings sorted by id.
env.substrate_count  # total number of substrates.
```

### PhysiCell cell_type
```physicell
env.cell_type_to_id  # dictionary with name string to integer id mapping.
env.cell_type_unique  # list of name strings sorted by id.
env.cell_type_count  # total number of cell_types.
env.cell_type_to_color  # dictionary with name string to color string mapping.
```

### Additional keyword arguments
```python
env.kwargs  # dictionary: additional key word arguments given to any of the main functions.
```

### Control standard output verbosity
```python
env.verbose  # boolean
```
