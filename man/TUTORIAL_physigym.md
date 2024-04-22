# physigym : Bridging PhysiCell and Gymnasium

Please install the latest version of the physigym user project, as described in the [HowTo](https://github.com/Dante-Berth/PhysiGym/blob/main/man/TUTORIAL_physigym.md) section.


## The most basic run.

```python
import gymnasium
import physigym  # import the Gymnasium PhysiCell bridge module
```

List the registered gymnasium classes.
in this listing you should find below the core classes, that ship with the basic installation, the just installed physigym/ModelPhysiCellEnv-v\* gymnasium enviroment class.

```python
gymnasium.envs.pprint_registry()
```

Let's make an instance the ModelPhysiCellEnv class and do a manual physicell run.
The output should look famililiar to PhysiCell users.
```python
env = gymnasium.make('physigym/ModelPhysiCellEnv-v0')

env.reset()  # initialize PhysiCell run
env.step(action={})  # do one gymnasium time step (similar to mcds timestep)
env.close()  # finalize PhysiCell run
```

And kill the python runtime.
```python
exit()
```

Let's reset the output folder:
```bash
make data-cleanup
```


## A more elaborate example.


This in this somewhat more realistic example we will tries to control the model so that the cell count for the "default" cell type over time stabilizes at about 128 cells.
For observation we will use cell count.
For action we will use a apoptosis inducing drug that kills the cells.


For this tutorial, we assume you have additionally [PhyiCell Studio](https://github.com/PhysiCell-Tools/PhysiCell-Studio) installed.

1. The PhysiCell level (C++ and Studio)

1.1 Fire up studio.

```bash
studio -c config/PhysiCell_settings.xml
```

1.2 In the studio, make the following changes and additions, and don't forget to save.

+ Config Basics: Max Time = 10080 [min] which are 7 [days].
+ Microenvironment: rename my_substrate to drug.
+ Microenvironment: set drug decay rate to 0.001 [1/min].
+ Cell Types / Death: death rate = 0.0 [1/min].
+ Cell Types / Custom Data: delete variable my_variable.
+ Cell Types / Custom Data: add variable Name: apoptosis_rate; Value 0.0; Units [1/min]
+ User Params: have a look at the time and dt_gym parameters!
+ User Params: delete parameters my_str, my_bool, my_int, my_float.
+ User Params: add a parameter drug_conc; Type double; Value 0.0; Units [fraction].
+ User Params: add a parameter cell_count; Type int; Value 1; Unit dimensionless.
+ Rules: drug increases apoptosis; Half-max 0.5; Saturation value: 1.0; Hill power 4; Add rule; enable.
+ File / Save.
+ Studio / Quit.


1.3 Compile and run the model the classic way.

For model development, it is sometimes useful to be able to compile and run the model the old-fashioned way.
In fact, this is the only reason why we kept the orignal main.cpp (which for emedding had to be ported to custom/physicellmodule.cpp) in the physigym code base.

In py3pc_embed we can compile and run the model the old-fashioned way like this:

```bash
make classic
./project
```

Let make clean slate agin, for a next physicell run.
```
rm -r output
mkdir output
```

1.4 Edit the custom_modules/custom.cpp file.

We don't need the custom data vector template.
But we need a function that will update the microenvironment with the drug we add.

At around line 146, delete the custom data vector.

```C++
// add custom data vector
for (int i = 0 ; i < all_cells->size(); i++) {
    std::vector<double> vector_double = VECTOR_ZERO;
    (*all_cells)[i]->custom_data.add_vector_variable("my_vector", vector_double);
}
```

At the bottom of the file, add this function to update the microenvironment.

```C++
int set_microenv(std::string s_substrate, double r_conc) {
    // update substrate concentration
    int k = microenvironment.find_density_index(s_substrate);
    for (unsigned int n=0; n < microenvironment.number_of_voxels(); n++) {
        microenvironment(n)[k] += r_conc;
    }
    return 0;
}
```


1.5 Edit the custom_modules/custom.h header file.

At the bottom of the file, add the fresh implemented function.

```C++
// add substrate
int set_microenv(std::string s_substrate, double r_conc);
```


2. The PhysiCell Python embedding level (C++)

2.1 Edit the custom_modules/embedding/physicellmodule.cpp file.

Parameters, custom variables, and custom vectors are only the interface.
We still have to connect them to something meaningful.
This is done in the custom_modules/embedding/physicellmodule.cpp in the physicell_step function.
Please have a look this function.

At line around 190, you will find already prepeared, commented out example code, for action and observation, for all possible parameter, variable and vector types.

Let's first focus on the action.
The only thig left to do is to connect our drug_conc parameter with the already implemented set_microenv function.
After the commented out action example code, at line 225 insert the following line.

```C++
// add drug
set_microenv("drug", parameters.doubles("drug_conc"));
```

Similar for observation.
We simply have to update our cell_count parameter with the actual cell count.
After the commented out obeservation example code, at around line 270 insert the following line.

```C++
// receive cell count
parameters.ints("cell_count") = (*all_cells).size();
```

For analysis purpose we transmit as well the apoptosis rate over the interface.
```C++
// receive apoptosis rate
for (Cell* pCell : (*all_cells)) {
    pCell->custom_data["apoptosis_rate"] = get_single_behavior(pCell, "apoptosis");
}
```


3. The PhysiCell Gymnasium level (Python3)

Finally, lets update the Gymnasium ModelPhysiCellEnv class, found in the custom_modules/physigym/physigym/envs/physicell_model.py.


3.1 Edit custom_modules/physigym/physigym/envs/physicell_model.py file.

3.1.1 \_get\_action\_space function

Let's decalre the action space.
In the studio we specifed the unit of the drug_conc parameter as fraction.
This means, in Gymnasium terms whe have a Box space.

First, let's comment out the default 'discrete': spaces.Discrete(2) !
Then let's declare a Box space labeled drug_conc
This is a single continuous vector with all possible real values from 0 to 1.

Replace the default with:
```python
action_space = spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float64)
```


3.1.2 \_get\_observation\_space function

We do a similar thing for the observation space.
Based on our classig run we do not exect to have more then 2^14 cells.

Replace the default with:
```python
observation_space = spaces.Box(low=0, high=(2**16 - 1), shape=(), dtype=np.uint16)
```


3.1.3 \_get\_img function

We will now implement a plot, to display drug concentrationa in the domain, as well as all cells, colored by the apoptosis rate.

```
##################
# substrate data #
##################

df_conc = pd.DataFrame(physicell.get_microenv('drug'), columns=['x','y','z','drug'])
df_conc = df_conc.loc[df_conc.z == 0.0, :]
df_mesh = df_conc.pivot(index='y', columns='x', values='drug')
ax.contourf(
    df_mesh.columns, df_mesh.index, df_mesh.values,
    vmin=0.0, vmax=0.2, cmap='Reds',
    #alpha=0.5,
)

######################
# substrate colorbar #
######################

fig.colorbar(
    mappable=cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=0.2), cmap='Reds'),
    label='drug_conc',
    ax=ax,
)

#############
# cell data #
#############

df_cell = pd.DataFrame(physicell.get_cell(), columns=['ID', 'x','y', 'z'])
df_variable = pd.DataFrame(physicell.get_variable("apoptosis_rate"), columns=['apoptosis_rate'])
df_cell = pd.merge(df_cell, df_variable, left_index=True, right_index=True, how='left')
df_cell = df_cell.loc[df_cell.z == 0.0, :]
df_cell.plot(
    kind='scatter', x='x', y='y', c='apoptosis_rate',
    xlim=[
        int(self.x_root.xpath('//domain/x_min')[0].text),
        int(self.x_root.xpath('//domain/x_max')[0].text),
    ],
    ylim=[
        int(self.x_root.xpath('//domain/y_min')[0].text),
        int(self.x_root.xpath('//domain/y_max')[0].text),
    ],
    vmin=0.0, vmax=0.1, cmap='viridis',
    grid=True,
    title=f'dt_gym step {str(self.iteration).zfill(3)}: {df_cell.shape[0]} / 128 [cell]',
    ax=ax,
)
```


3.1.4 \_get\_observation

In our model, the only thing we have to observe is the cell count.
We way we provide this information has to be compatible with the observation space we defined above, in our case a single integer number.

Replace the default `o_observation = {'discrete': True}` with:

```
o_observation = np.array(physicell.get_parameter('cell_count'), dtype=np.uint16)
```

3.1.5 \_get\_info

We coould provide additional information, importand for controlling the action the policy.
For example, if we do reinforcement learning on a [jump and run game](https://c64online.com/c64-games/the-great-giana-sisters/), the number of heart (lives left) from our character.

In our simple model we don't have such information.

So, just leave the default, the empty dictionary.


3.1.6 \get\_terminated

In our model run (epoche) will be terminated, if all cells are dead and the species died out.
Note that it is a huge difference, if the model is terminated (all cells are dead) or is truncated (simply runs out of max time).

Replace the default `b_terminated = False` with:

```
b_terminated = physicell.get_parameter('cell_count') <= 0
```


3.1.7 \_get\_reward

The revard have to be a float number between or equal to 0.0 and 1.0.
Delete the default `r_reward = 0.0`.
Our revard algorythm looks like this:

```
i_cellcount = np.clip(physicell.get_parameter('cell_count'), a_min=0, a_max=256)
if (i_cellcount == 128):
    r_reward == 1
elif (i_cellcount < 128):
    r_reward = i_cellcount / 128
elif (i_cellcount > 128):
    r_reward = (i_cellcount - 128) / 128
else:
    sys.exit('Error @ CorePhysiCellEnv._get_reward : strange clipped cell_count detected {i_cellcount}.')
```


4. Running the model (Python3 and Bash)

4.1 Compile the model.

This necessary, becasue of all the chages we did to the PhysiCell custom.cpp code, the embedding module, and the physigym module.

```bash
make
```

4.2 Python script that will run one epoch of the model.

Open a Pyton shell an execute the following code sequence (or write a Python script that does the same):

```Python
# library
import gymnasium
import physigym  # import the Gymnasium PhysiCell bridge module

# load PhysiCell Gymnasium enviroment
env = gymnasium.make('physigym/ModelPhysiCellEnv-v0')

# reset the enviroment
d_observation, d_info = env.reset()

# episode time step loop
b_episode_over = False
while not b_episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    o_observation, r_reward, b_terminated, b_truncated, d_info = env.step(action)
    b_episode_over = b_terminated or b_truncated

# episode finishing
env.close()

# kill the python runtime.
exit()
```

# BUE: i am here!

```python
# load libraries
import gymnasium
import physigym

# episode initialization
env = gymnasium.make('physigym/CorePhysiCell-v0', render_mode="human")
```

```
import gymnasium
gymnasium.utils.env_checker.check_env

gymnasium.utils.save_video.save_video

gymnasium.utils.performance.benchmark_step
gymnasium.utils.performance.benchmark_init
gymnasium.utils.performance.benchmark_render
```


5. The PhysiCell dataloader for data analysis (Python3 and Bash)

1.9 We can do similar plotting and even more in depth data analysis with the [pcdl](https://github.com/elmbeech/physicelldataloader) library on the dumped data.

Install the [pcdl](https://github.com/elmbeech/physicelldataloader) library.

```bash
pip3 install -U pcdl[all]
```

Open a Python3 shell and use the [pcdl](https://github.com/elmbeech/physicelldataloader) library to do a time series plot.

```python
# library
import matplotlib.pyplot as plt
import os
import pcdl

# load time series
mcdsts = pcdl.TimeSeries('output/')

# make scatter and contour plot overlays
os.makedirs('output/ctrl_pcdl/', exist_ok=True)
for i, mcds in enumerate(mcdsts.l_mcds):
    fig, ax = plt.subplots(figsize=(9,6))
    fig.suptitle(f'drug controlled time series: {int(mcds.get_time())}[min] {mcds.get_cell_df().shape[0]}[cell]')
    ax.axis('equal')
    mcds.plot_contour('drug', vmin=0.0, vmax=0.2, cmap='Reds', ax=ax)
    mcds.plot_scatter('apoptosis_rate', z_axis=[0.0,0.01], cmap='viridis', ax=ax)
    plt.tight_layout()
    fig.savefig(f'output/ctrl_pcdl/drug_ctrld_timeseries_time{str(int(mcds.get_time())).zfill(8)}min.jpeg', facecolor='white')
    plt.close()

# make a gif from the plots
mcdsts.make_gif('output/ctrl_pcdl/', interface='jpeg')

# additionally, make time series plots
mcdsts.plot_timeseries(title='total cell count over time', ext='jpeg')
mcdsts.plot_timeseries(focus_cat='cell_type', focus_num='apoptosis_rate', frame='cell_df', title='mean apoptosis rate over time', ext='jpeg')
mcdsts.plot_timeseries(focus_num='drug', frame='conc_df', title='mean drug concentration over time', ext='jpeg')
```

