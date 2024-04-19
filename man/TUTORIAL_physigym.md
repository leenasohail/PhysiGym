# py3pc_embedding : Embedding PhysiCell into a Python module

Please install the latest version of the py3pc_embed user project, as described in the [HowTo](https://github.com/elmbeech/physicellembedding?tab=readme-ov-file#how-to-install-the-py3pc_embed-physicell-user-project) section.


## Caution: there is no memory reset between PhysiCell runs!

If you do more than one Physicell run in series, the PhysiCell variables will not reset.
In the following section, we take a look at the problem and discuss some workarounds.
In the near future, we will tackle this issue by the root and implement a proper reset function straight in PhysiCell, so that these workarounds no longer are needed.


### 1. In the Python3 REPL let's do a series of PhysiCell runs and study what happens ...

1.1 Do a minimal but complete PhysiCell run.

```python
from embedding import physicell  # import the PhysiCell module

physicell.start()
physicell.step()
physicell.stop()
```


1.2 Move the output folder.

```python
import os
import shutil

shutil.move('output', 'output01')
os.makedirs('output', exist_ok=True)
```


1.3 Do another PhysiCell run ...

```python
from embedding import physicell  # import the PhysiCell module

physicell.start()
physicell.step()
```

**Dang! Wait. Did you realize, the number of agents has not reset?**

The **problem** is, that even if we run physicell.stop() to finalize a PhysiCell run, the variables in PhysiCell, the variables in the C++ extension module, are not reset to their initialization values.
Though the variables are reset, when we leave the PhysiCell main function by exiting the Python runtime.


1.4 Let's reset the output folder:

```bash
rm -r output01
make data-cleanup
```


### 2. Python3 REPL workaround.

A possibility to deal with the PhysiCell rest issue interactively is to start a fresh Python shell as a child process for each PhysiCell run and exit() the shell, when the run is complete, without leaving the Python shell that runs as the mother process.


2.1 Open a python shell, and execute these command blocks one by one.

```python
import os
import shutil
import subprocess
```

```python
subprocess.run(['python3'])  # from the running interactive python shell start a fresh interactive subprocess python shell
```

```python
from embedding import physicell  # import the PhysiCell module
physicell.start()
physicell.step()
physicell.stop()
exit()  # exit the subprocess python shell
```

```python
shutil.move('output', 'output001')
os.makedirs('output', exist_ok=True)
```

```python
subprocess.run(['python3']) # from the running interactive python shell start a fresh interactive python shell
```

``` python
from embedding import physicell  # import the PhysiCell module
physicell.start()
physicell.step()
physicell.stop()
exit()  # exit the subprocess python shell
```

``` python
shutil.move('output', 'output002')
os.makedirs('output', exist_ok=True)
```

This will work, when you run the command blocks one by one.
However, this will not work as a Python3 script because the subprocess Python3 shell is interactive.


2.2 Let's reset the output folder:

```bash
rm -r output*
make data-cleanup
```


### 3. Python3 script workaround ~ by running a subprocess code file.

A scriptable possibility to deal with the PhysiCell reset issue is to write a sole script file for the subprocess.


3.1 First, let's write the subprocess python script.

```bash
echo "from embedding import physicell;physicell.start();physicell.step();physicell.stop()" > custom_modules/subprocess_code.py
```


3.2 Now, let's run this script as a subprocess.

```python
import os
import shutil
import subprocess

for i in range(2):
    subprocess.run(['python3', 'custom_modules/subprocess_code.py'], check=True)

    shutil.move('output', f'output{str(i).zfill(3)}')
    os.makedirs('output', exist_ok=True)
```


3.3 Let's reset the output folder:

```bash
rm -r output*
make data-cleanup
```


### 4. Python3 script workaround ~ by running a subprocess code string.

If we use the lower level function Popen instead of run, we can even run a subprocess without writing the code to a separate file.


4.1 This is the all-in-one Python3 script:

```python
import os
import shutil
import subprocess

s_code = b"""
from embedding import physicell
physicell.start()
physicell.step()
physicell.stop()
"""

for i in range(2):
    p = subprocess.Popen(['python3'], stdin=subprocess.PIPE)
    p.stdin.write(s_code)
    p.stdin.close()
    p.communicate()

    shutil.move('output', f'output{str(i).zfill(3)}')
    os.makedirs('output', exist_ok=True)
```


4.2 Let's reset the output folder:

```bash
rm -r output*
make data-cleanup
```

+ If you like to learn more about subprocess, there is a very nice Real Python tutorial: https://realpython.com/python-subprocess/
+ Additionally, check out the official Python documentation: https://docs.python.org/3/library/subprocess.html#module-subprocess


### 1. Gymnamsiym enviroment workaround

```bash
sed -i 's/cp .\/user_projects\/$(PROJ)\/custom_modules\//cp -r .\/user_projects\/$(PROJ)\/custom_modules\//' ./Makefile
make load PROJ=physigym
```


## A more elaborate example.

For this somewhat more realistic tutorial, we assume you have additionally [PhyiCell Studio](https://github.com/PhysiCell-Tools/PhysiCell-Studio) installed.


1.0 Fire up studio.

```bash
studio -p
```


1.1 In the studio, make the following changes and additionsi, and don't forget to save.

+ Config Basics: Max Time = 10080 [min] which are 7 [days].
+ Microenvironment: rename my_substrate to drug.
+ Microenvironment: set drug decay rate to 0.001 [1/min].
+ Cell Types / Death: death rate = 0 [1/min].
+ Cell Types / Custom Data: delete variable my_variable.
+ Cell Types / Custom Data: add variable Name: apoptosis_rate; Value 0.0.
+ User Params: delete parameters my_float, my_int, my_bool, my_str.
+ User Params: add a parameter drug_conc; Type double; Value 0.0; Units fraction.
+ User Params: add a parameter cell_count; Type int; Value 0; Unit dimensionless.
+ Rules: drug increases apoptosis; Half-max 0.5; Saturation value: 1.0; Hill power 4; Add rule; Enable.
+ File / Save.
+ Studio / Quit.


1.2 Compile and run the model the classic way.

For model development, it is sometimes useful to be able to compile and run the model the old-fashioned way.
In fact, this is the only reason why we kept the orignal main.cpp (which for emedding had to be ported to custom/physicellmodule.cpp) in the py3pc_embed code base. 

In py3pc_embed we can compile and run the model the old-fashioned way like this:

```bash
make classic
./project
```


1.3 Edit the custom_modules/custom.cpp file.

Parameters, custom variables, and custom vectors are only the interface.
We still have to connect the custom variables to something meaningful.

At around line 150, delete the custom data vector.

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


1.4 Edit the custom_modules/custom.h header file.

At the bottom of the file, add the fresh implemented function.

```C++
int set_microenv(std::string s_substrate, double r_conc);
```


1.5 Edit the custom_modules/physicellmodule.cpp.

In the physicell_step function, we have to add code to interact with our parameter-custom-variable-custom-vector-interface.

The drug concentration should be based on the cell count, right before we drug.
To get this working, we need a global flag variables for drugging.
We can define this variable right before the physicell_step function (line 137).

```C++
bool b_drug = false;
```

We want to drug every 12 [h] \(720 [min]).
Because this is not in the phenotype (6 [min]), mechanics (0.1 [min]), or diffusion (0.01[min]) time step scale, we will set custom_dt to 720 [min] \(around line 147).

```C++
double custom_dt = 720; // min
```

Then place the following code as the "on custom time step" block (around line 188 onwards).

```C++
// administer drug
if (b_drug) {
    std::cout << "administer drug ... " << std::endl;
    // update drug concentration
    set_microenv("drug", parameters.doubles("drug_conc"));
    // set flag
    b_drug = false;
}

// on custom time step
if (custom_countdown <= 0) {
    custom_countdown += custom_dt;

    // Put custom timescale action here!
    std::cout << "processing custom time step action ... " << std::endl;
    // extract cell_count
    parameters.ints("cell_count") = (*all_cells).size();
    // extract apoptosis rate
    for (Cell* pCell : (*all_cells)) {
        pCell->custom_data["apoptosis_rate"] = get_single_behavior(pCell, "apoptosis");
    }
    // set flag
    b_drug = true;
    step = false;
}
```


1.6 Compile the model.

```bash
make
```


1.7 Open a Python3 shell and run the model.

This python code tries to control the model so that the cell count over time stabilizes at about 128 cells.

```python
# library
from embedding import physicell

# set variables
i_cell_target = 128

# start
physicell.start()

# step loop
for i in range(int(10080 / 720)):
    # extract data
    i_cell_count = physicell.get_parameter('cell_count')

    #  policy
    if (i_cell_count > i_cell_target):
        r_drug_conc = 0.1
    else:
        r_drug_conc = 0.0

    # action
    print(f'set drug_conc to: {r_drug_conc}')
    physicell.set_parameter('drug_conc', r_drug_conc)
    physicell.step()

# stop
physicell.stop()

# leave the python shell
exit()
```


1.8 Run the model again, this time additionally generating each custom time step an image, making therefor use of pandas and matplotlib.

```python
# library
from embedding import physicell
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import os
import pandas as pd

# set variables
i_cell_target = 128

# start
physicell.start()

# step loop
os.makedirs('output/ctrl_embed/', exist_ok=True)
for i in range(int(10080 / 720)):
    # extract cell count data
    i_cell_count = physicell.get_parameter('cell_count')

    # extract cell data
    df_cell = pd.DataFrame(physicell.get_cell(), columns=['ID', 'x','y', 'z'])
    df_apoptosis =  pd.DataFrame(physicell.get_variable("apoptosis_rate"), columns=['apoptosis_rate'])
    df_cell = pd.merge(df_cell, df_apoptosis, left_index=True, right_index=True, how='left')

    # extract substrate data
    df_conc = pd.DataFrame(physicell.get_microenv("drug"), columns=['x','y','z','drug'])

    # generate plot
    fig, ax = plt.subplots(figsize=(9,6))
    ax.axis('equal')
    # substrate data
    df_mesh = df_conc.pivot(index='y', columns='x', values='drug')
    ax.contourf(
        df_mesh.columns, df_mesh.index, df_mesh.values,
        vmin=0.0, vmax=0.2, cmap='Reds',
        #alpha=0.5,
    )
    fig.colorbar(
        mappable=cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=0.2), cmap='Reds'),
        label='drug',
        ax=ax,
    )
    # cell data
    df_cell.plot(
        kind='scatter', x='x', y='y', c='apoptosis_rate',
        xlim=[-500,500],
        ylim=[-500,500],
        vmin=0.0, vmax=0.01, cmap='viridis',
        grid=True,
        title=f'drug controlled time series step {str(i).zfill(3)}: {i_cell_count}/{i_cell_target} [cell]',
        ax=ax,
    )
    # save to file
    plt.tight_layout()
    fig.savefig(f'output/ctrl_embed/drug_ctrld_timeseries_step{str(i).zfill(3)}.jpeg', facecolor='white')
    plt.close()

    # policy
    if (i_cell_count > i_cell_target):
        r_drug_conc = 0.1
    else:
        r_drug_conc = 0.0
    print(f'set drug_conc to: {r_drug_conc}')

    # action
    physicell.set_parameter('drug_conc', r_drug_conc)
    physicell.step()

# stop
physicell.stop()

# leave the python shell
exit()
```


1.9 We can do similar plotting and even more in depth data analysis with the [pcdl](https://github.com/elmbeech/physicelldataloader) library on the dump data.

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

1.10  Run the model again, this time from the physigym layer.

write for physicell model, write a gym enviroment called PhysiCell-v0, making therefore use of the physigym/envs/ CorePhysiCell class


Load the physicell model into a gymnasium environment and run it.

```python
# load libraries
import gymnasium
import physigym

# episode initialization
env = gymnasium.make('physigym/CorePhysiCell-v0', render_mode="human")
d_observation, d_info = env.reset()

# episode time step loop
b_episode_over = False
while not b_episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    o_observation, r_reward, b_terminated, b_truncated, d_info = env.step(action)
    b_episode_over = b_terminated or b_truncated

# episode finishing
env.close()
```

```
import gymnasium
gymnasium.utils.env_checker.check_env

gymnasium.utils.save_video.save_video

gymnasium.utils.performance.benchmark_step
gymnasium.utils.performance.benchmark_init
gymnasium.utils.performance.benchmark_render


```

