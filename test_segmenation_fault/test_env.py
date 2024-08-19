import numpy as np
import os
import gymnasium as gym
from embedding import physicell
import shutil
import physigym
import pandas as pd

#############################
# copy tutorial model files #
#############################
os.chdir("../PhysiGym")
print("\nTUTORIAL: copy tutorial config files ...")
shutil.copyfile(
    "test_segmenation/config/PhysiCell_settings.xml",
    "../PhysiCell/config/PhysiCell_settings.xml",
)
shutil.copyfile(
    "test_segmenation/config/cell_rules.csv",
    "../PhysiCell/config/cell_rules.csv",
)
shutil.copyfile(
    "test_segmenation/config/cells.csv",
    "../PhysiCell/config/cells.csv",
)

os.chdir("../PhysiCell")

env = gym.make(
    "physigym/ModelPhysiCellEnv-v0",
    # settingxml='config/PhysiCell_settings.xml',
    # render_mode='rgb_array',
    # render_fps=10
)


"""
import plotly.graph_objs as go
from plotly.subplots import make_subplots

fig = go.Figure()
"""
maximum = 100000
for i in range(1, maximum):
    list_nb_cells = []
    list_drug_dose = []
    print("###############")
    print(f"Iteration:{i}")
    print("###############")
    env.reset()
    dose = 0
    list_drug_dose.append(dose)
    drug_dose = {"drug_dose": np.array([dose])}
    env.step(drug_dose)
    df_cell = pd.DataFrame(
        physicell.get_cell(), columns=["ID", "x", "y", "z", "death.dead"]
    )
    list_nb_cells.append(len(df_cell[df_cell["death.dead"] == 0]))

    dose = i / maximum
    # First 30 steps with a constant dose
    for _ in range(60):
        list_drug_dose.append(dose)
        drug_dose = {"drug_dose": np.array([dose])}
        env.step(drug_dose)
        df_cell = pd.DataFrame(
            physicell.get_cell(), columns=["ID", "x", "y", "z", "death.dead"]
        )

        list_nb_cells.append(len(df_cell[df_cell["death.dead"] == 0]))
"""
    # Add a 3D trace to the plot for this particular dose
    fig.add_trace(
        go.Scatter3d(
            x=list(range(1, 41)),
            y=list_nb_cells,
            z=list_drug_dose,  # Assuming 40 total time steps
            mode="lines",
            name=f"Dose {i/maximum:.4f}",
        )
    )

# Adding labels and title
fig.update_layout(
    title="3D Plot of Cell Counts Over Time with Varying Drug Doses",
    scene=dict(
        xaxis_title="Time Step", yaxis_title="Number of Cells", zaxis_title="Drug Dose"
    ),
    legend_title="Initial Dose",
)

# Display the interactive 3D plot
fig.show()
"""
