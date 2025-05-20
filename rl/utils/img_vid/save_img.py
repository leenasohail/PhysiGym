import os
import matplotlib.pyplot as plt
import numpy as np


def saving_img(
    image_folder: str,
    info: dict,
    step_episode: int,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    saving_title: str = "output_simulation_image_episode",
    color_mapping: dict = {},
):
    # Helper function to convert RGB tuple to hexadecimal color string
    def rgb_to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255) if rgb[0] <= 1 else int(rgb[0]),
            int(rgb[1] * 255) if rgb[1] <= 1 else int(rgb[1]),
            int(rgb[2] * 255) if rgb[2] <= 1 else int(rgb[2]),
        )

    # Ensure the directory to save images exists
    os.makedirs(image_folder, exist_ok=True)

    # Extract cell data and cancer cell count from the info dictionary
    df_cell = info["df_cell"]
    count_cancer_cell = info["number_cancer_cells"]

    # Create a subplot layout: 1 row, 3 columns (main plot + 2 sidebars)
    fig, ax = plt.subplots(
        1, 3, figsize=(10, 6), gridspec_kw={"width_ratios": [1, 0.2, 0.2]}
    )

    # Get list of unique cell types from the DataFrame
    unique_cell_types = df_cell["type"].unique().tolist()

    # Loop over each cell type to plot them with a different color
    for cell_type in unique_cell_types:
        tuple_color = color_mapping[cell_type]  # Get RGB color tuple for this type
        # Filter alive cells of the current type
        df_celltype = df_cell.loc[
            (df_cell.dead == 0.0) & (df_cell.type == cell_type), :
        ]
        # Plot the cells as scatter points
        df_celltype.plot(
            kind="scatter",
            x="x",
            y="y",
            c=rgb_to_hex(tuple_color),  # Convert color to hex format
            xlim=[x_min, x_max],
            ylim=[y_min, y_max],
            grid=True,
            label=cell_type,
            s=100,  # Marker size
            title=f"episode step {str(step_episode).zfill(3)}, cancer cell: {count_cancer_cell}",
            ax=ax[0],  # Main plot area
        ).legend(loc="lower left")

    # Define the colors to use for the two drug bars
    list_colors = ["royalblue", "darkorange"]

    # Inner function to draw a fluid-level bar to represent drug amount
    def create_fluid_bar(ax_bar, drug_amount, title, max_amount=1, color="cyan"):
        ax_bar.set_xlim(0, 1)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_title(title, fontsize=10)
        ax_bar.set_xticks([])
        ax_bar.set_yticks(np.linspace(0, 1, 5))  # Y ticks at 0%, 25%, 50%, 75%, 100%

        # Normalize the drug amount
        fill_level = drug_amount / max_amount

        # Fill the area up to the current level with the drug color
        ax_bar.fill_betweenx(np.linspace(0, fill_level, 100), 0, 1, color=color)

        # Draw container box
        ax_bar.spines["left"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.spines["top"].set_visible(True)
        ax_bar.spines["bottom"].set_visible(True)

    # Draw a drug bar for each drug in the action dictionary
    action = info["action"]
    for i, (key, value) in enumerate(
        action.items(), start=1
    ):  # Start from subplot index 1
        create_fluid_bar(ax[i], value[0], f"drug_{key}_{i}", color=list_colors[i - 1])

    # Save the entire figure to file
    plt.savefig(image_folder + f"/{saving_title} step {str(step_episode).zfill(3)}")
    plt.close(fig)  # Close the figure to free memory
