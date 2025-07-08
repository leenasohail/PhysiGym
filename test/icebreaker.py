####
# title: icebreaker.py
#
# language: python3
# author: Elmar Bucher
# date: 2024-05-07
# license: BSD 3-Clause
#
# run:
#     python3 test/test_episode.py
#
# description:
#     unit test code for the physigym project
#####

# bue 20250123: icebreaker have to fetch threads and seed from the xml (and maybe time).

# modules
import argparse
import glob
from lxml import etree
import matplotlib.pyplot as plt
import pcdl

def drift(b_plot=False):

    # processing
    b_settingxml = False

    ddf_cell = {}
    ddf_conc = {}
    for s_path in sorted(glob.glob("output/episode0*")):
        print(f"processing: {s_path} ...")

        # extract episode
        i_episode = int(s_path.replace("output/episode",""))

        # load settingxml
        if not b_settingxml:
            x_tree = etree.parse(f"{s_path}/PhysiCell_settings.xml")
            x_root = x_tree.getroot()
            b_settingxml = True

        # load data
        mcdsts = pcdl.TimeSeries(s_path, settingxml=f"PhysiCell_settings.xml", verbose=False)
        df_cell = mcdsts.get_cell_df().drop({"runtime"}, axis=1)
        df_conc = mcdsts.get_conc_df().drop({"runtime"}, axis=1)
        ddf_conc.update({i_episode: df_conc})
        ddf_cell.update({i_episode: df_cell})

        # plot timeseries
        if b_plot:
            s_file = f"timeseries_plot_episode{str(i_episode).zfill(8)}.png"
            print(f"generate plot: {s_file} ...")
            fig, axs = plt.subplots(nrows=4, ncols=2 ,figsize=(8,12))
            mcdsts.plot_timeseries("cell_type", ax=axs[0,0])
            mcdsts.plot_timeseries(None, "substrate_a", ax=axs[1,0])
            mcdsts.plot_timeseries(None, "substrate_b", ax=axs[2,0])
            mcdsts.plot_timeseries(None, "substrate_c", ax=axs[3,0])
            mcdsts.plot_timeseries("cell_type", "substrate_a", ax=axs[1,1])
            mcdsts.plot_timeseries("cell_type", "substrate_b", ax=axs[2,1])
            mcdsts.plot_timeseries("cell_type", "substrate_c", ax=axs[3,1])
            fig.suptitle(f"timeseries episode {str(i_episode).zfill(8)}")
            plt.tight_layout()
            fig.savefig(s_file)

    # check results
    print(f"checking for conc drift ice ...")
    b_drift_conc = False
    for i_episode in ddf_conc.keys():
        for s_column in  ddf_conc[0].columns:
            if any(ddf_conc[0][s_column] != ddf_conc[i_episode][s_column]):
                b_drift_conc = True

    print(f"checking for cell drift ice ...")
    b_drift_cell = False
    for i_episode in ddf_cell.keys():
        try:
            for s_column in  ddf_cell[0].columns:
                if any(ddf_cell[0][s_column] != ddf_cell[i_episode][s_column]):
                    b_drift_cell = True

        except ValueError:
            b_drift_cell = True
    o_seed = x_root.xpath("//random_seed")[0].text
    print(["seed", [int(o_seed) if o_seed.isdigit() else o_seed][0]])
    print(["thread", int(x_root.xpath("//omp_num_threads")[0].text)])
    print([b_drift_conc, b_drift_cell])
    print(any([b_drift_conc, b_drift_cell]))
    return 0


# run
if __name__ == "__main__":
    print(f"run icebreaker script ...")

    # argv
    parser = argparse.ArgumentParser(
        prog = f"icebreaker",
        description = f"check for drift ice, for an improper reset from one to the next episode.",
    )
    # b_plot
    parser.add_argument(
        "b_plot",
        #type = bool,
        nargs = "?",
        default = "false",
        help = ""
    )

    # parse arguments
    args = parser.parse_args()
    #print(args)

    # processing
    drift(b_plot = args.b_plot.lower() == "true")
