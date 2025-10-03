# library
import os
import subprocess
import time


# const default
s_settingxml="config/PhysiCell_settings.xml"

# const bigred200 run
i_thread=24  # based on node setting
b_gpu="true"  # based on node setting
s_name="sac_bigred200"
b_wandb="true"
s_entity="corporate-manu-sureli"  # name of the project in wandb
s_render_mode="none"
i_seed = "none"
r_max_time_episode=12900.0  # 8[d]=12900.0[min] development 1440.0
i_total_step_learn=1000000  # 1000000 development 8 (3[step/day])
i_tumor=512
i_cell_1=128

# variable bigred200 run (9[observation_mode] * 4[init_mode] * 3[seeding] = 108[run] * 3[repeat] = 324[run])
ls_observation_mode = [
    "scalars_cells","scalars_substrates","scalars_cells_substrates",
    "img_mc_cells","img_mc_substrates","img_mc_cells_substrates",
    "img_rgb",
    "graph_delaunay","graph_neighbor"
]
ls_init_mode = ["random_mode","circular_mode","hex_mode","robust"]
lr_cell_2_fraction = [0.0,0.5,1.0]

# generate slurm sbatch scripts
i = 0
for i_repeat in range(1):
    for s_observation_mode in ls_observation_mode[:]:  # 9
        for s_init_mode in ls_init_mode[:]:  # 4
            for r_cell_2_fraction in lr_cell_2_fraction[:]:  # 3
                s_label = f"sts{str(i).zfill(3)}"
                print(f"processing: {s_label}\t{i_repeat}\t{s_observation_mode}\t{s_init_mode}\t{r_cell_2_fraction}")

                # run command
                #s_cmd = "srun echo 'hallo world!'"
                #s_cmd = "srun python3 custom_modules/physigym/physigym/envs/run_physigym_tib.py"
                #s_cmd = f"srun python3 custom_modules/physigym/physigym/envs/run_physigym_tib_sac.py --name sac_bigred200 --gpu {b_gpu} --wandb true --max_time_episode {r_max_time_episode} --total_step_learn {i_total_step_learn}"
                s_cmd = f"srun python3 custom_modules/physigym/physigym/envs/run_physigym_tib_sac.py --gpu {b_gpu} --name sac_bigred200 --wandb true --entity {s_entity} --settingxml {s_settingxml} --render_mode {s_render_mode} --seed {i_seed} --max_time_episode={r_max_time_episode} --total_step_learn={i_total_step_learn} --max_time_episode {r_max_time_episode} --total_step_learn {i_total_step_learn} --observation_mode {s_observation_mode} --init_mode {s_init_mode} --tumor {i_tumor} --cell_1 {i_cell_1} --cell_2_fraction {r_cell_2_fraction}"

                # write script
                ls_script = [
                    "#!/bin/bash\n",
                    f"#SBATCH -J {s_label}\n",
                    "#SBATCH -p gpu  # gpu-debug gpu general\n",
                    f"#SBATCH -o {s_label}_%j.out\n",
                    f"#SBATCH -e {s_label}_%j.err\n",
                    "#SBATCH --mail-type=ALL\n",
                    "#SBATCH --mail-user=me@iu.edu  # user specific\n",
                    "#SBATCH --nodes=1\n",
                    "#SBATCH --ntasks-per-node=1  # 64\n",
                    f"#SBATCH --cpus-per-task={i_thread}  # 64\n",
                    "#SBATCH --gpus-per-node=1  # 4\n",
                    "#SBATCH --time=46:00:00  #hh:mm:ss 48:00:00\n",
                    "#SBATCH --mem=64G  # 512G\n",
                    "#SBATCH -A r00000  # user specific\n",
                    "\n",
                    "# Load any modules that your program needs\n",
                    "module --ignore_cache load python/gpu/3.11.5\n",
                    "source /N/slate/me/.local/lib/pcvenv/bin/activate\n",
                    "\n",
                    "# Run your program\n",
                    f"{s_cmd}\n",
                ]
                s_file = f"{s_label}.sbatch"
                f = open(s_file, "w")
                f.writelines(ls_script)
                f.close()

                # track script
                f = open("slurm_job_tracker.tsv", "a")
                f.write(f"{int(time.time())}\t{s_label}\t{s_observation_mode}\t{s_init_mode}\t{r_cell_2_fraction}\t{i_repeat}\t{s_file}\n")
                f.close()

                # submit script to slurm
                subprocess.run(["sbatch", s_file])

                # finalize
                time.sleep(2)
                i += 1

                # developement
                #break
            #break
        #break
    #break
