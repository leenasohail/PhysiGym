# library
import os
import subprocess
import time


# const bigred200 run
i_thread=24  # based on node setting

# comile
# 1. pcvenv
# 2. srun -A r00241 -p debug make data-cleanup clean  # gpu-debug debug
# 3. srun -A r00241 -p gpu-debug make  # gpu-debug debug

# generate slurm sbatch scripts
i = 0
for i_repeat in range(128):  # 128
    s_label = f"pgsts{str(i).zfill(3)}"
    print(f"\nprocessing: {s_label}\t{i_repeat}")

    # run command
    #s_cmd = "srun echo 'hallo world!'"
    s_cmd = "srun python3 custom_modules/physigym/physigym/envs/run_physigym_tib_episodes.py config/PhysiCell_settings.xml --max_time 1440.0 --thread 24 --seed none"

    # write slurm batch script (iu bigred200 USER@iu.edu specific)
    ls_script = [
        "#!/bin/bash\n",
        f"#SBATCH -J {s_label}\n",
        "#SBATCH -p general  # gpu-debug debug gpu general \n",
        f"#SBATCH -o {s_label}_%j.out\n",
        f"#SBATCH -e {s_label}_%j.err\n",
        "#SBATCH --mail-type=ALL\n",
        "#SBATCH --mail-user=USER@iu.edu  # specific\n",
        "#SBATCH --nodes=1\n",
        "#SBATCH --ntasks-per-node=1  # 64\n",
        f"#SBATCH --cpus-per-task={i_thread}  # 64\n",
        "#SBATCH --gpus-per-node=0  # 4\n",
        "#SBATCH --time=36:00:00  #hh:mm:ss 48:00:00\n",
        "#SBATCH --mem=64G  # 512G\n",
        "#SBATCH -A r00000  # specific\n",
        "\n",
        "# Load any modules that your program needs\n",
        "module --ignore_cache load python/gpu/3.11.5\n",
        "source /N/slate/USER/.local/lib/pcvenv/bin/activate\n",
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
    f.write(f"{int(time.time())}\t{s_label}\t{i_repeat}\t{s_file}\n")
    f.close()

    # submit script to slurm
    subprocess.run(["sbatch", s_file])

    # finalize
    time.sleep(2)
    i += 1

    # developement
    #break
