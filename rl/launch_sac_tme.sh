#!/bin/bash

# Number of instances to launch
NUM_INSTANCES_IMAGES=4
# Path to the Python file
SCRIPT_PATH="rl/sac_tme_model.py"


for ((i=1; i<=NUM_INSTANCES_IMAGES; i++)); do
    exp_name="sac_image_seed_${i}"
    echo "Launching instance $i with seed $i..."
    sleep 2
    nohup python "$SCRIPT_PATH" \
        --seed "$i" \
        --observation_type "image" \
        --name "$exp_name" > "output_image_$i.log" 2>&1 &
    echo "Instance $i launched with PID $!"
    sleep 2
done


