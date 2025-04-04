#!/bin/bash

# Number of instances to launch
NUM_INSTANCES_IMAGES=1
# Path to the Python file
SCRIPT_PATH="rl/sac_complex_tme.py"


for ((i=1; i<=NUM_INSTANCES_IMAGES; i++)); do
    exp_name="sac_image_complex_tme_seed_${i}"
    echo "Launching instance $i with seed $i..."
    sleep 2
    nohup python "$SCRIPT_PATH" \
        --seed "$i" \
        --observation_type "image_gray" \
        --name "$exp_name" > "output_image_complex_tme_$i.log" 2>&1 &
    echo "Instance $i launched with PID $!"
    sleep 2
done


