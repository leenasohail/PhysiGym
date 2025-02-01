#!/bin/bash

# Number of instances to launch
NUM_INSTANCES=5

# Path to the Python file
SCRIPT_PATH="rl/sac_tme_model.py"

# Loop to launch instances
for i in $(seq 1 $NUM_INSTANCES); do
    echo "Launching instance $i with seed $i..."
    nohup python "$SCRIPT_PATH" --seed "$i" > "output_$i.log" 2>&1 &
    echo "Instance $i launched with PID $!"
    sleep 2
done

echo "All $NUM_INSTANCES instances have been launched."
