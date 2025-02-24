#!/bin/bash

SCRIPT_PATH="rl/sb/stable_baselines.py"
ALGO_NAME="SAC"
NUM_INSTANCES=3
NAME="${ALGO_NAME}_sb"

for i in $(seq 1 $NUM_INSTANCES); do
    # Replace 0 by 255 (unclear in your script, so removed it)
    nohup python "$SCRIPT_PATH" --algo_name "$ALGO_NAME" --seed "$i" > "${NAME}_${i}.log" 2>&1 &
    echo "Instance $i launched with PID $!"
    sleep 10
done
