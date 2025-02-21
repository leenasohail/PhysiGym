#!/bin/bash

SCRIPT_PATH="rl/sb/sb_hyperopt_own.py"
ALGO="tqc"
NUM_INSTANCES=2
NAME="${ALGO}_sb_hyperopt_own"

for i in $(seq 1 $NUM_INSTANCES); do
    # Replace 0 by 255 (unclear in your script, so removed it)
    nohup python "$SCRIPT_PATH" --algo "$ALGO" > "${NAME}_${i}.log" 2>&1 &
    echo "Instance $i launched with PID $!"
    sleep 10
done
