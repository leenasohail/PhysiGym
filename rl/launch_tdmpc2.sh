#!/bin/bash



# Path to the Python file
SCRIPT_PATH="rl/tdmpc2/tdmpc2_jax/train_physigym.py"

nohup python3 "$SCRIPT_PATH" > "tdmpc2_output.log" 2>&1 &

echo "All $NUM_INSTANCES instances have been launched."
