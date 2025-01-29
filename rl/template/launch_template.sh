#!/bin/bash
# Path to the Python file
SCRIPT_PATH="rl/template/template.py"
echo "Launching instance..."
nohup python "$SCRIPT_PATH" > "output_template.log" 2>&1 &

