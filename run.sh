#!/bin/bash
# AirPointer launch script

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Default to custom1 if no model specified
MODEL=${1:-custom1}

# Launch with the specified model
/Users/vedantkhairnar/Documents/RIT/DL/FinalProject/.venv/bin/python main_advanced.py --model "$MODEL"
