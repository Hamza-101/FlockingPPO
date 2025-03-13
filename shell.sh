#!/bin/bash
# Run the first script
python3 PPO.py

# Check if the first script terminated successfully
if [ $? -eq 0 ]; then
    # Run the second script if the first one was successful
    python3 PlotAnimationRL.py
else
    echo "First script encountered an error."
fi